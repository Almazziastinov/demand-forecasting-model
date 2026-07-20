from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from pipelines.forecast_publish.production_dataset_refresh import (
    build_allocation_assortment,
    build_uplifted_daily_from_clickhouse_multipliers,
    delete_older_allocation_snapshot_rows,
)
from pipelines.forecast_publish.production_dataset_refresh import (
    create_client_with_retry,
)
from pipelines.forecast_publish.production_dataset_refresh import (
    refresh_weather_features_with_fallback,
)
from pipelines.forecast_publish.production_dataset_refresh import (
    resolve_default_refresh_dates,
)
from src.experiments_v2.build_bakery_weather_features import _enrich_weather


def test_resolve_default_refresh_dates_uses_moscow_business_day() -> None:
    dates = resolve_default_refresh_dates(
        pd.Timestamp("2026-06-10 22:30:00", tz="UTC"),
        timezone="Europe/Moscow",
    )

    assert dates.forecast_start_date == "2026-06-11"
    assert dates.history_end_date == "2026-06-10"


def test_build_allocation_assortment_keeps_all_categories() -> None:
    sales = pd.DataFrame(
        {
            "city": ["Kazan", "Kazan"],
            "bakery_id": [1, 1],
            "product_id": [10, 20],
            "product_name": ["Bread", "Coffee"],
            "category_name": ["Bread", "Hot drinks"],
        }
    )

    result = build_allocation_assortment(sales, valid_from="2026-07-19")

    assert set(result["product_id"]) == {"10", "20"}
    assert set(result["category_name"]) == {"Bread", "Hot drinks"}
    assert result["source"].unique().tolist() == ["recent_sales_window"]
    assert str(result["valid_from"].iloc[0]) == "2026-07-19"


def test_delete_older_allocation_snapshot_rows_is_scoped() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls = []

        def command(self, query, parameters=None):
            self.calls.append((query, parameters))

    client = FakeClient()
    cutoff = pd.Timestamp("2026-07-20 15:55:46.317", tz="UTC")

    delete_older_allocation_snapshot_rows(
        client,
        table="assortment_city_products",
        valid_from="2026-07-19",
        loaded_at_cutoff=cutoff,
    )

    query, parameters = client.calls[0]
    assert "valid_from = {valid_from:Date}" in query
    assert "source in {managed_sources:Array(String)}" in query
    assert "loaded_at < {loaded_at_cutoff:DateTime64(3)}" in query
    assert parameters["valid_from"] == "2026-07-19"
    assert parameters["managed_sources"] == [
        "recent_sales_window",
        "carried_forward_no_recent_sales",
    ]
    assert parameters["loaded_at_cutoff"] == cutoff.to_pydatetime()


def test_delete_older_allocation_snapshot_rows_rejects_naive_cutoff() -> None:
    class FakeClient:
        def command(self, query, parameters=None):
            raise AssertionError("command must not be called")

    with pytest.raises(ValueError, match="timezone-aware"):
        delete_older_allocation_snapshot_rows(
            FakeClient(),
            table="assortment_city_products",
            valid_from="2026-07-19",
            loaded_at_cutoff=pd.Timestamp("2026-07-20 15:55:46.317"),
        )


def test_create_client_with_retry_retries_transient_failure(monkeypatch):
    calls = {"count": 0}

    def factory(env_file):
        calls["count"] += 1
        if calls["count"] == 1:
            raise TimeoutError("clickhouse timeout")
        return {"env_file": env_file}

    monkeypatch.setattr(
        "pipelines.forecast_publish.production_dataset_refresh.time.sleep",
        lambda seconds: None,
    )

    client = create_client_with_retry(
        factory,
        "env",
        attempts=2,
        sleep_seconds=0.01,
    )

    assert client == {"env_file": "env"}
    assert calls["count"] == 2


def test_build_uplifted_daily_from_clickhouse_multipliers() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-10", "2026-06-10"]),
            "bakery_id": [1, 2],
            "bakery_name": ["B1", "B2"],
            "city": ["Kazan", "Kazan"],
            "bakery_sales": [100.0, 80.0],
            "line_amount_sum": [1000.0, 800.0],
            "priced_quantity": [100.0, 80.0],
            "price_x_qty_sum": [1000.0, 800.0],
            "avg_price": [10.0, 10.0],
            "dow": [2, 2],
            "day": [10, 10],
            "month": [6, 6],
            "iso_week": [24, 24],
            "is_weekend": [0, 0],
            "is_month_start": [0, 0],
            "is_month_end": [0, 0],
            "is_payday_week": [0, 0],
        }
    )
    profile = pd.DataFrame(
        {
            "bakery_id": [1, 1, 2, 2],
            "dow": [2, 2, 1, 1],
            "hour": [8, 9, 8, 9],
            "mean_hour_share_norm": [0.25, 0.75, 0.50, 0.50],
        }
    )
    exact_multipliers = pd.DataFrame(
        {
            "bakery_id": [1, 1],
            "dow": [2, 2],
            "hour": [8, 9],
            "sku_uplift_multiplier": [2.0, 1.0],
        }
    )
    fallback_multipliers = pd.DataFrame(
        {
            "bakery_id": [2, 2],
            "hour": [8, 9],
            "sku_uplift_multiplier": [1.5, 1.0],
        }
    )

    uplifted, summary = build_uplifted_daily_from_clickhouse_multipliers(
        daily,
        profile,
        exact_multipliers,
        fallback_multipliers,
    )

    b1 = uplifted[uplifted["bakery_id"] == 1].iloc[0]
    b2 = uplifted[uplifted["bakery_id"] == 2].iloc[0]
    assert round(float(b1["bakery_sales_uplifted"]), 4) == 125.0
    assert round(float(b2["bakery_sales_uplifted"]), 4) == 100.0
    assert summary["uplift_source"] == "clickhouse_uplift_multipliers"


def test_refresh_weather_features_falls_back_to_existing_file(monkeypatch):
    work_dir = Path("tests") / "_tmp_production_dataset_refresh"
    work_dir.mkdir(parents=True, exist_ok=True)
    weather_path = work_dir / "weather.csv"
    pd.DataFrame(
        {
            "date": ["2026-06-10"],
            "city": ["Kazan"],
            "temp_mean": [20.0],
        }
    ).to_csv(weather_path, index=False, encoding="utf-8-sig")
    dataset_path = work_dir / "daily.csv"
    pd.DataFrame(
        {
            "date": ["2026-06-10"],
            "city": ["Kazan"],
        }
    ).to_csv(dataset_path, index=False, encoding="utf-8-sig")

    fake_module = types.ModuleType("src.experiments_v2.build_bakery_weather_features")

    def _raise_fetch(*args, **kwargs):
        raise TimeoutError("openmeteo timeout")

    fake_module.fetch_weather_features = _raise_fetch
    fake_module.infer_weather_request = lambda paths, horizon_days: (
        ["Kazan"],
        "2026-06-10",
        "2026-06-24",
    )
    monkeypatch.setitem(
        sys.modules,
        "src.experiments_v2.build_bakery_weather_features",
        fake_module,
    )

    result = refresh_weather_features_with_fallback(
        dataset_paths=[dataset_path],
        horizon_days=14,
        weather_path=weather_path,
    )

    assert result["weather_status"] == "existing_file_fallback"
    assert result["weather_rows"] == 1
    assert result["weather_start_date"] == "2026-06-10"
    assert result["weather_end_date"] == "2026-06-24"
    assert "openmeteo timeout" in str(result["weather_error"])
    weather_path.unlink()
    dataset_path.unlink()


def test_enrich_weather_flags_heavy_precipitation_as_bad_weather() -> None:
    city = "\u041a\u0430\u0437\u0430\u043d\u044c"
    weather = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-23"]),
            "city": [city],
            "temp_max": [21.5],
            "temp_min": [17.1],
            "temp_mean": [19.4],
            "precipitation": [13.6],
            "rain": [0.0],
            "snowfall": [0.0],
            "windspeed_max": [12.1],
            "weathercode": [95],
        }
    )

    enriched = _enrich_weather(weather)

    assert int(enriched.loc[0, "weather_cat_code"]) == 5
    assert int(enriched.loc[0, "is_rainy"]) == 1
    assert int(enriched.loc[0, "is_bad_weather"]) == 1

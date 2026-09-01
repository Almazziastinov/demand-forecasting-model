import pandas as pd
import pytest

from scripts.backtest_daily_sku_allocation import run_backtest, validate_snapshot


def test_snapshot_rejects_mixed_runs() -> None:
    snapshot = pd.DataFrame(
        [
            {
                "date": "2026-08-01",
                "bakery_id": 1,
                "city": "Казань",
                "category": "Выпечка",
                "product_id": 10,
                "source_run_id": "run_a",
                "incumbent_sku_forecast": 10.0,
            },
            {
                "date": "2026-08-01",
                "bakery_id": 1,
                "city": "Казань",
                "category": "Выпечка",
                "product_id": 20,
                "source_run_id": "run_b",
                "incumbent_sku_forecast": 20.0,
            },
        ]
    )
    with pytest.raises(ValueError, match="mixed source_run_id"):
        validate_snapshot(snapshot)


def test_backtest_preserves_incumbent_category_total() -> None:
    history_rows = []
    for date in pd.date_range("2026-07-01", periods=14):
        for product_id, demand in [(10, 20.0), (20, 80.0)]:
            history_rows.append(
                {
                    "date": date,
                    "bakery_id": 1,
                    "city": "Казань",
                    "category": "Выпечка",
                    "product_id": product_id,
                    "demand_mid": demand,
                }
            )
    panel = pd.DataFrame(
        history_rows
        + [
            {
                "date": "2026-07-15",
                "bakery_id": 1,
                "city": "Казань",
                "category": "Выпечка",
                "product_id": 10,
                "demand_mid": 30.0,
            },
            {
                "date": "2026-07-15",
                "bakery_id": 1,
                "city": "Казань",
                "category": "Выпечка",
                "product_id": 20,
                "demand_mid": 70.0,
            },
        ]
    )
    snapshot = pd.DataFrame(
        [
            {
                "date": "2026-07-15",
                "bakery_id": 1,
                "city": "Казань",
                "category": "Выпечка",
                "product_id": 10,
                "source_run_id": "run_a",
                "incumbent_sku_forecast": 50.0,
            },
            {
                "date": "2026-07-15",
                "bakery_id": 1,
                "city": "Казань",
                "category": "Выпечка",
                "product_id": 20,
                "source_run_id": "run_a",
                "incumbent_sku_forecast": 50.0,
            },
        ]
    )

    rows, summary = run_backtest(snapshot, panel, target_col="demand_mid")

    assert rows["daily_profile_forecast"].sum() == pytest.approx(100.0)
    assert summary["conservation"]["daily_max_abs_delta"] == pytest.approx(0.0)
    assert summary["metrics"]["daily_profile_forecast"]["wape_pct"] < summary["metrics"][
        "incumbent_sku_forecast"
    ]["wape_pct"]

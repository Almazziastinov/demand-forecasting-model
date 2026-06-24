from __future__ import annotations

import pandas as pd

from src.experiments_v2.apply_bakery_profiles_clickhouse import (
    RAW_SALES_LINE_TABLE,
    _build_recent_correction_targets,
    _recent_sales_source_sql,
    fill_missing_bakery_hours,
    filter_by_active_assortment,
    renormalize_hourly_to_bakery_forecast,
)


def test_recent_correction_targets_filter_dead_sku_and_preserve_day_total() -> None:
    hourly = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-02", "2026-05-02"]),
            "dow": [5, 5],
            "bakery_id": [1, 1],
            "hour": [9, 9],
            "product_id": [10, 20],
            "sku_hour_forecast": [60.0, 40.0],
            "source": ["exact", "exact"],
        }
    )
    recent = pd.DataFrame(
        {
            "bakery_id": [1],
            "product_id": [10],
            "recent_qty": [100.0],
            "recent_days_sold": [10],
            "recent_share": [1.0],
        }
    )

    targets = _build_recent_correction_targets(
        hourly,
        recent,
        mode="dead_0d",
    )

    live = targets[targets["product_id"] == 10].iloc[0]
    dead = targets[targets["product_id"] == 20].iloc[0]
    assert live["corrected_daily_forecast"] == 100.0
    assert dead["corrected_daily_forecast"] == 0.0
    assert targets["corrected_daily_forecast"].sum() == 100.0


def test_blend_recent_50_can_add_recent_sku_absent_from_profile() -> None:
    hourly = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-02"]),
            "dow": [5],
            "bakery_id": [1],
            "hour": [9],
            "product_id": [10],
            "sku_hour_forecast": [100.0],
            "source": ["exact"],
        }
    )
    recent = pd.DataFrame(
        {
            "bakery_id": [1, 1],
            "product_id": [10, 30],
            "recent_qty": [50.0, 50.0],
            "recent_days_sold": [10, 10],
            "recent_share": [0.5, 0.5],
        }
    )

    targets = _build_recent_correction_targets(
        hourly,
        recent,
        mode="blend_recent_50",
    )

    existing = targets[targets["product_id"] == 10].iloc[0]
    new = targets[targets["product_id"] == 30].iloc[0]
    assert existing["corrected_daily_forecast"] == 75.0
    assert new["corrected_daily_forecast"] == 25.0
    assert targets["corrected_daily_forecast"].sum() == 100.0


def test_costly_pie_category_recent_correction_cannot_lift_above_base() -> None:
    pie_category = (
        "\u041f\u0438\u0440\u043e\u0433\u0438 "
        "\u0441\u044b\u0442\u043d\u044b\u0435"
    )
    savory_category = (
        "\u0412\u044b\u043f\u0435\u0447\u043a\u0430 "
        "\u0441\u044b\u0442\u043d\u0430\u044f"
    )
    hourly = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-02", "2026-05-02"]),
            "dow": [5, 5],
            "bakery_id": [1, 1],
            "hour": [9, 9],
            "product_id": [10, 20],
            "sku_hour_forecast": [20.0, 80.0],
            "source": ["exact", "exact"],
        }
    )
    recent = pd.DataFrame(
        {
            "bakery_id": [1, 1],
            "product_id": [10, 20],
            "category_name": [pie_category, savory_category],
            "recent_qty": [80.0, 20.0],
            "recent_days_sold": [10, 10],
            "recent_share": [0.8, 0.2],
        }
    )

    targets = _build_recent_correction_targets(
        hourly,
        recent,
        mode="blend_recent_50",
    )

    pie = targets[targets["product_id"] == 10].iloc[0]
    other = targets[targets["product_id"] == 20].iloc[0]
    assert pie["corrected_daily_forecast"] == 20.0
    assert other["corrected_daily_forecast"] == 80.0
    assert targets["corrected_daily_forecast"].sum() == 100.0


def test_costly_pie_category_can_fallback_to_recent_absolute_cap() -> None:
    pie_category = (
        "\u041f\u0438\u0440\u043e\u0433\u0438 "
        "\u0441\u044b\u0442\u043d\u044b\u0435"
    )
    savory_category = (
        "\u0412\u044b\u043f\u0435\u0447\u043a\u0430 "
        "\u0441\u044b\u0442\u043d\u0430\u044f"
    )
    hourly = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-02", "2026-05-02"]),
            "dow": [5, 5],
            "bakery_id": [1, 1],
            "hour": [9, 9],
            "product_id": [10, 20],
            "sku_hour_forecast": [20.0, 80.0],
            "source": ["exact", "exact"],
        }
    )
    recent = pd.DataFrame(
        {
            "bakery_id": [1, 1],
            "product_id": [10, 20],
            "category_name": [pie_category, savory_category],
            "recent_qty": [80.0, 20.0],
            "recent_days_sold": [10, 10],
            "recent_share": [0.8, 0.2],
        }
    )

    targets = _build_recent_correction_targets(
        hourly,
        recent,
        mode="blend_recent_50",
        category_recent_absolute_cap_days=10,
    )

    pie = targets[targets["product_id"] == 10].iloc[0]
    other = targets[targets["product_id"] == 20].iloc[0]
    assert pie["corrected_daily_forecast"] == 8.0
    assert other["corrected_daily_forecast"] == 92.0
    assert targets["corrected_daily_forecast"].sum() == 100.0


def test_costly_pie_category_capped_by_dow_recent_avg() -> None:
    pie_category = "Пироги сытные"
    savory_category = "Выпечка сытная"
    hourly = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-02", "2026-05-02"]),
            "dow": [5, 5],
            "bakery_id": [1, 1],
            "hour": [9, 9],
            "product_id": [10, 20],
            "sku_hour_forecast": [20.0, 80.0],
            "source": ["exact", "exact"],
        }
    )
    recent = pd.DataFrame(
        {
            "bakery_id": [1, 1],
            "product_id": [10, 20],
            "category_name": [pie_category, savory_category],
            "recent_qty": [80.0, 20.0],
            "recent_days_sold": [10, 10],
            "recent_share": [0.8, 0.2],
        }
    )
    # DOW=5 (суббота): среднее за последние 2 субботы = 8 шт для пирога
    recent_daily = pd.DataFrame(
        {
            "bakery_id": [1, 1],
            "product_id": [10, 20],
            "dow": [5, 5],
            "recent_dow_avg_qty": [8.0, 12.0],
        }
    )

    targets = _build_recent_correction_targets(
        hourly,
        recent,
        mode="blend_recent_50",
        recent_daily=recent_daily,
    )

    pie = targets[targets["product_id"] == 10].iloc[0]
    other = targets[targets["product_id"] == 20].iloc[0]
    # пирог capped по DOW avg: min(20, 8) = 8
    assert pie["corrected_daily_forecast"] == 8.0
    # другая категория не трогается
    assert other["corrected_daily_forecast"] == 92.0
    assert targets["corrected_daily_forecast"].sum() == 100.0


def test_runner_city_prior_soft_weekpart_lifts_city_top_runner() -> None:
    hourly = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-02", "2026-05-02"]),
            "dow": [5, 5],
            "bakery_id": [1, 1],
            "hour": [9, 9],
            "product_id": [10, 20],
            "sku_hour_forecast": [20.0, 80.0],
            "source": ["exact", "exact"],
        }
    )
    recent = pd.DataFrame(
        {
            "bakery_id": [1, 1, 2, 2],
            "product_id": [10, 20, 10, 20],
            "city": ["Казань", "Казань", "Казань", "Казань"],
            "product_name": [
                "Треугольник курица безд",
                "Беккен капуста",
                "Треугольник курица безд",
                "Беккен капуста",
            ],
            "category_name": [
                "Выпечка сытная",
                "Выпечка сытная",
                "Выпечка сытная",
                "Выпечка сытная",
            ],
            "recent_qty": [200.0, 800.0, 900.0, 100.0],
            "recent_days_sold": [21, 21, 21, 21],
            "recent_share": [0.20, 0.80, 0.90, 0.10],
        }
    )
    recent_daily = pd.DataFrame(
        {
            "bakery_id": [1, 1],
            "product_id": [10, 20],
            "is_weekend": [1, 1],
            "dow": [5, 5],
            "recent_share_daily_winsor": [0.20, 0.80],
            "recent_share_weekpart_winsor": [0.20, 0.80],
            "recent_weekpart_obs": [8, 8],
        }
    )

    targets = _build_recent_correction_targets(
        hourly,
        recent,
        mode="runner_city_prior_soft_weekpart",
        recent_daily=recent_daily,
    )

    lifted = targets[targets["product_id"] == 10].iloc[0]
    reduced = targets[targets["product_id"] == 20].iloc[0]
    assert round(float(lifted["corrected_daily_forecast"]), 4) > 20.0
    assert round(float(reduced["corrected_daily_forecast"]), 4) < 80.0
    assert round(float(targets["corrected_daily_forecast"].sum()), 4) == 100.0


def test_recent_sales_source_deduplicates_raw_check_lines() -> None:
    source = _recent_sales_source_sql(RAW_SALES_LINE_TABLE).lower()

    assert "select distinct" in source
    assert "svezhar.fct_check_lines" in source
    assert "hex(fcl.cash_event_type)" in source
    assert "fcl.check_date between %(recent_start)s and %(recent_end)s" in source


def test_assortment_renormalization_preserves_bakery_hour_total() -> None:
    sku_hourly = pd.DataFrame(
        {
            "date": ["2026-06-01", "2026-06-01"],
            "dow": [0, 0],
            "bakery_id": [1, 1],
            "hour": [9, 9],
            "product_id": [10, 20],
            "sku_hour_forecast": [30.0, 20.0],
        }
    )
    bakery_hourly = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-01")],
            "bakery_id": [1],
            "hour": [9],
            "bakery_hour_forecast": [100.0],
        }
    )

    normalized, stats = renormalize_hourly_to_bakery_forecast(
        sku_hourly,
        bakery_hourly,
    )

    assert normalized["sku_hour_forecast"].sum() == 100.0
    product_10_forecast = normalized.loc[
        normalized["product_id"].eq(10), "sku_hour_forecast"
    ].iloc[0]
    assert product_10_forecast == 60.0
    assert stats == {"groups_scaled": 1, "groups_without_sku": 0}


def test_missing_hour_uses_same_day_product_shares() -> None:
    sku_hourly = pd.DataFrame(
        {
            "date": ["2026-06-01", "2026-06-01"],
            "dow": [0, 0],
            "bakery_id": [1, 1],
            "hour": [9, 9],
            "product_id": [10, 20],
            "sku_hour_forecast": [30.0, 70.0],
            "source": ["exact", "exact"],
        }
    )
    bakery_hourly = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-01")],
            "dow": [0, 0],
            "bakery_id": [1, 1],
            "hour": [9, 10],
            "bakery_hour_forecast": [100.0, 50.0],
        }
    )

    filled, stats = fill_missing_bakery_hours(sku_hourly, bakery_hourly)
    hour_10 = filled[filled["hour"].eq(10)].sort_values("product_id")

    assert hour_10["sku_hour_forecast"].tolist() == [15.0, 35.0]
    assert stats == {"groups_filled": 1, "groups_unfilled": 0}


def test_missing_bakery_uses_city_hour_product_shares() -> None:
    sku_hourly = pd.DataFrame(
        {
            "date": ["2026-06-01", "2026-06-01"],
            "dow": [0, 0],
            "bakery_id": [1, 1],
            "hour": [9, 9],
            "product_id": [10, 20],
            "sku_hour_forecast": [30.0, 70.0],
            "source": ["exact", "exact"],
        }
    )
    bakery_hourly = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-01")],
            "dow": [0, 0],
            "bakery_id": [1, 2],
            "hour": [9, 9],
            "bakery_hour_forecast": [100.0, 50.0],
        }
    )
    bakery_city = pd.DataFrame(
        {"bakery_id": [1, 2], "city": ["Казань", "Казань"]}
    )

    filled, stats = fill_missing_bakery_hours(
        sku_hourly,
        bakery_hourly,
        bakery_city_lookup=bakery_city,
    )
    bakery_2 = filled[filled["bakery_id"].eq(2)].sort_values("product_id")

    assert bakery_2["sku_hour_forecast"].tolist() == [15.0, 35.0]
    assert stats == {"groups_filled": 1, "groups_unfilled": 0}


def test_missing_bakery_uses_own_recent_product_shares() -> None:
    sku_hourly = pd.DataFrame(
        {
            "date": ["2026-06-01"],
            "dow": [0],
            "bakery_id": [1],
            "hour": [9],
            "product_id": [10],
            "sku_hour_forecast": [100.0],
            "source": ["exact"],
        }
    )
    bakery_hourly = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-01")],
            "dow": [0, 0],
            "bakery_id": [1, 2],
            "hour": [9, 9],
            "bakery_hour_forecast": [100.0, 50.0],
        }
    )
    recent = pd.DataFrame(
        {
            "bakery_id": [2, 2],
            "product_id": [30, 40],
            "recent_share": [0.25, 0.75],
        }
    )

    filled, stats = fill_missing_bakery_hours(
        sku_hourly,
        bakery_hourly,
        recent_product_weights=recent,
    )
    bakery_2 = filled[filled["bakery_id"].eq(2)].sort_values("product_id")

    assert bakery_2["sku_hour_forecast"].tolist() == [12.5, 37.5]
    assert bakery_2["source"].unique().tolist() == [
        "assortment_recent_bakery_fallback"
    ]
    assert stats == {"groups_filled": 1, "groups_unfilled": 0}


def test_missing_bakery_uses_network_hour_product_shares_as_last_resort() -> None:
    sku_hourly = pd.DataFrame(
        {
            "date": ["2026-06-01", "2026-06-01"],
            "dow": [0, 0],
            "bakery_id": [1, 1],
            "hour": [9, 9],
            "product_id": [10, 20],
            "sku_hour_forecast": [30.0, 70.0],
            "source": ["exact", "exact"],
        }
    )
    bakery_hourly = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-01")],
            "dow": [0, 0],
            "bakery_id": [1, 2],
            "hour": [9, 9],
            "bakery_hour_forecast": [100.0, 50.0],
        }
    )

    filled, stats = fill_missing_bakery_hours(sku_hourly, bakery_hourly)
    bakery_2 = filled[filled["bakery_id"].eq(2)].sort_values("product_id")

    assert bakery_2["sku_hour_forecast"].tolist() == [15.0, 35.0]
    assert bakery_2["source"].unique().tolist() == [
        "assortment_network_hour_fallback"
    ]
    assert stats == {"groups_filled": 1, "groups_unfilled": 0}


def test_assortment_filter_keeps_unconfigured_city() -> None:
    hourly = pd.DataFrame(
        {
            "date": ["2026-06-01"],
            "dow": [0],
            "bakery_id": [2],
            "hour": [9],
            "product_id": [999],
            "sku_hour_forecast": [10.0],
        }
    )
    allowed = pd.DataFrame({"city": ["Казань"], "product_id": [1]})
    bakery_city = pd.DataFrame({"bakery_id": [2], "city": ["Иркутск"]})

    filtered, stats = filter_by_active_assortment(
        hourly,
        allowed_pairs=allowed,
        bakery_city_lookup=bakery_city,
        forecast_col="sku_hour_forecast",
    )

    assert filtered["product_id"].tolist() == [999]
    assert stats == {"rows_removed": 0, "forecast_removed": 0.0}

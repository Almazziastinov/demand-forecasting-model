from __future__ import annotations

import pandas as pd

from src.experiments_v2.apply_bakery_profiles_clickhouse import (
    RAW_SALES_LINE_TABLE,
    _build_recent_correction_targets,
    _recent_sales_source_sql,
    apply_hierarchical_haircut,
    build_hierarchical_haircut_coefficients,
    cap_sku_uplift_per_sku,
    cap_sku_uplift_to_bakery_forecast,
    compensate_for_assortment_exclusion,
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


def test_compensate_for_assortment_exclusion_redistributes_dropped_demand() -> None:
    # product 3 was excluded by assortment filtering (present pre-filter,
    # absent post-filter) — its 15.0 units should land proportionally on
    # products 1 and 2, not vanish.
    pre_filter = pd.DataFrame(
        {
            "date": ["2026-07-08"] * 3,
            "bakery_id": [257, 257, 257],
            "hour": [9, 9, 9],
            "product_id": [1, 2, 3],
            "sku_hour_forecast": [10.0, 5.0, 15.0],
        }
    )
    post_filter = pre_filter[pre_filter["product_id"] != 3].copy()

    compensated, stats = compensate_for_assortment_exclusion(
        pre_filter,
        post_filter,
        group_keys=["date", "bakery_id", "hour"],
        forecast_col="sku_hour_forecast",
    )

    assert stats == {"groups_scaled": 1, "groups_without_remaining_rows": 0}
    result = dict(zip(compensated["product_id"], compensated["sku_hour_forecast"]))
    assert result[1] == 20.0  # 10 * (30/15)
    assert result[2] == 10.0  # 5 * (30/15)
    # full pre-filter total preserved, not just cancelled by the filter
    assert round(compensated["sku_hour_forecast"].sum(), 6) == 30.0


def test_compensate_for_assortment_exclusion_is_noop_when_nothing_removed() -> None:
    pre_filter = pd.DataFrame(
        {
            "date": ["2026-07-08"],
            "bakery_id": [257],
            "hour": [9],
            "product_id": [1],
            "sku_hour_forecast": [42.0],
        }
    )
    post_filter = pre_filter.copy()

    compensated, stats = compensate_for_assortment_exclusion(
        pre_filter,
        post_filter,
        group_keys=["date", "bakery_id", "hour"],
        forecast_col="sku_hour_forecast",
    )

    assert stats == {"groups_scaled": 1, "groups_without_remaining_rows": 0}
    assert compensated["sku_hour_forecast"].tolist() == [42.0]


def test_compensate_leaves_fully_excluded_group_alone() -> None:
    # Every product in this group got filtered out — nothing left to
    # redistribute onto, so the group is reported but left as an empty gap
    # rather than fabricating a row.
    pre_filter = pd.DataFrame(
        {
            "date": ["2026-07-08"],
            "bakery_id": [257],
            "hour": [9],
            "product_id": [1],
            "sku_hour_forecast": [42.0],
        }
    )
    post_filter = pre_filter.iloc[0:0].copy()

    compensated, stats = compensate_for_assortment_exclusion(
        pre_filter,
        post_filter,
        group_keys=["date", "bakery_id", "hour"],
        forecast_col="sku_hour_forecast",
    )

    assert stats == {"groups_scaled": 0, "groups_without_remaining_rows": 1}
    assert compensated.empty


def test_cap_sku_uplift_scales_down_when_over_ratio() -> None:
    # bakery 1: SKU sum=260, bakery-day=200, ratio 1.30 > cap 1.20,
    # so scale = 1.20 / 1.30.
    # bakery 2: SKU sum = 220, bakery-day = 200 → ratio 1.10 < cap 1.20 → no scaling
    hourly = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-01"] * 4),
            "bakery_id": [1, 1, 2, 2],
            "hour": [9, 10, 9, 10],
            "product_id": [100, 100, 100, 100],
            "sku_hour_forecast": [130.0, 130.0, 110.0, 110.0],
        }
    )
    bakery_forecast = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-01", "2026-07-01"]),
            "bakery_id": [1, 2],
            "forecast_final": [200.0, 200.0],
        }
    )
    result, stats = cap_sku_uplift_to_bakery_forecast(
        hourly, bakery_forecast, max_ratio=1.20, forecast_col="forecast_final"
    )
    bak1 = result[result["bakery_id"] == 1]["sku_hour_forecast"].sum()
    bak2 = result[result["bakery_id"] == 2]["sku_hour_forecast"].sum()
    assert abs(bak1 - 200.0 * 1.20) < 0.01, (
        f"bakery 1 sum should be capped to 240, got {bak1}"
    )
    assert abs(bak2 - 220.0) < 0.01, (
        f"bakery 2 sum should be unchanged at 220, got {bak2}"
    )
    assert stats["bakery_days_capped"] == 1
    assert stats["bakery_days_total"] == 2


def test_cap_sku_uplift_is_noop_when_under_ratio() -> None:
    hourly = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-01"] * 2),
            "bakery_id": [1, 1],
            "hour": [9, 10],
            "product_id": [100, 100],
            "sku_hour_forecast": [100.0, 100.0],
        }
    )
    bakery_forecast = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-01"]),
            "bakery_id": [1],
            "forecast_final": [200.0],
        }
    )
    result, stats = cap_sku_uplift_to_bakery_forecast(
        hourly, bakery_forecast, max_ratio=1.35, forecast_col="forecast_final"
    )
    assert result["sku_hour_forecast"].sum() == 200.0
    assert stats["bakery_days_capped"] == 0


def test_cap_sku_uplift_per_sku_scales_down_when_over_ratio() -> None:
    # SKU 100: rolling_mean=100/day, cap=1.2x → sku_cap=120, fc_day=200 → scale=0.6
    # SKU 200: rolling_mean=200/day, cap=1.2x → sku_cap=240, fc_day=100 → no cap
    hourly = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-01"] * 4),
            "bakery_id": [1, 1, 1, 1],
            "hour": [9, 10, 9, 10],
            "product_id": [100, 100, 200, 200],
            "sku_hour_forecast": [100.0, 100.0, 50.0, 50.0],
        }
    )
    recent_stats = pd.DataFrame(
        {
            "bakery_id": [1, 1],
            "product_id": [100, 200],
            "recent_qty": [1000.0, 2000.0],
            "recent_days_sold": [10, 10],
        }
    )
    result, stats = cap_sku_uplift_per_sku(hourly, recent_stats, max_ratio=1.2)
    sku100 = result[result["product_id"] == 100]["sku_hour_forecast"].sum()
    sku200 = result[result["product_id"] == 200]["sku_hour_forecast"].sum()
    assert abs(sku100 - 120.0) < 0.01, f"SKU 100 should be capped to 120, got {sku100}"
    assert abs(sku200 - 100.0) < 0.01, (
        f"SKU 200 should be unchanged at 100, got {sku200}"
    )
    assert stats["sku_days_capped"] == 1
    assert stats["sku_days_total"] == 2


def test_cap_sku_uplift_per_sku_is_noop_when_under_ratio() -> None:
    # fc_day=120, rolling_mean=100, cap=1.2x, so sku_cap=120 and scale=1.0.
    hourly = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-01"] * 2),
            "bakery_id": [1, 1],
            "hour": [9, 10],
            "product_id": [100, 100],
            "sku_hour_forecast": [60.0, 60.0],
        }
    )
    recent_stats = pd.DataFrame(
        {
            "bakery_id": [1],
            "product_id": [100],
            "recent_qty": [1000.0],
            "recent_days_sold": [10],
        }
    )
    result, stats = cap_sku_uplift_per_sku(hourly, recent_stats, max_ratio=1.2)
    assert abs(result["sku_hour_forecast"].sum() - 120.0) < 0.01
    assert stats["sku_days_capped"] == 0


def test_cap_sku_uplift_per_sku_is_noop_when_no_recent_stats() -> None:
    # SKU not in recent_stats → rolling_mean unknown → no cap
    hourly = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-01"] * 2),
            "bakery_id": [1, 1],
            "hour": [9, 10],
            "product_id": [100, 100],
            "sku_hour_forecast": [500.0, 500.0],
        }
    )
    recent_stats = pd.DataFrame(
        {
            "bakery_id": pd.Series([], dtype="int64"),
            "product_id": pd.Series([], dtype="int64"),
            "recent_qty": pd.Series([], dtype="float64"),
            "recent_days_sold": pd.Series([], dtype="int64"),
        }
    )
    result, stats = cap_sku_uplift_per_sku(hourly, recent_stats, max_ratio=1.2)
    assert abs(result["sku_hour_forecast"].sum() - 1000.0) < 0.01
    assert stats["sku_days_capped"] == 0


def test_hierarchical_haircut_shrinks_pair_toward_bakery() -> None:
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-01", "2026-07-02"] * 2),
            "bakery_id": [1, 1, 1, 1],
            "product_id": [10, 10, 20, 20],
            "forecast_qty": [100.0, 100.0, 100.0, 100.0],
            "actual_qty": [50.0, 50.0, 100.0, 100.0],
        }
    )
    bakery, pair = build_hierarchical_haircut_coefficients(
        history,
        target_ratio=1.1,
        min_coefficient=0.75,
        pair_prior_days=2.0,
    )
    assert bakery.loc[0, "bakery_coefficient"] == 0.825
    sku10 = pair.loc[pair["product_id"] == 10, "hierarchical_coefficient"].iloc[0]
    assert abs(sku10 - 0.7875) < 1e-9


def test_hierarchical_haircut_protects_underforecast_bakery() -> None:
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-01", "2026-07-02"]),
            "bakery_id": [1, 1],
            "product_id": [10, 10],
            "forecast_qty": [50.0, 50.0],
            "actual_qty": [100.0, 100.0],
        }
    )
    bakery, pair = build_hierarchical_haircut_coefficients(
        history,
        target_ratio=1.15,
        min_coefficient=0.85,
        pair_prior_days=7.0,
    )
    assert bool(bakery.loc[0, "protect_from_haircut"])
    assert pair.loc[0, "hierarchical_coefficient"] == 1.0


def test_apply_hierarchical_haircut_uses_bakery_fallback_for_new_sku() -> None:
    hourly = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-08", "2026-07-08"]),
            "dow": [2, 2],
            "bakery_id": [1, 1],
            "hour": [9, 10],
            "product_id": [99, 99],
            "sku_hour_forecast": [50.0, 50.0],
        }
    )
    bakery = pd.DataFrame(
        {
            "bakery_id": [1],
            "bakery_coefficient": [0.9],
            "protect_from_haircut": [False],
        }
    )
    pair = pd.DataFrame(
        {
            "bakery_id": [1],
            "product_id": [10],
            "hierarchical_coefficient": [0.8],
        }
    )
    result, stats = apply_hierarchical_haircut(hourly, bakery, pair)
    assert result["sku_hour_forecast"].sum() == 90.0
    assert stats["overall_coefficient"] == 0.9


def test_sku_cap_before_assortment_compensation_preserves_capped_total() -> None:
    pre_filter = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-08", "2026-07-08"]),
            "bakery_id": [257, 257],
            "hour": [9, 9],
            "product_id": [100, 200],
            "sku_hour_forecast": [100.0, 100.0],
        }
    )
    recent_stats = pd.DataFrame(
        {
            "bakery_id": [257, 257],
            "product_id": [100, 200],
            "recent_qty": [1000.0, 500.0],
            "recent_days_sold": [10, 10],
        }
    )
    capped, _ = cap_sku_uplift_per_sku(pre_filter, recent_stats, max_ratio=1.2)
    post_filter = capped[capped["product_id"] == 100].copy()
    compensated, _ = compensate_for_assortment_exclusion(
        capped,
        post_filter,
        group_keys=["date", "bakery_id", "hour"],
        forecast_col="sku_hour_forecast",
    )
    assert capped["sku_hour_forecast"].sum() == 160.0
    assert compensated["sku_hour_forecast"].sum() == 160.0

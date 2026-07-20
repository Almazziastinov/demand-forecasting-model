from __future__ import annotations

import pandas as pd

from scripts.analyze_pilot_stockout_forecast_bias import (
    build_comparison,
    summarize_group,
)


def test_bias_groups_exclude_ambiguous_days() -> None:
    signals = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-07-01", "2026-07-02", "2026-07-03", "2026-06-30"]
            ),
            "bakery_id": [20, 20, 20, 20],
            "product_id": [100, 100, 100, 100],
            "daily_sold": [8.0, 10.0, 9.0, 7.0],
            "stockout_group": [
                "clear_stockout",
                "confirmed_non_stockout",
                "ambiguous",
                "clear_stockout",
            ],
        }
    )
    forecast = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-03"]),
            "bakery_id": [20, 20, 20],
            "product_id": [100, 100, 100],
            "forecast_qty": [12.0, 9.0, 20.0],
            "source_run_id": ["run"] * 3,
            "latest_generated_at": pd.to_datetime(["2026-06-30"] * 3),
        }
    )

    comparison = build_comparison(signals, forecast)

    assert comparison["stockout_group"].tolist() == [
        "clear_stockout",
        "confirmed_non_stockout",
    ]
    assert comparison["bias_qty"].tolist() == [4.0, -1.0]


def test_summary_reports_aggregate_and_row_level_bias() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-01", "2026-07-02"]),
            "bakery_id": [20, 20],
            "daily_sold": [10.0, 30.0],
            "forecast_qty": [15.0, 25.0],
            "bias_qty": [5.0, -5.0],
            "has_forecast": [True, True],
            "forecast_to_sales_ratio": [1.5, 25 / 30],
        }
    )

    summary = summarize_group(frame)

    assert summary["total_bias_qty"] == 0.0
    assert summary["aggregate_bias_pct_of_sales"] == 0.0
    assert summary["covered_aggregate_bias_pct_of_sales"] == 0.0
    assert summary["mean_bias_qty_per_sku_day"] == 0.0
    assert summary["positive_bias_share"] == 0.5
    assert summary["below_observed_share"] == 0.5

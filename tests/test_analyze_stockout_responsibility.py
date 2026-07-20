from __future__ import annotations

import pandas as pd

from scripts.analyze_stockout_responsibility import (
    BAKERY_UNDERPRODUCTION,
    FORECAST_MET_STILL_STOCKOUT,
    MODEL_UNDERFORECAST,
    classify_stockouts,
    summarize,
)


def test_classifies_only_accepted_stockouts_by_identifiable_cause() -> None:
    frame = pd.DataFrame(
        {
            "stockout_group": ["clear_stockout"] * 3 + ["confirmed_non_stockout"],
            "forecast_qty": [8.0, 12.0, 10.0, 1.0],
            "daily_sold": [10.0, 10.0, 10.0, 5.0],
            "qty_produced": [10.0, 10.0, 10.0, 5.0],
            "bakery_id": [1, 1, 2, 2],
            "product_id": [10, 11, 12, 13],
            "last_hour_gap": [3.0, 4.0, 2.0, 5.0],
            "bakery_sales_after_last": [100.0, 120.0, 80.0, 90.0],
        }
    )

    result = classify_stockouts(frame)

    assert result["responsibility_group"].tolist() == [
        MODEL_UNDERFORECAST,
        BAKERY_UNDERPRODUCTION,
        FORECAST_MET_STILL_STOCKOUT,
    ]
    assert result["confirmed_model_shortfall_qty"].tolist() == [2.0, 0.0, 0.0]
    assert result["bakery_execution_gap_qty"].tolist() == [0.0, 2.0, 0.0]


def test_summary_uses_all_stockouts_as_share_denominator() -> None:
    frame = pd.DataFrame(
        {
            "stockout_group": ["clear_stockout", "clear_stockout"],
            "forecast_qty": [8.0, 12.0],
            "daily_sold": [10.0, 10.0],
            "qty_produced": [10.0, 10.0],
            "bakery_id": [1, 1],
            "product_id": [10, 11],
            "last_hour_gap": [3.0, 4.0],
            "bakery_sales_after_last": [100.0, 120.0],
        }
    )

    summary = summarize(classify_stockouts(frame))

    assert set(summary["share_of_stockouts"]) == {0.5}

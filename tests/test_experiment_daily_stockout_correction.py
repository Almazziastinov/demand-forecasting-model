from __future__ import annotations

import pandas as pd

from scripts.experiment_daily_stockout_correction import (
    apply_daily_correction,
    evaluate,
)


def test_daily_correction_uses_only_prior_dates() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-06-01", periods=4),
            "bakery_id": [1] * 4,
            "product_id": [10] * 4,
            "stockout_group": ["clear_stockout"] * 3 + ["confirmed_non_stockout"],
            "daily_sold": [12.0, 12.0, 12.0, 10.0],
            "forecast_qty": [10.0, 10.0, 10.0, 10.0],
        }
    )

    result = apply_daily_correction(
        frame, lookback_days=28, min_history_days=2, min_stockouts=2, max_factor=1.5
    )

    assert result.loc[0, "daily_correction_factor"] == 1.0
    assert result.loc[1, "daily_correction_factor"] == 1.0
    assert result.loc[2, "daily_correction_factor"] == 1.2
    assert result.loc[3, "daily_correction_factor"] == 1.2


def test_evaluation_reports_stockout_gain_and_normal_cost() -> None:
    frame = pd.DataFrame(
        {
            "stockout_group": ["clear_stockout", "confirmed_non_stockout"],
            "daily_sold": [12.0, 10.0],
            "forecast_qty": [10.0, 10.0],
            "adjusted_forecast_qty": [12.0, 12.0],
            "daily_correction_factor": [1.2, 1.2],
        }
    )

    metrics = evaluate(frame)

    assert metrics["underforecast_cases_removed"] == 1
    assert metrics["adjusted_confirmed_shortfall_qty"] == 0.0
    assert metrics["normal_adjusted_forecast_to_sales"] == 1.2

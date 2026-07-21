from __future__ import annotations

import pandas as pd
import pytest

from scripts.experiment_regime_aware_sku_allocation import (
    apply_guarded_allocation,
    apply_positive_capacity_allocation,
    build_walk_forward_signals,
    choose_dominant_runs,
    prepare_universe,
)


def test_choose_dominant_runs_uses_most_supported_run() -> None:
    labels = pd.DataFrame(
        {
            "date": ["2026-06-01"] * 3,
            "bakery_id": [1, 1, 1],
            "product_id": [10, 20, 30],
            "source_run_id": ["main", "main", "old"],
            "latest_generated_at": ["2026-06-01T01:00:00Z"] * 3,
        }
    )
    result = choose_dominant_runs(labels)
    assert result.iloc[0]["source_run_id"] == "main"


def test_prepare_universe_uses_complete_forecast_total_for_baseline_share() -> None:
    forecasts = pd.DataFrame(
        {
            "date": ["2026-06-01"] * 3,
            "bakery_id": [1, 1, 1],
            "product_id": [10, 20, 30],
            "product_name": ["A", "B", "C"],
            "forecast_qty": [10.0, 20.0, 70.0],
        }
    )
    actual = pd.DataFrame(
        {
            "date": ["2026-06-01"] * 3,
            "bakery_id": [1, 1, 1],
            "product_id": [10, 20, 30],
            "daily_sold": [20.0, 20.0, 60.0],
        }
    )
    labels = pd.DataFrame(
        {
            "date": ["2026-06-01"],
            "bakery_id": [1],
            "product_id": [10],
            "stockout_group": ["confirmed_non_stockout"],
            "has_forecast": [True],
        }
    )
    stability = pd.DataFrame(
        {
            "bakery_id": [1],
            "product_id": [10],
            "recurrent_allocation": [True],
            "is_bakery_top5_by_sales": [True],
            "is_potentially_problematic": [True],
        }
    )
    result = prepare_universe(forecasts, actual, labels, stability)
    candidate = result[result["product_id"].eq(10)].iloc[0]
    assert candidate["baseline_share"] == 0.1
    assert candidate["observed_share"] == 0.2


def test_walk_forward_signal_does_not_use_current_day() -> None:
    dates = pd.date_range("2026-06-01", periods=6)
    frame = pd.DataFrame(
        {
            "date": dates,
            "bakery_id": [1] * 6,
            "product_id": [10] * 6,
            "is_labelled_candidate": [True] * 6,
            "stockout_group": ["confirmed_non_stockout"] * 6,
            "observed_log_residual": [0.2, 0.2, 0.2, 0.2, 0.2, 5.0],
            "daily_sold": [10.0] * 6,
        }
    )
    result = build_walk_forward_signals(frame)
    assert result.iloc[-1]["stable_log_residual"] == 0.2
    assert result.iloc[-1]["history_days"] == 5


def test_risk_signal_uses_only_prior_stockout_labels() -> None:
    dates = pd.date_range("2026-06-01", periods=6)
    frame = pd.DataFrame(
        {
            "date": dates,
            "bakery_id": [1] * 6,
            "product_id": [10] * 6,
            "is_labelled_candidate": [True] * 6,
            "stockout_group": [
                "confirmed_non_stockout",
                "confirmed_non_stockout",
                "confirmed_non_stockout",
                "confirmed_non_stockout",
                "clear_stockout",
                "clear_stockout",
            ],
            "observed_log_residual": [0.4, 0.4, 0.4, 0.4, 0.1, 9.0],
            "daily_sold": [10.0] * 6,
        }
    )
    result = build_walk_forward_signals(frame)
    assert result.iloc[-1]["prior_stockout_rate_14"] == 0.2
    assert result.iloc[-1]["risk_log_residual"] == pytest.approx(0.32)


def test_guarded_allocation_preserves_full_total_and_budget() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-01", "2026-06-01"]),
            "bakery_id": [1, 1],
            "baseline_share": [0.5, 0.5],
            "baseline_total": [100.0, 100.0],
            "forecast_qty": [50.0, 50.0],
            "stable_log_residual": [1.0, 0.0],
            "recent_log_residual": [1.0, 0.0],
            "regime_confirmed": [False, False],
            "signal_reliability": [1.0, 0.0],
            "allocation_eligible": [True, False],
        }
    )
    result = apply_guarded_allocation(
        frame,
        signal_mode="stable",
        strength=1.0,
        max_shift_fraction=0.01,
    )
    assert round(result["adjusted_forecast_qty"].sum(), 8) == 100.0
    assert round(result["shifted_qty"].sum() / 2.0, 8) == 1.0


def test_positive_capacity_uses_only_headroom_and_preserves_total() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-01"] * 3),
            "bakery_id": [1, 1, 1],
            "baseline_share": [0.2, 0.5, 0.3],
            "baseline_total": [100.0] * 3,
            "forecast_qty": [20.0, 50.0, 30.0],
            "stable_log_residual": [1.0, 0.0, 0.0],
            "recent_log_residual": [1.0, 0.0, 0.0],
            "regime_confirmed": [False] * 3,
            "signal_reliability": [1.0, 0.0, 0.0],
            "allocation_eligible": [True, False, False],
            "is_screened_pair": [True, True, False],
            "prior_sales_q75_28": [20.0, 45.0, 30.0],
            "prior_sales_q90_28": [20.0, 45.0, 30.0],
            "prior_sales_q95_28": [20.0, 45.0, 30.0],
            "prior_sales_days_28": [10.0] * 3,
        }
    )
    result = apply_positive_capacity_allocation(
        frame,
        signal_mode="stable",
        strength=1.0,
        max_shift_fraction=0.05,
        max_sku_uplift=0.50,
    )
    assert round(result["adjusted_forecast_qty"].sum(), 8) == 100.0
    assert result.loc[0, "adjusted_forecast_qty"] == 25.0
    assert result.loc[1, "adjusted_forecast_qty"] == 45.0
    assert result.loc[2, "adjusted_forecast_qty"] == 30.0

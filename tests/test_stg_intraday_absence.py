"""Small synthetic checks for the research-only intraday anomaly audit."""

from __future__ import annotations

import pandas as pd

from scripts.analyze_stg_intraday_absence import (
    HOURS,
    add_historical_reference,
    evaluate_signals,
)


def test_reference_uses_training_dates_only() -> None:
    rows = []
    for date, sold in [
        ("2026-06-22", 3.0),
        ("2026-06-29", 5.0),
        ("2026-07-06", 100.0),
    ]:
        rows.append(
            {
                "date": pd.Timestamp(date),
                "bakery_id": 20,
                "product_id": 36,
                "dow": 0,
                "hour": 12,
                "sold": sold,
                "bakery_hour_sales": 30.0,
                "release_qty": 20.0,
                "incoming_move_qty": 0.0,
                "outgoing_move_qty": 0.0,
            }
        )
    holdout = add_historical_reference(pd.DataFrame(rows))
    assert len(holdout) == 1
    assert holdout.iloc[0]["expected"] == 4.0
    assert holdout.iloc[0]["reference_days"] == 2


def test_interior_gap_is_candidate_not_verified_stockout() -> None:
    rows = []
    for hour in HOURS:
        rows.append(
            {
                "date": pd.Timestamp("2026-07-06"),
                "bakery_id": 20,
                "product_id": 36,
                "hour": hour,
                "sold": 2.0 if hour in {11, 14, 19} else 0.0,
                "bakery_hour_sales": 15.0,
                "expected": 3.0,
                "reference_days": 4,
                "release_qty": 20.0,
                "incoming_move_qty": 0.0,
                "outgoing_move_qty": 0.0,
                "written_off_qty": 0.0,
                "historical_median_daily_sales": 1.0,
                "same_window_days": 8,
                "same_window_zeros": 0,
                "all_window_days": 40,
                "all_window_zeros": 0,
            }
        )
    cases, summary = evaluate_signals(pd.DataFrame(rows))
    gap = cases[cases["signal"] == "interior_gap"]
    assert len(gap) == 1
    assert gap.iloc[0]["start_hour"] == 12
    assert not gap.iloc[0]["frequency_guard_pass"]
    assert summary["zero_sales_despite_daily_supply_review_only"] == 0

    high_volume = pd.DataFrame(rows)
    high_volume.loc[high_volume["hour"].isin([11, 14, 19]), "sold"] = 5.0
    high_volume["historical_median_daily_sales"] = 20.0
    high_cases, high_summary = evaluate_signals(high_volume)
    assert high_cases.loc[
        high_cases["signal"] == "interior_gap", "frequency_guard_pass"
    ].all()
    assert high_summary["interior_gap_frequency_guard_pass"] == 1

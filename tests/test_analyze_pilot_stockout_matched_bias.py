from __future__ import annotations

import pandas as pd

from scripts.analyze_pilot_stockout_matched_bias import (
    aggregate_matched_cases,
    build_matches,
    summarize,
)


def _row(
    date: str,
    group: str,
    *,
    sold: float,
    forecast: float,
    produced: float,
) -> dict[str, object]:
    return {
        "date": pd.Timestamp(date),
        "stockout_group": group,
        "bakery_id": 20,
        "bakery_name": "Bakery",
        "product_id": 100,
        "product_name": "Product",
        "category_name": "Category",
        "dow": 0,
        "daily_sold": sold,
        "forecast_qty": forecast,
        "qty_produced": produced,
        "stock_balance": 0.0 if group == "clear_stockout" else 3.0,
        "bias_qty": forecast - sold,
        "last_sale_hour": 16,
        "normal_last_hour": 20,
        "bakery_sales_after_last": 100.0,
        "normal_days": 4,
        "source_run_id": "run",
    }


def test_matches_same_pair_weekday_and_similar_production() -> None:
    frame = pd.DataFrame(
        [
            _row("2026-07-20", "clear_stockout", sold=10, forecast=9, produced=10),
            _row("2026-07-13", "confirmed_non_stockout", sold=8, forecast=12, produced=11),
            _row("2026-07-06", "confirmed_non_stockout", sold=9, forecast=10, produced=20),
        ]
    )

    matches = build_matches(frame, production_tolerance=0.25)
    cases = aggregate_matched_cases(matches)
    result = summarize(cases, total_stockout_cases=1)

    assert len(matches) == 1
    assert matches["date"].item() == pd.Timestamp("2026-07-13")
    assert result["matched_cases"] == 1
    assert result["forecast_below_observed_cases"] == 1
    assert result["stockout_ratio_below_matched_control_share"] == 1.0

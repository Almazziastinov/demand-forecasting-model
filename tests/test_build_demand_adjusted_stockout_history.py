from __future__ import annotations

import pandas as pd

from scripts.build_demand_adjusted_stockout_history import build_adjusted_history
from scripts.build_demand_adjusted_stockout_history import reconstruct_cases
from scripts.build_demand_adjusted_stockout_history import select_cases


def test_select_cases_supports_inverse_allocation_rule() -> None:
    classified = pd.DataFrame(
        {
            "case_type": ["allocation", "uncertain", "demand_loss"],
            "robust_case_type": ["allocation", "uncertain", "demand_loss"],
        }
    )

    selected = select_cases(classified, mode="not_robust_allocation")

    assert selected.index.tolist() == [1, 2]


def test_reconstruct_cases_only_fills_post_cutoff_hours() -> None:
    dates = pd.to_datetime(["2026-06-01", "2026-06-08", "2026-06-15", "2026-06-22"])
    rows = []
    for date in dates:
        for hour in [17, 18, 19]:
            rows.extend(
                [
                    {
                        "date": date,
                        "bakery_id": 1,
                        "product_id": 10,
                        "hour": hour,
                        "sold": 2.0,
                    },
                    {
                        "date": date,
                        "bakery_id": 1,
                        "product_id": 20,
                        "hour": hour,
                        "sold": 8.0,
                    },
                ]
            )
    hourly = pd.DataFrame(rows)
    cases = pd.DataFrame(
        {
            "date": [dates[-1]],
            "bakery_id": [1],
            "bakery_name": ["A"],
            "product_id": [10],
            "product_name": ["P"],
            "daily_sold": [2.0],
            "last_sale_hour": [17.0],
            "bakery_gap": [10.0],
            "bakery_ratio": [0.8],
            "robust_case_type": ["demand_loss"],
        }
    )
    stockouts = cases.iloc[0:0].copy()
    audit, hourly_audit = reconstruct_cases(hourly, cases, stockouts)
    assert audit.iloc[0]["imputed_demand"] > 0
    assert hourly_audit["hour"].tolist() == [18, 19]

    daily_sku, daily_bakery, profile = build_adjusted_history(hourly, audit)
    assert daily_sku["demand_adjusted_sales"].sum() > daily_sku["observed_sales"].sum()
    assert (
        daily_bakery["demand_adjusted_sales"].sum()
        > daily_bakery["observed_sales"].sum()
    )
    assert profile["share_delta"].max() > 0

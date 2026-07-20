from __future__ import annotations

import pandas as pd

from scripts.classify_stockout_mechanisms import add_weekday_counterfactual
from scripts.classify_stockout_mechanisms import classify_cases


def test_add_weekday_counterfactual_uses_only_prior_same_weekdays() -> None:
    frame = pd.DataFrame(
        {
            "bakery_id": [1] * 4,
            "date": pd.to_datetime(
                ["2026-06-01", "2026-06-08", "2026-06-15", "2026-06-22"]
            ),
            "sales": [100.0, 110.0, 120.0, 1000.0],
        }
    )
    result = add_weekday_counterfactual(
        frame, keys=["bakery_id"], value="sales", lags=(7, 14, 21)
    )
    assert result.iloc[-1]["expected_sales"] == 110.0
    assert result.iloc[-1]["reference_days_sales"] == 3


def test_classify_cases_separates_allocation_and_demand_loss() -> None:
    dates = pd.to_datetime(["2026-06-10", "2026-06-11"])
    cases = pd.DataFrame(
        {
            "date": dates,
            "bakery_id": [1, 1],
            "product_id": [10, 10],
            "daily_sold": [8.0, 8.0],
            "confirmed_model_shortfall_qty": [2.0, 2.0],
        }
    )
    bakery = pd.DataFrame(
        {
            "date": dates,
            "bakery_id": [1, 1],
            "actual_bakery": [98.0, 70.0],
            "expected_actual_bakery": [100.0, 100.0],
            "sigma_actual_bakery": [5.0, 5.0],
            "reference_days_actual_bakery": [6, 6],
        }
    )
    sku = pd.DataFrame(
        {
            "date": dates,
            "bakery_id": [1, 1],
            "product_id": [10, 10],
            "sold": [8.0, 8.0],
            "expected_sold": [10.0, 10.0],
            "sigma_sold": [1.0, 1.0],
            "reference_days_sold": [6, 6],
        }
    )
    result = classify_cases(cases, bakery, sku)
    assert result["case_type"].tolist() == ["allocation", "demand_loss"]

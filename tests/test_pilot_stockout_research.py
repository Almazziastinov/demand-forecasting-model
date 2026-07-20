from __future__ import annotations

import pandas as pd

from scripts.analyze_pilot_mart_zero_stockout_balance import (
    BAKEABLE_CATEGORIES,
    load_hourly,
)
from scripts.backtest_pilot_mart_zero_pseudo_stockout import apply_policy
from scripts import compare_pilot_mart_zero_demand_profiles as profile_comparison


def test_bakery_hour_total_includes_non_bakeable_sales(tmp_path) -> None:
    bakeable_category = next(iter(BAKEABLE_CATEGORIES))
    source = pd.DataFrame(
        [
            {
                "check_datetime": "2026-07-01T07:00:00Z",
                "check_date": "2026-07-01",
                "quantity": 2,
                "bakery_id": 20,
                "product_id": 100,
                "category_name": bakeable_category,
            },
            {
                "check_datetime": "2026-07-01T07:00:00Z",
                "check_date": "2026-07-01",
                "quantity": 8,
                "bakery_id": 20,
                "product_id": 200,
                "category_name": "Напитки",
            },
        ]
    )
    path = tmp_path / "hourly.csv"
    source.to_csv(path, index=False, encoding="utf-8-sig")

    hourly, bakery_hour = load_hourly(path)

    assert hourly["sold"].sum() == 2
    assert bakery_hour["bakery_hour_sales"].sum() == 10


def test_pseudo_policy_uses_training_volume_not_hidden_true_total() -> None:
    reconstructed = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-07-01")] * 2,
            "bakery_id": [20, 20],
            "product_id": [100, 100],
            "is_hidden_hour": [True, True],
            "normal_daily_sold": [4.0, 4.0],
            "daily_sold_hourly": [100.0, 100.0],
            "imputed_demand": [5.0, 5.0],
            "sold_observed": [0.0, 0.0],
        }
    )

    result = apply_policy(reconstructed)

    assert result["policy_imputed_demand"].sum() == 4.0


def test_profile_comparison_reconstructs_train_without_holdout(monkeypatch) -> None:
    dates = pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-03"])
    frame = pd.DataFrame(
        {
            "date": dates,
            "bakery_id": [20, 20, 20],
            "product_id": [100, 100, 100],
            "dow": dates.dayofweek,
            "hour": [10, 10, 10],
            "sold": [2.0, 2.0, 20.0],
            "balance_is_consistent": [True, True, True],
            "hourly_daily_sales_agree": [True, True, True],
            "is_inventory_stockout": [False, False, False],
        }
    )
    observed_dates: list[pd.Timestamp] = []

    def fake_reconstruct(train: pd.DataFrame) -> pd.DataFrame:
        observed_dates.extend(train["date"].tolist())
        result = train.copy()
        result["sold_demand"] = result["sold"]
        result["is_policy_adjusted"] = False
        result["imputed_demand"] = 0.0
        return result

    monkeypatch.setattr(profile_comparison, "reconstruct_training_window", fake_reconstruct)

    profile_comparison.compare_window(frame, history_days=2, holdout_days=1)

    assert max(observed_dates) == pd.Timestamp("2026-07-02")

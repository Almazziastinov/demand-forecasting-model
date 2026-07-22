from __future__ import annotations

import pandas as pd
import pytest

from scripts.experiment_stockout_adjusted_bakery_targets import (
    CONSERVATIVE,
    WEIGHTED,
    build_adjustment_variants,
    select_non_overlapping_cutoffs,
    summarize_predictions,
)


def test_build_adjustment_variants_applies_weight_and_conservative_cap() -> None:
    demand = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-01", "2026-06-01"]),
            "bakery_id": [1, 1],
            "product_id": [10, 20],
            "is_clear_stockout": [True, False],
            "demand_lower_bound": [4.0, 8.0],
            "raw_imputed_demand": [10.0, 0.0],
            "imputed_demand": [3.0, 0.0],
            "suggested_training_weight": [0.8, 1.0],
            "reference_days": [5, 0],
        }
    )

    result = build_adjustment_variants(demand).set_index("variant")

    assert result.loc[WEIGHTED, "imputed_demand"] == pytest.approx(2.4)
    assert result.loc[CONSERVATIVE, "imputed_demand"] == 2.0


def test_summary_separates_clean_and_stockout_evaluation_targets() -> None:
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-01", "2026-06-02"]),
            "bakery_id": [1, 1],
            "bakery_name": ["A", "A"],
            "bakery_sales": [100.0, 100.0],
            "prediction": [105.0, 105.0],
        }
    )
    adjustments = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-02")],
            "bakery_id": [1],
            "full_imputed": [10.0],
            "conservative_imputed": [5.0],
            "stockout_skus": [1],
        }
    )

    result = summarize_predictions(predictions, adjustments, variant="v")
    indexed = result.set_index("scope")

    assert indexed.loc["clean_days_observed_sales", "bias_qty"] == 5.0
    assert indexed.loc["stockout_days_observed_lower_bound", "bias_qty"] == 5.0
    assert indexed.loc["stockout_days_conservative_point", "bias_qty"] == 0.0
    assert indexed.loc["stockout_days_full_point", "bias_qty"] == -5.0


def test_non_overlapping_cutoffs_skip_overlapping_middle_window() -> None:
    cutoffs = pd.Series(["2026-06-21", "2026-06-28", "2026-07-05"])

    result = select_non_overlapping_cutoffs(cutoffs, holdout_days=14)

    assert result == ["2026-06-21", "2026-07-05"]

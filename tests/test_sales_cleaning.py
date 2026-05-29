from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments_v2.sales_cleaning import add_base_training_policy_flags  # noqa: E402
from src.experiments_v2.sales_cleaning import add_capped_base_target  # noqa: E402
from src.experiments_v2.sales_cleaning import add_quantile_capped_base_target  # noqa: E402
from src.experiments_v2.sales_cleaning import add_robust_sales_outlier_flags  # noqa: E402
from src.experiments_v2.sales_cleaning import add_rolling_median_capped_base_target  # noqa: E402
from src.experiments_v2.sales_cleaning import add_rolling_quantile_capped_base_target  # noqa: E402


def test_add_robust_sales_outlier_flags_marks_high_spike():
    dates = pd.date_range("2026-05-01", periods=12, freq="D")
    values = [10.0] * 11 + [40.0]
    df = pd.DataFrame(
        {
            "date": dates,
            "bakery_id": [1] * 12,
            "dow": [0] * 12,
            "bakery_sales": values,
        }
    )

    result = add_robust_sales_outlier_flags(
        df,
        value_col="bakery_sales",
        entity_cols=["bakery_id"],
        seasonal_cols=["dow"],
        robust_z_threshold=3.5,
        high_ratio_threshold=2.0,
    )

    assert result.iloc[-1]["sales_high_outlier_flag"] == 1
    assert result.iloc[-1]["expected_base_source"] == "seasonal"
    assert result.iloc[-1]["sales_to_expected_ratio"] == 4.0


def test_base_training_policy_uses_explicit_context_flags_only():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-01", "2026-05-02"]),
            "bakery_id": [1, 1],
            "sales_high_outlier_flag": [1, 1],
            "sales_low_outlier_flag": [0, 0],
            "event_context_flag": [0, 1],
        }
    )

    result = add_base_training_policy_flags(
        df,
        contextual_flag_cols=["event_context_flag"],
    )

    no_context = result.iloc[0]
    with_context = result.iloc[1]

    assert no_context["unexplained_high_outlier_flag"] == 1
    assert no_context["correction_candidate_flag"] == 0
    assert no_context["base_model_sample_weight"] == 0.35

    assert with_context["contextual_high_outlier_flag"] == 1
    assert with_context["correction_candidate_flag"] == 1
    assert with_context["base_model_sample_weight"] == 0.60


def test_base_training_policy_downweights_imputed_rows():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-01", "2026-05-02"]),
            "bakery_id": [1, 1],
            "sales_high_outlier_flag": [0, 0],
            "sales_low_outlier_flag": [0, 0],
            "sales_missing_flag": [0, 1],
        }
    )

    result = add_base_training_policy_flags(df)

    assert result.loc[0, "base_model_sample_weight"] == 1.0
    assert result.loc[1, "base_model_sample_weight"] == 0.25


def test_add_capped_base_target_caps_unexplained_high_outlier_only():
    df = pd.DataFrame(
        {
            "bakery_sales": [100.0, 100.0, 10.0],
            "expected_base_qty": [40.0, 40.0, 40.0],
            "sales_high_outlier_flag": [1, 1, 0],
            "sales_low_outlier_flag": [0, 0, 1],
            "contextual_high_outlier_flag": [0, 1, 0],
            "unexplained_high_outlier_flag": [1, 0, 0],
        }
    )

    result = add_capped_base_target(
        df,
        value_col="bakery_sales",
        upper_multiplier=1.5,
        lower_multiplier=0.5,
    )

    assert result.loc[0, "bakery_sales_base_capped"] == 60.0
    assert result.loc[0, "base_target_capped_flag"] == 1

    assert result.loc[1, "bakery_sales_base_capped"] == 100.0
    assert result.loc[1, "base_target_capped_flag"] == 0

    assert result.loc[2, "bakery_sales_base_capped"] == 20.0
    assert result.loc[2, "base_target_capped_flag"] == 1


def test_add_quantile_capped_base_target_uses_weekday_bucket():
    df = pd.DataFrame(
        {
            "bakery_id": [1] * 10 + [1] * 10,
            "dow": [1] * 10 + [5] * 10,
            "sales": [10.0] * 9 + [100.0] + [40.0] * 10,
        }
    )

    result = add_quantile_capped_base_target(
        df,
        value_col="sales",
        entity_cols=["bakery_id"],
        seasonal_cols=["dow"],
        lower_quantile=0.10,
        upper_quantile=0.90,
        min_seasonal_rows=8,
    )

    assert result.loc[9, "quantile_cap_source"] == "seasonal"
    assert round(result.loc[9, "sales_base_quantile_capped"], 6) == 19.0
    assert result.loc[9, "quantile_base_target_capped_flag"] == 1
    assert result.loc[10, "sales_base_quantile_capped"] == 40.0


def test_add_quantile_capped_base_target_falls_back_when_weekday_is_thin():
    df = pd.DataFrame(
        {
            "bakery_id": [1] * 10,
            "dow": [1, 1] + [2] * 8,
            "sales": [10.0, 100.0] + [40.0] * 8,
        }
    )

    result = add_quantile_capped_base_target(
        df,
        value_col="sales",
        entity_cols=["bakery_id"],
        seasonal_cols=["dow"],
        lower_quantile=0.10,
        upper_quantile=0.90,
        min_seasonal_rows=8,
    )

    assert result.loc[1, "quantile_cap_source"] == "entity"
    assert round(result.loc[1, "sales_base_quantile_capped"], 6) == 46.0


def test_add_quantile_capped_base_target_preserves_contextual_high_by_default():
    df = pd.DataFrame(
        {
            "bakery_id": [1] * 10,
            "dow": [1] * 10,
            "sales": [10.0] * 9 + [100.0],
            "contextual_high_outlier_flag": [0] * 9 + [1],
        }
    )

    result = add_quantile_capped_base_target(
        df,
        value_col="sales",
        entity_cols=["bakery_id"],
        seasonal_cols=["dow"],
        lower_quantile=0.10,
        upper_quantile=0.90,
        min_seasonal_rows=8,
    )

    assert result.loc[9, "sales_base_quantile_capped"] == 100.0
    assert result.loc[9, "quantile_base_target_capped_flag"] == 0


def test_add_rolling_median_capped_base_target_follows_trend():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=10, freq="7D"),
            "bakery_id": [1] * 10,
            "dow": [3] * 10,
            "sales": [100, 110, 120, 130, 140, 150, 160, 170, 180, 300],
        }
    )

    result = add_rolling_median_capped_base_target(
        df,
        value_col="sales",
        entity_cols=["bakery_id"],
        seasonal_cols=["dow"],
        window=4,
        min_periods=4,
        upper_multiplier=1.6,
        lower_multiplier=0.5,
    )

    assert result.loc[0, "sales_base_rolling_capped"] == 100
    assert result.loc[8, "sales_base_rolling_capped"] == 180
    assert result.loc[9, "rolling_base_median"] == 165
    assert result.loc[9, "sales_base_rolling_capped"] == 264
    assert result.loc[9, "rolling_base_target_capped_flag"] == 1


def test_add_rolling_quantile_capped_target_uses_trailing_window():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=12, freq="7D"),
            "bakery_id": [1] * 12,
            "dow": [3] * 12,
            "sales": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 200],
        }
    )

    result = add_rolling_quantile_capped_base_target(
        df,
        value_col="sales",
        entity_cols=["bakery_id"],
        seasonal_cols=["dow"],
        window=8,
        min_periods=4,
        lower_quantile=0.05,
        upper_quantile=0.95,
    )

    spike = result.iloc[11]
    assert spike["rolling_quantile_base_target_capped_flag"] == 1
    assert spike["sales_base_rolling_quantile_capped"] < 50
    assert pd.notna(spike["rolling_q_upper"])
    assert spike["rolling_q_cap_source"] == "rolling_same_bucket"

    cold_start = result.iloc[0]
    assert cold_start["sales_base_rolling_quantile_capped"] == 10
    assert cold_start["rolling_quantile_base_target_capped_flag"] == 0


def test_add_rolling_quantile_capped_target_preserves_contextual_high():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=12, freq="7D"),
            "bakery_id": [1] * 12,
            "dow": [3] * 12,
            "sales": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 200],
            "contextual_high_outlier_flag": [0] * 11 + [1],
        }
    )

    result = add_rolling_quantile_capped_base_target(
        df,
        value_col="sales",
        entity_cols=["bakery_id"],
        seasonal_cols=["dow"],
        window=8,
        min_periods=4,
    )

    spike = result.iloc[11]
    assert spike["sales_base_rolling_quantile_capped"] == 200
    assert spike["rolling_quantile_base_target_capped_flag"] == 0

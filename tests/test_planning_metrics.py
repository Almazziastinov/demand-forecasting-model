from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments_v2.planning_metrics import add_planning_error_columns  # noqa: E402
from src.experiments_v2.planning_metrics import aggregate_planning_metrics  # noqa: E402
from src.experiments_v2.planning_metrics import planning_metrics  # noqa: E402
from src.experiments_v2.planning_metrics import summarize_models_by_planning_metrics  # noqa: E402


def test_add_planning_error_columns_marks_large_material_errors():
    df = pd.DataFrame(
        {
            "actual": [100.0, 100.0, 100.0, 10.0],
            "prediction": [40.0, 45.0, 70.0, 0.0],
        }
    )

    result = add_planning_error_columns(
        df,
        actual_col="actual",
        prediction_col="prediction",
        abs_error_threshold=50.0,
        rel_error_threshold=0.20,
    )

    assert result["large_error_flag"].tolist() == [1, 1, 0, 0]
    assert result["underforecast_flag"].tolist() == [1, 1, 1, 1]


def test_planning_metrics_counts_bias_and_large_error_direction():
    df = pd.DataFrame(
        {
            "actual": [100.0, 100.0, 100.0],
            "prediction": [40.0, 180.0, 90.0],
        }
    )

    metrics = planning_metrics(
        df,
        actual_col="actual",
        prediction_col="prediction",
        abs_error_threshold=50.0,
        rel_error_threshold=0.20,
    )

    assert metrics["rows"] == 3
    assert metrics["actual_sum"] == 300.0
    assert metrics["prediction_sum"] == 310.0
    assert metrics["bias"] == -10.0
    assert metrics["large_error_rows"] == 2
    assert metrics["large_underforecast_rows"] == 1
    assert metrics["large_overforecast_rows"] == 1


def test_aggregate_planning_metrics_groups_by_business_level():
    df = pd.DataFrame(
        {
            "city": ["Kazan", "Kazan", "Kursk"],
            "category": ["Bread", "Bread", "Bread"],
            "actual": [100.0, 100.0, 80.0],
            "prediction": [40.0, 100.0, 70.0],
        }
    )

    result = aggregate_planning_metrics(
        df,
        group_cols=["city", "category"],
        actual_col="actual",
        prediction_col="prediction",
        abs_error_threshold=50.0,
        rel_error_threshold=0.20,
    )

    kazan = result[result["city"] == "Kazan"].iloc[0]
    kursk = result[result["city"] == "Kursk"].iloc[0]

    assert kazan["rows"] == 2
    assert kazan["large_error_rows"] == 1
    assert kursk["large_error_rows"] == 0


def test_summarize_models_by_planning_metrics_sorts_by_large_error_share():
    df = pd.DataFrame(
        {
            "model": ["base", "base", "corrected", "corrected"],
            "actual": [100.0, 100.0, 100.0, 100.0],
            "prediction": [40.0, 40.0, 90.0, 95.0],
        }
    )

    result = summarize_models_by_planning_metrics(
        df,
        model_col="model",
        actual_col="actual",
        prediction_col="prediction",
    )

    assert result.iloc[0]["model"] == "corrected"
    assert result.iloc[0]["large_error_rows"] == 0

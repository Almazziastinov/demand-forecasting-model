from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


RUN_PATH = Path("src/experiments_v2/78_bakery_target_cleaning/run.py")
SPEC = importlib.util.spec_from_file_location("exp78_run", RUN_PATH)
EXP78 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(EXP78)


def test_select_feature_columns_drops_constant_and_missing_columns():
    frame = pd.DataFrame(
        {
            "bakery_id": [1, 2],
            "city": ["A", "B"],
            "dow": [0, 1],
            "month": [1, 1],
            "is_weekend": [0, 0],
        }
    )

    selected = EXP78.select_feature_columns(frame)

    assert "bakery_id" in selected
    assert "city" in selected
    assert "dow" in selected
    assert "month" not in selected
    assert "is_weekend" not in selected


def test_build_model_summary_counts_bakery_wins():
    model_metrics = [
        {"model": "raw", "mae": 10.0, "mse": 100.0, "wmape": 5.0, "bias": 1.0},
        {"model": "clean", "mae": 8.0, "mse": 80.0, "wmape": 4.0, "bias": 0.5},
    ]
    bakery_metrics = pd.DataFrame(
        [
            {"model": "raw", "bakery_id": 1, "mae": 10.0},
            {"model": "clean", "bakery_id": 1, "mae": 8.0},
            {"model": "raw", "bakery_id": 2, "mae": 7.0},
            {"model": "clean", "bakery_id": 2, "mae": 9.0},
        ]
    )

    summary = EXP78.build_model_summary(model_metrics, bakery_metrics)

    clean = summary[summary["model"] == "clean"].iloc[0]
    raw = summary[summary["model"] == "raw"].iloc[0]
    assert clean["win_count"] == 1
    assert raw["win_count"] == 1
    assert summary.iloc[0]["model"] == "clean"

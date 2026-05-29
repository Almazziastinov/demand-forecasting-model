from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


RUN_PATH = Path("src/experiments_v2/79_structural_top_down_correction/run.py")
SPEC = importlib.util.spec_from_file_location("exp79_run", RUN_PATH)
EXP79 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(EXP79)


def _train_frame() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2026-01-01", periods=42, freq="D")
    for dt in dates:
        dow = dt.dayofweek
        rows.append(
            {
                "date": dt,
                "bakery_id": "B1",
                "bakery_name": "Bakery 1",
                "city": "Kazan",
                "dow": dow,
                "bakery_sales": 200.0 if dow >= 5 else 100.0,
            }
        )
    return pd.DataFrame(rows)


def test_build_structural_profile_learns_weekend_shape():
    profile = EXP79.build_structural_profile(_train_frame(), recent_days=28)

    weekday = profile[(profile["bakery_id"] == "B1") & (profile["dow"] == 0)].iloc[0]
    weekend = profile[(profile["bakery_id"] == "B1") & (profile["dow"] == 5)].iloc[0]

    assert len(profile) == 7
    assert weekend["structural_baseline_shape"] > weekday["structural_baseline_shape"]
    assert weekend["structural_top_down_baseline"] > weekday[
        "structural_top_down_baseline"
    ]


def test_attach_structural_baseline_adds_forecast_for_holdout_rows():
    train = _train_frame()
    profile = EXP79.build_structural_profile(train)
    holdout = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-02-12", "2026-02-14"]),
            "bakery_id": ["B1", "B1"],
            "bakery_name": ["Bakery 1", "Bakery 1"],
            "city": ["Kazan", "Kazan"],
            "dow": [3, 5],
            "bakery_sales": [100.0, 220.0],
        }
    )

    result = EXP79.attach_structural_baseline(holdout, profile)

    assert result["structural_top_down_baseline"].notna().all()
    assert result.loc[1, "structural_top_down_baseline"] > result.loc[
        0, "structural_top_down_baseline"
    ]


def test_select_feature_columns_includes_baseline_feature():
    frame = pd.DataFrame(
        {
            "bakery_id": ["B1", "B2"],
            "city": ["Kazan", "Kursk"],
            "dow": [1, 2],
            "structural_top_down_baseline": [100.0, 200.0],
            "constant": [1, 1],
        }
    )

    selected = EXP79.select_feature_columns(frame)

    assert "bakery_id" in selected
    assert "structural_top_down_baseline" in selected
    assert "constant" not in selected

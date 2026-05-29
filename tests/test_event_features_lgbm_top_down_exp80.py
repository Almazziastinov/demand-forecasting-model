from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


RUN_PATH = Path("src/experiments_v2/80_event_features_lgbm_top_down/run.py")
SPEC = importlib.util.spec_from_file_location("exp80_run", RUN_PATH)
EXP80 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(EXP80)


def test_add_payday_distance_features_marks_payday_window():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-04", "2026-01-05", "2026-01-06"]),
            "dow": [6, 0, 1],
        }
    )

    result = EXP80.add_payday_distance_features(df)

    assert result.loc[0, "is_pre_payday_1d"] == 1
    assert result.loc[1, "is_payday"] == 1
    assert result.loc[2, "is_post_payday_1d"] == 1
    assert result.loc[1, "payday_window_type"] == "payday"


def test_add_enriched_event_features_marks_event_windows():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-03-19", "2026-03-20", "2026-03-21"]),
            "city": ["Kazan", "Kazan", "Kazan"],
            "dow": [3, 4, 5],
            "holiday_name": ["", "uraza_bayram", ""],
            "is_holiday": [0, 1, 0],
            "days_to_next_event": [1, 0, 999],
            "days_since_prev_event": [999, 0, 1],
            "current_event_cluster": ["cluster_none", "cluster_1", "cluster_none"],
            "next_event_cluster": ["cluster_1", "cluster_1", "cluster_none"],
            "prev_event_cluster": ["cluster_none", "cluster_1", "cluster_1"],
        }
    )

    result = EXP80.add_enriched_event_features(df)

    assert result.loc[0, "event_window_type"] == "pre_event_1_3"
    assert result.loc[1, "event_window_type"] == "event_day"
    assert result.loc[2, "event_window_type"] == "post_event_1_3"
    assert result.loc[1, "holiday_name_feature"] == "uraza_bayram"
    assert result.loc[0, "nearest_event_city"] == "cluster_1|Kazan"


def test_select_feature_columns_drops_constant_enriched_features():
    frame = pd.DataFrame(
        {
            "bakery_id": ["B1", "B2"],
            "city": ["Kazan", "Kursk"],
            "holiday_name_feature": ["no_holiday", "womens_day"],
            "is_pre_event_1d": [0, 0],
        }
    )

    selected = EXP80.select_feature_columns(
        frame,
        ["bakery_id", "city", "holiday_name_feature", "is_pre_event_1d"],
    )

    assert "bakery_id" in selected
    assert "holiday_name_feature" in selected
    assert "is_pre_event_1d" not in selected

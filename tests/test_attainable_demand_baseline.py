"""Tests for first attainable-demand baseline layer."""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments_v2.attainable_demand_baseline import build_attainable_baseline  # noqa: E402


def _make_profile_input() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Дата": pd.Timestamp("2026-01-05"),
                "Пекарня": "B1",
                "Категория": "Cat1",
                "Номенклатура": "P1",
                "ДеньНедели": 0,
                "Продано": 2.0,
                "sku_sales_total": 2.0,
                "bakery_sales_total": 100.0,
                "category_sales_total": 20.0,
                "good_execution_day": False,
                "early_stop_flag": False,
                "stockout_like_hours": 2,
            }
        ]
    )


def _make_blended_profiles() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Пекарня": "B1",
                "Категория": "Cat1",
                "Номенклатура": "P1",
                "ДеньНедели": 0,
                "final_expected_share": 0.2,
                "share_source_primary": "bakery_category",
                "blend_confidence_score": 0.8,
            }
        ]
    )


def test_attainable_baseline_uses_category_when_category_profile_primary():
    baseline = build_attainable_baseline(_make_profile_input(), _make_blended_profiles())
    row = baseline.iloc[0]
    assert row["attainable_sales_from_category"] == 4.0
    assert row["attainable_sales_baseline"] == 4.0


def test_attainable_baseline_marks_opportunity():
    baseline = build_attainable_baseline(_make_profile_input(), _make_blended_profiles())
    row = baseline.iloc[0]
    assert row["attainable_gap"] == 2.0
    assert bool(row["opportunity_flag"])

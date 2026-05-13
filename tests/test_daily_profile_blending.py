"""Tests for profile blending / shrinkage layer."""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments_v2.daily_profile_blending import build_blended_profiles  # noqa: E402


def _make_profiles() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "profile_level": "bakery_sku",
                "Пекарня": "B1",
                "Номенклатура": "P1",
                "Категория": "Cat1",
                "ДеньНедели": 0,
                "mean_share_of_bakery": 0.10,
                "mean_share_of_category": 0.30,
                "n_good_days": 3,
                "profile_reliability_score": 0.20,
            },
            {
                "profile_level": "sku_global",
                "Номенклатура": "P1",
                "Категория": "Cat1",
                "ДеньНедели": 0,
                "mean_share_of_bakery": 0.08,
                "mean_share_of_category": 0.25,
                "n_good_days": 20,
                "profile_reliability_score": 0.80,
            },
            {
                "profile_level": "bakery_category",
                "Пекарня": "B1",
                "Категория": "Cat1",
                "ДеньНедели": 0,
                "mean_share_of_bakery": 0.40,
                "mean_share_of_category": 0.20,
                "n_good_days": 15,
                "profile_reliability_score": 0.60,
            },
            {
                "profile_level": "category_global",
                "Категория": "Cat1",
                "ДеньНедели": 0,
                "mean_share_of_bakery": 0.35,
                "mean_share_of_category": 0.15,
                "n_good_days": 40,
                "profile_reliability_score": 0.90,
            },
        ]
    )


def test_blending_builds_final_expected_share():
    blended = build_blended_profiles(_make_profiles())
    row = blended.iloc[0]
    assert 0 < row["final_expected_share"] < 1
    assert row["share_source_primary"] in {
        "bakery_sku",
        "sku_global",
        "bakery_category",
        "category_global",
    }


def test_blending_downweights_weak_local_profile():
    blended = build_blended_profiles(_make_profiles())
    row = blended.iloc[0]
    assert row["w_bakery_sku"] < row["w_sku_global"]

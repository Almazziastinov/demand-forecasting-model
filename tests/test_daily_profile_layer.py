"""Tests for hierarchical daily share profiles."""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments_v2.daily_profile_layer import (  # noqa: E402
    BAKERY_COL,
    CATEGORY_COL,
    DATE_COL,
    DOW_COL,
    PRODUCT_COL,
    TARGET_COL,
    build_daily_profiles,
    build_profile_input,
)


def _make_backbone() -> pd.DataFrame:
    rows = []
    dates = pd.to_datetime(["2026-01-05", "2026-01-12", "2026-01-19"])
    for date in dates:
        rows.extend(
            [
                {
                    DATE_COL: date,
                    BAKERY_COL: "B1",
                    CATEGORY_COL: "Cat1",
                    PRODUCT_COL: "P1",
                    TARGET_COL: 2.0,
                    DOW_COL: int(date.dayofweek),
                },
                {
                    DATE_COL: date,
                    BAKERY_COL: "B1",
                    CATEGORY_COL: "Cat1",
                    PRODUCT_COL: "P2",
                    TARGET_COL: 6.0,
                    DOW_COL: int(date.dayofweek),
                },
            ]
        )
    return pd.DataFrame(rows)


def _make_availability() -> pd.DataFrame:
    rows = []
    dates = pd.to_datetime(["2026-01-05", "2026-01-12", "2026-01-19"])
    for date in dates:
        for product, sales in [("P1", 2.0), ("P2", 6.0)]:
            rows.append(
                {
                    DATE_COL: date,
                    BAKERY_COL: "B1",
                    CATEGORY_COL: "Cat1",
                    PRODUCT_COL: product,
                    "sku_sales_total": sales,
                    "bakery_sales_total": 8.0,
                    "good_execution_day": True,
                    "availability_score": 0.9,
                    "stockout_like_hours": 0,
                    "zero_under_traffic_hours": 0,
                    "early_stop_flag": False,
                }
            )
    return pd.DataFrame(rows)


def test_build_profile_input_calculates_shares():
    profile_input = build_profile_input(_make_backbone(), _make_availability())
    p1 = profile_input[profile_input[PRODUCT_COL] == "P1"].iloc[0]
    p2 = profile_input[profile_input[PRODUCT_COL] == "P2"].iloc[0]

    assert p1["share_of_bakery"] == 0.25
    assert p2["share_of_bakery"] == 0.75
    assert p1["share_of_category"] == 0.25


def test_build_daily_profiles_creates_hierarchical_levels():
    profile_input = build_profile_input(_make_backbone(), _make_availability())
    profiles = build_daily_profiles(profile_input)

    levels = set(profiles["profile_level"].unique())
    assert {"bakery_sku", "sku_global", "bakery_category", "category_global"} <= levels

    row = profiles[
        (profiles["profile_level"] == "bakery_sku")
        & (profiles[BAKERY_COL] == "B1")
        & (profiles[PRODUCT_COL] == "P1")
    ].iloc[0]
    assert row["n_good_days"] == 3
    assert round(float(row["mean_share_of_bakery"]), 4) == 0.25

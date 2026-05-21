import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "analysis" / "normative_anchor_analysis.py"
SPEC = importlib.util.spec_from_file_location("normative_anchor_analysis", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_build_anchor_profile_identifies_bakery_driven_anchor() -> None:
    dates = pd.date_range("2026-01-01", periods=14, freq="D")
    rows = pd.DataFrame(
        {
            "date": dates,
            "bakery_id": ["b1"] * len(dates),
            "product_id": ["p1"] * len(dates),
            "bakery_name": ["Bakery 1"] * len(dates),
            "product_name": ["SKU 1"] * len(dates),
            "category_name": ["Bread"] * len(dates),
            "city": ["Kazan"] * len(dates),
            "dow": [d.weekday() for d in dates],
            "observed_sales_qty": [10, 12, 11, 13, 12, 8, 7, 10, 12, 11, 13, 12, 8, 7],
            "bakery_sales_qty_total": [100, 120, 110, 130, 120, 80, 70, 100, 120, 110, 130, 120, 80, 70],
            "category_sales_qty_in_bakery_day": [20, 24, 22, 26, 24, 16, 14, 20, 24, 22, 26, 24, 16, 14],
            "sku_sales_share_in_bakery_day": [0.10] * len(dates),
            "sku_sales_share_in_category_day": [0.50] * len(dates),
        }
    )
    segment_map = pd.DataFrame(
        {
            "bakery_id": ["b1"],
            "product_id": ["p1"],
            "primary_segment": ["bakery_driven"],
            "weekly_seasonality_strength": [0.9],
            "weekday_profile_stability": [0.8],
            "release_coverage_share": [0.1],
            "release_corr_with_sales": [0.1],
            "bakery_sales_corr": [0.95],
            "zero_share": [0.0],
            "predictability_score": [0.7],
        }
    )

    profile = MODULE.build_anchor_profile(rows, segment_map)
    row = profile.iloc[0]

    assert row["dominant_anchor"] == "bakery_scale"
    assert row["bakery_anchor_strength"] > row["release_anchor_strength"]


def test_build_anchor_summary_returns_segment_rollup() -> None:
    profile = pd.DataFrame(
        {
            "primary_segment": ["stable", "stable", "intermittent"],
            "observed_mean": [10.0, 12.0, 1.0],
            "zero_share": [0.1, 0.2, 0.8],
            "self_pattern_strength": [0.8, 0.7, 0.2],
            "release_anchor_strength": [0.9, 0.85, 0.1],
            "bakery_anchor_strength": [0.6, 0.65, 0.2],
            "category_anchor_strength": [0.7, 0.75, 0.3],
            "product_weekday_alignment": [0.9, 0.8, 0.4],
            "share_in_bakery_cv": [0.1, 0.2, 0.9],
            "share_in_category_cv": [0.2, 0.3, 0.8],
            "dominant_anchor": ["release", "release", "self_pattern"],
        }
    )

    summary = MODULE.build_anchor_summary(profile)
    dominance = MODULE.build_anchor_dominance(profile)

    stable = summary.loc[summary["primary_segment"] == "stable"].iloc[0]
    assert stable["pairs"] == 2
    assert stable["release_anchor_strength"] > stable["bakery_anchor_strength"]
    assert dominance.loc[dominance["primary_segment"] == "stable", "share"].max() == 1.0

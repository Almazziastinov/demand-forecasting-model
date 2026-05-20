import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "experiments_v2"
    / "75_normative_demand_map"
    / "run.py"
)

SPEC = importlib.util.spec_from_file_location("normative_demand_map_run", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_build_pair_profile_map_returns_expected_segments() -> None:
    dates = pd.date_range("2026-01-01", periods=14, freq="D")

    stable_sales = [10, 12, 11, 13, 12, 8, 7, 10, 12, 11, 13, 12, 8, 7]
    intermittent_sales = [0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 4, 0, 0]

    stable_rows = pd.DataFrame(
        {
            "date": dates,
            "bakery_id": ["b1"] * len(dates),
            "bakery_name": ["Bakery A"] * len(dates),
            "category_name": ["Bread"] * len(dates),
            "product_id": ["p1"] * len(dates),
            "product_name": ["SKU Stable"] * len(dates),
            "city": ["Kazan"] * len(dates),
            "dow": [d.weekday() for d in dates],
            "observed_sales_qty": stable_sales,
            "bakery_sales_qty_total": [v * 10 for v in stable_sales],
            "sku_sales_share_in_bakery_day": [0.10] * len(dates),
            "release_present_flag": [1] * len(dates),
            "moves_present_flag": [1] * len(dates),
            "release_qty": [v + 2 for v in stable_sales],
            "net_move_qty": [1.0] * len(dates),
            "row_quality_score": [1.0] * len(dates),
        }
    )

    intermittent_rows = pd.DataFrame(
        {
            "date": dates,
            "bakery_id": ["b1"] * len(dates),
            "bakery_name": ["Bakery A"] * len(dates),
            "category_name": ["Bread"] * len(dates),
            "product_id": ["p2"] * len(dates),
            "product_name": ["SKU Intermittent"] * len(dates),
            "city": ["Kazan"] * len(dates),
            "dow": [d.weekday() for d in dates],
            "observed_sales_qty": intermittent_sales,
            "bakery_sales_qty_total": [max(v, 1) * 5 for v in intermittent_sales],
            "sku_sales_share_in_bakery_day": [0.02 if v > 0 else 0.0 for v in intermittent_sales],
            "release_present_flag": [0] * len(dates),
            "moves_present_flag": [0] * len(dates),
            "release_qty": [0.0] * len(dates),
            "net_move_qty": [0.0] * len(dates),
            "row_quality_score": [0.9] * len(dates),
        }
    )

    df = pd.concat([stable_rows, intermittent_rows], ignore_index=True)
    profile_map = MODULE.build_pair_profile_map(df)

    assert len(profile_map) == 2

    stable = profile_map.loc[profile_map["product_name"] == "SKU Stable"].iloc[0]
    intermittent = profile_map.loc[profile_map["product_name"] == "SKU Intermittent"].iloc[0]

    assert stable["primary_segment"] == "stable"
    assert stable["lag7_r2"] > 0.9
    assert stable["predictability_score"] > intermittent["predictability_score"]

    assert intermittent["primary_segment"] == "intermittent"
    assert intermittent["zero_share"] > 0.6


def test_assign_segment_high_censoring_priority() -> None:
    row = pd.Series(
        {
            "zero_share": 0.1,
            "active_days_share": 0.9,
            "release_coverage_share": 0.8,
            "lag7_r2": 0.7,
            "cv_sales": 1.8,
            "bakery_sales_corr": 0.2,
            "trend_corr": 0.1,
            "weekly_seasonality_strength": 0.4,
            "weekly_amplitude_cv": 0.2,
        }
    )
    assert MODULE.assign_segment(row) == "high_censoring"

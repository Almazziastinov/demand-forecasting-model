import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "experiments_v2"
    / "77_segmented_normative_constructors"
    / "run.py"
)
SPEC = importlib.util.spec_from_file_location("segmented_normative_run", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_build_segmented_normative_builds_stable_and_bakery_driven_candidates() -> None:
    dates = pd.date_range("2026-01-01", periods=21, freq="D")
    stable_rows = pd.DataFrame(
        {
            "date": dates,
            "bakery_id": ["b1"] * len(dates),
            "product_id": ["p1"] * len(dates),
            "dow": [d.weekday() for d in dates],
            "observed_sales_qty": [10, 12, 11, 13, 12, 8, 7] * 3,
            "release_qty": [11, 12, 11, 14, 13, 8, 7] * 3,
            "bakery_sales_qty_total": [100, 120, 110, 130, 120, 80, 70] * 3,
            "sku_sales_share_in_bakery_day": [0.1] * len(dates),
        }
    )
    bakery_rows = pd.DataFrame(
        {
            "date": dates,
            "bakery_id": ["b1"] * len(dates),
            "product_id": ["p2"] * len(dates),
            "dow": [d.weekday() for d in dates],
            "observed_sales_qty": [20, 24, 22, 26, 24, 16, 14] * 3,
            "release_qty": [0.0] * len(dates),
            "bakery_sales_qty_total": [100, 120, 110, 130, 120, 80, 70] * 3,
            "sku_sales_share_in_bakery_day": [0.2] * len(dates),
        }
    )
    daily_df = pd.concat([stable_rows, bakery_rows], ignore_index=True)
    segment_map = pd.DataFrame(
        {
            "bakery_id": ["b1", "b1"],
            "product_id": ["p1", "p2"],
            "primary_segment": ["stable", "bakery_driven"],
        }
    )

    result = MODULE.build_segmented_normative(daily_df, segment_map)
    stable = result.loc[result["product_id"] == "p1"]
    bakery = result.loc[result["product_id"] == "p2"]

    assert stable["segment_constructor_name"].eq("stable_release_weekday").all()
    assert bakery["segment_constructor_name"].eq("bakery_total_x_sku_share").all()
    assert stable["segment_normative_candidate"].notna().all()
    assert bakery["segment_normative_candidate"].notna().all()
    assert (bakery["segment_normative_candidate"] >= 0).all()


def test_build_pair_summary_returns_correlations_for_built_pairs() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=7, freq="D"),
            "bakery_id": ["b1"] * 7,
            "product_id": ["p1"] * 7,
            "primary_segment": ["stable"] * 7,
            "segment_constructor_name": ["stable_release_weekday"] * 7,
            "observed_sales_qty": [10, 12, 11, 13, 12, 8, 7],
            "segment_normative_candidate": [10, 11, 10, 12, 11, 8, 7],
            "release_qty": [11, 12, 11, 13, 12, 8, 7],
            "bakery_sales_qty_total": [100, 120, 110, 130, 120, 80, 70],
        }
    )

    pair_summary = MODULE.build_pair_summary(df)
    row = pair_summary.iloc[0]
    assert row["candidate_corr_with_observed"] > 0.8
    assert row["candidate_corr_with_release"] > 0.8

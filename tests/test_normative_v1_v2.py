import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "experiments_v2"
    / "76_normative_v1_v2"
    / "run.py"
)

SPEC = importlib.util.spec_from_file_location("normative_v1_v2_run", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_build_normative_candidates_preserves_rows_and_assigns_expected_variant() -> None:
    dates = pd.date_range("2026-01-01", periods=21, freq="D")
    stable_sales = [10, 12, 11, 13, 12, 8, 7] * 3
    amplitude_sales = [8, 14, 10, 16, 12, 6, 5, 9, 18, 12, 20, 15, 7, 6, 10, 22, 13, 24, 16, 8, 6]

    stable = pd.DataFrame(
        {
            "date": dates,
            "bakery_id": ["b1"] * len(dates),
            "product_id": ["p1"] * len(dates),
            "dow": [d.weekday() for d in dates],
            "observed_sales_qty": stable_sales,
        }
    )
    amplitude = pd.DataFrame(
        {
            "date": dates,
            "bakery_id": ["b1"] * len(dates),
            "product_id": ["p2"] * len(dates),
            "dow": [d.weekday() for d in dates],
            "observed_sales_qty": amplitude_sales,
        }
    )
    daily = pd.concat([stable, amplitude], ignore_index=True)

    segment_map = pd.DataFrame(
        {
            "bakery_id": ["b1", "b1"],
            "product_id": ["p1", "p2"],
            "primary_segment": ["stable", "amplitude_unstable"],
            "predictability_score": [0.8, 0.5],
        }
    )

    result = MODULE.build_normative_candidates(daily, segment_map)

    assert len(result) == len(daily)
    assert {"normative_v1", "normative_v2", "normative_candidate", "normative_candidate_name"} <= set(result.columns)
    assert (result["normative_candidate"] >= 0).all()

    stable_result = result.loc[result["product_id"] == "p1"]
    amplitude_result = result.loc[result["product_id"] == "p2"]

    assert stable_result["normative_candidate_name"].eq("normative_v1").all()
    assert amplitude_result["normative_candidate_name"].eq("normative_v2").all()


def test_build_pair_summary_shows_smoother_candidate_than_observed_for_stable_series() -> None:
    dates = pd.date_range("2026-02-01", periods=21, freq="D")
    observed = [10, 12, 11, 13, 12, 8, 7] * 3
    daily = pd.DataFrame(
        {
            "date": dates,
            "bakery_id": ["b1"] * len(dates),
            "product_id": ["p1"] * len(dates),
            "dow": [d.weekday() for d in dates],
            "observed_sales_qty": observed,
        }
    )
    segment_map = pd.DataFrame(
        {
            "bakery_id": ["b1"],
            "product_id": ["p1"],
            "primary_segment": ["stable"],
            "predictability_score": [0.9],
        }
    )

    with_normative = MODULE.build_normative_candidates(daily, segment_map)
    pair_summary = MODULE.build_pair_summary(with_normative)

    row = pair_summary.iloc[0]
    assert row["normative_candidate_name"] == "normative_v1"
    assert row["normative_v1_cv"] <= row["observed_cv"]

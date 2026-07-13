"""Tests for the rolling bakery-day bias correction."""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.forecast_publish.rolling_bakery_bias import (  # noqa: E402
    build_effective_bias_table,
    compute_rolling_bias,
)


def test_compute_rolling_bias_averages_residuals_per_bakery():
    perf = pd.DataFrame(
        {
            "bakery_id": [1, 1, 1, 2, 2, 2],
            "forecast_base": [100.0, 100.0, 100.0, 200.0, 200.0, 200.0],
            "actual_qty": [110.0, 90.0, 100.0, 220.0, 220.0, 220.0],
        }
    )
    result = compute_rolling_bias(perf, min_days=3)
    result = result.set_index("bakery_id")

    assert result.loc[1, "bias"] == 0.0
    assert result.loc[2, "bias"] == 20.0
    assert result.loc[1, "n_days"] == 3
    assert result.loc[2, "n_days"] == 3


def test_compute_rolling_bias_drops_bakeries_below_min_days():
    perf = pd.DataFrame(
        {
            "bakery_id": [1, 1, 2, 2, 2],
            "forecast_base": [100.0, 100.0, 50.0, 50.0, 50.0],
            "actual_qty": [90.0, 90.0, 60.0, 60.0, 60.0],
        }
    )
    result = compute_rolling_bias(perf, min_days=3)

    assert set(result["bakery_id"]) == {2}


def test_compute_rolling_bias_empty_input_returns_empty_frame():
    result = compute_rolling_bias(pd.DataFrame(), min_days=3)
    assert result.empty
    assert list(result.columns) == ["bakery_id", "bias", "n_days"]


def test_build_effective_bias_table_prefers_rolling_over_static():
    rolling = pd.DataFrame({"bakery_id": [1], "bias": [15.0]})
    static = pd.DataFrame({"bakery_id": [1, 2], "bias": [-5.0, -8.0]})

    result = build_effective_bias_table(
        rolling, static, bakery_ids=[1, 2, 3]
    ).set_index("bakery_id")

    assert result.loc[1, "bias"] == 15.0   # rolling available -> wins over static
    assert result.loc[2, "bias"] == -8.0   # no rolling -> falls back to static
    assert result.loc[3, "bias"] == 0.0    # neither -> defaults to 0


def test_build_effective_bias_table_handles_empty_static_and_rolling():
    empty = pd.DataFrame(columns=["bakery_id", "bias"])
    result = build_effective_bias_table(empty, empty, bakery_ids=[1, 2])

    assert (result["bias"] == 0.0).all()
    assert sorted(result["bakery_id"]) == [1, 2]

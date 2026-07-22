from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_stockout_direction_shadow import (
    MEMBERSHIP_SEED_REQUIRED_SCOPES,
    MEMBERSHIP_SEED_VARIANT,
    load_membership_seed_candidate,
)


def test_load_membership_seed_candidate_requires_three_winning_folds(tmp_path):
    rows = []
    for scope in MEMBERSHIP_SEED_REQUIRED_SCOPES:
        for cutoff, delta in zip(["a", "b", "c"], [-0.1, -0.2, -0.3], strict=True):
            rows.append(
                {
                    "cutoff": cutoff,
                    "variant": MEMBERSHIP_SEED_VARIANT,
                    "scope": scope,
                    "delta": delta,
                    "under": -1.0,
                    "over": 0.5,
                }
            )
    path = tmp_path / "summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    result = load_membership_seed_candidate(
        path,
        evaluated_through="2026-07-19",
    )

    assert result["historical_gates_pass"]
    assert result["prospective_days_observed"] == 0
    assert result["status"] == "historical_pass_pending_prospective"


def test_load_membership_seed_candidate_rejects_missing_scope(tmp_path):
    path = tmp_path / "summary.csv"
    pd.DataFrame(
        {
            "cutoff": ["a"],
            "variant": [MEMBERSHIP_SEED_VARIANT],
            "scope": ["all_holdout"],
            "delta": [-0.1],
            "under": [-1.0],
            "over": [0.5],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="scope not found"):
        load_membership_seed_candidate(path, evaluated_through="2026-07-19")

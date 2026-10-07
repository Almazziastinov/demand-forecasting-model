"""The volatility gate distinguishes stable and noisy weekly histories."""

from __future__ import annotations

import pandas as pd

from src.model_tournament.volatility_guard import (
    attach_supply_context,
    attach_volatility_guard,
)


def _policy(values: list[float], *, bakery_id: int = 1) -> pd.DataFrame:
    dates = pd.date_range("2026-01-05", periods=len(values), freq="7D")
    target = len(values) - 2
    rows = pd.DataFrame(
        {
            "date": dates,
            "bakery_id": bakery_id,
            "product_id": 10,
            "observed_sales_qty": values,
            "bridge_reference": 125.0,
            "regularization_candidate": False,
            "release_uplift": 0.0,
            "unverified_uplift": 0.0,
        }
    )
    rows.loc[target, "regularization_candidate"] = True
    rows.loc[target, "unverified_uplift"] = 45.0
    return rows


def test_stable_120_80_130_passes_but_noisy_series_does_not() -> None:
    stable = _policy([120.0] * 8 + [80.0, 130.0])
    noisy = _policy(
        [70.0, 170.0, 80.0, 160.0, 90.0, 150.0, 100.0, 120.0, 80.0, 130.0],
        bakery_id=2,
    )
    result = attach_volatility_guard(pd.concat([stable, noisy], ignore_index=True))
    selected = result.loc[result["regularization_candidate"]].set_index("bakery_id")
    assert selected.loc[1, "volatility_guard_pass"]
    assert selected.loc[1, "guarded_unverified_uplift"] == 45.0
    assert not selected.loc[2, "volatility_guard_pass"]
    assert selected.loc[2, "guarded_unverified_uplift"] == 0.0


def test_future_values_cannot_change_past_volatility_score() -> None:
    source = _policy([120.0] * 8 + [80.0, 130.0])
    original = attach_volatility_guard(source)
    changed = source.copy()
    changed.loc[changed.index[-1], "observed_sales_qty"] = 1000.0
    later = attach_volatility_guard(changed)
    target = source.index[-2]
    assert original.loc[target, "dip_z_score"] == later.loc[target, "dip_z_score"]
    assert (
        original.loc[target, "volatility_guard_pass"]
        == later.loc[target, "volatility_guard_pass"]
    )


def test_too_few_past_weeks_are_rejected() -> None:
    source = _policy([120.0] * 5 + [80.0, 130.0])
    result = attach_volatility_guard(source)
    assert not result.loc[source.index[-2], "volatility_guard_pass"]


def test_incoming_moves_can_explain_a_sales_dip_without_release_drop() -> None:
    source = _policy([120.0] * 8 + [80.0, 130.0])
    source["reason_code"] = "unexplained_weekly_dip"
    guarded = attach_volatility_guard(source)
    flows = source[["date", "bakery_id", "product_id"]].copy()
    flows["release_qty"] = 0.0
    flows["incoming_move_qty"] = 120.0
    flows["outgoing_move_qty"] = 0.0
    flows.loc[flows.index[-2], "incoming_move_qty"] = 55.0
    result = attach_supply_context(guarded, flows)
    case = result.loc[result.index[-2]]
    assert case["supply_shortfall_candidate"]
    assert case["guarded_supply_uplift"] == 45.0
    assert case["guarded_no_supply_signal_uplift"] == 0.0

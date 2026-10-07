"""Inventory uncertainty is causal and never asserts physical stock."""

import pandas as pd
import pytest

from src.model_tournament.inventory_evidence import inventory_uncertainty


def _row(date: str, *, release: float, sales: float) -> dict[str, object]:
    return {
        "date": date,
        "bakery_id": 1,
        "product_id": 2,
        "release_qty": release,
        "incoming_move_qty": 0.0,
        "outgoing_move_qty": 0.0,
        "observed_sales_qty": sales,
        "written_off_qty": 0.0,
    }


def test_positive_previous_residual_withholds_restoration() -> None:
    frame = pd.DataFrame(
        [
            _row("2026-01-01", release=12, sales=7),
            _row("2026-01-02", release=10, sales=10),
        ]
    )
    result = inventory_uncertainty(frame)
    second = result.iloc[1]
    assert second["prior_day_positive_residual_qty"] == pytest.approx(5)
    assert bool(second["possible_carryover"])
    assert not bool(second["high_confidence_inventory_context"])


def test_negative_residual_is_ambiguous_but_gap_does_not_carry() -> None:
    frame = pd.DataFrame(
        [
            _row("2026-01-01", release=12, sales=7),
            _row("2026-01-03", release=5, sales=10),
        ]
    )
    second = inventory_uncertainty(frame).iloc[1]
    assert second["prior_day_positive_residual_qty"] == pytest.approx(0)
    assert bool(second["same_day_supply_inconsistent"])


def test_later_observations_do_not_change_earlier_flags() -> None:
    frame = pd.DataFrame(
        [
            _row("2026-01-01", release=12, sales=7),
            _row("2026-01-02", release=10, sales=10),
        ]
    )
    earlier = inventory_uncertainty(frame)
    with_future = inventory_uncertainty(
        pd.concat([frame, pd.DataFrame([_row("2026-01-03", release=5, sales=100)])])
    )
    pd.testing.assert_frame_equal(earlier, with_future.iloc[:2].reset_index(drop=True))

"""Research-control definitions must stay identical across time windows."""

from __future__ import annotations

import pandas as pd

from scripts.analyze_stg_intraday_absence import HOURS
from scripts.audit_intraday_gap_controls import eligible_fixed_tails, flow_band


def test_flow_band_does_not_treat_surplus_as_verified_stock() -> None:
    assert flow_band(10.0, 10.0, 0.0) == "near_supply_no_writeoff"
    assert flow_band(5.0, 10.0, 2.0) == "flow_surplus_with_writeoff"
    assert flow_band(5.0, 10.0, 0.0) == "other"
    assert flow_band(10.0, 0.0, 0.0) == "unknown"


def test_fixed_tail_compares_same_hours_in_control_and_test() -> None:
    rows = []
    for date in ("2026-06-20", "2026-07-10"):
        for hour in HOURS:
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "bakery_id": 20,
                    "product_id": 36,
                    "hour": hour,
                    "sold": 10.0 if hour == 17 else 0.0,
                    "bakery_hour_sales": 30.0,
                    "expected": 2.0,
                    "reference_days": 4,
                    "release_qty": 10.0,
                    "incoming_move_qty": 0.0,
                    "outgoing_move_qty": 0.0,
                    "written_off_qty": 0.0,
                }
            )
    tails = eligible_fixed_tails(pd.DataFrame(rows))
    assert tails["period"].tolist() == ["control", "test"]
    assert tails["zero_tail"].tolist() == [True, True]

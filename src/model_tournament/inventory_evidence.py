"""Conservative inventory-uncertainty flags for a research demand target.

The daily flow residual is not a measured physical stock balance. In
particular, yesterday's positive residual may or may not carry into today.
We therefore use it only to *withhold* uncertain restoration, never to
claim a confirmed stockout or add assumed supply to sales.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.model_tournament.fixed_origin import KEYS


FLOW_COLUMNS = [
    "release_qty",
    "incoming_move_qty",
    "outgoing_move_qty",
    "observed_sales_qty",
    "written_off_qty",
]


def inventory_uncertainty(
    panel: pd.DataFrame, *, tolerance: float = 1.0
) -> pd.DataFrame:
    """Return causal day/pair flags, without interpreting sparse days as stock.

    A positive previous-day flow residual may represent carryover. A negative
    current residual means observed sales exceed same-day recorded supply.
    Either case makes the old sell-through stockout heuristic ambiguous.
    """
    required = {"date", *KEYS, *FLOW_COLUMNS}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"Inventory evidence lacks columns: {missing}")
    if tolerance < 0:
        raise ValueError("Tolerance must be nonnegative")
    work = panel[["date", *KEYS, *FLOW_COLUMNS]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="raise").dt.normalize()
    if work.duplicated(["date", *KEYS]).any():
        raise ValueError("Inventory evidence has duplicate SKU-days")
    for column in FLOW_COLUMNS:
        values = pd.to_numeric(work[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all() or values.lt(0).any():
            raise ValueError(f"Inventory evidence has invalid {column}")
        work[column] = values
    work = work.sort_values([*KEYS, "date"]).reset_index(drop=True)
    residual = (
        work["release_qty"]
        + work["incoming_move_qty"]
        - work["outgoing_move_qty"]
        - work["observed_sales_qty"]
        - work["written_off_qty"]
    )
    grouped = work.groupby(KEYS, sort=False)
    previous_day = grouped["date"].shift(1)
    previous_residual = residual.groupby(
        [work["bakery_id"], work["product_id"]], sort=False
    ).shift(1)
    consecutive = (work["date"] - previous_day).dt.days.eq(1)
    work["prior_day_positive_residual_qty"] = (
        previous_residual.clip(lower=0).where(consecutive, 0).fillna(0.0)
    )
    work["same_day_flow_residual_qty"] = residual
    work["possible_carryover"] = work["prior_day_positive_residual_qty"].gt(tolerance)
    work["same_day_supply_inconsistent"] = residual.lt(-tolerance)
    work["high_confidence_inventory_context"] = ~(
        work["possible_carryover"] | work["same_day_supply_inconsistent"]
    )
    return work[
        [
            "date", *KEYS, "prior_day_positive_residual_qty",
            "same_day_flow_residual_qty", "possible_carryover",
            "same_day_supply_inconsistent", "high_confidence_inventory_context",
        ]
    ]

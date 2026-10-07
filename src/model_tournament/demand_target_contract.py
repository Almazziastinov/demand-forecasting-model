"""Research demand-target invariants with auditable up/down adjustments."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_demand_target_contract(
    frame: pd.DataFrame,
    *,
    sales_column: str = "demand_lower_bound",
    demand_column: str = "demand_point_estimate",
    uplift_column: str | None = "imputed_demand",
    reduction_column: str | None = None,
    require_global_uplift: bool = False,
    tolerance: float = 1e-8,
) -> None:
    """Validate point targets and optionally gate a complete target panel.

    Individual SKU-days may be below observed sales. Only set
    ``require_global_uplift`` on the full declared panel, not a batch or case.
    Keep positive restoration and outlier reduction in distinct columns.
    """
    columns = [sales_column, demand_column]
    if uplift_column is not None:
        columns.append(uplift_column)
    if reduction_column is not None:
        columns.append(reduction_column)
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Demand target lacks columns: {missing}")
    if frame.empty:
        raise ValueError("Demand target panel is empty")
    values = {
        column: pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        for column in columns
    }
    if any(not np.isfinite(value).all() for value in values.values()):
        raise ValueError("Demand target contains missing or non-finite quantities")
    sales = values[sales_column]
    demand = values[demand_column]
    if (sales < -tolerance).any():
        raise ValueError("Observed sales cannot be negative")
    if (demand < -tolerance).any():
        raise ValueError("Reconstructed demand cannot be negative")
    uplift = np.zeros(len(frame), dtype=float)
    if uplift_column is not None:
        uplift = values[uplift_column]
        if (uplift < -tolerance).any():
            raise ValueError("Imputed demand cannot be negative")
    reduction = np.zeros(len(frame), dtype=float)
    if reduction_column is not None:
        reduction = values[reduction_column]
        if (reduction < -tolerance).any():
            raise ValueError("Outlier reduction cannot be negative")
    if not np.allclose(
        demand, sales + uplift - reduction, atol=tolerance, rtol=0.0
    ):
        raise ValueError("Demand must equal sales plus restoration minus reduction")
    if require_global_uplift and demand.sum() <= sales.sum() + tolerance:
        raise ValueError("Total restored demand must exceed total observed sales")

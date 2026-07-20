from __future__ import annotations

import pandas as pd

from scripts.experiment_dynamic_sku_allocation import apply_constrained_allocation


def test_apply_constrained_allocation_preserves_bakery_total() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-01", "2026-06-01"]),
            "bakery_id": [1, 1],
            "baseline_share": [0.4, 0.6],
            "baseline_total": [100.0, 100.0],
            "correction": [1.0, -1.0],
        }
    )
    result = apply_constrained_allocation(
        frame, correction_column="correction", strength=0.5
    )
    assert round(result["adjusted_forecast_qty"].sum(), 8) == 100.0
    assert result.iloc[0]["adjusted_share"] > 0.4

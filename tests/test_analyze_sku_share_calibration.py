from __future__ import annotations

import pandas as pd

from scripts.analyze_sku_share_calibration import build_share_comparison


def test_share_comparison_removes_bakery_total_error() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-01"]),
            "bakery_id": [1],
            "forecast_qty": [10.0],
            "daily_sold": [20.0],
        }
    )
    bakery = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-01"]),
            "bakery_id": [1],
            "bakery_actual_qty": [200.0],
            "bakery_forecast_qty": [100.0],
        }
    )
    totals = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-01"]),
            "bakery_id": [1],
            "sku_forecast_total_qty": [100.0],
        }
    )

    result = build_share_comparison(frame, bakery, totals)

    assert result.loc[0, "forecast_share"] == 0.1
    assert result.loc[0, "allocated_qty_at_actual_bakery_total"] == 20.0
    assert result.loc[0, "allocation_bias_qty"] == 0.0

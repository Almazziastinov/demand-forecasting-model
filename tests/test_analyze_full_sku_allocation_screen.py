from __future__ import annotations

import pandas as pd

from scripts.analyze_full_sku_allocation_screen import aggregate_pairs


def test_aggregate_pairs_separates_missing_and_low_allocations() -> None:
    rows = []
    for product_id, allocated in [(1, 0.0), (2, 60.0), (3, 105.0)]:
        for day in range(5):
            rows.append(
                {
                    "date": pd.Timestamp("2026-07-01") + pd.Timedelta(days=day),
                    "bakery_id": 10,
                    "bakery_name": "Test",
                    "product_id": product_id,
                    "product_name": f"SKU {product_id}",
                    "stockout_group": "confirmed_non_stockout",
                    "daily_sold": 20.0,
                    "allocated_qty_at_actual_bakery_total": allocated / 5,
                    "forecast_qty": allocated / 5,
                }
            )

    result = aggregate_pairs(pd.DataFrame(rows)).set_index("product_id")

    assert result.loc[1, "issue_type"] == "missing_allocation"
    assert result.loc[2, "issue_type"] == "persistent_local_underallocation"
    assert result.loc[3, "issue_type"] == "no_material_issue"


from __future__ import annotations

import pandas as pd

from scripts.backtest_pseudo_stockout_reconstruction import build_guarded_hybrid


def test_guarded_hybrid_caps_case_level_imputation() -> None:
    base = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-03-22", "2026-03-22"]),
            "bakery_id": [1, 1],
            "product_id": [10, 10],
            "sold": [2.0, 0.0],
            "sold_demand": [2.0, 10.0],
        }
    )
    share = base.copy()
    share["sold_demand"] = [2.0, 8.0]
    result = build_guarded_hybrid(base, share, max_case_uplift_ratio=0.75)
    assert round(result["imputed_demand"].sum(), 6) == 3.0

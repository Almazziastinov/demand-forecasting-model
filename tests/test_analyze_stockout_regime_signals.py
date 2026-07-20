from __future__ import annotations

import pandas as pd

from scripts.analyze_stockout_regime_signals import build_features


def test_lag_features_do_not_use_current_day() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-01", "2026-06-02"]),
            "bakery_id": [1, 1],
            "product_id": [10, 10],
            "stockout_group": ["confirmed_non_stockout", "clear_stockout"],
            "daily_sold": [10.0, 100.0],
            "observed_share": [0.1, 0.9],
            "forecast_qty": [10.0, 10.0],
            "forecast_share": [0.1, 0.1],
        }
    )

    result = build_features(frame)

    assert pd.isna(result.loc[0, "prior_sold_mean_7"])
    assert result.loc[1, "prior_sold_mean_7"] == 10.0
    assert result.loc[1, "prior_stockout_rate_7"] == 0.0

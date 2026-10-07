from __future__ import annotations

import pandas as pd

from scripts.train_old_stock_sellability_model import _rolling_share


def test_rolling_share_uses_only_prior_dates() -> None:
    dates = pd.date_range("2026-01-01", periods=8)
    rows = pd.DataFrame(
        {
            "date": dates,
            "product_id": [1] * 8,
            "old_qty": [1.0] * 7 + [1000.0],
            "label_qty": [10.0] * 7 + [1000.0],
        }
    )

    result = _rolling_share(rows, ["product_id"], "old_share_prior")
    day_eight = result.loc[result["date"].eq(dates[-1]), "old_share_prior"].iloc[0]

    assert day_eight == (7.0 + 5.0) / (70.0 + 100.0)


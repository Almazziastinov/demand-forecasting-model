from __future__ import annotations

import pandas as pd

from scripts.backtest_assortment_coverage_guard import (
    build_threshold_sensitivity,
    select_allowed_products_asof,
)


def test_select_allowed_products_uses_latest_batch_available_before_run() -> None:
    assortment = pd.DataFrame(
        {
            "city": ["A", "A", "A"],
            "product_id": [10, 20, 30],
            "valid_from": pd.to_datetime(
                ["2026-06-01", "2026-06-02", "2026-06-02"]
            ),
            "valid_to": [pd.NaT, pd.NaT, pd.NaT],
            "loaded_at": pd.to_datetime(
                [
                    "2026-06-01T01:00:00Z",
                    "2026-06-02T01:00:00Z",
                    "2026-06-02T04:00:00Z",
                ],
                utc=True,
            ),
        }
    )

    products, batch = select_allowed_products_asof(
        assortment,
        city="A",
        forecast_date=pd.Timestamp("2026-06-02"),
        run_generated_at=pd.Timestamp("2026-06-02T02:00:00Z"),
    )

    assert products == {20}
    assert batch == pd.Timestamp("2026-06-02")


def test_threshold_sensitivity_counts_established_known_cases() -> None:
    known = pd.DataFrame(
        {
            "recent_days_sold": [0, 1, 2, 3],
            "recent_qty": [0.0, 1.0, 2.0, 4.0],
        }
    )

    result = build_threshold_sensitivity(known)

    assert result["known_cases_caught"].tolist() == [3, 2, 1]

from __future__ import annotations

import pandas as pd
import pytest

from src.experiments_v2.sku_cold_start import (
    ColdStartConfig,
    apply_category_neutral_cold_start,
    build_cold_start_registry,
)


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-07-01", periods=5),
            "bakery_id": 20,
            "product_id": 11573,
            "sold_qty": [2.0, 4.0, 6.0, 8.0, 100.0],
            "forecast_qty": [0.0, 0.0, 0.0, 1.0, 100.0],
        }
    )


def test_registry_uses_only_prior_own_sales() -> None:
    registry = build_cold_start_registry(
        _history(),
        as_of_date="2026-07-05",
    )

    assert registry.iloc[0]["forecast_days"] == 1
    assert registry.iloc[0]["cold_start_floor"] == pytest.approx(7.778)


def test_mature_forecast_leaves_cold_start_registry() -> None:
    history = pd.concat([_history()] * 4, ignore_index=True)
    history["date"] = pd.date_range("2026-06-15", periods=len(history))
    history["forecast_qty"] = 1.0

    registry = build_cold_start_registry(
        history,
        as_of_date="2026-07-06",
        config=ColdStartConfig(max_forecast_days=13),
    )

    assert registry.empty


def test_application_preserves_category_total() -> None:
    forecast = pd.DataFrame(
        {
            "date": ["2026-07-05", "2026-07-05"],
            "bakery_id": [20, 20],
            "product_id": [11573, 1],
            "category_name": ["Сытная", "Сытная"],
            "forecast_qty": [1.0, 9.0],
        }
    )
    registry = pd.DataFrame(
        {
            "bakery_id": [20],
            "product_id": [11573],
            "cold_start_floor": [5.0],
        }
    )

    corrected = apply_category_neutral_cold_start(forecast, registry)

    assert corrected["cold_start_forecast_qty"].sum() == pytest.approx(10.0)
    assert corrected.loc[
        corrected["product_id"].eq(11573),
        "cold_start_forecast_qty",
    ].iloc[0] > 1.0

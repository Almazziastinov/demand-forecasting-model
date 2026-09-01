from __future__ import annotations

import pandas as pd
import pytest

from src.experiments_v2.sku_cold_start import (
    ColdStartConfig,
    add_missing_cold_start_candidates,
    apply_category_neutral_cold_start,
    apply_independent_cold_start,
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


def test_registry_discovers_products_without_hardcoded_ids() -> None:
    history = _history().copy()
    history["product_id"] = 11575

    registry = build_cold_start_registry(history, as_of_date="2026-07-05")

    assert registry["product_id"].tolist() == [11575]


def test_missing_effective_assortment_sku_can_receive_cold_start_floor() -> None:
    forecast = pd.DataFrame(
        {
            "date": ["2026-08-20"],
            "bakery_id": [270],
            "product_id": [1],
            "product_name": ["Existing"],
            "category_name": ["Пироги сладкие"],
            "forecast_qty": [10.0],
        }
    )
    candidates = pd.DataFrame(
        {
            "bakery_id": [270],
            "product_id": [11575],
            "product_name": ["Кексовый с манго"],
            "category_name": ["Пироги сладкие"],
        }
    )
    registry = pd.DataFrame(
        {"bakery_id": [270], "product_id": [11575], "cold_start_floor": [2.0]}
    )

    expanded = add_missing_cold_start_candidates(forecast, candidates)
    corrected = apply_category_neutral_cold_start(expanded, registry)

    assert corrected["cold_start_forecast_qty"].sum() == pytest.approx(10.0)
    assert corrected.loc[
        corrected["product_id"].eq(11575), "cold_start_forecast_qty"
    ].iat[0] > 0.0


def test_independent_cold_start_is_added_above_full_mature_total() -> None:
    forecast = pd.DataFrame(
        {
            "date": ["2026-08-20"] * 3,
            "bakery_id": [270] * 3,
            "product_id": [1, 2, 11575],
            "category_name": ["Сытная", "Сладкая", "Сладкая"],
            "forecast_qty": [60.0, 30.0, 10.0],
        }
    )
    registry = pd.DataFrame(
        {"bakery_id": [270], "product_id": [11575], "cold_start_floor": [8.0]}
    )

    result = apply_independent_cold_start(forecast, registry)
    mature = result[~result["is_cold_start"]]
    cold = result[result["is_cold_start"]]

    assert mature["independent_forecast_qty"].sum() == pytest.approx(100.0)
    assert cold["independent_forecast_qty"].sum() == pytest.approx(8.0)
    assert result["independent_forecast_qty"].sum() == pytest.approx(108.0)
    mature_values = mature.set_index("product_id")[
        "independent_forecast_qty"
    ].to_dict()
    assert mature_values == pytest.approx(
        {1: 100.0 * 60.0 / 90.0, 2: 100.0 * 30.0 / 90.0}
    )


def test_all_cold_bakery_falls_back_to_original_allocation() -> None:
    forecast = pd.DataFrame(
        {
            "date": ["2026-08-20", "2026-08-20"],
            "bakery_id": [275, 275],
            "product_id": [11575, 11615],
            "forecast_qty": [10.0, 20.0],
        }
    )
    registry = pd.DataFrame(
        {
            "bakery_id": [275, 275],
            "product_id": [11575, 11615],
            "cold_start_floor": [2.0, 3.0],
        }
    )

    result = apply_independent_cold_start(forecast, registry)

    assert not result["is_cold_start"].any()
    assert result["independent_forecast_qty"].tolist() == pytest.approx(
        [10.0, 20.0]
    )

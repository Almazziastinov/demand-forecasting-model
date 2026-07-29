from __future__ import annotations

import pandas as pd
import pytest

from src.experiments_v2.sku_systematic_correction import (
    CorrectionConfig,
    apply_category_neutral_corrections,
    build_correction_registry,
)


def _history() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    rows = []
    for date in dates:
        rows.extend(
            [
                {
                    "date": date,
                    "bakery_id": 20,
                    "product_id": 1,
                    "product_name": "Under",
                    "category_name": "Сытная",
                    "forecast_qty": 8.0,
                    "demand_qty": 12.0,
                },
                {
                    "date": date,
                    "bakery_id": 20,
                    "product_id": 2,
                    "product_name": "Balanced",
                    "category_name": "Сытная",
                    "forecast_qty": 10.0,
                    "demand_qty": 10.0,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_registry_uses_adaptive_unbounded_multiplier() -> None:
    registry = build_correction_registry(
        _history(),
        as_of_date="2026-01-31",
        config=CorrectionConfig(min_age_days=21),
    )

    assert registry["product_id"].tolist() == [1]
    assert registry.iloc[0]["direction"] == "underforecast"
    assert registry.iloc[0]["multiplier"] > 1.0
    assert registry.iloc[0]["multiplier"] == pytest.approx(
        registry.iloc[0]["full_multiplier"]
        ** registry.iloc[0]["smoothing"]
    )
    assert 0.10 <= registry.iloc[0]["smoothing"] <= 0.30


def test_registry_does_not_use_rows_from_forecast_date() -> None:
    history = _history()
    same_day = history.iloc[[0]].copy()
    same_day["date"] = pd.Timestamp("2026-01-31")
    same_day["forecast_qty"] = 1000.0
    history = pd.concat([history, same_day], ignore_index=True)

    registry = build_correction_registry(
        history,
        as_of_date="2026-01-31",
        config=CorrectionConfig(min_age_days=21),
    )

    assert registry.iloc[0]["forecast_qty"] == pytest.approx(240.0)


def test_registry_requires_mature_positive_forecast_history() -> None:
    history = _history()
    product_mask = history["product_id"].eq(1)
    history.loc[product_mask, "forecast_qty"] = 0.0
    history.loc[
        product_mask & history["date"].ge("2026-01-26"),
        "forecast_qty",
    ] = 10.0

    registry = build_correction_registry(
        history,
        as_of_date="2026-01-31",
        config=CorrectionConfig(min_age_days=21),
    )

    assert registry.empty


def test_application_preserves_category_total() -> None:
    registry = build_correction_registry(
        _history(),
        as_of_date="2026-01-31",
        config=CorrectionConfig(min_age_days=21),
    )
    forecast = pd.DataFrame(
        [
            {
                "date": "2026-01-31",
                "bakery_id": 20,
                "product_id": 1,
                "category_name": "Сытная",
                "forecast_qty": 40.0,
            },
            {
                "date": "2026-01-31",
                "bakery_id": 20,
                "product_id": 2,
                "category_name": "Сытная",
                "forecast_qty": 60.0,
            },
        ]
    )

    corrected = apply_category_neutral_corrections(forecast, registry)

    assert corrected["corrected_forecast_qty"].sum() == pytest.approx(100.0)
    corrected_under = corrected.loc[
        corrected["product_id"].eq(1),
        "corrected_forecast_qty",
    ].iloc[0]
    assert corrected_under > 40.0


def test_balanced_oscillation_is_not_corrected() -> None:
    history = _history()
    mask = history["product_id"].eq(1)
    history.loc[mask, "forecast_qty"] = [
        8.0 if index % 2 == 0 else 16.0 for index in range(mask.sum())
    ]
    history.loc[mask, "demand_qty"] = 12.0

    registry = build_correction_registry(
        history,
        as_of_date="2026-01-31",
        config=CorrectionConfig(
            min_age_days=21,
            min_abs_bias=0.10,
            min_directionality=0.50,
        ),
    )

    assert registry.empty

"""Tests for bakery-level daily forecasting helpers."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments_v2.bakery_day_forecast import build_model_frame  # noqa: E402
from src.experiments_v2.bakery_day_forecast import build_future_feature_rows  # noqa: E402
from src.experiments_v2.bakery_day_forecast import recursive_forecast  # noqa: E402


class _ConstantModel:
    def predict(self, x):
        return np.full(len(x), 10.0, dtype=float)


def _history() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2026-01-01", periods=8, freq="D")
    for bakery_id, city, base in [("B1", "Kazan", 10.0), ("B2", "Moscow", 20.0)]:
        for i, dt in enumerate(dates):
            rows.append(
                {
                    "date": dt,
                    "bakery_id": bakery_id,
                    "bakery_name": bakery_id,
                    "city": city,
                    "bakery_sales": base + i,
                    "avg_price": 100.0 + i,
                    "dow": dt.dayofweek,
                    "day": dt.day,
                    "month": dt.month,
                    "iso_week": int(dt.isocalendar().week),
                    "is_weekend": int(dt.dayofweek >= 5),
                    "is_month_start": int(dt.day <= 5),
                    "is_month_end": int(dt.day >= 25),
                    "is_payday_week": int(dt.day in [4, 5, 6, 19, 20, 21]),
                    "bakery_sales_lag1": 0.0,
                    "bakery_sales_lag2": 0.0,
                    "bakery_sales_lag3": 0.0,
                    "bakery_sales_lag7": 0.0,
                    "bakery_sales_lag14": 0.0,
                    "bakery_sales_lag30": 0.0,
                    "bakery_sales_roll_mean3": 0.0,
                    "bakery_sales_roll_mean7": 0.0,
                    "bakery_sales_roll_mean14": 0.0,
                    "bakery_sales_roll_mean30": 0.0,
                    "bakery_sales_roll_std7": 0.0,
                }
            )
    return build_model_frame(pd.DataFrame(rows))


def test_build_future_feature_rows_uses_latest_history():
    history = _history()
    future = build_future_feature_rows(history, pd.Timestamp("2026-01-09"))
    row = future[future["bakery_id"] == "B1"].iloc[0]

    assert len(future) == 2
    assert row["bakery_sales_lag1"] == 17.0
    assert row["bakery_sales_lag7"] == 11.0
    assert round(float(row["bakery_sales_roll_mean3"]), 4) == 16.0
    assert round(float(row["avg_price"]), 4) == 107.0


def test_recursive_forecast_returns_horizon_for_each_bakery():
    history = _history()
    feature_cols = [
        "bakery_id",
        "city",
        "dow",
        "day",
        "month",
        "iso_week",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "is_payday_week",
        "is_holiday",
        "is_pre_holiday",
        "is_post_holiday",
        "avg_price",
        "bakery_sales_lag1",
        "bakery_sales_lag2",
        "bakery_sales_lag3",
        "bakery_sales_lag7",
        "bakery_sales_lag14",
        "bakery_sales_lag30",
        "bakery_sales_roll_mean3",
        "bakery_sales_roll_mean7",
        "bakery_sales_roll_mean14",
        "bakery_sales_roll_mean30",
        "bakery_sales_roll_std7",
        "bakery_sales_dow_mean",
        "bakery_sales_trend",
        "bakery_sales_cv_7d",
    ]
    forecast = recursive_forecast(history, _ConstantModel(), feature_cols, horizon_days=3)

    assert len(forecast) == 6
    assert forecast["date"].nunique() == 3
    assert set(forecast["bakery_id"]) == {"B1", "B2"}
    assert float(forecast["bakery_day_forecast"].min()) == 10.0

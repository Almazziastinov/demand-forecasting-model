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


def test_build_model_frame_adds_event_cluster_features():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-03-19", "2026-03-20", "2026-03-21"]),
            "bakery_id": ["B1", "B1", "B1"],
            "bakery_name": ["B1", "B1", "B1"],
            "city": ["Kazan", "Kazan", "Kazan"],
            "bakery_sales": [10.0, 12.0, 11.0],
            "avg_price": [100.0, 100.0, 100.0],
            "dow": [3, 4, 5],
            "day": [19, 20, 21],
            "month": [3, 3, 3],
            "iso_week": [12, 12, 12],
            "is_weekend": [0, 0, 1],
            "is_month_start": [0, 0, 0],
            "is_month_end": [0, 0, 0],
            "is_payday_week": [1, 1, 1],
            "bakery_sales_lag1": [0.0, 10.0, 12.0],
            "bakery_sales_lag2": [0.0, 0.0, 10.0],
            "bakery_sales_lag3": [0.0, 0.0, 0.0],
            "bakery_sales_lag7": [0.0, 0.0, 0.0],
            "bakery_sales_lag14": [0.0, 0.0, 0.0],
            "bakery_sales_lag30": [0.0, 0.0, 0.0],
            "bakery_sales_roll_mean3": [0.0, 0.0, 0.0],
            "bakery_sales_roll_mean7": [0.0, 0.0, 0.0],
            "bakery_sales_roll_mean14": [0.0, 0.0, 0.0],
            "bakery_sales_roll_mean30": [0.0, 0.0, 0.0],
            "bakery_sales_roll_std7": [0.0, 0.0, 0.0],
        }
    )
    frame = build_model_frame(df)
    row_before = frame[frame["date"] == pd.Timestamp("2026-03-19")].iloc[0]
    row_event = frame[frame["date"] == pd.Timestamp("2026-03-20")].iloc[0]

    assert row_before["next_event_cluster"] == "cluster_1"
    assert int(row_before["days_to_next_event"]) == 1
    assert row_event["current_event_cluster"] == "cluster_1"
    assert int(row_event["is_near_event_window"]) == 1


def test_build_model_frame_adds_enriched_event_features():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-03-19", "2026-03-20", "2026-03-21"]),
            "bakery_id": ["B1", "B1", "B1"],
            "bakery_name": ["B1", "B1", "B1"],
            "city": ["Kazan", "Kazan", "Kazan"],
            "bakery_sales": [10.0, 12.0, 11.0],
            "avg_price": [100.0, 100.0, 100.0],
            "dow": [3, 4, 5],
            "day": [19, 20, 21],
            "month": [3, 3, 3],
            "iso_week": [12, 12, 12],
            "is_weekend": [0, 0, 1],
            "is_month_start": [0, 0, 0],
            "is_month_end": [0, 0, 0],
            "is_payday_week": [1, 1, 1],
            "bakery_sales_lag1": [0.0, 10.0, 12.0],
            "bakery_sales_lag2": [0.0, 0.0, 10.0],
            "bakery_sales_lag3": [0.0, 0.0, 0.0],
            "bakery_sales_lag7": [0.0, 0.0, 0.0],
            "bakery_sales_lag14": [0.0, 0.0, 0.0],
            "bakery_sales_lag30": [0.0, 0.0, 0.0],
            "bakery_sales_roll_mean3": [0.0, 0.0, 0.0],
            "bakery_sales_roll_mean7": [0.0, 0.0, 0.0],
            "bakery_sales_roll_mean14": [0.0, 0.0, 0.0],
            "bakery_sales_roll_mean30": [0.0, 0.0, 0.0],
            "bakery_sales_roll_std7": [0.0, 0.0, 0.0],
        }
    )

    frame = build_model_frame(df)
    row_before = frame[frame["date"] == pd.Timestamp("2026-03-19")].iloc[0]
    row_event = frame[frame["date"] == pd.Timestamp("2026-03-20")].iloc[0]
    row_after = frame[frame["date"] == pd.Timestamp("2026-03-21")].iloc[0]

    assert row_before["event_window_type"] == "pre_event_1_3"
    assert row_event["event_window_type"] == "event_day"
    assert row_after["event_window_type"] == "post_event_1_3"
    assert row_event["holiday_name_feature"] == "uraza_bayram"
    assert row_before["nearest_event_city"] == "cluster_1|Kazan"


def test_build_future_feature_rows_adds_enriched_event_features():
    history = _history()
    future = build_future_feature_rows(history, pd.Timestamp("2026-01-05"))

    assert "payday_window_type" in future.columns
    assert "event_window_type" in future.columns
    assert set(future["payday_window_type"]) == {"payday"}


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
        "current_event_cluster",
        "prev_event_cluster",
        "next_event_cluster",
        "days_since_prev_event",
        "days_to_next_event",
        "is_near_event_window",
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
    forecast = recursive_forecast(
        history,
        _ConstantModel(),
        feature_cols,
        horizon_days=3,
    )

    assert len(forecast) == 6
    assert forecast["date"].nunique() == 3
    assert set(forecast["bakery_id"]) == {"B1", "B2"}
    assert float(forecast["bakery_day_forecast"].min()) == 10.0

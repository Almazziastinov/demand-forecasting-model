"""Tests for hourly availability layer heuristics."""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments_v2.availability_layer import (  # noqa: E402
    BAKERY_COL,
    CATEGORY_COL,
    DATE_COL,
    DOW_COL,
    HOUR_COL,
    PRODUCT_COL,
    TARGET_COL,
    add_hourly_availability_signals,
    build_daily_availability,
    build_hourly_frame,
)


def _make_hourly_sales() -> pd.DataFrame:
    rows = []
    for date in pd.to_datetime(["2026-01-05", "2026-01-12"]):
        dow = int(date.dayofweek)
        bakery = "B1"
        category = "Cat1"

        # Bakery traffic through the whole day.
        for hour, bakery_qty in [(8, 10.0), (9, 12.0), (10, 11.0), (11, 9.0)]:
            rows.append(
                {
                    DATE_COL: date,
                    DOW_COL: dow,
                    BAKERY_COL: bakery,
                    CATEGORY_COL: category,
                    PRODUCT_COL: "Other",
                    HOUR_COL: hour,
                    TARGET_COL: bakery_qty,
                }
            )

        # SKU sells at 8 and 10, zero at 9 should look like stockout-like.
        for hour, sku_qty in [(8, 2.0), (10, 2.0)]:
            rows.append(
                {
                    DATE_COL: date,
                    DOW_COL: dow,
                    BAKERY_COL: bakery,
                    CATEGORY_COL: category,
                    PRODUCT_COL: "P1",
                    HOUR_COL: hour,
                    TARGET_COL: sku_qty,
                }
            )
    return pd.DataFrame(rows)


def test_hourly_signals_detect_zero_under_traffic_and_neighbor_pattern():
    hourly_sales = _make_hourly_sales()
    hourly_frame = build_hourly_frame(hourly_sales)
    signals = add_hourly_availability_signals(hourly_frame)

    slot = signals[
        (signals[PRODUCT_COL] == "P1")
        & (signals[DATE_COL] == pd.Timestamp("2026-01-05"))
        & (signals[HOUR_COL] == 9)
    ].iloc[0]

    assert bool(slot["has_normal_traffic"])
    assert bool(slot["has_hist_demand"])
    assert bool(slot["has_neighbor_sales"])
    assert bool(slot["zero_under_traffic"])
    assert bool(slot["stockout_like_hour"])


def test_daily_availability_rolls_up_hourly_signals():
    hourly_sales = _make_hourly_sales()
    hourly_frame = build_hourly_frame(hourly_sales)
    signals = add_hourly_availability_signals(hourly_frame)
    daily = build_daily_availability(signals)

    row = daily[(daily[PRODUCT_COL] == "P1") & (daily[DATE_COL] == pd.Timestamp("2026-01-05"))].iloc[0]
    assert row["stockout_like_hours"] >= 1
    assert row["zero_under_traffic_hours"] >= 1
    assert 0.0 <= row["availability_score"] <= 1.0

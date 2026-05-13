"""Tests for the sales-first factual backbone builder."""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments_v2.sales_first_backbone import (  # noqa: E402
    BAKERY_COL,
    CATEGORY_COL,
    CITY_COL,
    DATE_COL,
    PRODUCT_COL,
    TARGET_COL,
    build_bakery_category_daily_sales,
    build_bakery_daily_sales,
    build_sales_backbone,
)


def _make_sales_df() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2026-01-01", periods=20, freq="D")
    for date in dates:
        for product, sales in [("P1", 2.0), ("P2", 3.0)]:
            rows.append(
                {
                    DATE_COL: date,
                    BAKERY_COL: "B1",
                    CATEGORY_COL: "Cat1",
                    PRODUCT_COL: product,
                    TARGET_COL: sales,
                    CITY_COL: "Kazan",
                    "ДеньНедели": date.dayofweek,
                    "День": date.day,
                    "IsWeekend": int(date.dayofweek >= 5),
                    "Месяц": date.month,
                    "НомерНедели": int(date.isocalendar().week),
                    "is_holiday": 0,
                    "sales_lag1": 1.0,
                    "sales_roll_mean7": 2.0,
                    "Спрос": sales + 1.0,
                    "lost_qty": 1.0,
                    "is_censored": 1,
                }
            )
    return pd.DataFrame(rows)


def test_build_sales_backbone_excludes_legacy_demand_columns():
    df = _make_sales_df()
    backbone = build_sales_backbone(df)
    assert "Спрос" not in backbone.columns
    assert "lost_qty" not in backbone.columns
    assert "is_censored" not in backbone.columns
    assert TARGET_COL in backbone.columns


def test_build_bakery_daily_sales_aggregates_sales():
    df = _make_sales_df()
    backbone = build_sales_backbone(df)
    bakery = build_bakery_daily_sales(backbone)

    assert "bakery_sales_total" in bakery.columns
    assert len(bakery) == backbone[DATE_COL].nunique()
    assert float(bakery.iloc[0]["bakery_sales_total"]) == 5.0
    assert int(bakery.iloc[0]["items_in_bakery_today"]) == 2


def test_build_bakery_category_daily_sales_has_share():
    df = _make_sales_df()
    backbone = build_sales_backbone(df)
    category = build_bakery_category_daily_sales(backbone)

    assert "category_sales_total" in category.columns
    assert "category_share_in_bakery" in category.columns
    assert float(category.iloc[0]["category_sales_total"]) == 5.0
    assert float(category.iloc[0]["category_share_in_bakery"]) == 1.0

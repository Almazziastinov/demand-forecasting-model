"""Tests for scripts/build_city_assortment_from_sales.py's pure DataFrame logic."""

from __future__ import annotations

import datetime
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_city_assortment_from_sales import build_layers  # noqa: E402


def _sample_sales() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"city": "Казань", "bakery_id": 1, "product_id": "10", "product_name": "Ватрушка", "category_name": "Выпечка"},
            {"city": "Казань", "bakery_id": 2, "product_id": "10", "product_name": "Ватрушка", "category_name": "Выпечка"},
        ]
    )


def _sample_bakery_counts() -> pd.DataFrame:
    return pd.DataFrame([{"city": "Казань", "total_bakeries": 2}])


def test_valid_from_is_a_real_date_object_not_a_string():
    # Regression test: this frame gets inserted directly into ClickHouse via
    # client.insert_df into a `Date`-typed column. clickhouse-connect's Date
    # serializer does `(value - epoch).days` per cell, which raises
    # "unsupported operand type(s) for -: 'str' and 'datetime.date'" if
    # valid_from is a string (e.g. from a stray .isoformat() call) instead of
    # an actual date object. Confirmed by direct reproduction against a
    # throwaway ClickHouse table before this fix landed.
    result = build_layers(
        _sample_sales(),
        _sample_bakery_counts(),
        city_threshold=0.80,
        category_patterns=["выпечка"],
        valid_from="2026-07-14",
    )

    assert not result.empty
    assert all(isinstance(v, datetime.date) for v in result["valid_from"])
    assert not any(isinstance(v, str) for v in result["valid_from"])
    assert result["valid_from"].iloc[0] == datetime.date(2026, 7, 14)

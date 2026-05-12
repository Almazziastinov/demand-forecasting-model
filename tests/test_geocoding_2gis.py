"""Tests for 2GIS geocoding helpers."""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geocoding_2gis import build_query_variants, geocode_bakery_row


def test_build_query_variants_orders_best_first():
    row = pd.Series(
        {
            "bakery_name": "Ямашева 71 Казань",
            "city": "Казань",
            "address_raw": "пр-т Ямашева, 71",
            "address_normalized": "пр-т Ямашева, 71, Казань",
        }
    )

    variants = build_query_variants(row)

    assert variants[0] == "пр-т Ямашева, 71, Казань"
    assert "Ямашева 71 Казань, Казань" in variants


def test_geocode_bakery_row_returns_failed_shape_when_all_attempts_fail(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("network error")

    monkeypatch.setattr("src.geocoding_2gis.geocode_query_2gis", _boom)
    row = pd.Series(
        {
            "bakery_id": "bakery_1",
            "bakery_name": "Test Bakery",
            "city": "Казань",
            "address_raw": None,
            "address_normalized": None,
        }
    )

    result = geocode_bakery_row(row, api_key="dummy")

    assert result["geo_status"] == "failed"
    assert result["geo_confidence"] == 0.0
    assert result["Пекарня"] == "Test Bakery"

"""Tests for 2GIS building enrichment helpers."""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geocoding_2gis import GeocodeCandidate, geocode_bakery_building_row


def test_geocode_bakery_building_row_extracts_building_fields(monkeypatch):
    payload = {
        "result": {
            "items": [
                {
                    "id": "70030076123456789",
                    "name": "Test building",
                    "address_name": "Test street, 1",
                    "full_address_name": "Kazan, Test street, 1",
                    "type": "building",
                    "point": {"lat": 55.1, "lon": 49.1},
                    "floors": {"ground_count": 9, "underground_count": 1},
                    "structure_info": {
                        "year_of_construction": 1988,
                        "apartments_count": 72,
                        "porch_count": 3,
                        "material": "brick",
                    },
                    "purpose_code": "residential",
                    "has_apartments_info": True,
                }
            ]
        }
    }

    def _fake_query(*args, **kwargs):
        return GeocodeCandidate(
            query="Test street, 1, Kazan",
            result_name="Test building",
            address_name="Test street, 1",
            full_name="Kazan, Test street, 1",
            lat=55.1,
            lon=49.1,
            result_type="building",
            confidence=0.95,
            status="matched_exact",
            raw_payload=payload,
        )

    monkeypatch.setattr("src.geocoding_2gis.geocode_building_query_2gis", _fake_query)
    row = pd.Series(
        {
            "bakery_id": "1",
            "bakery_name": "Test street 1 Kazan",
            "city": "Kazan",
            "price_region": "1",
        }
    )

    result = geocode_bakery_building_row(row, api_key="dummy")

    assert result["geo_status"] == "matched_exact"
    assert result["building_id"] == "70030076123456789"
    assert result["building_year_of_construction"] == 1988
    assert result["building_ground_floors"] == 9
    assert result["building_apartments_count"] == 72

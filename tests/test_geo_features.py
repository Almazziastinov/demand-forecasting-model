"""Tests for offline geo feature pipeline."""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geo_features import (
    aggregate_poi_features,
    build_bakery_geo_master,
    haversine_distance_m,
)


def test_haversine_distance_is_zero_for_same_point():
    assert haversine_distance_m(55.0, 49.0, 55.0, 49.0) == 0.0


def test_build_bakery_geo_master_uses_city_centroid_fallback():
    sales_df = pd.DataFrame(
        {
            "Дата": ["2026-01-01", "2026-01-02"],
            "Пекарня": ["B1", "B1"],
            "Город": ["Казань", "Казань"],
        }
    )

    master = build_bakery_geo_master(sales_df)
    row = master.iloc[0]

    assert row["geo_status"] == "city_only"
    assert row["geo_confidence"] == 0.2
    assert pd.notna(row["lat"])
    assert pd.notna(row["lon"])
    assert abs(row["x_km_local"]) < 1e-9
    assert abs(row["y_km_local"]) < 1e-9


def test_aggregate_poi_features_builds_distance_and_counts():
    master_df = pd.DataFrame(
        {
            "bakery_id": ["bakery_1"],
            "lat": [55.7879],
            "lon": [49.1233],
        }
    )
    poi_df = pd.DataFrame(
        {
            "bakery_id": ["bakery_1", "bakery_1", "bakery_1"],
            "poi_category": ["school", "park", "park"],
            "poi_lat": [55.7880, 55.7882, 55.7925],
            "poi_lon": [49.1234, 49.1235, 49.1275],
        }
    )

    features = aggregate_poi_features(master_df, poi_df, radii_m=(300, 1000))
    row = features.iloc[0]

    assert row["n_schools_300m"] == 1
    assert row["n_parks_300m"] == 1
    assert row["n_parks_1000m"] == 2
    assert row["dist_to_nearest_school_m"] < row["dist_to_nearest_park_m"] * 2
    assert row["education_poi_score"] > 0
    assert row["leisure_poi_score"] > 0

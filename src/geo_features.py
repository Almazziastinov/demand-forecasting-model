"""Offline geo feature pipeline for bakery-level modeling.

This module keeps the geo layer simple:
- build a bakery master table from historical sales datasets
- merge optional trusted geo inputs and manual overrides
- fall back to city centroids when exact coordinates are missing
- aggregate nearby POI objects into stable model-ready features
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.experiments_v2.common import CITY_COORDS

BAKERY_COL = "Пекарня"
CITY_COL = "Город"
DATE_COL = "Дата"

MASTER_COLUMNS = [
    "bakery_id",
    "bakery_name",
    "city",
    "first_seen_date",
    "last_seen_date",
    "n_rows",
    "n_dates",
    "address_raw",
    "address_normalized",
    "lat",
    "lon",
    "geo_source",
    "geo_confidence",
    "geo_status",
    "is_active",
    "x_km_local",
    "y_km_local",
    "dist_to_city_center_km",
]

POI_CATEGORY_ALIASES = {
    "park": "park",
    "embankment": "embankment",
    "theater": "theater",
    "concert_hall": "concert_hall",
    "cinema": "cinema",
    "school": "school",
    "college": "college",
    "university": "university",
    "business_center": "business_center",
    "mall": "mall",
    "market": "market",
    "metro": "metro",
    "bus_stop": "bus_stop",
    "stadium": "stadium",
    "sports_facility": "sports_facility",
}

POI_SCORE_GROUPS = {
    "education_poi_score": ["school", "college", "university"],
    "office_poi_score": ["business_center"],
    "leisure_poi_score": [
        "park",
        "embankment",
        "theater",
        "concert_hall",
        "cinema",
        "stadium",
        "sports_facility",
    ],
    "transit_access_score": ["metro", "bus_stop"],
}

DEFAULT_RADII_M = (300, 500, 1000)


def _stable_bakery_id(name: str) -> str:
    raw = str(name).strip()
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return f"bakery_{digest}"


def _normalize_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance between two points in meters."""
    radius_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * radius_m * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _distance_vectorized(lat: float, lon: float, poi_df: pd.DataFrame) -> np.ndarray:
    poi_pairs = zip(poi_df["poi_lat"], poi_df["poi_lon"], strict=False)
    distances = [
        haversine_distance_m(lat, lon, float(poi_lat), float(poi_lon))
        for poi_lat, poi_lon in poi_pairs
    ]
    return np.array(
        distances,
        dtype=float,
    )


def build_bakery_geo_master(
    sales_df: pd.DataFrame,
    existing_geo_df: pd.DataFrame | None = None,
    manual_overrides_df: pd.DataFrame | None = None,
    city_coords: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Build a bakery geo master table from sales history and optional geo inputs."""
    required = {BAKERY_COL, CITY_COL, DATE_COL}
    missing = required - set(sales_df.columns)
    if missing:
        raise KeyError(f"sales_df missing required columns: {sorted(missing)}")

    work = sales_df[[BAKERY_COL, CITY_COL, DATE_COL]].copy()
    work[DATE_COL] = pd.to_datetime(work[DATE_COL], errors="coerce")
    work = work.dropna(subset=[BAKERY_COL, CITY_COL, DATE_COL])

    master = (
        work.groupby([BAKERY_COL, CITY_COL], as_index=False)
        .agg(
            first_seen_date=(DATE_COL, "min"),
            last_seen_date=(DATE_COL, "max"),
            n_rows=(DATE_COL, "size"),
            n_dates=(DATE_COL, "nunique"),
        )
        .rename(columns={BAKERY_COL: "bakery_name", CITY_COL: "city"})
    )
    master["bakery_id"] = master["bakery_name"].map(_stable_bakery_id)
    master["address_raw"] = None
    master["address_normalized"] = None
    master["lat"] = np.nan
    master["lon"] = np.nan
    master["geo_source"] = "missing"
    master["geo_confidence"] = 0.0
    master["geo_status"] = "missing"
    master["is_active"] = True

    if existing_geo_df is not None and not existing_geo_df.empty:
        geo = existing_geo_df.copy()
        rename_map = {
            "Пекарня": "bakery_name",
            "Город": "city",
            "address": "address_raw",
            "address_raw": "address_raw",
            "address_normalized": "address_normalized",
            "latitude": "lat",
            "longitude": "lon",
            "geo_source": "geo_source",
            "geocoder_source": "geo_source",
            "geo_confidence": "geo_confidence",
            "geocoder_confidence": "geo_confidence",
            "geo_status": "geo_status",
            "geocoder_status": "geo_status",
        }
        geo = geo.rename(columns=rename_map)
        keep = [col for col in rename_map.values() if col in geo.columns]
        geo = geo[keep].copy()
        geo["bakery_name"] = geo["bakery_name"].map(_normalize_text)
        master = master.merge(
            geo,
            on="bakery_name",
            how="left",
            suffixes=("", "_geo"),
        )

        geo_cols = [
            "address_raw",
            "address_normalized",
            "lat",
            "lon",
            "geo_source",
            "geo_confidence",
            "geo_status",
        ]
        for col in geo_cols:
            geo_col = f"{col}_geo"
            if geo_col in master.columns:
                master[col] = master[geo_col].combine_first(master[col])
                master = master.drop(columns=[geo_col])

    if manual_overrides_df is not None and not manual_overrides_df.empty:
        overrides = manual_overrides_df.copy()
        overrides = overrides.rename(
            columns={
                "Пекарня": "bakery_name",
                "Город": "city",
                "latitude": "lat",
                "longitude": "lon",
            }
        )
        overrides["bakery_name"] = overrides["bakery_name"].map(_normalize_text)
        master = master.merge(
            overrides,
            on="bakery_name",
            how="left",
            suffixes=("", "_override"),
        )
        for col in ["address_raw", "address_normalized", "lat", "lon"]:
            override_col = f"{col}_override"
            if override_col in master.columns:
                master[col] = master[override_col].combine_first(master[col])
                master = master.drop(columns=[override_col])

        has_override = master["lat"].notna() & master["lon"].notna()
        master.loc[has_override, "geo_source"] = "manual_override"
        master.loc[has_override, "geo_confidence"] = 1.0
        master.loc[has_override, "geo_status"] = "manual_fix"

    city_coords = city_coords or CITY_COORDS
    city_centers = pd.DataFrame(
        [
            {
                "city": city,
                "city_lat": float(coords["lat"]),
                "city_lon": float(coords["lon"]),
            }
            for city, coords in city_coords.items()
        ]
    )
    master = master.merge(city_centers, on="city", how="left")

    missing_coords = master["lat"].isna() | master["lon"].isna()
    master.loc[missing_coords, "lat"] = master.loc[missing_coords, "city_lat"]
    master.loc[missing_coords, "lon"] = master.loc[missing_coords, "city_lon"]

    city_fallback = (
        missing_coords & master["city_lat"].notna() & master["city_lon"].notna()
    )
    master.loc[city_fallback, "geo_source"] = "city_centroid"
    master.loc[city_fallback, "geo_confidence"] = 0.2
    master.loc[city_fallback, "geo_status"] = "city_only"

    failed = master["lat"].isna() | master["lon"].isna()
    master.loc[failed, "geo_status"] = "failed"
    master.loc[failed, "geo_source"] = "missing"
    master.loc[failed, "geo_confidence"] = 0.0

    master = add_local_coordinates(master, city_coords=city_coords)
    master["address_raw"] = master["address_raw"].map(_normalize_text)
    master["address_normalized"] = master["address_normalized"].map(_normalize_text)
    master["is_active"] = master["last_seen_date"] == master["last_seen_date"].max()

    return (
        master[MASTER_COLUMNS]
        .sort_values(["city", "bakery_name"])
        .reset_index(drop=True)
    )


def add_local_coordinates(
    master_df: pd.DataFrame,
    city_coords: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Express bakery coordinates in local city-centric kilometers."""
    city_coords = city_coords or CITY_COORDS
    city_centers = pd.DataFrame(
        [
            {
                "city": city,
                "city_lat": float(coords["lat"]),
                "city_lon": float(coords["lon"]),
            }
            for city, coords in city_coords.items()
        ]
    )
    drop_cols = [col for col in ["city_lat", "city_lon"] if col in master_df.columns]
    df = master_df.drop(columns=drop_cols).copy()
    df = df.merge(city_centers, on="city", how="left")

    df["x_km_local"] = np.nan
    df["y_km_local"] = np.nan
    df["dist_to_city_center_km"] = np.nan

    valid = (
        df["lat"].notna()
        & df["lon"].notna()
        & df["city_lat"].notna()
        & df["city_lon"].notna()
    )
    if valid.any():
        lat = np.radians(df.loc[valid, "lat"].astype(float))
        lon = np.radians(df.loc[valid, "lon"].astype(float))
        city_lat = np.radians(df.loc[valid, "city_lat"].astype(float))
        city_lon = np.radians(df.loc[valid, "city_lon"].astype(float))

        earth_radius_km = 6371.0
        mean_lat = (lat + city_lat) / 2.0
        df.loc[valid, "x_km_local"] = (
            (lon - city_lon) * np.cos(mean_lat) * earth_radius_km
        )
        df.loc[valid, "y_km_local"] = (lat - city_lat) * earth_radius_km
        df.loc[valid, "dist_to_city_center_km"] = np.sqrt(
            df.loc[valid, "x_km_local"] ** 2 + df.loc[valid, "y_km_local"] ** 2
        )

    return df.drop(columns=["city_lat", "city_lon"])


def _normalize_poi_category(value: object) -> str | None:
    text = _normalize_text(value)
    if text is None:
        return None
    key = text.lower()
    return POI_CATEGORY_ALIASES.get(key, key)


def aggregate_poi_features(
    master_df: pd.DataFrame,
    poi_df: pd.DataFrame,
    radii_m: Iterable[int] = DEFAULT_RADII_M,
) -> pd.DataFrame:
    """Aggregate raw POI rows into stable bakery-level features."""
    required_master = {"bakery_id", "lat", "lon"}
    required_poi = {"bakery_id", "poi_category", "poi_lat", "poi_lon"}
    missing_master = required_master - set(master_df.columns)
    missing_poi = required_poi - set(poi_df.columns)
    if missing_master:
        raise KeyError(f"master_df missing required columns: {sorted(missing_master)}")
    if missing_poi:
        raise KeyError(f"poi_df missing required columns: {sorted(missing_poi)}")

    radii = sorted({int(radius) for radius in radii_m})
    work_master = master_df.copy()
    poi = poi_df.copy()
    poi["poi_category"] = poi["poi_category"].map(_normalize_poi_category)
    poi = poi.dropna(subset=["bakery_id", "poi_category", "poi_lat", "poi_lon"])

    categories = sorted(set(poi["poi_category"]))
    rows: list[dict[str, object]] = []

    for _, bakery in work_master.iterrows():
        row = {"bakery_id": bakery["bakery_id"]}
        lat = bakery.get("lat")
        lon = bakery.get("lon")
        bakery_poi = poi[poi["bakery_id"] == bakery["bakery_id"]].copy()
        if pd.notna(lat) and pd.notna(lon) and not bakery_poi.empty:
            bakery_poi["distance_m"] = _distance_vectorized(
                float(lat),
                float(lon),
                bakery_poi,
            )
        else:
            bakery_poi["distance_m"] = np.nan

        for category in categories:
            cat_df = bakery_poi[bakery_poi["poi_category"] == category]
            nearest_col = f"dist_to_nearest_{category}_m"
            row[nearest_col] = (
                float(cat_df["distance_m"].min()) if not cat_df.empty else np.nan
            )
            for radius in radii:
                count = (
                    int((cat_df["distance_m"] <= radius).sum())
                    if not cat_df.empty
                    else 0
                )
                row[f"n_{category}s_{radius}m"] = count

        for score_name, score_categories in POI_SCORE_GROUPS.items():
            score = 0.0
            for category in score_categories:
                cat_df = bakery_poi[bakery_poi["poi_category"] == category]
                if cat_df.empty:
                    continue
                score += float((1.0 / (1.0 + cat_df["distance_m"] / 100.0)).sum())
            row[score_name] = round(score, 6)

        rows.append(row)

    features = pd.DataFrame(rows)
    return work_master.merge(features, on="bakery_id", how="left")

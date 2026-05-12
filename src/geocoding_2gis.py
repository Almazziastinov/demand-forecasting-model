"""Helpers for geocoding bakery locations via 2GIS Geocoder API."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


GEOCODER_URL = "https://catalog.api.2gis.com/3.0/items/geocode"
DEFAULT_TIMEOUT_S = 20


@dataclass
class GeocodeCandidate:
    """A single geocoding attempt result."""

    query: str
    result_name: str | None
    address_name: str | None
    full_name: str | None
    lat: float | None
    lon: float | None
    result_type: str | None
    confidence: float
    status: str
    raw_payload: dict[str, Any]


def build_query_variants(row: pd.Series) -> list[str]:
    """Generate ordered query variants for one bakery row."""
    bakery_name = str(row.get("bakery_name", "")).strip()
    city = str(row.get("city", "")).strip()
    address_normalized = str(row.get("address_normalized", "")).strip()
    address_raw = str(row.get("address_raw", "")).strip()

    variants: list[str] = []
    for query in [
        address_normalized,
        f"{address_raw}, {city}" if address_raw else "",
        f"{bakery_name}, {city}" if bakery_name else "",
        bakery_name,
    ]:
        query = query.strip().strip(",")
        if query and query not in variants:
            variants.append(query)
    return variants


def _http_get_json(url: str, params: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    query_string = urlencode(params)
    with urlopen(f"{url}?{query_string}", timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _score_item(item: dict[str, Any], query: str, city: str) -> tuple[float, str]:
    """Heuristic confidence for a 2GIS geocoder response item."""
    score = 0.0
    status = "matched_fuzzy"

    point = item.get("point") or {}
    if "lat" in point and "lon" in point:
        score += 0.45

    item_type = str(item.get("type", "")).lower()
    if item_type == "building":
        score += 0.25
        status = "matched_exact"
    elif item_type:
        score += 0.1

    full_name = str(item.get("full_name", "")).lower()
    address_name = str(item.get("address_name", "")).lower()
    query_l = query.lower()
    city_l = city.lower()

    if city_l and city_l in full_name:
        score += 0.15
    if query_l and (query_l in full_name or query_l in address_name):
        score += 0.15

    if score < 0.55:
        status = "matched_fuzzy"
    return min(score, 1.0), status


def geocode_query_2gis(
    query: str,
    api_key: str,
    city: str = "",
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> GeocodeCandidate:
    """Geocode a single query using 2GIS."""
    payload = _http_get_json(
        GEOCODER_URL,
        {
            "q": query,
            "fields": "items.point",
            "key": api_key,
        },
        timeout_s=timeout_s,
    )

    items = payload.get("result", {}).get("items", [])
    if not items:
        return GeocodeCandidate(
            query=query,
            result_name=None,
            address_name=None,
            full_name=None,
            lat=None,
            lon=None,
            result_type=None,
            confidence=0.0,
            status="failed",
            raw_payload=payload,
        )

    item = items[0]
    point = item.get("point") or {}
    confidence, status = _score_item(item, query=query, city=city)
    return GeocodeCandidate(
        query=query,
        result_name=item.get("name"),
        address_name=item.get("address_name"),
        full_name=item.get("full_name"),
        lat=point.get("lat"),
        lon=point.get("lon"),
        result_type=item.get("type"),
        confidence=confidence,
        status=status,
        raw_payload=payload,
    )


def geocode_bakery_row(
    row: pd.Series,
    api_key: str,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Geocode one bakery row using ordered query variants."""
    city = str(row.get("city", "")).strip()
    bakery_name = str(row.get("bakery_name", "")).strip()
    variants = build_query_variants(row)

    best: GeocodeCandidate | None = None
    last_error: str | None = None

    for query in variants:
        try:
            candidate = geocode_query_2gis(
                query=query,
                api_key=api_key,
                city=city,
                timeout_s=timeout_s,
            )
        except Exception as exc:
            last_error = str(exc)
            continue

        if best is None or candidate.confidence > best.confidence:
            best = candidate
        if candidate.status == "matched_exact":
            break

    if best is None:
        return {
            "bakery_id": row["bakery_id"],
            "Пекарня": bakery_name,
            "Город": city,
            "address_raw": row.get("address_raw"),
            "address_normalized": row.get("address_normalized"),
            "lat": None,
            "lon": None,
            "geo_source": "2gis",
            "geo_confidence": 0.0,
            "geo_status": "failed",
            "geocode_query": None,
            "geocode_result_name": None,
            "geocode_full_name": None,
            "geocode_result_type": None,
            "error": last_error,
        }

    return {
        "bakery_id": row["bakery_id"],
        "Пекарня": bakery_name,
        "Город": city,
        "address_raw": row.get("address_raw"),
        "address_normalized": best.full_name or row.get("address_normalized"),
        "lat": best.lat,
        "lon": best.lon,
        "geo_source": "2gis",
        "geo_confidence": best.confidence,
        "geo_status": best.status,
        "geocode_query": best.query,
        "geocode_result_name": best.result_name,
        "geocode_full_name": best.full_name,
        "geocode_result_type": best.result_type,
        "error": None,
    }


def get_2gis_api_key(cli_value: str | None = None) -> str:
    """Resolve 2GIS API key from CLI or environment."""
    api_key = cli_value or os.getenv("DGIS_API_KEY")
    if not api_key:
        raise ValueError("2GIS API key is required: pass --api-key or set DGIS_API_KEY")
    return api_key

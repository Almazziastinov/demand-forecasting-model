"""Helpers for geocoding bakery locations via geopy + Nominatim."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd


DEFAULT_USER_AGENT = "demand-forecasting-model-geocoder/1.0"
DEFAULT_TIMEOUT_S = 20


@dataclass
class GeocodeCandidate:
    """A single geocoding attempt result."""

    query: str
    address: str | None
    lat: float | None
    lon: float | None
    raw_type: str | None
    raw_class: str | None
    confidence: float
    status: str
    raw_payload: dict[str, Any] | None


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


def get_geopy_user_agent(cli_value: str | None = None) -> str:
    """Resolve user agent from CLI or environment."""
    return cli_value or os.getenv("GEOPY_USER_AGENT") or DEFAULT_USER_AGENT


def _get_nominatim(user_agent: str, timeout_s: int):
    try:
        from geopy.geocoders import Nominatim
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "geopy is not installed. "
            "Run `.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt`"
        ) from exc

    return Nominatim(user_agent=user_agent, timeout=timeout_s)


def _tokenize(text: str) -> set[str]:
    return {token for token in re.split(r"[^0-9A-Za-zА-Яа-я]+", text.lower()) if token}


def _score_location(location: Any, query: str, city: str) -> tuple[float, str]:
    raw = getattr(location, "raw", {}) or {}
    address = str(getattr(location, "address", "") or "").lower()
    query_tokens = _tokenize(query)
    address_tokens = _tokenize(address)
    city_l = city.lower()

    score = 0.2

    raw_type = str(raw.get("type", "")).lower()
    raw_class = str(raw.get("class", "")).lower()
    if raw_type in {"house", "building", "apartments", "commercial"}:
        score += 0.35
    elif raw_type in {"road", "street"}:
        score += 0.15

    importance = raw.get("importance")
    try:
        score += min(float(importance), 0.2)
    except (TypeError, ValueError):
        pass

    if city_l and city_l in address:
        score += 0.1

    if query_tokens:
        overlap = len(query_tokens & address_tokens) / len(query_tokens)
        score += 0.25 * overlap

    if raw_class == "building" and score >= 0.75:
        status = "matched_exact"
    elif score >= 0.45:
        status = "matched_fuzzy"
    else:
        status = "city_only"
    return min(score, 1.0), status


def geocode_query_geopy(
    geolocator: Any,
    query: str,
    city: str = "",
) -> GeocodeCandidate:
    """Geocode a single query using geopy Nominatim."""
    location = geolocator.geocode(
        query,
        exactly_one=True,
        addressdetails=True,
        language="ru",
    )
    if location is None:
        return GeocodeCandidate(
            query=query,
            address=None,
            lat=None,
            lon=None,
            raw_type=None,
            raw_class=None,
            confidence=0.0,
            status="failed",
            raw_payload=None,
        )

    raw = getattr(location, "raw", {}) or {}
    confidence, status = _score_location(location, query=query, city=city)
    return GeocodeCandidate(
        query=query,
        address=getattr(location, "address", None),
        lat=getattr(location, "latitude", None),
        lon=getattr(location, "longitude", None),
        raw_type=raw.get("type"),
        raw_class=raw.get("class"),
        confidence=confidence,
        status=status,
        raw_payload=raw,
    )


def geocode_bakery_row(
    row: pd.Series,
    user_agent: str,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Geocode one bakery row using ordered query variants."""
    city = str(row.get("city", "")).strip()
    bakery_name = str(row.get("bakery_name", "")).strip()
    variants = build_query_variants(row)
    geolocator = _get_nominatim(user_agent=user_agent, timeout_s=timeout_s)

    best: GeocodeCandidate | None = None
    last_error: str | None = None

    for query in variants:
        try:
            candidate = geocode_query_geopy(
                geolocator=geolocator,
                query=query,
                city=city,
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
            "geo_source": "geopy_nominatim",
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
        "address_normalized": best.address or row.get("address_normalized"),
        "lat": best.lat,
        "lon": best.lon,
        "geo_source": "geopy_nominatim",
        "geo_confidence": best.confidence,
        "geo_status": best.status,
        "geocode_query": best.query,
        "geocode_result_name": best.address,
        "geocode_full_name": best.address,
        "geocode_result_type": best.raw_type,
        "error": None,
    }

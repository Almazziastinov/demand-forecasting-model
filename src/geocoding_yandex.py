"""Helpers for geocoding bakery locations via Yandex Geocoder API."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from src.experiments_v2.common import CITY_COORDS


GEOCODER_URL = "https://geocode-maps.yandex.ru/v1"
DEFAULT_TIMEOUT_S = 20
DEFAULT_LANG = "ru_RU"
DEFAULT_RESULTS = 1
DEFAULT_CITY_SPN = "0.4,0.4"


@dataclass
class GeocodeCandidate:
    """A single geocoding attempt result."""

    query: str
    name: str | None
    description: str | None
    formatted_address: str | None
    lat: float | None
    lon: float | None
    kind: str | None
    precision: str | None
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


def _extract_first_geo_object(payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        members = payload["response"]["GeoObjectCollection"]["featureMember"]
    except KeyError:
        return None
    if not members:
        return None
    return members[0].get("GeoObject")


def _score_yandex_result(
    geo_object: dict[str, Any],
    query: str,
    city: str,
) -> tuple[float, str]:
    meta = (
        geo_object.get("metaDataProperty", {})
        .get("GeocoderMetaData", {})
    )
    precision = str(meta.get("precision", "")).lower()
    kind = str(meta.get("kind", "")).lower()
    formatted = str(meta.get("Address", {}).get("formatted", "")).lower()
    query_l = query.lower()
    city_l = city.lower()

    precision_weights = {
        "exact": 0.95,
        "number": 0.85,
        "near": 0.7,
        "range": 0.6,
        "street": 0.45,
        "other": 0.3,
    }
    base_score = precision_weights.get(precision, 0.25)

    if city_l and city_l in formatted:
        base_score += 0.05
    if query_l and query_l in formatted:
        base_score += 0.03

    score = min(base_score, 1.0)
    if precision == "exact" and kind == "house":
        status = "matched_exact"
    elif score >= 0.5:
        status = "matched_fuzzy"
    else:
        status = "city_only"
    return score, status


def geocode_query_yandex(
    query: str,
    api_key: str,
    city: str = "",
    timeout_s: int = DEFAULT_TIMEOUT_S,
    lang: str = DEFAULT_LANG,
) -> GeocodeCandidate:
    """Geocode a single query using Yandex Geocoder API."""
    params: dict[str, Any] = {
        "apikey": api_key,
        "geocode": query,
        "lang": lang,
        "format": "json",
        "results": DEFAULT_RESULTS,
    }

    city_coords = CITY_COORDS.get(city)
    if city_coords is not None:
        params["ll"] = f"{city_coords['lon']},{city_coords['lat']}"
        params["spn"] = DEFAULT_CITY_SPN
        params["rspn"] = 1

    payload = _http_get_json(GEOCODER_URL, params, timeout_s=timeout_s)
    if "error" in payload:
        raise RuntimeError(payload.get("message") or payload["error"])

    geo_object = _extract_first_geo_object(payload)
    if geo_object is None:
        return GeocodeCandidate(
            query=query,
            name=None,
            description=None,
            formatted_address=None,
            lat=None,
            lon=None,
            kind=None,
            precision=None,
            confidence=0.0,
            status="failed",
            raw_payload=payload,
        )

    meta = (
        geo_object.get("metaDataProperty", {})
        .get("GeocoderMetaData", {})
    )
    address = meta.get("Address", {})
    point = geo_object.get("Point", {})
    pos = str(point.get("pos", "")).strip()
    lon, lat = (None, None)
    if pos:
        lon_str, lat_str = pos.split()
        lon = float(lon_str)
        lat = float(lat_str)

    confidence, status = _score_yandex_result(geo_object, query=query, city=city)
    return GeocodeCandidate(
        query=query,
        name=geo_object.get("name"),
        description=geo_object.get("description"),
        formatted_address=address.get("formatted"),
        lat=lat,
        lon=lon,
        kind=meta.get("kind"),
        precision=meta.get("precision"),
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
            candidate = geocode_query_yandex(
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
            "geo_source": "yandex",
            "geo_confidence": 0.0,
            "geo_status": "failed",
            "geocode_query": None,
            "geocode_result_name": None,
            "geocode_full_name": None,
            "geocode_result_type": None,
            "geocode_precision": None,
            "error": last_error,
        }

    return {
        "bakery_id": row["bakery_id"],
        "Пекарня": bakery_name,
        "Город": city,
        "address_raw": row.get("address_raw"),
        "address_normalized": best.formatted_address or row.get("address_normalized"),
        "lat": best.lat,
        "lon": best.lon,
        "geo_source": "yandex",
        "geo_confidence": best.confidence,
        "geo_status": best.status,
        "geocode_query": best.query,
        "geocode_result_name": best.name,
        "geocode_full_name": best.formatted_address,
        "geocode_result_type": best.kind,
        "geocode_precision": best.precision,
        "error": None,
    }


def get_yandex_api_key(cli_value: str | None = None) -> str:
    """Resolve Yandex Geocoder API key from CLI or environment."""
    api_key = cli_value or os.getenv("YANDEX_GEOCODER_API_KEY")
    if not api_key:
        raise ValueError(
            "Yandex API key is required: pass --api-key or set YANDEX_GEOCODER_API_KEY"
        )
    return api_key

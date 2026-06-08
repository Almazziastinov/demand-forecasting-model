"""Enrich a bakery dimension CSV with no-key OpenStreetMap building tags.

This is a free fallback when paid/private geodata APIs are unavailable. Coverage
depends on OSM tags: coordinates and building type are usually easier to find
than construction year or apartments count.

Examples:
  python scripts/enrich_bakery_buildings_osm.py ^
    --input-csv C:\\Users\\dns\\Downloads\\dim_bakeries_202606021647.csv
  python scripts/enrich_bakery_buildings_osm.py --limit 10 --sleep-seconds 1.2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.geo_features import haversine_distance_m  # noqa: E402
from src.geocoding_geopy import (  # noqa: E402
    DEFAULT_TIMEOUT_S,
    build_query_variants,
    get_geopy_user_agent,
)


DEFAULT_INPUT_CSV = Path(r"C:\Users\dns\Downloads\dim_bakeries_202606021647.csv")
DEFAULT_OUT_CSV = ROOT / "data" / "processed" / "bakery_building_geo_osm.csv"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

OSM_TYPE_TO_OVERPASS = {
    "node": "node",
    "way": "way",
    "relation": "relation",
    "N": "node",
    "W": "way",
    "R": "relation",
}


def _get_nominatim(user_agent: str, timeout_s: int):
    try:
        from geopy.geocoders import Nominatim
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "geopy is not installed. "
            "Run `.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt`"
        ) from exc

    return Nominatim(user_agent=user_agent, timeout=timeout_s)


def _normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    required = {"bakery_id", "bakery_name", "city"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"input CSV missing required columns: {sorted(missing)}")

    work = df.copy()
    work["bakery_name"] = work["bakery_name"].astype(str).str.strip()
    work["city"] = work["city"].astype(str).str.strip()
    if "address_raw" not in work.columns:
        work["address_raw"] = work["bakery_name"]
    if "address_normalized" not in work.columns:
        work["address_normalized"] = work["bakery_name"] + ", " + work["city"]
    return work


def _coerce_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().replace(",", ".")
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _first_tag(tags: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = tags.get(key)
        if value not in (None, ""):
            return value
    return None


def extract_osm_building_fields(tags: dict[str, Any]) -> dict[str, Any]:
    """Map raw OSM tags to the building columns used by enrichment outputs."""
    return {
        "building_purpose_code": _first_tag(tags, ["building", "building:use"]),
        "building_ground_floors": _coerce_int(
            _first_tag(tags, ["building:levels", "levels"])
        ),
        "building_min_ground_floors": _coerce_int(tags.get("building:min_level")),
        "building_underground_floors": _coerce_int(
            tags.get("building:levels:underground")
        ),
        "building_year_of_construction": _coerce_int(
            _first_tag(
                tags,
                [
                    "start_date",
                    "building:start_date",
                    "building:year_built",
                    "year_of_construction",
                ],
            )
        ),
        "building_apartments_count": _coerce_int(
            _first_tag(tags, ["building:flats", "addr:flats", "apartments"])
        ),
        "building_porch_count": _coerce_int(
            _first_tag(tags, ["building:entrances", "entrances"])
        ),
        "building_material": _first_tag(tags, ["building:material", "material"]),
        "building_floor_type": _first_tag(tags, ["roof:shape", "building:structure"]),
        "osm_building_tag": tags.get("building"),
        "osm_addr_street": tags.get("addr:street"),
        "osm_addr_housenumber": tags.get("addr:housenumber"),
    }


def _overpass_query(query: str, timeout_s: int) -> dict[str, Any]:
    data = urlencode({"data": query}).encode("utf-8")
    request = Request(
        OVERPASS_URL,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_osm_element_tags(
    osm_type: str | None,
    osm_id: int | str | None,
    timeout_s: int,
) -> dict[str, Any] | None:
    element_type = OSM_TYPE_TO_OVERPASS.get(str(osm_type))
    if element_type is None or osm_id is None:
        return None

    query = f"""
[out:json][timeout:{timeout_s}];
{element_type}({osm_id});
out center tags;
"""
    payload = _overpass_query(query, timeout_s=timeout_s)
    elements = payload.get("elements") or []
    return elements[0] if elements else None


def _fetch_nearest_building(
    lat: float,
    lon: float,
    timeout_s: int,
    radius_m: int,
) -> dict[str, Any] | None:
    query = f"""
[out:json][timeout:{timeout_s}];
(
  way(around:{radius_m},{lat},{lon})["building"];
  relation(around:{radius_m},{lat},{lon})["building"];
);
out center tags;
"""
    payload = _overpass_query(query, timeout_s=timeout_s)
    elements = payload.get("elements") or []
    if not elements:
        return None

    def _distance(element: dict[str, Any]) -> float:
        center = element.get("center") or {}
        element_lat = element.get("lat") or center.get("lat")
        element_lon = element.get("lon") or center.get("lon")
        if element_lat is None or element_lon is None:
            return float("inf")
        return haversine_distance_m(lat, lon, float(element_lat), float(element_lon))

    return min(elements, key=_distance)


def _element_point(element: dict[str, Any] | None) -> tuple[float | None, float | None]:
    if not element:
        return None, None
    center = element.get("center") or {}
    lat = element.get("lat") or center.get("lat")
    lon = element.get("lon") or center.get("lon")
    return lat, lon


def enrich_bakery_row_osm(
    row: pd.Series,
    geolocator: Any,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    overpass_radius_m: int = 45,
    skip_overpass: bool = False,
) -> dict[str, Any]:
    """Geocode one bakery and enrich it with free OSM building tags."""
    city = str(row.get("city", "")).strip()
    variants = build_query_variants(row)
    best_location = None
    best_query = None
    last_error = None

    for query in variants:
        try:
            location = geolocator.geocode(
                query,
                exactly_one=True,
                addressdetails=True,
                extratags=True,
                namedetails=False,
                language="ru",
            )
        except Exception as exc:
            last_error = str(exc)
            continue
        if location is not None:
            best_location = location
            best_query = query
            break

    raw = getattr(best_location, "raw", {}) or {}
    lat = getattr(best_location, "latitude", None)
    lon = getattr(best_location, "longitude", None)
    address = getattr(best_location, "address", None)
    element = None
    osm_tags = dict(raw.get("extratags") or {})

    if best_location is not None and not skip_overpass:
        try:
            element = _fetch_osm_element_tags(
                raw.get("osm_type"),
                raw.get("osm_id"),
                timeout_s=timeout_s,
            )
            element_tags = (element or {}).get("tags") or {}
            if element_tags:
                osm_tags.update(element_tags)
            if "building" not in osm_tags and lat is not None and lon is not None:
                element = _fetch_nearest_building(
                    float(lat),
                    float(lon),
                    timeout_s=timeout_s,
                    radius_m=overpass_radius_m,
                )
                element_tags = (element or {}).get("tags") or {}
                if element_tags:
                    osm_tags.update(element_tags)
        except Exception as exc:
            last_error = str(exc)

    element_lat, element_lon = _element_point(element)
    if element_lat is not None and element_lon is not None:
        lat = element_lat
        lon = element_lon

    building_fields = extract_osm_building_fields(osm_tags)
    status = "failed"
    confidence = 0.0
    if best_location is not None:
        status = "matched_fuzzy"
        confidence = 0.55
    if osm_tags.get("building"):
        status = "matched_exact"
        confidence = 0.8
    if building_fields["building_year_of_construction"] is not None:
        confidence = min(confidence + 0.1, 1.0)

    building_id = (
        f"{(element or raw).get('type') or raw.get('osm_type')}:"
        f"{(element or raw).get('id') or raw.get('osm_id')}"
    )

    return {
        "bakery_id": row["bakery_id"],
        "bakery_name": str(row.get("bakery_name", "")).strip(),
        "city": city,
        "price_region": row.get("price_region"),
        "address_raw": row.get("address_raw"),
        "address_normalized": address or row.get("address_normalized"),
        "lat": lat,
        "lon": lon,
        "geo_source": "osm_nominatim_overpass",
        "geo_confidence": confidence,
        "geo_status": status,
        "geocode_query": best_query,
        "geocode_result_name": address,
        "geocode_full_name": address,
        "geocode_result_type": raw.get("type"),
        "building_id": building_id,
        "building_name": osm_tags.get("name"),
        "building_address_name": ", ".join(
            part
            for part in [
                building_fields["osm_addr_street"],
                building_fields["osm_addr_housenumber"],
            ]
            if part
        )
        or None,
        "building_full_address_name": address,
        **building_fields,
        "osm_raw_type": raw.get("type"),
        "osm_raw_class": raw.get("class"),
        "error": last_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich bakery dimension rows with no-key OSM building tags"
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Input CSV with bakery_id, bakery_name, city",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_OUT_CSV,
        help="Output enriched CSV",
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default=None,
        help="Optional custom user agent. If omitted, GEOPY_USER_AGENT is used.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke limit")
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.2,
        help="Delay between public service requests. Keep >= 1.0.",
    )
    parser.add_argument(
        "--overpass-radius-m",
        type=int,
        default=45,
        help="Radius for nearest building lookup around geocoded point",
    )
    parser.add_argument(
        "--skip-overpass",
        action="store_true",
        help="Use only Nominatim extratags; faster but lower building coverage",
    )
    parser.add_argument("--encoding", type=str, default="utf-8-sig")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv, encoding=args.encoding)
    df = _normalize_input(df)
    if args.limit is not None:
        df = df.head(args.limit).copy()

    geolocator = _get_nominatim(
        user_agent=get_geopy_user_agent(args.user_agent),
        timeout_s=DEFAULT_TIMEOUT_S,
    )

    rows = []
    total = len(df)
    print(f"Enriching {total} bakeries from {args.input_csv} via OSM ...")
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        result = enrich_bakery_row_osm(
            row,
            geolocator=geolocator,
            overpass_radius_m=args.overpass_radius_m,
            skip_overpass=args.skip_overpass,
        )
        rows.append(result)
        print(
            f"[{idx}/{total}] {result['bakery_name']} -> "
            f"{result['geo_status']} ({result['geo_confidence']:.2f}), "
            f"type={result['building_purpose_code']}, "
            f"year={result['building_year_of_construction']}, "
            f"floors={result['building_ground_floors']}, "
            f"apartments={result['building_apartments_count']}"
        )
        time.sleep(args.sleep_seconds)

    out_df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved: {args.out_csv}")
    print(
        "Status counts: "
        + str(out_df["geo_status"].value_counts(dropna=False).to_dict())
    )
    coverage_cols = [
        "building_purpose_code",
        "building_year_of_construction",
        "building_ground_floors",
        "building_apartments_count",
        "building_material",
    ]
    coverage = {
        col: int(out_df[col].notna().sum())
        for col in coverage_cols
        if col in out_df.columns
    }
    print(f"Building field coverage: {coverage}")


if __name__ == "__main__":
    main()

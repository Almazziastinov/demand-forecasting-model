"""Enrich a bakery dimension CSV with 2GIS building attributes.

Examples:
  python scripts/enrich_bakery_buildings_2gis.py ^
    --input-csv C:\\Users\\dns\\Downloads\\dim_bakeries_202606021647.csv ^
    --api-key YOUR_KEY
  python scripts/enrich_bakery_buildings_2gis.py --limit 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.geocoding_2gis import geocode_bakery_building_row, get_2gis_api_key  # noqa: E402


DEFAULT_INPUT_CSV = Path(r"C:\Users\dns\Downloads\dim_bakeries_202606021647.csv")
DEFAULT_OUT_CSV = ROOT / "data" / "processed" / "bakery_building_geo_2gis.csv"


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich bakery dimension rows with 2GIS building attributes"
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
        "--api-key",
        type=str,
        default=None,
        help="2GIS API key. If omitted, DGIS_API_KEY env var is used.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for a smoke run",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Delay between API calls",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8-sig",
        help="Input CSV encoding",
    )
    args = parser.parse_args()

    api_key = get_2gis_api_key(args.api_key)
    df = pd.read_csv(args.input_csv, encoding=args.encoding)
    df = _normalize_input(df)
    if args.limit is not None:
        df = df.head(args.limit).copy()

    rows = []
    total = len(df)
    print(f"Enriching {total} bakeries from {args.input_csv} ...")

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        result = geocode_bakery_building_row(row, api_key=api_key)
        rows.append(result)
        print(
            f"[{idx}/{total}] {result['bakery_name']} -> "
            f"{result['geo_status']} ({result['geo_confidence']:.2f}), "
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

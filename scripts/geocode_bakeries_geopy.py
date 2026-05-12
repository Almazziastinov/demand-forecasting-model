"""Geocode bakeries from bakery_geo_master.csv via geopy + Nominatim.

Examples:
  python scripts/geocode_bakeries_geopy.py
  python scripts/geocode_bakeries_geopy.py --limit 20 --sleep-seconds 1.2
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.geocoding_geopy import geocode_bakery_row, get_geopy_user_agent  # noqa: E402


DEFAULT_MASTER_CSV = ROOT / "data" / "processed" / "bakery_geo_master.csv"
DEFAULT_OUT_CSV = ROOT / "data" / "processed" / "bakery_geo_manual.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Geocode bakeries from bakery_geo_master.csv via geopy + Nominatim"
    )
    parser.add_argument(
        "--master-csv",
        type=Path,
        default=DEFAULT_MASTER_CSV,
        help="Input bakery_geo_master.csv",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_OUT_CSV,
        help="Output CSV compatible with build_geo_mvp.py",
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default=None,
        help="Optional custom user agent. If omitted, GEOPY_USER_AGENT is used.",
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
        default=1.2,
        help="Delay between requests. Keep >= 1.0 for public Nominatim.",
    )
    args = parser.parse_args()

    user_agent = get_geopy_user_agent(args.user_agent)
    master_df = pd.read_csv(args.master_csv, encoding="utf-8-sig")
    if args.limit is not None:
        master_df = master_df.head(args.limit).copy()

    rows = []
    total = len(master_df)
    print(f"Geocoding {total} bakeries from {args.master_csv} ...")

    for idx, (_, row) in enumerate(master_df.iterrows(), start=1):
        result = geocode_bakery_row(row, user_agent=user_agent)
        rows.append(result)
        print(
            f"[{idx}/{total}] {result['Пекарня']} -> "
            f"{result['geo_status']} ({result['geo_confidence']:.2f})"
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


if __name__ == "__main__":
    main()


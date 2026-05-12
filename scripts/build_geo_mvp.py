"""Build offline bakery geo master and optional POI feature table.

Examples:
  python scripts/build_geo_mvp.py
  python scripts/build_geo_mvp.py --sales-csv data/processed/daily_sales_8m_demand.csv
  python scripts/build_geo_mvp.py --existing-geo data/processed/bakery_geo_manual.csv `
    --poi-raw data/processed/bakery_poi_raw.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.geo_features import aggregate_poi_features, build_bakery_geo_master  # noqa: E402

DEFAULT_SALES_CSV = ROOT / "data" / "processed" / "daily_sales_8m_demand.csv"
DEFAULT_MASTER_OUT = ROOT / "data" / "processed" / "bakery_geo_master.csv"
DEFAULT_FEATURES_OUT = ROOT / "data" / "processed" / "bakery_geo_features.csv"


def _read_optional_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build offline bakery geo master and geo features"
    )
    parser.add_argument(
        "--sales-csv",
        type=Path,
        default=DEFAULT_SALES_CSV,
        help="Sales dataset with Дата/Пекарня/Город",
    )
    parser.add_argument(
        "--existing-geo",
        type=Path,
        default=None,
        help="Optional trusted geo master CSV",
    )
    parser.add_argument(
        "--manual-overrides",
        type=Path,
        default=None,
        help="Optional manual fixes CSV",
    )
    parser.add_argument(
        "--poi-raw",
        type=Path,
        default=None,
        help="Optional raw POI CSV for bakery-level aggregation",
    )
    parser.add_argument(
        "--master-out",
        type=Path,
        default=DEFAULT_MASTER_OUT,
        help="Output CSV for bakery_geo_master",
    )
    parser.add_argument(
        "--features-out",
        type=Path,
        default=DEFAULT_FEATURES_OUT,
        help="Output CSV for bakery_geo_features",
    )
    args = parser.parse_args()

    print(f"[1/3] Loading sales data: {args.sales_csv}")
    sales_df = pd.read_csv(args.sales_csv, encoding="utf-8-sig")

    existing_geo_df = _read_optional_csv(args.existing_geo)
    manual_overrides_df = _read_optional_csv(args.manual_overrides)

    print("[2/3] Building bakery_geo_master...")
    master_df = build_bakery_geo_master(
        sales_df=sales_df,
        existing_geo_df=existing_geo_df,
        manual_overrides_df=manual_overrides_df,
    )
    args.master_out.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(args.master_out, index=False, encoding="utf-8-sig")
    print(f"  saved: {args.master_out}")
    print(
        "  coverage: "
        f"{master_df['lat'].notna().sum()} / {len(master_df)} with coordinates, "
        f"{(master_df['geo_status'] == 'city_only').sum()} city centroid fallback"
    )

    if args.poi_raw is None:
        print("[3/3] No --poi-raw passed, skipping geo feature aggregation.")
        return

    print(f"[3/3] Aggregating POI features: {args.poi_raw}")
    poi_df = pd.read_csv(args.poi_raw, encoding="utf-8-sig")
    feature_df = aggregate_poi_features(master_df, poi_df)
    args.features_out.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(args.features_out, index=False, encoding="utf-8-sig")
    print(f"  saved: {args.features_out}")


if __name__ == "__main__":
    main()

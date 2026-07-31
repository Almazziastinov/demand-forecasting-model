"""
Create uplift profile 'pilots_evening_YYYYMMDD' from weekly_20260714:
  - Pilot bakeries, hours 0–15: sku_uplift_multiplier = 1.0  (no uplift)
  - Pilot bakeries, hours 16–23: keep original multiplier     (stockout correction handles this)
  - All other bakeries: copy unchanged

Run:
  # dry-run (local dev)
  .venv/Scripts/python.exe -m scripts.build_pilots_evening_uplift --dry-run

  # write to dev
  .venv/Scripts/python.exe -m scripts.build_pilots_evening_uplift --write-table

  # write to prod (on VM)
  python -m scripts.build_pilots_evening_uplift --env-file .env --write-table
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from dotenv import dotenv_values
except ImportError:
    def dotenv_values(path):
        vals = {}
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            vals[k.strip()] = v.strip().strip('"').strip("'")
        return vals

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import clickhouse_connect

TABLE          = "sku_hour_uplift_multiplier_embedded"
SOURCE_VERSION = "weekly_20260714"
PILOT_IDS = {
    1, 20, 21, 22, 28, 39, 41, 56, 57, 66, 67,
    69, 80, 89, 107, 125, 149, 155, 160, 221, 222, 257,
}
# All hours set to 1.0 for pilot bakeries — stockout correction is the sole
# uplift mechanism for them; mean-share floor would double-count.


def build(client, new_version: str) -> pd.DataFrame:
    df = client.query_df(f"""
        SELECT bakery_id, dow, hour,
               argMax(sku_uplift_multiplier, generated_at) AS sku_uplift_multiplier
        FROM {TABLE}
        WHERE profile_version = %(src)s
        GROUP BY bakery_id, dow, hour
    """, parameters={"src": SOURCE_VERSION})

    print(f"  Loaded {len(df)} rows from {SOURCE_VERSION}")
    print(f"  Bakeries: {df['bakery_id'].nunique()}")

    pilot_mask  = df["bakery_id"].isin(PILOT_IDS)
    before_mean = df.loc[pilot_mask, "sku_uplift_multiplier"].mean()
    df.loc[pilot_mask, "sku_uplift_multiplier"] = 1.0
    n_zeroed = pilot_mask.sum()
    print(f"  Pilot bakeries all hours: {n_zeroed} rows set to 1.0 "
          f"(was avg {before_mean:.3f})")

    df["profile_version"] = new_version
    df["generated_at"]    = datetime.now(tz=timezone.utc)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=str(ROOT / ".env.dev"))
    parser.add_argument("--new-version", default=None,
                        help="profile_version tag, default pilots_evening_YYYYMMDD")
    parser.add_argument("--write-table", action="store_true")
    parser.add_argument("--dry-run",     action="store_true")
    args = parser.parse_args()

    new_version = args.new_version or f"pilots_evening_{date.today().strftime('%Y%m%d')}"

    env     = dotenv_values(args.env_file)
    _secure = str(env.get("CLICKHOUSE_SECURE", "true")).lower() not in ("0", "false", "no")
    _verify = str(env.get("CLICKHOUSE_VERIFY", "true")).lower() not in ("0", "false", "no")
    client  = clickhouse_connect.get_client(
        host=env["CLICKHOUSE_HOST"], port=int(env["CLICKHOUSE_PORT"]),
        username=env["CLICKHOUSE_USER"], password=env["CLICKHOUSE_PASSWORD"],
        database=env["CLICKHOUSE_DATABASE"], secure=_secure, verify=_verify,
    )

    print(f"Source version : {SOURCE_VERSION}")
    print(f"New version    : {new_version}")
    print(f"Pilot bakeries : {sorted(PILOT_IDS)}")
    print(f"Pilot uplift   : 1.0 for all hours (stockout correction is sole mechanism)")
    print()

    df = build(client, new_version)

    # Stats
    pilot = df[df["bakery_id"].isin(PILOT_IDS)]
    others = df[~df["bakery_id"].isin(PILOT_IDS)]
    print()
    print(f"  Pilot bakeries: {len(pilot)} rows = 1.0")
    print(f"  Other bakeries: {len(others)} rows unchanged (avg {others['sku_uplift_multiplier'].mean():.3f})")
    print()
    print(f"Total rows to write: {len(df)}")

    if args.dry_run or not args.write_table:
        print("[dry-run] not writing")
        return

    print(f"Inserting {len(df)} rows into {TABLE} ...")
    client.insert_df(TABLE, df)
    print("Done.")
    print()
    print("Next steps:")
    print(f"  1. On VM: update FORECAST_UPLIFT_PROFILE_VERSION={new_version} in .env")
    print("  2. systemctl start forecast-production.service")


if __name__ == "__main__":
    main()

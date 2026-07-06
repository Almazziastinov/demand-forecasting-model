"""Migrate bakeable_products table: add scope + bakery_id columns.

Because ORDER BY cannot be changed via ALTER TABLE, we:
  1. Rename existing table to bakeable_products_backup_<timestamp>
  2. Create new table with updated schema
  3. Copy all existing rows (with scope='city', bakery_id=NULL as defaults)

Safe to re-run: checks if new columns already exist before migrating.

Usage
-----
.venv\\Scripts\\python.exe scripts\\migrate_bakeable_products_add_scope.py
.venv\\Scripts\\python.exe scripts\\migrate_bakeable_products_add_scope.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipelines.forecast_publish.load_forecast_run import DEFAULT_ENV_PATH, create_client
from pipelines.forecast_publish.table_names import get_table_suffix_from_env_file, table_name


def already_migrated(client, table: str) -> bool:
    df = client.query_df(
        f"SELECT name FROM system.columns WHERE table = '{table}' AND name = 'scope'"
    )
    return not df.empty


def get_row_count(client, table: str) -> int:
    df = client.query_df(f"SELECT count() as cnt FROM {table}")
    return int(df.iloc[0]["cnt"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=DEFAULT_ENV_PATH, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    client = create_client(args.env_file)
    suffix = get_table_suffix_from_env_file(args.env_file)
    tbl = table_name("bakeable_products", suffix=suffix)
    backup_ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    backup_tbl = f"{tbl}_backup_{backup_ts}"

    print(f"target table : {tbl}")
    print(f"backup table : {backup_tbl}")

    if already_migrated(client, tbl):
        print("Already migrated (scope column exists). Nothing to do.")
        return

    row_count = get_row_count(client, tbl)
    print(f"existing rows: {row_count}")

    if args.dry_run:
        print("[dry-run] would rename, recreate, and reinsert. Exiting.")
        return

    print(f"Step 1: renaming {tbl} -> {backup_tbl}", flush=True)
    client.command(f"RENAME TABLE {tbl} TO {backup_tbl}")

    print("Step 2: creating new table with scope + bakery_id", flush=True)
    client.command(f"""
        CREATE TABLE {tbl} (
          city LowCardinality(String),
          product_id String,
          product_name String,
          category_name String,
          is_bakeable UInt8,
          source LowCardinality(String),
          source_file String,
          scope LowCardinality(String),
          bakery_id Nullable(Int64),
          valid_from Date,
          valid_to Nullable(Date),
          is_active UInt8,
          loaded_at DateTime64(3),
          comment String
        )
        ENGINE = ReplacingMergeTree(loaded_at)
        ORDER BY (city, product_id, scope, coalesce(bakery_id, -1), valid_from)
    """)

    print(f"Step 3: copying rows from {backup_tbl} (scope='city', bakery_id=NULL)", flush=True)
    client.command(f"""
        INSERT INTO {tbl}
        SELECT
            city, product_id, product_name, category_name,
            is_bakeable, source, source_file,
            'city' AS scope,
            NULL  AS bakery_id,
            valid_from, valid_to, is_active, loaded_at, comment
        FROM {backup_tbl} FINAL
        WHERE is_active = 1
    """)

    new_count = get_row_count(client, tbl)
    print(f"Migration done. new table rows: {new_count} (backup: {backup_tbl})")


if __name__ == "__main__":
    main()

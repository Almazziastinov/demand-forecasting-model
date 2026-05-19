from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipelines.forecast_publish.load_forecast_run import create_client
from pipelines.forecast_publish.load_forecast_run import DEFAULT_ENV_PATH
from pipelines.forecast_publish.load_forecast_run import DEFAULT_PROFILE_PATH
from pipelines.forecast_publish.load_forecast_run import load_schema

DEFAULT_SCHEMA_PATH = ROOT / "apps" / "forecast_embedded" / "sql" / "schema.sql"
DEFAULT_EXPORT_PATH = ROOT / "data" / "processed" / "sku_hour_share_profile_smoothed.clickhouse.csv"
PROFILE_TABLE = "sku_hour_share_profile_smoothed_embedded"
CSV_CHUNK_SIZE = 200_000

PROFILE_COLUMNS = [
    "bakery_id",
    "bakery_name",
    "product_id",
    "product_name",
    "category_name",
    "dow",
    "hour",
    "n_days",
    "mean_sku_share_in_hour",
    "mean_sku_hour_sales",
    "median_sku_share_in_hour",
    "std_sku_share_in_hour",
    "mean_sku_share_in_hour_norm",
]


def normalize_profile_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    work = chunk.copy()
    for col in PROFILE_COLUMNS:
        if col not in work.columns:
            work[col] = pd.NA

    int_cols = ["bakery_id", "product_id", "dow", "hour", "n_days"]
    float_cols = [
        "mean_sku_share_in_hour",
        "mean_sku_hour_sales",
        "median_sku_share_in_hour",
        "std_sku_share_in_hour",
        "mean_sku_share_in_hour_norm",
    ]
    str_cols = ["bakery_name", "product_name", "category_name"]

    for col in int_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0).astype("int64")
    for col in float_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0).astype("float64")
    for col in str_cols:
        work[col] = work[col].fillna("").astype(str)
        work[col] = work[col].replace({"": None})

    return work[PROFILE_COLUMNS]


def truncate_profile_table(client, table: str = PROFILE_TABLE) -> None:
    client.command(f"truncate table {table}")


def load_profile_to_clickhouse(
    *,
    profile_path: str | Path,
    env_file: str | Path = DEFAULT_ENV_PATH,
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
    table: str = PROFILE_TABLE,
    chunk_size: int = CSV_CHUNK_SIZE,
    truncate: bool = False,
) -> dict[str, int]:
    client = create_client(env_file)
    load_schema(client, Path(schema_path))
    if truncate:
        truncate_profile_table(client, table)

    rows_loaded = 0
    reader = pd.read_csv(profile_path, encoding="utf-8-sig", chunksize=chunk_size)
    for chunk in reader:
        prepared = normalize_profile_chunk(chunk)
        client.insert_df(table, prepared)
        rows_loaded += len(prepared)

    return {"rows_loaded": rows_loaded}


def export_profile_from_clickhouse(
    *,
    output_path: str | Path,
    env_file: str | Path = DEFAULT_ENV_PATH,
    table: str = PROFILE_TABLE,
) -> dict[str, int]:
    client = create_client(env_file)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    query = f"""
        select {", ".join(PROFILE_COLUMNS)}
        from {table}
        order by bakery_id, dow, hour, product_id
    """
    rows_written = 0
    wrote_header = False
    with client.query_df_stream(query) as stream:
        for block in stream:
            if block.empty:
                continue
            block.to_csv(
                output_path,
                mode="a",
                index=False,
                encoding="utf-8-sig",
                header=not wrote_header,
            )
            rows_written += len(block)
            wrote_header = True
    return {"rows_written": rows_written}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load/export the smoothed SKU hour profile in ClickHouse")
    parser.add_argument("--mode", choices=["load", "export"], required=True)
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH))
    parser.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument("--profile-path", default=str(DEFAULT_PROFILE_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_EXPORT_PATH))
    parser.add_argument("--table", default=PROFILE_TABLE)
    parser.add_argument("--chunk-size", type=int, default=CSV_CHUNK_SIZE)
    parser.add_argument("--truncate", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "load":
        result = load_profile_to_clickhouse(
            profile_path=args.profile_path,
            env_file=args.env_file,
            schema_path=args.schema_path,
            table=args.table,
            chunk_size=args.chunk_size,
            truncate=args.truncate,
        )
        print("=" * 72)
        print("SKU PROFILE LOADED")
        print("=" * 72)
        print(f"table: {args.table}")
        print(f"rows_loaded: {result['rows_loaded']}")
    else:
        result = export_profile_from_clickhouse(
            output_path=args.output_path,
            env_file=args.env_file,
            table=args.table,
        )
        print("=" * 72)
        print("SKU PROFILE EXPORTED")
        print("=" * 72)
        print(f"table: {args.table}")
        print(f"rows_written: {result['rows_written']}")
        print(f"output: {args.output_path}")


if __name__ == "__main__":
    main()

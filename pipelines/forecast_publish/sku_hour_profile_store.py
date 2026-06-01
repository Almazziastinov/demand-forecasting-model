from __future__ import annotations

# ruff: noqa: E402,E501

import argparse
from pathlib import Path
import sys

import numpy as np
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
UPLIFT_MULTIPLIER_TABLE = "sku_hour_uplift_multiplier_embedded"
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

UPLIFT_MULTIPLIER_COLUMNS = [
    "bakery_id",
    "dow",
    "hour",
    "sku_uplift_multiplier",
    "profile_version",
    "generated_at",
]

APPLIED_UPLIFT_SHARE_COL = "sku_share_in_hour_adj"
APPLIED_UPLIFT_SHARE_NORM_COL = "sku_share_in_hour_adj_norm"
APPLIED_BAKERY_HOUR_SALES_COL = "bakery_hour_sales"


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


def truncate_table(client, table: str) -> None:
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


def build_uplift_multiplier_frame(
    applied_path: str | Path,
    *,
    profile_version: str,
    chunk_size: int = 1_000_000,
) -> pd.DataFrame:
    usecols = {
        "date",
        "bakery_id",
        "dow",
        "hour",
        APPLIED_BAKERY_HOUR_SALES_COL,
        APPLIED_UPLIFT_SHARE_COL,
        APPLIED_UPLIFT_SHARE_NORM_COL,
    }
    parts: list[pd.DataFrame] = []
    reader = pd.read_csv(
        applied_path,
        encoding="utf-8-sig",
        usecols=lambda c: c in usecols,
        chunksize=chunk_size,
    )
    for chunk in reader:
        for col in [
            APPLIED_BAKERY_HOUR_SALES_COL,
            APPLIED_UPLIFT_SHARE_COL,
            APPLIED_UPLIFT_SHARE_NORM_COL,
        ]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0.0)
        hourly = (
            chunk.groupby(["date", "bakery_id", "dow", "hour"], as_index=False)
            .agg(
                bakery_hour_sales=(APPLIED_BAKERY_HOUR_SALES_COL, "max"),
                raw_share_sum=(APPLIED_UPLIFT_SHARE_COL, "sum"),
                norm_share_sum=(APPLIED_UPLIFT_SHARE_NORM_COL, "sum"),
            )
        )
        hourly["sku_uplift_multiplier"] = (
            hourly["raw_share_sum"] / hourly["norm_share_sum"].replace(0.0, np.nan)
        ).fillna(1.0).clip(lower=1.0)
        hourly["weighted_multiplier"] = (
            hourly["sku_uplift_multiplier"] * hourly["bakery_hour_sales"]
        )
        parts.append(
            hourly[
                [
                    "bakery_id",
                    "dow",
                    "hour",
                    "bakery_hour_sales",
                    "weighted_multiplier",
                ]
            ]
        )

    if not parts:
        return pd.DataFrame(columns=UPLIFT_MULTIPLIER_COLUMNS)

    combined = pd.concat(parts, ignore_index=True)
    result = (
        combined.groupby(["bakery_id", "dow", "hour"], as_index=False)
        .agg(
            bakery_hour_sales=("bakery_hour_sales", "sum"),
            weighted_multiplier=("weighted_multiplier", "sum"),
        )
    )
    result["sku_uplift_multiplier"] = (
        result["weighted_multiplier"] / result["bakery_hour_sales"].replace(0.0, np.nan)
    ).fillna(1.0).clip(lower=1.0)
    result["profile_version"] = profile_version
    result["generated_at"] = pd.Timestamp.utcnow().tz_localize(None)

    fallback = (
        combined.groupby(["bakery_id", "hour"], as_index=False)
        .agg(
            bakery_hour_sales=("bakery_hour_sales", "sum"),
            weighted_multiplier=("weighted_multiplier", "sum"),
        )
    )
    fallback["sku_uplift_multiplier"] = (
        fallback["weighted_multiplier"]
        / fallback["bakery_hour_sales"].replace(0.0, np.nan)
    ).fillna(1.0).clip(lower=1.0)
    fallback["dow"] = -1
    fallback["profile_version"] = profile_version
    fallback["generated_at"] = result["generated_at"].iloc[0]
    result = pd.concat(
        [
            result,
            fallback[
                [
                    "bakery_id",
                    "dow",
                    "hour",
                    "sku_uplift_multiplier",
                    "profile_version",
                    "generated_at",
                ]
            ],
        ],
        ignore_index=True,
    )

    int_cols = ["bakery_id", "dow", "hour"]
    for col in int_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0).astype("int64")
    result["sku_uplift_multiplier"] = pd.to_numeric(
        result["sku_uplift_multiplier"],
        errors="coerce",
    ).fillna(1.0).astype("float64")
    return result[UPLIFT_MULTIPLIER_COLUMNS]


def load_uplift_multipliers_to_clickhouse(
    *,
    applied_path: str | Path,
    env_file: str | Path = DEFAULT_ENV_PATH,
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
    table: str = UPLIFT_MULTIPLIER_TABLE,
    profile_version: str = "default",
    chunk_size: int = CSV_CHUNK_SIZE,
    truncate: bool = False,
) -> dict[str, int]:
    client = create_client(env_file)
    load_schema(client, Path(schema_path))
    if truncate:
        truncate_table(client, table)

    multipliers = build_uplift_multiplier_frame(
        applied_path,
        profile_version=profile_version,
        chunk_size=chunk_size,
    )
    if len(multipliers):
        client.insert_df(table, multipliers)
    return {"rows_loaded": int(len(multipliers))}


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
    parser.add_argument(
        "--mode",
        choices=["load", "export", "load-uplift-multipliers"],
        required=True,
    )
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH))
    parser.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument("--profile-path", default=str(DEFAULT_PROFILE_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_EXPORT_PATH))
    parser.add_argument("--table", default=PROFILE_TABLE)
    parser.add_argument("--uplift-table", default=UPLIFT_MULTIPLIER_TABLE)
    parser.add_argument("--applied-path", default="")
    parser.add_argument("--profile-version", default="default")
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
    elif args.mode == "export":
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
    else:
        result = load_uplift_multipliers_to_clickhouse(
            applied_path=args.applied_path,
            env_file=args.env_file,
            schema_path=args.schema_path,
            table=args.uplift_table,
            profile_version=args.profile_version,
            chunk_size=args.chunk_size,
            truncate=args.truncate,
        )
        print("=" * 72)
        print("SKU UPLIFT MULTIPLIERS LOADED")
        print("=" * 72)
        print(f"table: {args.uplift_table}")
        print(f"profile_version: {args.profile_version}")
        print(f"rows_loaded: {result['rows_loaded']}")


if __name__ == "__main__":
    main()

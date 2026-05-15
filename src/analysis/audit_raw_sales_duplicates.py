from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.experiments_v2.raw_snapshot_schema import normalize_snapshot_chunk
from src.experiments_v2.raw_snapshot_schema import parse_snapshot_date
from src.experiments_v2.raw_snapshot_schema import parse_snapshot_datetime


DEFAULT_SOURCE_PATH = ROOT / "data" / "raw" / "sales_hrs_all_clickhouse.csv"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "raw_sales_duplicate_audit"
CHUNK_SIZE = 1_000_000
SALES_EVENTS = {"Продажа"}

REQUIRED_COLS = [
    "check_datetime",
    "check_date",
    "cash_event_type",
    "quantity",
    "price",
    "line_amount",
    "bakery_id",
    "bakery_name",
    "city",
    "product_id",
    "product_name",
    "category_name",
]

STRICT_DUP_KEYS = [
    "check_datetime",
    "bakery_id",
    "product_id",
    "quantity",
    "price",
    "line_amount",
    "cash_event_type",
]

RELAXED_DUP_KEYS = [
    "check_datetime",
    "bakery_id",
    "product_id",
]


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _prepare_chunk(chunk: pd.DataFrame, target_dates: set[pd.Timestamp]) -> pd.DataFrame:
    work = normalize_snapshot_chunk(chunk)
    for col in REQUIRED_COLS:
        if col not in work.columns:
            work[col] = pd.NA

    work["check_date"] = parse_snapshot_date(work["check_date"]).dt.normalize()
    work = work[work["check_date"].isin(target_dates)].copy()
    if work.empty:
        return work

    work["check_datetime"] = parse_snapshot_datetime(work["check_datetime"])
    work = work[work["cash_event_type"].isin(SALES_EVENTS)].copy()
    if work.empty:
        return work

    work["quantity"] = pd.to_numeric(work["quantity"], errors="coerce").fillna(0.0)
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work["line_amount"] = pd.to_numeric(work["line_amount"], errors="coerce")
    work["quantity"] = work["quantity"].clip(lower=0.0)
    return work[REQUIRED_COLS].copy()


def load_sales_rows_for_dates(
    source_path: str | Path,
    target_dates: list[pd.Timestamp],
    *,
    chunk_size: int = CHUNK_SIZE,
) -> pd.DataFrame:
    target_date_set = {pd.Timestamp(dt).normalize() for dt in target_dates}
    parts: list[pd.DataFrame] = []

    reader = pd.read_csv(source_path, encoding="utf-8-sig", chunksize=chunk_size)
    for i, chunk in enumerate(reader, start=1):
        part = _prepare_chunk(chunk, target_date_set)
        if not part.empty:
            parts.append(part)
        if i % 5 == 0:
            print(f"processed chunks: {i}", flush=True)

    if not parts:
        return pd.DataFrame(columns=REQUIRED_COLS)
    return pd.concat(parts, ignore_index=True)


def summarize_duplicate_groups(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=keys + ["duplicate_rows", "quantity_sum", "line_amount_sum"])

    grouped = (
        df.groupby(keys, dropna=False, as_index=False)
        .agg(
            duplicate_rows=("check_date", "size"),
            quantity_sum=("quantity", "sum"),
            line_amount_sum=("line_amount", "sum"),
        )
        .sort_values(["duplicate_rows", "quantity_sum", "line_amount_sum"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    return grouped[grouped["duplicate_rows"] > 1].reset_index(drop=True)


def build_date_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "check_date",
                "rows",
                "bakeries",
                "unique_checks",
                "unique_products",
                "quantity_sum",
                "line_amount_sum",
            ]
        )

    summary = (
        df.groupby("check_date", as_index=False)
        .agg(
            rows=("check_date", "size"),
            bakeries=("bakery_id", "nunique"),
            unique_checks=("check_datetime", "nunique"),
            unique_products=("product_id", "nunique"),
            quantity_sum=("quantity", "sum"),
            line_amount_sum=("line_amount", "sum"),
        )
        .sort_values("check_date")
        .reset_index(drop=True)
    )
    return summary


def build_bakery_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "check_date",
                "bakery_id",
                "bakery_name",
                "city",
                "rows",
                "unique_checks",
                "unique_products",
                "quantity_sum",
                "line_amount_sum",
            ]
        )

    return (
        df.groupby(["check_date", "bakery_id", "bakery_name", "city"], as_index=False)
        .agg(
            rows=("check_date", "size"),
            unique_checks=("check_datetime", "nunique"),
            unique_products=("product_id", "nunique"),
            quantity_sum=("quantity", "sum"),
            line_amount_sum=("line_amount", "sum"),
        )
        .sort_values(["check_date", "quantity_sum"], ascending=[True, False])
        .reset_index(drop=True)
    )


def build_suspicious_check_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "check_date",
                "check_datetime",
                "bakery_id",
                "bakery_name",
                "city",
                "rows",
                "unique_products",
                "quantity_sum",
                "line_amount_sum",
            ]
        )

    checks = (
        df.groupby(["check_date", "check_datetime", "bakery_id", "bakery_name", "city"], as_index=False)
        .agg(
            rows=("product_id", "size"),
            unique_products=("product_id", "nunique"),
            quantity_sum=("quantity", "sum"),
            line_amount_sum=("line_amount", "sum"),
        )
        .sort_values(["rows", "quantity_sum"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return checks


def build_overview(
    df: pd.DataFrame,
    strict_dups: pd.DataFrame,
    relaxed_dups: pd.DataFrame,
    target_dates: list[pd.Timestamp],
    source_path: str | Path,
) -> dict:
    raw_qty = float(df["quantity"].sum()) if not df.empty else 0.0
    raw_amount = float(df["line_amount"].fillna(0.0).sum()) if not df.empty else 0.0

    strict_dedup = df.drop_duplicates(subset=STRICT_DUP_KEYS, keep="first") if not df.empty else df
    relaxed_dedup = df.drop_duplicates(subset=RELAXED_DUP_KEYS, keep="first") if not df.empty else df

    strict_qty_gap = raw_qty - float(strict_dedup["quantity"].sum()) if not strict_dedup.empty else raw_qty
    relaxed_qty_gap = raw_qty - float(relaxed_dedup["quantity"].sum()) if not relaxed_dedup.empty else raw_qty
    strict_amount_gap = raw_amount - float(strict_dedup["line_amount"].fillna(0.0).sum()) if not strict_dedup.empty else raw_amount
    relaxed_amount_gap = raw_amount - float(relaxed_dedup["line_amount"].fillna(0.0).sum()) if not relaxed_dedup.empty else raw_amount

    return {
        "source_path": str(source_path),
        "target_dates": [str(pd.Timestamp(dt).date()) for dt in target_dates],
        "rows_filtered": int(len(df)),
        "date_min": None if df.empty else str(df["check_date"].min().date()),
        "date_max": None if df.empty else str(df["check_date"].max().date()),
        "bakeries": int(df["bakery_id"].nunique()) if not df.empty else 0,
        "unique_checks": int(df["check_datetime"].nunique()) if not df.empty else 0,
        "unique_products": int(df["product_id"].nunique()) if not df.empty else 0,
        "raw_quantity_sum": round(raw_qty, 6),
        "raw_line_amount_sum": round(raw_amount, 6),
        "exact_duplicate_rows": int(df.duplicated().sum()) if not df.empty else 0,
        "strict_duplicate_groups": int(len(strict_dups)),
        "strict_duplicate_extra_rows": int(strict_dups["duplicate_rows"].sub(1).sum()) if not strict_dups.empty else 0,
        "strict_duplicate_quantity_gap": round(strict_qty_gap, 6),
        "strict_duplicate_line_amount_gap": round(strict_amount_gap, 6),
        "relaxed_duplicate_groups": int(len(relaxed_dups)),
        "relaxed_duplicate_extra_rows": int(relaxed_dups["duplicate_rows"].sub(1).sum()) if not relaxed_dups.empty else 0,
        "relaxed_duplicate_quantity_gap_upper_bound": round(relaxed_qty_gap, 6),
        "relaxed_duplicate_line_amount_gap_upper_bound": round(relaxed_amount_gap, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit duplicate risk in raw check-level sales snapshot")
    parser.add_argument("--source-path", default=str(DEFAULT_SOURCE_PATH))
    parser.add_argument("--date", action="append", dest="dates", required=True, help="Target date in YYYY-MM-DD format. Repeatable.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--save-filtered-rows", action="store_true")
    args = parser.parse_args()

    target_dates = [pd.Timestamp(dt).normalize() for dt in args.dates]
    output_dir = Path(args.output_dir)
    source_path = Path(args.source_path)

    filtered = load_sales_rows_for_dates(source_path, target_dates, chunk_size=args.chunk_size)
    date_summary = build_date_summary(filtered)
    bakery_summary = build_bakery_summary(filtered)
    suspicious_checks = build_suspicious_check_summary(filtered)
    strict_dups = summarize_duplicate_groups(filtered, STRICT_DUP_KEYS)
    relaxed_dups = summarize_duplicate_groups(filtered, RELAXED_DUP_KEYS)
    overview = build_overview(filtered, strict_dups, relaxed_dups, target_dates, source_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_filtered_rows:
        save_csv(filtered, output_dir / "filtered_sales_rows.csv")
    save_csv(date_summary, output_dir / "date_summary.csv")
    save_csv(bakery_summary, output_dir / "bakery_summary.csv")
    save_csv(suspicious_checks, output_dir / "suspicious_checks.csv")
    save_csv(strict_dups, output_dir / "strict_duplicate_groups.csv")
    save_csv(relaxed_dups, output_dir / "relaxed_duplicate_groups.csv")
    (output_dir / "overview.json").write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved: {output_dir}")
    print(json.dumps(overview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Extract checkout freshness facts for the fixed pilot/SKU research scope."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ROOT / "data/raw/sales_stg_2025_2026.csv",
    ROOT / "data/raw/pilot_stg_check_lines_2026-04-30_2026-07-19.csv",
]
PILOT = ROOT / ".codex_tmp/pilot_management_sep01_09/detail.csv"
TRUSTED = ROOT / "reports/prod_direct_end_of_day_economics_august_folds_20260910/daily_rows.parquet"
OUTPUT = ROOT / ".codex_tmp/pilot_freshness_history_20260911.parquet"
USECOLS = [
    "check_datetime",
    "check_date",
    "cash_event_type",
    "quantity",
    "bakery_id",
    "product_id",
    "freshness",
    "price",
    "line_amount",
]


def main() -> None:
    bakery_ids = set(pd.read_csv(PILOT, usecols=["bakery_id"])["bakery_id"].dropna().astype(int))
    product_ids = set(pd.read_parquet(TRUSTED, columns=["product_id"])["product_id"].astype(int).unique())
    parts = []
    for source in SOURCES:
        for chunk in pd.read_csv(source, usecols=USECOLS, chunksize=1_000_000, low_memory=False):
            chunk["bakery_id"] = pd.to_numeric(chunk["bakery_id"], errors="coerce")
            chunk["product_id"] = pd.to_numeric(chunk["product_id"], errors="coerce")
            chunk = chunk[
                chunk["bakery_id"].isin(bakery_ids)
                & chunk["product_id"].isin(product_ids)
                & chunk["check_date"].astype(str).between("2025-01-01", "2026-07-31")
            ]
            if len(chunk):
                parts.append(chunk)
    rows = pd.concat(parts, ignore_index=True)
    rows = rows.drop_duplicates(
        ["check_datetime", "check_date", "cash_event_type", "quantity", "bakery_id", "product_id", "freshness", "price", "line_amount"]
    )
    rows["quantity"] = pd.to_numeric(rows["quantity"], errors="coerce").fillna(0)
    rows["line_amount"] = pd.to_numeric(rows["line_amount"], errors="coerce").fillna(0)
    rows["freshness"] = rows["freshness"].fillna("Не указано")
    result = rows.groupby(
        ["check_date", "bakery_id", "product_id", "freshness"], as_index=False
    ).agg(quantity=("quantity", "sum"), amount=("line_amount", "sum"), lines=("quantity", "size"))
    result["check_date"] = pd.to_datetime(result["check_date"])
    result.to_parquet(OUTPUT, index=False)
    print(result.groupby("freshness").agg(quantity=("quantity", "sum"), amount=("amount", "sum"), lines=("lines", "sum")).to_string())
    print("rows", len(result), "dates", result["check_date"].nunique(), "bakeries", result["bakery_id"].nunique())


if __name__ == "__main__":
    main()

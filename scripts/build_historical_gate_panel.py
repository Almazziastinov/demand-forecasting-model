"""Prepare a compact causal SKU-day panel for the historical gate backtest."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/sku_daily_research_panel.csv"
PILOT_DETAIL = ROOT / ".codex_tmp/pilot_management_sep01_09/detail.csv"
OUTPUT = ROOT / ".codex_tmp/historical_gate_backtest/panel.parquet"
CHUNK_SIZE = 250_000
START = pd.Timestamp("2025-01-15")
END = pd.Timestamp("2026-05-12")
USECOLS = [
    "date",
    "bakery_id",
    "bakery_name",
    "city",
    "product_id",
    "product_name",
    "category_name",
    "observed_sales_qty",
    "observed_sales_amount",
    "first_sale_hour",
    "last_sale_hour",
    "avg_sales_price",
    "release_qty",
    "incoming_move_qty",
    "outgoing_move_qty",
    "release_present_flag",
    "all_sources_present_flag",
]


def main() -> None:
    pilot = pd.read_csv(PILOT_DETAIL, usecols=["bakery_id"])
    bakery_ids = set(pd.to_numeric(pilot["bakery_id"]).dropna().astype(int))
    parts = []
    for index, chunk in enumerate(
        pd.read_csv(SOURCE, usecols=USECOLS, chunksize=CHUNK_SIZE), start=1
    ):
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk["bakery_id"] = pd.to_numeric(
            chunk["bakery_id"], errors="coerce"
        ).astype("Int64")
        selected = chunk[
            chunk["bakery_id"].isin(bakery_ids)
            & chunk["date"].between(START, END)
        ].copy()
        if not selected.empty:
            selected["bakery_id"] = selected["bakery_id"].astype(int)
            selected["product_id"] = pd.to_numeric(
                selected["product_id"], errors="coerce"
            ).astype("Int64")
            selected = selected.dropna(subset=["product_id"])
            selected["product_id"] = selected["product_id"].astype(int)
            parts.append(selected)
        if index % 20 == 0:
            print(f"chunks={index} selected={sum(len(part) for part in parts):,}")
    panel = pd.concat(parts, ignore_index=True)
    panel = panel.drop_duplicates(["date", "bakery_id", "product_id"], keep="last")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUTPUT, index=False)
    coverage = panel.groupby(panel["date"].dt.to_period("M")).agg(
        rows=("product_id", "size"),
        bakeries=("bakery_id", "nunique"),
        sales=("observed_sales_qty", "sum"),
        production=("release_qty", "sum"),
    )
    coverage.to_csv(OUTPUT.with_name("coverage.csv"))
    print(coverage.to_string())


if __name__ == "__main__":
    main()

"""Combine historical warm-up with fresh ClickHouse facts and add causal maturity."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / ".codex_tmp/historical_gate_backtest/causal_mature_panel.parquet"
FRESH = ROOT / ".codex_tmp/historical_clickhouse_panel"
OUTPUT = ROOT / ".codex_tmp/historical_clickhouse_panel/causal_panel.parquet"
KEYS = ["date", "bakery_id", "product_id"]


def main() -> None:
    old = pd.read_parquet(OLD)
    old = old[old["date"].lt(pd.Timestamp("2026-01-01"))].copy()
    fresh = pd.concat(
        [
            pd.read_csv(path, low_memory=False)
            for path in sorted(FRESH.glob("2026??.csv.gz"))
        ],
        ignore_index=True,
    )
    fresh["date"] = pd.to_datetime(fresh["date"]).dt.normalize()
    columns = [
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
        "written_off_qty",
    ]
    for frame in (old, fresh):
        for column in columns:
            if column not in frame:
                frame[column] = 0.0 if column.endswith("_qty") else pd.NA
    rows = pd.concat([old[columns], fresh[columns]], ignore_index=True)
    rows = (
        rows.drop_duplicates(KEYS, keep="last").sort_values(KEYS).reset_index(drop=True)
    )
    numeric = [
        "observed_sales_qty",
        "observed_sales_amount",
        "release_qty",
        "incoming_move_qty",
        "outgoing_move_qty",
        "written_off_qty",
    ]
    for column in numeric:
        rows[column] = pd.to_numeric(rows[column], errors="coerce").fillna(0.0)
    rows["activity"] = rows[numeric].abs().sum(axis=1).gt(0)
    rows["sales_positive"] = rows["observed_sales_qty"].gt(0).astype(float)
    rows["first_observed_date"] = rows.groupby(["bakery_id", "product_id"])[
        "date"
    ].transform("min")
    rows["pair_age_days"] = (rows["date"] - rows["first_observed_date"]).dt.days
    # Calendar-time rolling counts, explicitly excluding the current day.
    indexed = rows.set_index("date")
    grouped = indexed.groupby(["bakery_id", "product_id"], sort=False)
    active = grouped["activity"].rolling("28D", closed="left").sum()
    sales = grouped["sales_positive"].rolling("28D", closed="left").sum()
    rows["prior_active_days_28"] = active.reset_index(
        level=[0, 1], drop=True
    ).to_numpy()
    rows["prior_sales_days_28"] = sales.reset_index(level=[0, 1], drop=True).to_numpy()
    rows["is_mature_causal"] = (
        rows["prior_active_days_28"].gt(0)
        & rows["prior_sales_days_28"].ge(3)
        & rows["pair_age_days"].ge(14)
    )
    rows["is_cold_start_causal"] = rows["activity"] & ~rows["is_mature_causal"]
    rows = rows.drop(columns="sales_positive")
    rows.to_parquet(OUTPUT, index=False)
    coverage = (
        rows[rows["date"].between("2026-01-01", "2026-08-31")]
        .groupby(
            rows.loc[
                rows["date"].between("2026-01-01", "2026-08-31"), "date"
            ].dt.to_period("M")
        )
        .agg(
            rows=("product_id", "size"),
            bakeries=("bakery_id", "nunique"),
            sales=("observed_sales_qty", "sum"),
        )
    )
    coverage.to_csv(FRESH / "causal_coverage.csv")
    print(coverage.to_string())


if __name__ == "__main__":
    main()

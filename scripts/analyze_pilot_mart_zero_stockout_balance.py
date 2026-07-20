"""Build stockout balance diagnostics from pilot mart_zero_sales_60d export.

This is the first dataset-forming step for the current stockout-demand
research vector. It uses only local CSV extracts:

- data/raw/pilot_mart_zero_sales_2026-04-30_2026-07-19.csv
- data/raw/pilot_stg_check_lines_2026-04-30_2026-07-19.csv

No production state is read or written by this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DAILY_PATH = ROOT / "data" / "raw" / "pilot_mart_zero_sales_2026-04-30_2026-07-19.csv"
DEFAULT_HOURLY_PATH = ROOT / "data" / "raw" / "pilot_stg_check_lines_2026-04-30_2026-07-19.csv"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "pilot_mart_zero_stockout_balance"

BAKEABLE_CATEGORIES = {
    "Пироги сытные",
    "Пироги сладкие",
    "Выпечка сытная",
    "Выпечка сладкая",
    "Фастфуд",
}

PILOT_BAKERY_IDS = {16, 20, 21, 22, 28, 80, 89, 107, 221, 222, 257}


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def load_daily(path: Path) -> pd.DataFrame:
    daily = pd.read_csv(path, encoding="utf-8-sig")
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily["bakery_id"] = pd.to_numeric(daily["bakery_id"], errors="coerce").astype("Int64")
    daily["product_id"] = pd.to_numeric(daily["product_id"], errors="coerce").astype("Int64")
    for column in ["qty_sold", "qty_produced", "qty_received", "qty_sent", "stock_balance", "revenue"]:
        if column in daily.columns:
            daily[column] = _to_numeric(daily[column])
    daily = daily[daily["bakery_id"].isin(PILOT_BAKERY_IDS)].copy()
    daily = daily[daily["category_name"].isin(BAKEABLE_CATEGORIES)].copy()
    return daily


def load_hourly(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    usecols = ["check_datetime", "check_date", "quantity", "bakery_id", "product_id", "category_name"]
    sku_parts: list[pd.DataFrame] = []
    bakery_parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, encoding="utf-8-sig", usecols=usecols, chunksize=250_000):
        chunk["bakery_id"] = pd.to_numeric(chunk["bakery_id"], errors="coerce").astype("Int64")
        chunk["product_id"] = pd.to_numeric(chunk["product_id"], errors="coerce").astype("Int64")
        chunk = chunk[chunk["bakery_id"].isin(PILOT_BAKERY_IDS)].copy()
        if chunk.empty:
            continue
        chunk["date"] = pd.to_datetime(chunk["check_date"], errors="coerce").dt.normalize()
        dt = pd.to_datetime(chunk["check_datetime"], errors="coerce", utc=True)
        chunk["hour"] = dt.dt.tz_convert("Europe/Moscow").dt.hour
        chunk["sold"] = _to_numeric(chunk["quantity"]).clip(lower=0.0)
        bakery_parts.append(
            chunk.groupby(["date", "bakery_id", "hour"], as_index=False)["sold"].sum()
        )
        bakeable = chunk[chunk["category_name"].isin(BAKEABLE_CATEGORIES)]
        if not bakeable.empty:
            sku_parts.append(
                bakeable.groupby(
                    ["date", "bakery_id", "product_id", "hour"],
                    as_index=False,
                )["sold"].sum()
            )

    if not sku_parts:
        empty_hourly = pd.DataFrame(columns=["date", "bakery_id", "product_id", "hour", "sold"])
        empty_bakery = pd.DataFrame(columns=["date", "bakery_id", "hour", "bakery_hour_sales"])
        return empty_hourly, empty_bakery

    hourly = (
        pd.concat(sku_parts, ignore_index=True)
        .groupby(["date", "bakery_id", "product_id", "hour"], as_index=False)["sold"]
        .sum()
    )
    bakery_hour = (
        pd.concat(bakery_parts, ignore_index=True)
        .groupby(["date", "bakery_id", "hour"], as_index=False)["sold"]
        .sum()
        .rename(columns={"sold": "bakery_hour_sales"})
    )
    return hourly, bakery_hour


def build_balance(daily: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    work = daily.copy()
    hourly_daily = (
        hourly.groupby(["date", "bakery_id", "product_id"], as_index=False)["sold"]
        .sum()
        .rename(columns={"sold": "hourly_sold"})
    )
    work = work.merge(hourly_daily, on=["date", "bakery_id", "product_id"], how="left")
    work["hourly_sold"] = work["hourly_sold"].fillna(0.0)

    work["expected_stock_balance"] = (
        work["qty_produced"] + work["qty_received"] - work["qty_sent"] - work["qty_sold"]
    )
    work["balance_error"] = work["stock_balance"] - work["expected_stock_balance"]
    work["balance_is_consistent"] = work["balance_error"].abs() <= 1.0
    work["hourly_daily_sales_error"] = work["hourly_sold"] - work["qty_sold"]
    work["hourly_daily_sales_agree"] = work["hourly_daily_sales_error"].abs() <= 1.0
    work["sell_through"] = work["qty_sold"] / work["qty_produced"].replace(0.0, np.nan)
    work["is_simple_stockout"] = work["sell_through"] >= 0.90
    work["is_inventory_stockout"] = (
        work["balance_is_consistent"]
        & (work["stock_balance"] <= 1.0)
        & ((work["qty_produced"] + work["qty_received"] - work["qty_sent"]) > 0)
    )
    work["is_reliable_inventory_stockout"] = (
        work["is_inventory_stockout"] & work["hourly_daily_sales_agree"]
    )
    return work


def build_hourly_frame(
    balance: pd.DataFrame,
    hourly: pd.DataFrame,
    bakery_hour: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "date",
        "bakery_id",
        "bakery_name",
        "city",
        "product_id",
        "product_name",
        "category_name",
        "qty_sold",
        "qty_produced",
        "qty_received",
        "qty_sent",
        "stock_balance",
        "hourly_sold",
        "balance_is_consistent",
        "hourly_daily_sales_agree",
        "is_inventory_stockout",
        "is_reliable_inventory_stockout",
    ]
    base = balance[keys].copy()
    hours = pd.DataFrame({"hour": range(6, 24)})
    frame = base.merge(hours, how="cross")
    frame = frame.merge(hourly, on=["date", "bakery_id", "product_id", "hour"], how="left")
    frame = frame.merge(bakery_hour, on=["date", "bakery_id", "hour"], how="left")
    frame["sold"] = frame["sold"].fillna(0.0)
    frame["bakery_hour_sales"] = frame["bakery_hour_sales"].fillna(0.0)
    frame["dow"] = frame["date"].dt.dayofweek
    return frame


def summarize(balance: pd.DataFrame, hourly_frame: pd.DataFrame) -> dict[str, Any]:
    stockouts = balance[balance["is_inventory_stockout"]]
    reliable = balance[balance["is_reliable_inventory_stockout"]]
    by_bakery = (
        balance.groupby(["bakery_id", "bakery_name"], as_index=False)
        .agg(
            rows=("date", "size"),
            dates=("date", "nunique"),
            products=("product_id", "nunique"),
            balance_consistent_share=("balance_is_consistent", "mean"),
            hourly_sales_agree_share=("hourly_daily_sales_agree", "mean"),
            inventory_stockouts=("is_inventory_stockout", "sum"),
            reliable_inventory_stockouts=("is_reliable_inventory_stockout", "sum"),
            qty_sold=("qty_sold", "sum"),
            qty_produced=("qty_produced", "sum"),
        )
        .sort_values("bakery_id")
    )
    return {
        "rows": int(len(balance)),
        "hourly_frame_rows": int(len(hourly_frame)),
        "date_min": str(balance["date"].min().date()) if len(balance) else None,
        "date_max": str(balance["date"].max().date()) if len(balance) else None,
        "dates": int(balance["date"].nunique()),
        "bakeries": sorted(int(value) for value in balance["bakery_id"].dropna().unique()),
        "products": int(balance["product_id"].nunique()),
        "balance_consistent_rows": int(balance["balance_is_consistent"].sum()),
        "balance_consistent_share": float(balance["balance_is_consistent"].mean()),
        "hourly_daily_sales_agree_rows": int(balance["hourly_daily_sales_agree"].sum()),
        "hourly_daily_sales_agree_share": float(balance["hourly_daily_sales_agree"].mean()),
        "inventory_stockouts": int(len(stockouts)),
        "reliable_inventory_stockouts": int(len(reliable)),
        "simple_stockouts": int(balance["is_simple_stockout"].sum()),
        "both_simple_and_inventory": int((balance["is_simple_stockout"] & balance["is_inventory_stockout"]).sum()),
        "median_abs_balance_error": float(balance["balance_error"].abs().median()),
        "p90_abs_balance_error": float(balance["balance_error"].abs().quantile(0.90)),
        "median_abs_hourly_daily_sales_error": float(balance["hourly_daily_sales_error"].abs().median()),
        "p90_abs_hourly_daily_sales_error": float(balance["hourly_daily_sales_error"].abs().quantile(0.90)),
        "by_bakery": by_bakery.to_dict(orient="records"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze pilot mart_zero stockout balance")
    parser.add_argument("--daily-path", default=str(DEFAULT_DAILY_PATH))
    parser.add_argument("--hourly-path", default=str(DEFAULT_HOURLY_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    daily = load_daily(Path(args.daily_path))
    hourly, bakery_hour = load_hourly(Path(args.hourly_path))
    balance = build_balance(daily, hourly)
    hourly_frame = build_hourly_frame(balance, hourly, bakery_hour)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    balance.to_csv(output_dir / "inventory_balance.csv", index=False, encoding="utf-8-sig")
    hourly_frame.to_csv(output_dir / "hourly_frame.csv", index=False, encoding="utf-8-sig")
    balance[~balance["balance_is_consistent"]].sort_values(
        "balance_error", key=lambda series: series.abs(), ascending=False
    ).head(500).to_csv(output_dir / "largest_balance_errors.csv", index=False, encoding="utf-8-sig")
    balance[balance["is_reliable_inventory_stockout"]].to_csv(
        output_dir / "reliable_inventory_stockouts.csv", index=False, encoding="utf-8-sig"
    )
    summary = summarize(balance, hourly_frame)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

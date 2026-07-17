"""Overlay inventory-confirmed stockouts on local hourly sales."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments_v2.stockout_demand_preprocessing import (  # noqa: E402
    build_bakery_share_reference,
    build_uncensored_hour_reference,
    reconstruct_stockout_demand,
    reconstruct_stockout_demand_from_bakery_share,
)

PILOT_BAKERY_IDS = {20, 21, 22, 28, 80, 89, 107, 221, 222, 257}


def _numeric_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype("int64")


def load_hourly_sales(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    usecols = [
        "check_datetime",
        "check_date",
        "cash_event_type",
        "quantity",
        "bakery_id",
        "product_id",
    ]
    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=750_000):
        chunk["bakery_id"] = _numeric_id(chunk["bakery_id"])
        chunk = chunk[
            chunk["bakery_id"].isin(PILOT_BAKERY_IDS)
            & (chunk["cash_event_type"] == "Продажа")
        ].copy()
        if chunk.empty:
            continue
        chunk["date"] = pd.to_datetime(chunk["check_date"], errors="coerce")
        chunk = chunk[
            chunk["date"].between(
                pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-29")
            )
        ]
        if chunk.empty:
            continue
        chunk["product_id"] = _numeric_id(chunk["product_id"])
        chunk["hour"] = (
            pd.to_datetime(chunk["check_datetime"], errors="coerce", utc=True)
            .dt.tz_convert("Europe/Moscow")
            .dt.hour
        )
        chunk["sold"] = pd.to_numeric(chunk["quantity"], errors="coerce").fillna(0.0)
        parts.append(chunk[["date", "bakery_id", "product_id", "hour", "sold"]])
    sales = (
        pd.concat(parts, ignore_index=True)
        .groupby(["date", "bakery_id", "product_id", "hour"], as_index=False)["sold"]
        .sum()
    )
    bakery_hour = (
        sales.groupby(["date", "bakery_id", "hour"], as_index=False)["sold"]
        .sum()
        .rename(columns={"sold": "bakery_hour_sales"})
    )
    return sales, bakery_hour


def build_hourly_frame(
    balance: pd.DataFrame,
    sales: pd.DataFrame,
    bakery_hour: pd.DataFrame,
) -> pd.DataFrame:
    balance["date"] = pd.to_datetime(balance["date"])
    balance = balance.rename(columns={"sold": "daily_sold"})
    hours = pd.DataFrame({"hour": range(6, 24)})
    frame = balance.merge(hours, how="cross")
    frame = frame.merge(
        sales,
        on=["date", "bakery_id", "product_id", "hour"],
        how="left",
    ).merge(bakery_hour, on=["date", "bakery_id", "hour"], how="left")
    frame["sold"] = frame["sold"].fillna(0.0)
    frame["bakery_hour_sales"] = frame["bakery_hour_sales"].fillna(0.0)
    frame["dow"] = frame["date"].dt.dayofweek
    frame["is_production_observed"] = frame["balance_is_consistent"]
    frame["is_stockout_day"] = frame["is_inventory_stockout"]
    return frame


def build_daily_diagnostics(
    frame: pd.DataFrame,
    good: pd.DataFrame,
    share: pd.DataFrame,
) -> pd.DataFrame:
    work = frame.copy()
    work["positive_hour"] = work["hour"].where(work["sold"] > 0)
    daily = work.groupby(
        ["date", "bakery_id", "product_id", "product_name", "dow"],
        as_index=False,
    ).agg(
        sold=("sold", "sum"),
        closing_stock=("closing_stock", "first"),
        is_stockout=("is_inventory_stockout", "first"),
        balance_consistent=("balance_is_consistent", "first"),
        last_sale_hour=("positive_hour", "max"),
    )
    last = daily[["date", "bakery_id", "product_id", "last_sale_hour"]]
    after = work.merge(last, on=["date", "bakery_id", "product_id"], how="left")
    after["bakery_sales_after_last"] = np.where(
        after["hour"] > after["last_sale_hour"], after["bakery_hour_sales"], 0.0
    )
    after = after.groupby(["date", "bakery_id", "product_id"], as_index=False)[
        "bakery_sales_after_last"
    ].sum()
    daily = daily.merge(after, on=["date", "bakery_id", "product_id"])

    benchmark = (
        daily[daily["balance_consistent"] & ~daily["is_stockout"]]
        .groupby(["bakery_id", "product_id", "dow"], as_index=False)
        .agg(
            non_stockout_days=("date", "nunique"),
            non_stockout_median_sold=("sold", "median"),
            non_stockout_median_last_hour=("last_sale_hour", "median"),
        )
    )
    daily = daily.merge(benchmark, on=["bakery_id", "product_id", "dow"], how="left")
    daily["last_hour_gap"] = (
        daily["non_stockout_median_last_hour"] - daily["last_sale_hour"]
    )
    daily["sold_vs_non_stockout"] = daily["sold"] / daily[
        "non_stockout_median_sold"
    ].replace(0.0, np.nan)
    for label, reconstructed in [("good_day", good), ("bakery_share", share)]:
        added = reconstructed.groupby(
            ["date", "bakery_id", "product_id"], as_index=False
        ).agg(
            **{
                f"{label}_imputed": ("imputed_demand", "sum"),
                f"{label}_hours": ("is_censored_hour", "sum"),
            }
        )
        daily = daily.merge(added, on=["date", "bakery_id", "product_id"])
    daily["strong_temporal_signal"] = (
        daily["is_stockout"]
        & (daily["last_hour_gap"] >= 2)
        & (daily["bakery_sales_after_last"] >= 50)
    )
    return daily


def main() -> None:
    balance = pd.read_csv(
        ROOT / "reports/stockout_inventory_balance_10/inventory_balance.csv"
    )
    sales, bakery_hour = load_hourly_sales(
        ROOT / "data/raw/sales_stg_2025_2026.csv"
    )
    frame = build_hourly_frame(balance, sales, bakery_hour)
    reference_train = frame[frame["balance_is_consistent"]].copy()
    good_reference = build_uncensored_hour_reference(reference_train)
    share_reference = build_bakery_share_reference(reference_train)
    good = reconstruct_stockout_demand(frame, good_reference)
    share = reconstruct_stockout_demand_from_bakery_share(frame, share_reference)
    daily = build_daily_diagnostics(frame, good, share)

    output_dir = ROOT / "reports/inventory_stockout_hourly_10"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "hourly_frame.csv", index=False)
    daily.to_csv(output_dir / "daily_cases.csv", index=False)
    stockouts = daily[daily["is_stockout"]].copy()
    stockouts.sort_values("bakery_share_imputed", ascending=False).head(200).to_csv(
        output_dir / "top_stockout_cases.csv", index=False
    )
    strong = stockouts[stockouts["strong_temporal_signal"]].copy()
    strong.to_csv(output_dir / "strong_temporal_stockouts.csv", index=False)
    summary = {
        "hourly_rows": int(len(frame)),
        "stockout_days": int(len(stockouts)),
        "stockouts_with_benchmark": int(
            stockouts["non_stockout_days"].fillna(0).gt(0).sum()
        ),
        "stockouts_ending_2h_early": int((stockouts["last_hour_gap"] >= 2).sum()),
        "stockouts_with_50_sales_after": int(
            (stockouts["bakery_sales_after_last"] >= 50).sum()
        ),
        "strong_temporal_stockouts": int(len(strong)),
        "median_last_hour_gap": float(stockouts["last_hour_gap"].median()),
        "median_sales_after_last": float(stockouts["bakery_sales_after_last"].median()),
        "good_day_imputed_hours": int(good["is_censored_hour"].sum()),
        "good_day_imputed_units": float(good["imputed_demand"].sum()),
        "bakery_share_imputed_hours": int(share["is_censored_hour"].sum()),
        "bakery_share_imputed_units": float(share["imputed_demand"].sum()),
        "strong_bakery_share_imputed_units": float(
            strong["bakery_share_imputed"].sum()
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

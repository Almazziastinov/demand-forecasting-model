"""Build temporal stockout labels and demand reconstruction for pilot data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments_v2.stockout_demand_preprocessing import (  # noqa: E402
    build_bakery_share_reference,
    reconstruct_stockout_demand_from_bakery_share,
)


DEFAULT_INPUT_DIR = ROOT / "reports" / "pilot_mart_zero_stockout_balance"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "pilot_mart_zero_demand_reconstruction"


def load_hourly_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    for column in [
        "balance_is_consistent",
        "hourly_daily_sales_agree",
        "is_inventory_stockout",
        "is_reliable_inventory_stockout",
    ]:
        frame[column] = frame[column].fillna(False).astype(bool)
    frame["dow"] = frame["date"].dt.dayofweek
    frame["is_production_observed"] = frame["balance_is_consistent"]
    frame["is_stockout_day"] = frame["is_reliable_inventory_stockout"]
    frame["daily_sold"] = pd.to_numeric(frame["qty_sold"], errors="coerce").fillna(0.0)
    return frame


def build_daily_signals(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["positive_hour"] = work["hour"].where(work["sold"] > 0)
    daily = work.groupby(
        ["date", "bakery_id", "bakery_name", "product_id", "product_name", "category_name", "dow"],
        as_index=False,
    ).agg(
        hourly_sold=("sold", "sum"),
        daily_sold=("daily_sold", "first"),
        qty_produced=("qty_produced", "first"),
        stock_balance=("stock_balance", "first"),
        is_inventory_stockout=("is_inventory_stockout", "first"),
        is_reliable_inventory_stockout=("is_reliable_inventory_stockout", "first"),
        balance_consistent=("balance_is_consistent", "first"),
        hourly_daily_sales_agree=("hourly_daily_sales_agree", "first"),
        last_sale_hour=("positive_hour", "max"),
    )

    normal = daily[
        daily["balance_consistent"]
        & daily["hourly_daily_sales_agree"]
        & ~daily["is_inventory_stockout"]
    ]
    benchmark = (
        normal.groupby(["bakery_id", "product_id", "dow"], as_index=False)
        .agg(
            normal_days=("date", "nunique"),
            normal_daily_sold=("hourly_sold", "median"),
            normal_last_hour=("last_sale_hour", "median"),
        )
    )
    daily = daily.merge(benchmark, on=["bakery_id", "product_id", "dow"], how="left")

    last = daily[["date", "bakery_id", "product_id", "last_sale_hour"]]
    after = work.merge(last, on=["date", "bakery_id", "product_id"], how="left")
    after["bakery_sales_after_last"] = np.where(
        after["hour"] > after["last_sale_hour"],
        after["bakery_hour_sales"],
        0.0,
    )
    after = after.groupby(["date", "bakery_id", "product_id"], as_index=False)[
        "bakery_sales_after_last"
    ].sum()
    daily = daily.merge(after, on=["date", "bakery_id", "product_id"], how="left")
    daily["last_hour_gap"] = daily["normal_last_hour"] - daily["last_sale_hour"]
    daily["is_strong_temporal_stockout"] = (
        daily["is_reliable_inventory_stockout"]
        & daily["normal_days"].fillna(0).gt(0)
        & daily["last_hour_gap"].ge(2)
        & daily["bakery_sales_after_last"].ge(50)
    )
    return daily


def apply_reconstruction_policy(
    frame: pd.DataFrame,
    reconstructed: pd.DataFrame,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    strong = daily[daily["is_strong_temporal_stockout"]][
        ["date", "bakery_id", "product_id", "normal_daily_sold"]
    ].copy()
    work = reconstructed.merge(
        strong.assign(_strong=True),
        on=["date", "bakery_id", "product_id"],
        how="left",
    )
    work["_strong"] = work["_strong"].eq(True)
    work["raw_imputed_demand"] = work["imputed_demand"]
    work["imputed_demand"] = np.where(work["_strong"], work["imputed_demand"], 0.0)

    day_added = work.groupby(["date", "bakery_id", "product_id"])["imputed_demand"].transform("sum")
    low_volume = work["normal_daily_sold"] <= 10
    cap = np.maximum(4.0, 0.5 * work["normal_daily_sold"])
    scale = np.where(low_volume & (day_added > cap), cap / day_added, 1.0)
    work["policy_scale"] = np.nan_to_num(scale, nan=1.0, posinf=1.0)
    work["imputed_demand"] *= work["policy_scale"]
    work["sold_observed"] = work["sold"]
    work["sold_demand"] = work["sold_observed"] + work["imputed_demand"]
    work["is_policy_adjusted"] = work["imputed_demand"] > 0
    return work


def summarize(daily: pd.DataFrame, adjusted: pd.DataFrame) -> dict[str, Any]:
    stockouts = daily[daily["is_inventory_stockout"]]
    reliable = daily[daily["is_reliable_inventory_stockout"]]
    strong = daily[daily["is_strong_temporal_stockout"]]
    adjusted_hours = adjusted[adjusted["is_policy_adjusted"]]
    by_bakery = (
        daily.groupby(["bakery_id", "bakery_name"], as_index=False)
        .agg(
            rows=("date", "size"),
            products=("product_id", "nunique"),
            inventory_stockouts=("is_inventory_stockout", "sum"),
            reliable_inventory_stockouts=("is_reliable_inventory_stockout", "sum"),
            strong_temporal_stockouts=("is_strong_temporal_stockout", "sum"),
        )
        .merge(
            adjusted.groupby("bakery_id", as_index=False).agg(
                adjusted_hours=("is_policy_adjusted", "sum"),
                imputed_units=("imputed_demand", "sum"),
            ),
            on="bakery_id",
            how="left",
        )
        .sort_values("bakery_id")
    )
    return {
        "daily_rows": int(len(daily)),
        "date_min": str(daily["date"].min().date()) if len(daily) else None,
        "date_max": str(daily["date"].max().date()) if len(daily) else None,
        "inventory_stockouts": int(len(stockouts)),
        "reliable_inventory_stockouts": int(len(reliable)),
        "strong_temporal_stockouts": int(len(strong)),
        "strong_temporal_stockout_share_of_reliable": float(len(strong) / len(reliable)) if len(reliable) else 0.0,
        "adjusted_hours": int(adjusted["is_policy_adjusted"].sum()),
        "imputed_units": float(adjusted["imputed_demand"].sum()),
        "raw_imputed_units_before_policy": float(adjusted["raw_imputed_demand"].sum()),
        "adjusted_bakeries": sorted(int(value) for value in adjusted_hours["bakery_id"].dropna().unique()),
        "adjusted_products": int(adjusted_hours["product_id"].nunique()),
        "by_bakery": by_bakery.to_dict(orient="records"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pilot demand reconstruction from mart_zero stockout labels")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_hourly_frame(input_dir / "hourly_frame.csv")
    daily = build_daily_signals(frame)
    reference_train = frame[
        frame["balance_is_consistent"]
        & frame["hourly_daily_sales_agree"]
        & ~frame["is_inventory_stockout"]
    ].copy()
    reference = build_bakery_share_reference(reference_train)
    reconstructed = reconstruct_stockout_demand_from_bakery_share(frame, reference)
    adjusted = apply_reconstruction_policy(frame, reconstructed, daily)

    daily.to_csv(output_dir / "daily_stockout_signals.csv", index=False, encoding="utf-8-sig")
    daily[daily["is_strong_temporal_stockout"]].to_csv(
        output_dir / "strong_temporal_stockouts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    adjusted.to_csv(
        output_dir / "hourly_reconstructed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    adjusted[adjusted["is_policy_adjusted"]].to_csv(
        output_dir / "adjusted_hours.csv",
        index=False,
        encoding="utf-8-sig",
    )
    adjusted[adjusted["is_policy_adjusted"]].groupby(
        ["bakery_id", "product_id"],
        as_index=False,
    ).agg(
        product_name=("product_name", "first"),
        adjusted_hours=("is_policy_adjusted", "sum"),
        imputed_units=("imputed_demand", "sum"),
        observed_units=("sold_observed", "sum"),
    ).sort_values("imputed_units", ascending=False).head(200).to_csv(
        output_dir / "top_adjusted_bakery_products.csv",
        index=False,
        encoding="utf-8-sig",
    )
    adjusted[adjusted["is_policy_adjusted"]].groupby(
        ["hour"],
        as_index=False,
    ).agg(
        adjusted_hours=("is_policy_adjusted", "sum"),
        imputed_units=("imputed_demand", "sum"),
    ).sort_values("hour").to_csv(
        output_dir / "adjusted_by_hour.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = summarize(daily, adjusted)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

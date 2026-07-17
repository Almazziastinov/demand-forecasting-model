"""Build baseline and inventory-aware demand profiles on the March pilot."""

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
    reconstruct_stockout_demand_from_bakery_share,
)

TRAIN_END = pd.Timestamp("2026-03-21")
HOLDOUT_START = pd.Timestamp("2026-03-22")
PROFILE_KEYS = ["bakery_id", "product_id", "dow", "hour"]


def load_frame() -> pd.DataFrame:
    frame = pd.read_csv(ROOT / "reports/inventory_stockout_hourly_10/hourly_frame.csv")
    frame["date"] = pd.to_datetime(frame["date"])
    for column in [
        "balance_is_consistent",
        "is_inventory_stockout",
        "is_production_observed",
        "is_stockout_day",
    ]:
        frame[column] = frame[column].astype(bool)
    return frame


def daily_quality_and_signal(train: pd.DataFrame) -> pd.DataFrame:
    work = train.copy()
    work["positive_hour"] = work["hour"].where(work["sold"] > 0)
    daily = work.groupby(
        ["date", "bakery_id", "product_id", "dow"], as_index=False
    ).agg(
        hourly_sold=("sold", "sum"),
        balance_sold=("daily_sold", "first"),
        is_stockout=("is_inventory_stockout", "first"),
        balance_consistent=("balance_is_consistent", "first"),
        last_sale_hour=("positive_hour", "max"),
    )
    daily["sales_agree"] = (daily["hourly_sold"] - daily["balance_sold"]).abs() <= 1
    benchmark = (
        daily[
            daily["balance_consistent"] & ~daily["is_stockout"] & daily["sales_agree"]
        ]
        .groupby(["bakery_id", "product_id", "dow"], as_index=False)
        .agg(
            normal_days=("date", "nunique"),
            normal_daily_sold=("hourly_sold", "median"),
            normal_last_hour=("last_sale_hour", "median"),
        )
    )
    daily = daily.merge(benchmark, on=["bakery_id", "product_id", "dow"], how="left")
    last = daily[["date", "bakery_id", "product_id", "last_sale_hour"]]
    after = work.merge(last, on=["date", "bakery_id", "product_id"], how="left")
    after["sales_after_last"] = np.where(
        after["hour"] > after["last_sale_hour"], after["bakery_hour_sales"], 0.0
    )
    after = after.groupby(["date", "bakery_id", "product_id"], as_index=False)[
        "sales_after_last"
    ].sum()
    daily = daily.merge(after, on=["date", "bakery_id", "product_id"])
    daily["last_hour_gap"] = daily["normal_last_hour"] - daily["last_sale_hour"]
    daily["is_strong_stockout"] = (
        daily["balance_consistent"]
        & daily["sales_agree"]
        & daily["is_stockout"]
        & (daily["normal_days"] > 0)
        & (daily["last_hour_gap"] >= 2)
        & (daily["sales_after_last"] >= 50)
    )
    return daily


def apply_segmented_policy(
    train: pd.DataFrame,
    reconstructed: pd.DataFrame,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    strong = daily[daily["is_strong_stockout"]][
        ["date", "bakery_id", "product_id", "normal_daily_sold"]
    ].copy()
    work = reconstructed.merge(
        strong.assign(_strong=True),
        on=["date", "bakery_id", "product_id"],
        how="left",
    )
    work["_strong"] = work["_strong"].fillna(False)
    work["raw_imputed_demand"] = work["imputed_demand"]
    work["imputed_demand"] = np.where(work["_strong"], work["imputed_demand"], 0.0)

    day_added = work.groupby(["date", "bakery_id", "product_id"])[
        "imputed_demand"
    ].transform("sum")
    low_volume = work["normal_daily_sold"] <= 10
    cap = np.maximum(4.0, 0.5 * work["normal_daily_sold"])
    scale = np.where(low_volume & (day_added > cap), cap / day_added, 1.0)
    work["policy_scale"] = np.nan_to_num(scale, nan=1.0, posinf=1.0)
    work["imputed_demand"] *= work["policy_scale"]
    work["sold_observed"] = work["sold"]
    work["sold_demand"] = work["sold_observed"] + work["imputed_demand"]
    work["is_policy_adjusted"] = work["imputed_demand"] > 0
    return work


def build_profile(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    work = frame.copy()
    total = work.groupby(["date", "bakery_id", "hour"])[value_col].transform("sum")
    work["share"] = work[value_col] / total.replace(0.0, np.nan)
    profile = work.groupby(PROFILE_KEYS, as_index=False).agg(
        profile_share=("share", "mean"), profile_days=("date", "nunique")
    )
    norm = profile.groupby(["bakery_id", "dow", "hour"])["profile_share"].transform(
        "sum"
    )
    profile["profile_share"] /= norm.replace(0.0, np.nan)
    return profile


def evaluate(profile: pd.DataFrame, holdout: pd.DataFrame) -> dict[str, float | int]:
    work = holdout.copy()
    work["actual_total"] = work.groupby(["date", "bakery_id", "hour"])[
        "sold"
    ].transform("sum")
    work["actual_share"] = work["sold"] / work["actual_total"].replace(0.0, np.nan)
    work = work.merge(profile, on=PROFILE_KEYS, how="left")
    valid = work[work["actual_share"].notna() & work["profile_share"].notna()]
    error = (valid["profile_share"] - valid["actual_share"]).abs()
    return {
        "rows": int(len(valid)),
        "coverage": float(len(valid) / len(work)),
        "share_mae": float(error.mean()),
        "weighted_share_mae": float(np.average(error, weights=valid["actual_total"])),
    }


def main() -> None:
    frame = load_frame()
    train = frame[frame["date"] <= TRAIN_END].copy()
    holdout = frame[frame["date"] >= HOLDOUT_START].copy()
    daily = daily_quality_and_signal(train)
    reference = build_bakery_share_reference(train[train["balance_is_consistent"]])
    reconstructed = reconstruct_stockout_demand_from_bakery_share(train, reference)
    adjusted = apply_segmented_policy(train, reconstructed, daily)

    baseline_profile = build_profile(train, "sold")
    demand_profile = build_profile(adjusted, "sold_demand")
    holdout_daily = holdout.groupby(["date", "bakery_id", "product_id"])[
        "sold"
    ].transform("sum")
    reliable_holdout = holdout[
        holdout["balance_is_consistent"]
        & ~holdout["is_inventory_stockout"]
        & ((holdout_daily - holdout["daily_sold"]).abs() <= 1)
    ].copy()

    output_dir = ROOT / "reports/demand_adjusted_profile_10"
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_profile.to_csv(output_dir / "baseline_profile.csv", index=False)
    demand_profile.to_csv(output_dir / "demand_profile.csv", index=False)
    adjusted[adjusted["is_policy_adjusted"]].to_csv(
        output_dir / "adjusted_hours_audit.csv", index=False
    )
    daily.to_csv(output_dir / "train_daily_signals.csv", index=False)

    comparison = baseline_profile.merge(
        demand_profile,
        on=PROFILE_KEYS,
        suffixes=("_baseline", "_demand"),
    )
    comparison["share_delta"] = (
        comparison["profile_share_demand"] - comparison["profile_share_baseline"]
    )
    comparison.to_csv(output_dir / "profile_comparison.csv", index=False)
    summary = {
        "train_end": str(TRAIN_END.date()),
        "holdout_start": str(HOLDOUT_START.date()),
        "strong_stockout_days": int(daily["is_strong_stockout"].sum()),
        "adjusted_hours": int(adjusted["is_policy_adjusted"].sum()),
        "imputed_units": float(adjusted["imputed_demand"].sum()),
        "profile_rows_changed": int((comparison["share_delta"].abs() > 1e-12).sum()),
        "mean_abs_profile_delta": float(comparison["share_delta"].abs().mean()),
        "max_abs_profile_delta": float(comparison["share_delta"].abs().max()),
        "baseline_holdout": evaluate(baseline_profile, reliable_holdout),
        "demand_holdout": evaluate(demand_profile, reliable_holdout),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

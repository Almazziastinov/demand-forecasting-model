"""Pseudo-stockout backtest for the pilot mart_zero demand reconstruction.

The test hides the final N hours of known non-stockout SKU-days, runs the same
bakery-share reconstruction logic, and measures how much hidden demand is
recovered. It is intentionally local-only and reads the prepared research CSVs.
"""

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
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "pilot_mart_zero_pseudo_stockout_backtest"


def load_frame(path: Path) -> pd.DataFrame:
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


def build_synthetic_cases(
    frame: pd.DataFrame,
    *,
    holdout_start: pd.Timestamp,
    gap_hours: int,
    min_daily_sold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdout = frame[frame["date"] >= holdout_start].copy()
    holdout["positive_hour"] = holdout["hour"].where(holdout["sold"] > 0)
    daily = holdout.groupby(
        ["date", "bakery_id", "bakery_name", "product_id", "product_name", "category_name"],
        as_index=False,
    ).agg(
        daily_sold_hourly=("sold", "sum"),
        daily_sold_balance=("daily_sold", "first"),
        balance_consistent=("balance_is_consistent", "first"),
        hourly_daily_sales_agree=("hourly_daily_sales_agree", "first"),
        inventory_stockout=("is_inventory_stockout", "first"),
        last_sale_hour=("positive_hour", "max"),
    )
    daily["cutoff_hour"] = daily["last_sale_hour"] - gap_hours
    candidates = daily[
        daily["balance_consistent"]
        & daily["hourly_daily_sales_agree"]
        & ~daily["inventory_stockout"]
        & daily["last_sale_hour"].notna()
        & daily["daily_sold_hourly"].ge(min_daily_sold)
        & daily["cutoff_hour"].ge(6)
    ].copy()
    synthetic = holdout.merge(
        candidates[
            [
                "date",
                "bakery_id",
                "product_id",
                "cutoff_hour",
                "daily_sold_hourly",
                "last_sale_hour",
            ]
        ],
        on=["date", "bakery_id", "product_id"],
        how="inner",
    )
    synthetic["true_sold"] = synthetic["sold"]
    synthetic["is_hidden_hour"] = synthetic["hour"] > synthetic["cutoff_hour"]
    synthetic["sold"] = np.where(synthetic["is_hidden_hour"], 0.0, synthetic["sold"])
    synthetic["is_stockout_day"] = True
    synthetic["is_production_observed"] = True
    return synthetic, candidates


def apply_policy(
    reconstructed: pd.DataFrame,
    *,
    low_volume_threshold: float = 10.0,
) -> pd.DataFrame:
    work = reconstructed.copy()
    hidden = work[work["is_hidden_hour"]]
    daily_normal = hidden.groupby(["date", "bakery_id", "product_id"])["daily_sold_hourly"].transform("first")
    day_added = hidden.groupby(["date", "bakery_id", "product_id"])["imputed_demand"].transform("sum")
    low_volume = daily_normal <= low_volume_threshold
    cap = np.maximum(4.0, 0.5 * daily_normal)
    scale = np.where(low_volume & (day_added > cap), cap / day_added, 1.0)
    scaled = hidden["imputed_demand"] * np.nan_to_num(scale, nan=1.0, posinf=1.0)
    work.loc[hidden.index, "policy_imputed_demand"] = scaled
    work["policy_imputed_demand"] = work["policy_imputed_demand"].fillna(0.0)
    work["sold_demand_policy"] = work["sold_observed"] + work["policy_imputed_demand"]
    return work


def evaluate_cases(
    reconstructed: pd.DataFrame,
    *,
    gap_hours: int,
    history_days: int,
) -> pd.DataFrame:
    hidden = reconstructed[reconstructed["is_hidden_hour"]].copy()
    cases = hidden.groupby(
        ["date", "bakery_id", "bakery_name", "product_id", "product_name", "category_name"],
        as_index=False,
    ).agg(
        true_hidden=("true_sold", "sum"),
        predicted_hidden_raw=("sold_demand", "sum"),
        predicted_hidden_policy=("sold_demand_policy", "sum"),
        daily_sold=("daily_sold_hourly", "first"),
        hidden_hours=("is_hidden_hour", "sum"),
        predicted_hours=("is_censored_hour", "sum"),
    )
    cases["error_raw"] = cases["predicted_hidden_raw"] - cases["true_hidden"]
    cases["error_policy"] = cases["predicted_hidden_policy"] - cases["true_hidden"]
    cases["abs_error_raw"] = cases["error_raw"].abs()
    cases["abs_error_policy"] = cases["error_policy"].abs()
    cases["gap_hours"] = gap_hours
    cases["history_days"] = history_days
    cases["volume_band"] = np.where(cases["daily_sold"] <= 10, "<=10", ">10")
    return cases


def summarize_cases(cases: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_cols = ["history_days", "gap_hours", "volume_band"]
    for keys, group in cases.groupby(group_cols):
        history_days, gap_hours, volume_band = keys
        true_total = group["true_hidden"].sum()
        for variant, pred_col, abs_col in [
            ("raw", "predicted_hidden_raw", "abs_error_raw"),
            ("policy", "predicted_hidden_policy", "abs_error_policy"),
        ]:
            predicted = group[pred_col].sum()
            rows.append(
                {
                    "history_days": int(history_days),
                    "gap_hours": int(gap_hours),
                    "volume_band": volume_band,
                    "variant": variant,
                    "cases": int(len(group)),
                    "true_hidden_units": float(true_total),
                    "predicted_units": float(predicted),
                    "recovery_ratio": float(predicted / true_total) if true_total else 0.0,
                    "bias_pct": float(100 * (predicted - true_total) / true_total) if true_total else 0.0,
                    "wape_pct": float(100 * group[abs_col].sum() / true_total) if true_total else 0.0,
                    "mae": float(group[abs_col].mean()) if len(group) else 0.0,
                    "underpredict_share": float((group[pred_col] < group["true_hidden"]).mean()) if len(group) else 0.0,
                }
            )
    return sorted(rows, key=lambda row: (row["history_days"], row["gap_hours"], row["volume_band"], row["variant"]))


def run_backtest(
    frame: pd.DataFrame,
    *,
    history_days: int,
    gap_hours: int,
    holdout_days: int,
    min_daily_sold: float,
) -> pd.DataFrame:
    max_date = frame["date"].max()
    holdout_start = max_date - pd.Timedelta(days=holdout_days - 1)
    train_end = holdout_start - pd.Timedelta(days=1)
    train_start = train_end - pd.Timedelta(days=history_days - 1)
    train = frame[
        (frame["date"] >= train_start)
        & (frame["date"] <= train_end)
        & frame["balance_is_consistent"]
        & frame["hourly_daily_sales_agree"]
        & ~frame["is_inventory_stockout"]
    ].copy()
    reference = build_bakery_share_reference(train)
    synthetic, _ = build_synthetic_cases(
        frame,
        holdout_start=holdout_start,
        gap_hours=gap_hours,
        min_daily_sold=min_daily_sold,
    )
    reconstructed = reconstruct_stockout_demand_from_bakery_share(synthetic, reference)
    reconstructed = apply_policy(reconstructed)
    cases = evaluate_cases(reconstructed, gap_hours=gap_hours, history_days=history_days)
    cases["train_start"] = train_start.date().isoformat()
    cases["train_end"] = train_end.date().isoformat()
    cases["holdout_start"] = holdout_start.date().isoformat()
    cases["holdout_end"] = max_date.date().isoformat()
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest pilot mart_zero pseudo-stockout reconstruction")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--history-days", nargs="+", type=int, default=[28, 42, 56])
    parser.add_argument("--gap-hours", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--holdout-days", type=int, default=14)
    parser.add_argument("--min-daily-sold", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_frame(Path(args.input_dir) / "hourly_frame.csv")
    all_cases = []
    for history_days in args.history_days:
        for gap_hours in args.gap_hours:
            all_cases.append(
                run_backtest(
                    frame,
                    history_days=history_days,
                    gap_hours=gap_hours,
                    holdout_days=args.holdout_days,
                    min_daily_sold=args.min_daily_sold,
                )
            )
    cases = pd.concat(all_cases, ignore_index=True)
    summary_rows = summarize_cases(cases)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases.to_csv(output_dir / "cases.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    payload = {
        "holdout_days": args.holdout_days,
        "min_daily_sold": args.min_daily_sold,
        "rows": summary_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()

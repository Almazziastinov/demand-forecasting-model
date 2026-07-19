"""Compare baseline vs demand-adjusted hourly SKU profiles for pilot data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "reports" / "pilot_mart_zero_demand_reconstruction"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "pilot_mart_zero_profile_comparison"
PROFILE_KEYS = ["bakery_id", "product_id", "dow", "hour"]


def load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    for column in [
        "balance_is_consistent",
        "hourly_daily_sales_agree",
        "is_inventory_stockout",
        "is_policy_adjusted",
    ]:
        frame[column] = frame[column].fillna(False).astype(bool)
    return frame


def build_profile(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    work = frame.copy()
    total = work.groupby(["date", "bakery_id", "hour"])[value_col].transform("sum")
    work["share"] = work[value_col] / total.replace(0.0, np.nan)
    profile = work.groupby(PROFILE_KEYS, as_index=False).agg(
        profile_share=("share", "mean"),
        profile_days=("date", "nunique"),
    )
    norm = profile.groupby(["bakery_id", "dow", "hour"])["profile_share"].transform("sum")
    profile["profile_share"] = profile["profile_share"] / norm.replace(0.0, np.nan)
    return profile


def evaluate(profile: pd.DataFrame, holdout: pd.DataFrame) -> dict[str, float | int]:
    work = holdout.copy()
    total = work.groupby(["date", "bakery_id", "hour"])["sold"].transform("sum")
    work["actual_share"] = work["sold"] / total.replace(0.0, np.nan)
    work["actual_total"] = total
    work = work.merge(profile, on=PROFILE_KEYS, how="left")
    valid = work[work["actual_share"].notna() & work["profile_share"].notna()].copy()
    error = (valid["profile_share"] - valid["actual_share"]).abs()
    return {
        "rows": int(len(valid)),
        "coverage": float(len(valid) / len(work)) if len(work) else 0.0,
        "share_mae": float(error.mean()) if len(valid) else 0.0,
        "weighted_share_mae": float(np.average(error, weights=valid["actual_total"])) if len(valid) else 0.0,
    }


def compare_window(frame: pd.DataFrame, *, history_days: int, holdout_days: int) -> dict[str, Any]:
    max_date = frame["date"].max()
    holdout_start = max_date - pd.Timedelta(days=holdout_days - 1)
    train_end = holdout_start - pd.Timedelta(days=1)
    train_start = train_end - pd.Timedelta(days=history_days - 1)

    train = frame[(frame["date"] >= train_start) & (frame["date"] <= train_end)].copy()
    holdout = frame[
        (frame["date"] >= holdout_start)
        & frame["balance_is_consistent"]
        & frame["hourly_daily_sales_agree"]
        & ~frame["is_inventory_stockout"]
    ].copy()
    baseline = build_profile(train, "sold")
    demand = build_profile(train, "sold_demand")
    comparison = baseline.merge(demand, on=PROFILE_KEYS, suffixes=("_baseline", "_demand"))
    comparison["share_delta"] = comparison["profile_share_demand"] - comparison["profile_share_baseline"]

    return {
        "history_days": history_days,
        "train_start": str(train_start.date()),
        "train_end": str(train_end.date()),
        "holdout_start": str(holdout_start.date()),
        "holdout_end": str(max_date.date()),
        "train_rows": int(len(train)),
        "holdout_rows": int(len(holdout)),
        "adjusted_train_hours": int(train["is_policy_adjusted"].sum()),
        "train_imputed_units": float(train["imputed_demand"].sum()),
        "profile_rows": int(len(comparison)),
        "profile_rows_changed": int((comparison["share_delta"].abs() > 1e-12).sum()),
        "mean_abs_profile_delta": float(comparison["share_delta"].abs().mean()) if len(comparison) else 0.0,
        "p99_abs_profile_delta": float(comparison["share_delta"].abs().quantile(0.99)) if len(comparison) else 0.0,
        "max_abs_profile_delta": float(comparison["share_delta"].abs().max()) if len(comparison) else 0.0,
        "baseline_holdout": evaluate(baseline, holdout),
        "demand_holdout": evaluate(demand, holdout),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and demand-adjusted pilot profiles")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--history-days", nargs="+", type=int, default=[28, 42, 56])
    parser.add_argument("--holdout-days", type=int, default=14)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_frame(Path(args.input_dir) / "hourly_reconstructed.csv")
    results = [
        compare_window(frame, history_days=history_days, holdout_days=args.holdout_days)
        for history_days in args.history_days
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "holdout_days": args.holdout_days,
        "results": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "history_days": result["history_days"],
                "train_start": result["train_start"],
                "train_end": result["train_end"],
                "holdout_start": result["holdout_start"],
                "holdout_end": result["holdout_end"],
                "adjusted_train_hours": result["adjusted_train_hours"],
                "train_imputed_units": result["train_imputed_units"],
                "profile_rows_changed": result["profile_rows_changed"],
                "mean_abs_profile_delta": result["mean_abs_profile_delta"],
                "baseline_weighted_share_mae": result["baseline_holdout"]["weighted_share_mae"],
                "demand_weighted_share_mae": result["demand_holdout"]["weighted_share_mae"],
            }
            for result in results
        ]
    ).to_csv(output_dir / "window_summary.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

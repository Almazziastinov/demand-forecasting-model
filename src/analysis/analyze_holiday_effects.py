from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.experiments_v2.bakery_day_forecast import DATE_COL
from src.experiments_v2.bakery_day_forecast import TARGET_COL
from src.experiments_v2.bakery_day_forecast import build_model_frame
from src.experiments_v2.bakery_day_forecast import load_dataset


DEFAULT_DATASET_PATH = ROOT / "data" / "processed" / "bakery_daily_sales.csv"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "holiday_effects"


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def build_network_daily(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby(DATE_COL, as_index=False)
        .agg(
            actual_sales=(TARGET_COL, "sum"),
            n_bakeries=("bakery_id", "nunique"),
            is_holiday=("is_holiday", "max"),
            is_pre_holiday=("is_pre_holiday", "max"),
            is_post_holiday=("is_post_holiday", "max"),
            holiday_name=("holiday_name", lambda s: next((x for x in s if x), "")),
        )
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )

    daily["prev_day_sales"] = daily["actual_sales"].shift(1)
    daily["next_day_sales"] = daily["actual_sales"].shift(-1)
    daily["roll7_before"] = daily["actual_sales"].shift(1).rolling(7, min_periods=3).mean()
    daily["roll7_after"] = daily["actual_sales"].shift(-1)[::-1].rolling(7, min_periods=3).mean()[::-1]
    daily["pct_vs_prev_day"] = (daily["actual_sales"] / (daily["prev_day_sales"] + 1e-8) - 1.0) * 100.0
    daily["pct_vs_next_day"] = (daily["actual_sales"] / (daily["next_day_sales"] + 1e-8) - 1.0) * 100.0
    daily["pct_vs_roll7_before"] = (daily["actual_sales"] / (daily["roll7_before"] + 1e-8) - 1.0) * 100.0
    daily["pct_vs_roll7_after"] = (daily["actual_sales"] / (daily["roll7_after"] + 1e-8) - 1.0) * 100.0
    return daily


def build_holiday_summary(network_daily: pd.DataFrame) -> pd.DataFrame:
    holiday_days = network_daily[
        (network_daily["is_holiday"] == 1)
        | (network_daily["is_pre_holiday"] == 1)
        | (network_daily["is_post_holiday"] == 1)
    ].copy()
    holiday_days["holiday_bucket"] = np.select(
        [
            holiday_days["is_holiday"] == 1,
            holiday_days["is_pre_holiday"] == 1,
            holiday_days["is_post_holiday"] == 1,
        ],
        [
            "holiday",
            "pre_holiday",
            "post_holiday",
        ],
        default="other",
    )

    summary = (
        holiday_days.groupby(["holiday_bucket", "holiday_name"], as_index=False)
        .agg(
            n_dates=(DATE_COL, "count"),
            mean_sales=("actual_sales", "mean"),
            median_sales=("actual_sales", "median"),
            mean_pct_vs_prev_day=("pct_vs_prev_day", "mean"),
            mean_pct_vs_next_day=("pct_vs_next_day", "mean"),
            mean_pct_vs_roll7_before=("pct_vs_roll7_before", "mean"),
            mean_pct_vs_roll7_after=("pct_vs_roll7_after", "mean"),
        )
        .sort_values(["holiday_bucket", "holiday_name"])
        .reset_index(drop=True)
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze holiday effects on the full bakery dataset")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    df = build_model_frame(load_dataset(args.dataset_path))
    network_daily = build_network_daily(df)
    holiday_summary = build_holiday_summary(network_daily)

    save_csv(network_daily, output_dir / "network_daily.csv")
    save_csv(
        network_daily[
            (network_daily["is_holiday"] == 1)
            | (network_daily["is_pre_holiday"] == 1)
            | (network_daily["is_post_holiday"] == 1)
        ],
        output_dir / "holiday_daily.csv",
    )
    save_csv(holiday_summary, output_dir / "holiday_summary.csv")

    overview = {
        "dataset_path": str(args.dataset_path),
        "date_min": str(network_daily[DATE_COL].min().date()),
        "date_max": str(network_daily[DATE_COL].max().date()),
        "holiday_dates": int(
            (
                (network_daily["is_holiday"] == 1)
                | (network_daily["is_pre_holiday"] == 1)
                | (network_daily["is_post_holiday"] == 1)
            ).sum()
        ),
        "rows_total": int(len(network_daily)),
    }
    (output_dir / "overview.json").write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {output_dir}")


if __name__ == "__main__":
    main()

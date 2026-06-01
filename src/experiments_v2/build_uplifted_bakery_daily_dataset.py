"""
Build bakery-level daily training data with non-normalized SKU uplift.

The uplift target follows the business logic:
  bakery_hour_sales_uplifted = bakery_hour_sales * sum(sku_share_in_hour_adj)
  bakery_sales_uplifted = sum(bakery_hour_sales_uplifted)

The resulting CSV keeps the original base bakery sales in audit columns, replaces
``bakery_sales`` with the uplifted target, and recomputes lag/rolling features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.experiments_v2.build_bakery_daily_dataset import BAKERY_ID_COL
from src.experiments_v2.build_bakery_daily_dataset import BAKERY_NAME_COL
from src.experiments_v2.build_bakery_daily_dataset import CHUNK_SIZE
from src.experiments_v2.build_bakery_daily_dataset import DATE_COL
from src.experiments_v2.build_bakery_daily_dataset import TARGET_COL
from src.experiments_v2.build_bakery_daily_dataset import add_cleaning_features
from src.experiments_v2.build_bakery_daily_dataset import add_lag_features
from src.experiments_v2.build_bakery_daily_dataset import build_summary


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DAILY_PATH = ROOT / "data" / "processed" / "bakery_daily_sales.csv"
DEFAULT_SMOOTHED_APPLIED_PATH = (
    ROOT / "data" / "processed" / "sku_hour_share_profile_daily_smoothed.csv"
)
DEFAULT_OUTPUT_PATH = ROOT / "data" / "processed" / "bakery_daily_sales_uplifted.csv"
DEFAULT_SUMMARY_PATH = (
    ROOT / "data" / "processed" / "bakery_daily_sales_uplifted_summary.json"
)

HOUR_COL = "hour"
BAKERY_HOUR_SALES_COL = "bakery_hour_sales"
UPLIFT_SHARE_COL = "sku_share_in_hour_adj"
NORM_UPLIFT_SHARE_COL = "sku_share_in_hour_adj_norm"

BASE_TARGET_COL = "bakery_sales_base"
UPLIFTED_TARGET_COL = "bakery_sales_uplifted"
UPLIFT_DELTA_COL = "bakery_sales_uplift_delta"
UPLIFT_RATE_COL = "bakery_sales_uplift_rate"
UPLIFT_MULTIPLIER_COL = "sku_share_uplift_multiplier"


def build_hourly_uplift(
    applied_path: str | Path,
    *,
    chunk_size: int = CHUNK_SIZE,
) -> pd.DataFrame:
    usecols = [
        DATE_COL,
        BAKERY_ID_COL,
        BAKERY_NAME_COL,
        HOUR_COL,
        BAKERY_HOUR_SALES_COL,
        UPLIFT_SHARE_COL,
        NORM_UPLIFT_SHARE_COL,
    ]
    parts: list[pd.DataFrame] = []
    reader = pd.read_csv(
        applied_path,
        encoding="utf-8-sig",
        usecols=usecols,
        chunksize=chunk_size,
    )
    for i, chunk in enumerate(reader, start=1):
        chunk[DATE_COL] = pd.to_datetime(
            chunk[DATE_COL],
            errors="coerce",
        ).dt.normalize()
        chunk[BAKERY_ID_COL] = pd.to_numeric(chunk[BAKERY_ID_COL], errors="coerce")
        chunk[HOUR_COL] = pd.to_numeric(chunk[HOUR_COL], errors="coerce")
        chunk[BAKERY_HOUR_SALES_COL] = pd.to_numeric(
            chunk[BAKERY_HOUR_SALES_COL],
            errors="coerce",
        ).fillna(0.0)
        chunk[UPLIFT_SHARE_COL] = pd.to_numeric(
            chunk[UPLIFT_SHARE_COL],
            errors="coerce",
        ).fillna(0.0)
        chunk[NORM_UPLIFT_SHARE_COL] = pd.to_numeric(
            chunk[NORM_UPLIFT_SHARE_COL],
            errors="coerce",
        ).fillna(0.0)
        chunk = chunk.dropna(subset=[DATE_COL, BAKERY_ID_COL, HOUR_COL])
        if chunk.empty:
            continue

        part = (
            chunk.groupby([DATE_COL, BAKERY_ID_COL, HOUR_COL], as_index=False)
            .agg(
                bakery_hour_sales=(BAKERY_HOUR_SALES_COL, "max"),
                sku_share_uplift_sum=(UPLIFT_SHARE_COL, "sum"),
                sku_share_uplift_norm_sum=(NORM_UPLIFT_SHARE_COL, "sum"),
            )
        )
        parts.append(part)
        if i % 5 == 0:
            print(f"processed uplift chunks: {i}", flush=True)

    if not parts:
        return pd.DataFrame(
            columns=[
                DATE_COL,
                BAKERY_ID_COL,
                HOUR_COL,
                "bakery_hour_sales",
                "sku_share_uplift_sum",
                "sku_share_uplift_norm_sum",
            ]
        )

    hourly = pd.concat(parts, ignore_index=True)
    hourly = (
        hourly.groupby([DATE_COL, BAKERY_ID_COL, HOUR_COL], as_index=False)
        .agg(
            bakery_hour_sales=("bakery_hour_sales", "max"),
            sku_share_uplift_sum=("sku_share_uplift_sum", "sum"),
            sku_share_uplift_norm_sum=("sku_share_uplift_norm_sum", "sum"),
        )
        .reset_index(drop=True)
    )
    hourly[UPLIFT_MULTIPLIER_COL] = (
        hourly["sku_share_uplift_sum"]
        / hourly["sku_share_uplift_norm_sum"].replace(0.0, pd.NA)
    ).fillna(1.0).clip(lower=1.0)
    hourly["bakery_hour_sales_uplifted"] = (
        hourly["bakery_hour_sales"] * hourly[UPLIFT_MULTIPLIER_COL]
    )
    return hourly


def build_daily_uplift(hourly: pd.DataFrame) -> pd.DataFrame:
    if hourly.empty:
        return pd.DataFrame(
            columns=[
                DATE_COL,
                BAKERY_ID_COL,
                BASE_TARGET_COL,
                UPLIFTED_TARGET_COL,
                UPLIFT_DELTA_COL,
                UPLIFT_RATE_COL,
                UPLIFT_MULTIPLIER_COL,
            ]
        )

    daily = (
        hourly.groupby([DATE_COL, BAKERY_ID_COL], as_index=False)
        .agg(
            bakery_sales_from_hours=("bakery_hour_sales", "sum"),
            bakery_sales_uplifted_from_hours=("bakery_hour_sales_uplifted", "sum"),
        )
        .reset_index(drop=True)
    )
    daily[UPLIFT_MULTIPLIER_COL] = (
        daily["bakery_sales_uplifted_from_hours"]
        / daily["bakery_sales_from_hours"].replace(0.0, pd.NA)
    ).fillna(1.0)
    daily[UPLIFT_DELTA_COL] = (
        daily["bakery_sales_uplifted_from_hours"] - daily["bakery_sales_from_hours"]
    )
    daily[UPLIFT_RATE_COL] = daily[UPLIFT_MULTIPLIER_COL] - 1.0
    return daily


def rebuild_target_features(df: pd.DataFrame) -> pd.DataFrame:
    lag_cols = [
        col
        for col in df.columns
        if col.startswith("bakery_sales_lag")
        or col.startswith("bakery_sales_roll_")
        or col
        in {
            "robust_sales_z",
            "sales_to_expected_ratio",
            "sales_high_outlier_flag",
            "sales_low_outlier_flag",
            "contextual_high_outlier_flag",
            "unexplained_high_outlier_flag",
            "base_model_downweight_flag",
            "correction_candidate_flag",
            "base_model_sample_weight",
            "quantile_cap_lower",
            "quantile_cap_upper",
            "quantile_cap_rows",
            "quantile_cap_source",
            "bakery_sales_base_quantile_capped",
            "quantile_base_target_capped_flag",
            "quantile_base_target_cap_delta",
            "rolling_base_median",
            "rolling_base_rows",
            "rolling_cap_source",
            "rolling_cap_lower",
            "rolling_cap_upper",
            "bakery_sales_base_rolling_capped",
            "rolling_base_target_capped_flag",
            "rolling_base_target_cap_delta",
            "rolling_q_lower",
            "rolling_q_upper",
            "rolling_q_rows",
            "rolling_q_cap_source",
            "bakery_sales_base_rolling_quantile_capped",
            "rolling_quantile_base_target_capped_flag",
            "rolling_quantile_base_target_cap_delta",
        }
    ]
    work = df.drop(columns=lag_cols, errors="ignore")
    work = add_cleaning_features(work)
    return add_lag_features(work)


def build_uplifted_daily_dataset(
    daily_path: str | Path,
    applied_path: str | Path,
    *,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[pd.DataFrame, dict]:
    base = pd.read_csv(daily_path, encoding="utf-8-sig")
    base[DATE_COL] = pd.to_datetime(base[DATE_COL], errors="coerce").dt.normalize()
    base[BAKERY_ID_COL] = pd.to_numeric(base[BAKERY_ID_COL], errors="coerce")

    hourly = build_hourly_uplift(applied_path, chunk_size=chunk_size)
    uplift = build_daily_uplift(hourly)

    work = base.merge(
        uplift,
        on=[DATE_COL, BAKERY_ID_COL],
        how="left",
    )
    work[BASE_TARGET_COL] = pd.to_numeric(work[TARGET_COL], errors="coerce").fillna(0.0)
    work["bakery_sales_from_hours"] = pd.to_numeric(
        work["bakery_sales_from_hours"],
        errors="coerce",
    ).fillna(work[BASE_TARGET_COL])
    work[UPLIFT_MULTIPLIER_COL] = pd.to_numeric(
        work[UPLIFT_MULTIPLIER_COL],
        errors="coerce",
    ).fillna(1.0).clip(lower=1.0)
    work["bakery_sales_uplifted_from_hours"] = pd.to_numeric(
        work.get("bakery_sales_uplifted_from_hours"),
        errors="coerce",
    ).fillna(work["bakery_sales_from_hours"] * work[UPLIFT_MULTIPLIER_COL])
    work[UPLIFTED_TARGET_COL] = work[BASE_TARGET_COL] * work[UPLIFT_MULTIPLIER_COL]
    work[UPLIFT_DELTA_COL] = work[UPLIFTED_TARGET_COL] - work[BASE_TARGET_COL]
    work[UPLIFT_RATE_COL] = (
        work[UPLIFT_DELTA_COL] / work[BASE_TARGET_COL].replace(0.0, pd.NA)
    ).fillna(0.0)
    work[UPLIFT_MULTIPLIER_COL] = (
        work[UPLIFTED_TARGET_COL] / work[BASE_TARGET_COL].replace(0.0, pd.NA)
    ).fillna(1.0)
    work[TARGET_COL] = work[UPLIFTED_TARGET_COL].clip(lower=0.0)
    work = rebuild_target_features(work)

    summary = build_summary(work)
    summary.update(
        {
            "target": TARGET_COL,
            "base_target_col": BASE_TARGET_COL,
            "uplifted_target_col": UPLIFTED_TARGET_COL,
            "mean_base_bakery_sales": round(float(work[BASE_TARGET_COL].mean()), 6),
            "mean_uplifted_bakery_sales": round(
                float(work[UPLIFTED_TARGET_COL].mean()),
                6,
            ),
            "mean_uplift_delta": round(float(work[UPLIFT_DELTA_COL].mean()), 6),
            "mean_uplift_rate": round(float(work[UPLIFT_RATE_COL].mean()), 6),
            "p95_uplift_rate": round(float(work[UPLIFT_RATE_COL].quantile(0.95)), 6),
            "max_uplift_rate": round(float(work[UPLIFT_RATE_COL].max()), 6),
            "uplifted_rows": int((work[UPLIFT_DELTA_COL] > 1e-9).sum()),
        }
    )
    return work, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build uplifted bakery daily dataset")
    parser.add_argument("--daily-path", default=str(DEFAULT_DAILY_PATH))
    parser.add_argument("--applied-path", default=str(DEFAULT_SMOOTHED_APPLIED_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    df, summary = build_uplifted_daily_dataset(
        args.daily_path,
        args.applied_path,
        chunk_size=args.chunk_size,
    )
    output_path = Path(args.output_path)
    summary_path = Path(args.summary_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 72)
    print("UPLIFTED BAKERY DAILY DATASET")
    print("=" * 72)
    print(f"dataset: {output_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()

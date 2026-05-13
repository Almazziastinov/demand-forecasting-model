"""
Build an alternative demand target for weak SKU pairs using bakery-share hourly profiles.

This layer is intentionally conservative:
- only applies to weak SKU pairs;
- uses the observed bakery hourly traffic and the SKU's typical share of bakery sales;
- never predicts less than observed sales;
- caps the uplift to avoid explosive corrections on unstable rows.

The goal is to test the heuristic from notebooks/hourly_sales_day.ipynb in a
reproducible form and only on bad / weak SKU pairs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments_v2.common import DEMAND_8M_PATH, SALES_HRS_PATH
from src.experiments_v2.hourly_positive_profile import (
    build_daily_from_applied,
    build_positive_profiles,
    load_sales,
    aggregate_hourly_sales,
    apply_profiles,
)


DATE_COL = "Дата"
BAKERY_COL = "Пекарня"
CATEGORY_COL = "Категория"
PRODUCT_COL = "Номенклатура"
TARGET_COL = "Продано"
DEMAND_COL = "Спрос"

OUTPUT_NAME = "daily_sales_8m_demand_weak_hourly_share.csv"
SUMMARY_OUTPUT_NAME = "daily_sales_8m_demand_weak_hourly_share_summary.json"
WEAK_SKU_PATH = Path(__file__).resolve().parents[2] / "reports" / "hybrid_research" / "sku_r2_summary.csv"

WEAK_R2_THRESHOLD = 0.0
MIN_PROFILED_HOURS = 4
MIN_EXPECTED_UPLIFT = 0.5
MAX_UPLIFT_MULTIPLIER = 3.0
MIN_POSITIVE_SLOTS = 3


def load_daily_demand(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    return df


def load_weak_sku_map(path: str | Path, *, threshold: float = WEAK_R2_THRESHOLD) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {BAKERY_COL, PRODUCT_COL, "best_r2"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing columns in weak SKU file: {sorted(missing)}")
    work = df[[BAKERY_COL, PRODUCT_COL, "best_r2"]].copy()
    work["best_r2"] = pd.to_numeric(work["best_r2"], errors="coerce")
    work["is_weak_sku"] = work["best_r2"] < threshold
    return work


def build_hourly_share_daily(source_path: str | Path) -> pd.DataFrame:
    sales = load_sales(source_path)
    hourly = aggregate_hourly_sales(sales)
    profiles = build_positive_profiles(hourly)
    applied = apply_profiles(hourly, profiles)
    daily = build_daily_from_applied(applied)

    support = (
        profiles.groupby([BAKERY_COL, CATEGORY_COL, PRODUCT_COL], as_index=False)
        .agg(
            hourly_profile_rows=("n_positive_slots", "size"),
            hourly_profile_positive_slots=("n_positive_slots", "sum"),
            hourly_profile_mean_cv=("cv_share_positive", "mean"),
        )
    )
    daily = daily.merge(
        support,
        on=[BAKERY_COL, CATEGORY_COL, PRODUCT_COL],
        how="left",
    )
    daily["hourly_profile_rows"] = daily["hourly_profile_rows"].fillna(0).astype(int)
    daily["hourly_profile_positive_slots"] = daily["hourly_profile_positive_slots"].fillna(0).astype(int)
    daily["hourly_profile_mean_cv"] = pd.to_numeric(
        daily["hourly_profile_mean_cv"], errors="coerce"
    ).fillna(np.nan)
    return daily


def build_weak_hourly_share_target(
    daily_df: pd.DataFrame,
    weak_map: pd.DataFrame,
    hourly_daily: pd.DataFrame,
    *,
    min_profiled_hours: int = MIN_PROFILED_HOURS,
    min_expected_uplift: float = MIN_EXPECTED_UPLIFT,
    max_uplift_multiplier: float = MAX_UPLIFT_MULTIPLIER,
    min_positive_slots: int = MIN_POSITIVE_SLOTS,
) -> pd.DataFrame:
    merge_cols = [DATE_COL, BAKERY_COL, CATEGORY_COL, PRODUCT_COL]
    keep_cols = [
        *merge_cols,
        "observed_sales",
        "expected_sales_from_hourly_profile",
        "total_hourly_gap",
        "profiled_hours",
        "positive_hours_observed",
        "hourly_profile_rows",
        "hourly_profile_positive_slots",
        "hourly_profile_mean_cv",
    ]
    keep_cols = [col for col in keep_cols if col in hourly_daily.columns]

    work = daily_df.merge(
        weak_map[[BAKERY_COL, PRODUCT_COL, "best_r2", "is_weak_sku"]],
        on=[BAKERY_COL, PRODUCT_COL],
        how="left",
    ).merge(
        hourly_daily[keep_cols],
        on=merge_cols,
        how="left",
    )

    work["is_weak_sku"] = work["is_weak_sku"].fillna(False)
    work["best_r2"] = pd.to_numeric(work["best_r2"], errors="coerce")
    work["expected_sales_from_hourly_profile"] = pd.to_numeric(
        work.get("expected_sales_from_hourly_profile", np.nan), errors="coerce"
    )
    work["total_hourly_gap"] = pd.to_numeric(work.get("total_hourly_gap", np.nan), errors="coerce").fillna(0.0)
    work["profiled_hours"] = pd.to_numeric(work.get("profiled_hours", 0), errors="coerce").fillna(0).astype(int)
    work["hourly_profile_positive_slots"] = pd.to_numeric(
        work.get("hourly_profile_positive_slots", 0), errors="coerce"
    ).fillna(0).astype(int)

    observed = pd.to_numeric(work[TARGET_COL], errors="coerce").fillna(0.0)
    expected = work["expected_sales_from_hourly_profile"].fillna(observed)
    expected = np.maximum(expected, observed)
    capped_expected = np.minimum(expected, observed * max_uplift_multiplier)

    eligible = (
        work["is_weak_sku"].astype(bool)
        & (work["profiled_hours"] >= min_profiled_hours)
        & (work["hourly_profile_positive_slots"] >= min_positive_slots)
        & ((capped_expected - observed) >= min_expected_uplift)
    )

    base_demand = pd.to_numeric(work.get(DEMAND_COL, observed), errors="coerce").fillna(observed)
    work["weak_hourly_share_expected"] = capped_expected
    work["weak_hourly_share_gap"] = capped_expected - observed
    work["weak_hourly_share_eligible"] = eligible
    work["Спрос_weak_hourly_share"] = np.where(
        eligible,
        np.maximum(base_demand, capped_expected),
        base_demand,
    )
    work["weak_hourly_share_uplift_vs_base"] = work["Спрос_weak_hourly_share"] - base_demand
    return work


def build_summary(df: pd.DataFrame) -> dict:
    eligible = df["weak_hourly_share_eligible"].fillna(False)
    uplift = pd.to_numeric(df["weak_hourly_share_uplift_vs_base"], errors="coerce").fillna(0.0)
    return {
        "rows": int(len(df)),
        "weak_rows": int(df["is_weak_sku"].fillna(False).sum()),
        "eligible_rows": int(eligible.sum()),
        "eligible_share_of_weak": round(
            float(eligible.sum() / max(df["is_weak_sku"].fillna(False).sum(), 1)),
            6,
        ),
        "mean_uplift_all": round(float(uplift.mean()), 6),
        "mean_uplift_eligible": round(float(uplift[eligible].mean()), 6) if eligible.any() else 0.0,
        "median_uplift_eligible": round(float(uplift[eligible].median()), 6) if eligible.any() else 0.0,
        "base_demand_mean": round(float(pd.to_numeric(df.get(DEMAND_COL), errors="coerce").mean()), 6),
        "adjusted_demand_mean": round(float(pd.to_numeric(df["Спрос_weak_hourly_share"], errors="coerce").mean()), 6),
    }


def save_outputs(output_dir: str | Path, df: pd.DataFrame, summary: dict) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / OUTPUT_NAME
    summary_path = out_dir / SUMMARY_OUTPUT_NAME
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"dataset": csv_path, "summary": summary_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build weak-SKU hourly-share alternative demand target")
    parser.add_argument("--daily-demand-path", default=str(DEMAND_8M_PATH))
    parser.add_argument("--source-path", default=str(SALES_HRS_PATH))
    parser.add_argument("--weak-sku-path", default=str(WEAK_SKU_PATH))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[2] / "data" / "processed"))
    args = parser.parse_args()

    daily_df = load_daily_demand(args.daily_demand_path)
    weak_map = load_weak_sku_map(args.weak_sku_path)
    hourly_daily = build_hourly_share_daily(args.source_path)
    result = build_weak_hourly_share_target(daily_df, weak_map, hourly_daily)
    summary = build_summary(result)
    paths = save_outputs(args.output_dir, result, summary)

    print("=" * 72)
    print("WEAK SKU HOURLY SHARE TARGET")
    print("=" * 72)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

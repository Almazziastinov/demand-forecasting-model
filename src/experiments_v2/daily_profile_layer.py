"""
Build daily share profiles for sales-first attainable-demand research.

Profiles are built only from days marked as `good_execution_day` in the
availability layer. The output is hierarchical so weak SKU rows can later
borrow strength from broader levels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATE_COL = "Дата"
BAKERY_COL = "Пекарня"
CATEGORY_COL = "Категория"
PRODUCT_COL = "Номенклатура"
TARGET_COL = "Продано"
DOW_COL = "ДеньНедели"

PROFILE_OUTPUT_NAME = "daily_share_profiles.csv"
PROFILE_INPUT_OUTPUT_NAME = "daily_profile_input.csv"
SUMMARY_OUTPUT_NAME = "daily_profile_layer_summary.json"


def load_backbone(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    return df


def load_availability(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    return df


def build_profile_input(backbone: pd.DataFrame, availability: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        DATE_COL,
        BAKERY_COL,
        CATEGORY_COL,
        PRODUCT_COL,
        TARGET_COL,
        DOW_COL,
    ]
    avail_cols = [
        DATE_COL,
        BAKERY_COL,
        CATEGORY_COL,
        PRODUCT_COL,
        "sku_sales_total",
        "bakery_sales_total",
        "good_execution_day",
        "availability_score",
        "stockout_like_hours",
        "zero_under_traffic_hours",
        "early_stop_flag",
    ]
    avail_df = availability[[col for col in avail_cols if col in availability.columns]].copy()
    work = avail_df.merge(
        backbone[keep_cols].copy(),
        on=[DATE_COL, BAKERY_COL, CATEGORY_COL, PRODUCT_COL],
        how="left",
        suffixes=("_avail", ""),
    )

    if "sku_sales_total" in work.columns:
        work["sku_sales_total"] = pd.to_numeric(work["sku_sales_total"], errors="coerce").fillna(work[TARGET_COL])
    else:
        work["sku_sales_total"] = pd.to_numeric(work[TARGET_COL], errors="coerce").fillna(0.0)

    work["good_execution_day"] = work["good_execution_day"].fillna(False).astype(bool)
    work["availability_score"] = pd.to_numeric(work.get("availability_score", 0.0), errors="coerce").fillna(0.0)
    work["stockout_like_hours"] = pd.to_numeric(work.get("stockout_like_hours", 0.0), errors="coerce").fillna(0.0)
    work["zero_under_traffic_hours"] = pd.to_numeric(
        work.get("zero_under_traffic_hours", 0.0), errors="coerce"
    ).fillna(0.0)
    work["early_stop_flag"] = work.get("early_stop_flag", False)
    work["early_stop_flag"] = work["early_stop_flag"].fillna(False).astype(bool)

    bakery_daily = (
        work.groupby([DATE_COL, BAKERY_COL], as_index=False)["sku_sales_total"]
        .sum()
        .rename(columns={"sku_sales_total": "bakery_sales_total_calc"})
    )
    category_daily = (
        work.groupby([DATE_COL, BAKERY_COL, CATEGORY_COL], as_index=False)["sku_sales_total"]
        .sum()
        .rename(columns={"sku_sales_total": "category_sales_total"})
    )

    work = work.merge(bakery_daily, on=[DATE_COL, BAKERY_COL], how="left")
    work = work.merge(category_daily, on=[DATE_COL, BAKERY_COL, CATEGORY_COL], how="left")

    work["bakery_sales_total"] = pd.to_numeric(
        work.get("bakery_sales_total", work["bakery_sales_total_calc"]), errors="coerce"
    ).fillna(work["bakery_sales_total_calc"])
    work["share_of_bakery"] = (
        work["sku_sales_total"] / work["bakery_sales_total"].replace(0, np.nan)
    ).fillna(0.0)
    work["share_of_category"] = (
        work["sku_sales_total"] / work["category_sales_total"].replace(0, np.nan)
    ).fillna(0.0)

    return work.sort_values([BAKERY_COL, PRODUCT_COL, DATE_COL]).reset_index(drop=True)


def _profile_stats(df: pd.DataFrame, share_col: str, sales_col: str) -> pd.DataFrame:
    grouped = df.groupby(df.columns.tolist()[:-2], as_index=False)
    profile = grouped.agg(
        n_good_days=(sales_col, "size"),
        mean_sales=(sales_col, "mean"),
        median_sales=(sales_col, "median"),
        positive_day_rate=(sales_col, lambda s: float((s > 0).mean())),
        mean_share=(share_col, "mean"),
        median_share=(share_col, "median"),
        share_std=(share_col, "std"),
    )
    profile["share_std"] = profile["share_std"].fillna(0.0)
    profile["cv_share"] = profile["share_std"] / profile["mean_share"].replace(0, np.nan)
    profile["cv_share"] = profile["cv_share"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    profile["share_zero"] = 1.0 - profile["positive_day_rate"]
    profile["profile_reliability_score"] = np.clip(
        (profile["n_good_days"] / 20.0) * (1.0 - profile["share_zero"]) * (1.0 / (1.0 + profile["cv_share"])),
        0.0,
        1.0,
    )
    return profile


def build_daily_profiles(profile_input: pd.DataFrame) -> pd.DataFrame:
    good = profile_input[profile_input["good_execution_day"]].copy()
    if good.empty:
        return pd.DataFrame()

    level_specs = [
        ("bakery_sku", [BAKERY_COL, PRODUCT_COL, CATEGORY_COL, DOW_COL]),
        ("sku_global", [PRODUCT_COL, CATEGORY_COL, DOW_COL]),
        ("bakery_category", [BAKERY_COL, CATEGORY_COL, DOW_COL]),
        ("category_global", [CATEGORY_COL, DOW_COL]),
    ]

    profile_frames = []
    for profile_level, keys in level_specs:
        bakery_level = _profile_stats(
            good[keys + ["share_of_bakery", "sku_sales_total"]].copy(),
            share_col="share_of_bakery",
            sales_col="sku_sales_total",
        ).rename(
            columns={
                "mean_share": "mean_share_of_bakery",
                "median_share": "median_share_of_bakery",
                "share_std": "share_std_of_bakery",
                "cv_share": "cv_share_of_bakery",
            }
        )

        category_level = _profile_stats(
            good[keys + ["share_of_category", "sku_sales_total"]].copy(),
            share_col="share_of_category",
            sales_col="sku_sales_total",
        ).rename(
            columns={
                "mean_share": "mean_share_of_category",
                "median_share": "median_share_of_category",
                "share_std": "share_std_of_category",
                "cv_share": "cv_share_of_category",
                "n_good_days": "n_good_days_category",
                "mean_sales": "mean_sales_category",
                "median_sales": "median_sales_category",
                "positive_day_rate": "positive_day_rate_category",
                "share_zero": "share_zero_category",
                "profile_reliability_score": "profile_reliability_score_category",
            }
        )

        merged = bakery_level.merge(keys and category_level, on=keys, how="outer")
        merged["profile_level"] = profile_level
        profile_frames.append(merged)

    profiles = pd.concat(profile_frames, ignore_index=True, sort=False)
    profiles["n_good_days"] = profiles["n_good_days"].fillna(0).astype(int)
    profiles["n_good_days_category"] = profiles["n_good_days_category"].fillna(0).astype(int)
    return profiles


def build_summary(profile_input: pd.DataFrame, profiles: pd.DataFrame) -> dict:
    good = profile_input["good_execution_day"].fillna(False)
    out = {
        "input_rows": int(len(profile_input)),
        "good_execution_rows": int(good.sum()),
        "good_execution_share": round(float(good.mean()), 4) if len(profile_input) else 0.0,
        "products_total": int(profile_input[PRODUCT_COL].nunique()) if len(profile_input) else 0,
        "bakeries_total": int(profile_input[BAKERY_COL].nunique()) if len(profile_input) else 0,
        "profiles_rows": int(len(profiles)),
        "date_min": None if profile_input.empty else str(profile_input[DATE_COL].min().date()),
        "date_max": None if profile_input.empty else str(profile_input[DATE_COL].max().date()),
    }
    if not profiles.empty:
        out["profiles_by_level"] = profiles["profile_level"].value_counts().to_dict()
    return out


def save_outputs(
    output_dir: str | Path,
    profile_input: pd.DataFrame,
    profiles: pd.DataFrame,
    summary: dict,
    *,
    output_suffix: str = "",
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{output_suffix}" if output_suffix else ""

    profile_input_path = out_dir / PROFILE_INPUT_OUTPUT_NAME.replace(".csv", f"{suffix}.csv")
    profile_path = out_dir / PROFILE_OUTPUT_NAME.replace(".csv", f"{suffix}.csv")
    summary_path = out_dir / SUMMARY_OUTPUT_NAME.replace(".json", f"{suffix}.json")

    profile_input.to_csv(profile_input_path, index=False, encoding="utf-8-sig")
    profiles.to_csv(profile_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "profile_input": profile_input_path,
        "profiles": profile_path,
        "summary": summary_path,
    }


def build_and_save_daily_profile_layer(
    backbone_path: str | Path,
    availability_path: str | Path,
    output_dir: str | Path,
    *,
    output_suffix: str = "",
) -> dict[str, Path]:
    backbone = load_backbone(backbone_path)
    availability = load_availability(availability_path)
    profile_input = build_profile_input(backbone, availability)
    profiles = build_daily_profiles(profile_input)
    summary = build_summary(profile_input, profiles)
    return save_outputs(
        output_dir,
        profile_input,
        profiles,
        summary,
        output_suffix=output_suffix,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build daily hierarchical share profiles")
    parser.add_argument(
        "--backbone-path",
        default="data/processed/daily_sales_backbone.csv",
        help="Path to canonical daily sales backbone",
    )
    parser.add_argument(
        "--availability-path",
        default="data/processed/availability_daily_signals.csv",
        help="Path to daily availability signals",
    )
    parser.add_argument("--output-suffix", default="", help="Suffix for output files")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    paths = build_and_save_daily_profile_layer(
        root / args.backbone_path,
        root / args.availability_path,
        root / "data" / "processed",
        output_suffix=args.output_suffix,
    )

    print("=" * 72)
    print("DAILY PROFILE LAYER")
    print("=" * 72)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

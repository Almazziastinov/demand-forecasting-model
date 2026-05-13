"""
First attainable-demand baseline based on blended daily share profiles.

This layer does not claim to estimate true demand yet. It produces a
conservative baseline of what a SKU could have sold given stable bakery/category
context and the best available hierarchical profile.
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
DOW_COL = "ДеньНедели"
TARGET_COL = "Продано"

OUTPUT_NAME = "attainable_demand_baseline.csv"
SUMMARY_OUTPUT_NAME = "attainable_demand_baseline_summary.json"


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    return df


def build_attainable_baseline(profile_input: pd.DataFrame, blended_profiles: pd.DataFrame) -> pd.DataFrame:
    merge_cols = [BAKERY_COL, PRODUCT_COL, CATEGORY_COL, DOW_COL]
    blended_keep = [
        BAKERY_COL,
        PRODUCT_COL,
        CATEGORY_COL,
        DOW_COL,
        "final_expected_share",
        "share_source_primary",
        "blend_confidence_score",
        "share_bakery_sku",
        "share_sku_global",
        "share_bakery_category",
        "share_category_global",
    ]
    blended_keep = [col for col in blended_keep if col in blended_profiles.columns]

    work = profile_input.merge(
        blended_profiles[blended_keep],
        on=merge_cols,
        how="left",
    )

    work["sku_sales_total"] = pd.to_numeric(work["sku_sales_total"], errors="coerce").fillna(work[TARGET_COL])
    work["bakery_sales_total"] = pd.to_numeric(work["bakery_sales_total"], errors="coerce").fillna(0.0)
    work["category_sales_total"] = pd.to_numeric(work["category_sales_total"], errors="coerce").fillna(0.0)
    work["final_expected_share"] = pd.to_numeric(work["final_expected_share"], errors="coerce")
    work["blend_confidence_score"] = pd.to_numeric(work["blend_confidence_score"], errors="coerce").fillna(0.0)

    # Two baseline views:
    # 1. bakery-based expected sales
    # 2. category-based expected sales
    work["attainable_sales_from_bakery"] = work["bakery_sales_total"] * work["final_expected_share"].fillna(0.0)
    work["attainable_sales_from_category"] = work["category_sales_total"] * work["final_expected_share"].fillna(0.0)

    work["attainable_sales_baseline"] = np.where(
        work["share_source_primary"].eq("bakery_category") | work["share_source_primary"].eq("category_global"),
        work["attainable_sales_from_category"],
        work["attainable_sales_from_bakery"],
    )

    work["attainable_sales_baseline"] = np.maximum(work["attainable_sales_baseline"], work["sku_sales_total"])
    work["attainable_sales_baseline"] = np.where(
        work["blend_confidence_score"] >= 0.5,
        work["attainable_sales_baseline"],
        work["sku_sales_total"],
    )

    work["attainable_gap"] = work["attainable_sales_baseline"] - work["sku_sales_total"]
    work["attainable_uplift_pct"] = np.where(
        work["sku_sales_total"] > 0,
        work["attainable_gap"] / work["sku_sales_total"] * 100.0,
        np.nan,
    )

    work["opportunity_flag"] = (
        (work["attainable_gap"] > 0)
        & (~work["good_execution_day"].fillna(False))
        & (work["blend_confidence_score"] >= 0.5)
    )
    work["opportunity_type"] = np.select(
        [
            work["opportunity_flag"] & work["early_stop_flag"].fillna(False),
            work["opportunity_flag"] & (work["stockout_like_hours"].fillna(0) > 0),
            work["opportunity_flag"],
        ],
        [
            "early_stop",
            "stockout_like",
            "general_execution_gap",
        ],
        default="none",
    )

    return work.sort_values([BAKERY_COL, PRODUCT_COL, DATE_COL]).reset_index(drop=True)


def build_summary(baseline: pd.DataFrame) -> dict:
    opportunity = baseline["opportunity_flag"].fillna(False)
    return {
        "rows": int(len(baseline)),
        "rows_with_profiles": int(baseline["final_expected_share"].notna().sum()),
        "attainable_gap_mean": round(float(baseline["attainable_gap"].mean()), 6),
        "attainable_gap_median": round(float(baseline["attainable_gap"].median()), 6),
        "opportunity_rows": int(opportunity.sum()),
        "opportunity_share": round(float(opportunity.mean()), 6) if len(baseline) else 0.0,
        "opportunity_type_counts": baseline["opportunity_type"].value_counts().to_dict(),
        "share_source_counts": baseline["share_source_primary"].value_counts(dropna=False).to_dict(),
        "confidence_mean": round(float(baseline["blend_confidence_score"].mean()), 6),
    }


def save_outputs(
    output_dir: str | Path,
    baseline: pd.DataFrame,
    summary: dict,
    *,
    output_suffix: str = "",
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{output_suffix}" if output_suffix else ""

    baseline_path = out_dir / OUTPUT_NAME.replace(".csv", f"{suffix}.csv")
    summary_path = out_dir / SUMMARY_OUTPUT_NAME.replace(".json", f"{suffix}.json")

    baseline.to_csv(baseline_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "baseline": baseline_path,
        "summary": summary_path,
    }


def build_and_save_attainable_baseline(
    profile_input_path: str | Path,
    blended_profiles_path: str | Path,
    output_dir: str | Path,
    *,
    output_suffix: str = "",
) -> dict[str, Path]:
    profile_input = load_csv(profile_input_path)
    blended_profiles = load_csv(blended_profiles_path)
    baseline = build_attainable_baseline(profile_input, blended_profiles)
    summary = build_summary(baseline)
    return save_outputs(
        output_dir,
        baseline,
        summary,
        output_suffix=output_suffix,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build first attainable demand baseline")
    parser.add_argument(
        "--profile-input-path",
        default="data/processed/daily_profile_input.csv",
        help="Path to daily profile input table",
    )
    parser.add_argument(
        "--blended-profiles-path",
        default="data/processed/daily_profile_blended.csv",
        help="Path to blended profile table",
    )
    parser.add_argument("--output-suffix", default="", help="Suffix for output files")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    paths = build_and_save_attainable_baseline(
        root / args.profile_input_path,
        root / args.blended_profiles_path,
        root / "data" / "processed",
        output_suffix=args.output_suffix,
    )

    print("=" * 72)
    print("ATTAINABLE DEMAND BASELINE")
    print("=" * 72)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

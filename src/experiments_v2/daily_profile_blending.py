"""
Blend hierarchical daily profiles into one usable expected-share layer.

The output of this step is a transparent rule-based shrinkage baseline:
for each bakery x SKU x day-of-week we combine available profile levels and
produce a single `final_expected_share`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BAKERY_COL = "Пекарня"
CATEGORY_COL = "Категория"
PRODUCT_COL = "Номенклатура"
DOW_COL = "ДеньНедели"

BLEND_OUTPUT_NAME = "daily_profile_blended.csv"
SUMMARY_OUTPUT_NAME = "daily_profile_blending_summary.json"


def load_profiles(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _rename_level(df: pd.DataFrame, level: str, suffix: str, key_cols: list[str]) -> pd.DataFrame:
    level_df = df[df["profile_level"] == level].copy()
    keep = key_cols + [
        "mean_share_of_bakery",
        "median_share_of_bakery",
        "mean_share_of_category",
        "median_share_of_category",
        "n_good_days",
        "profile_reliability_score",
    ]
    keep = [col for col in keep if col in level_df.columns]
    level_df = level_df[keep].copy()
    rename_map = {}
    for col in level_df.columns:
        if col not in key_cols:
            rename_map[col] = f"{col}_{suffix}"
    return level_df.rename(columns=rename_map)


def build_blended_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    bakery_sku = _rename_level(
        profiles,
        level="bakery_sku",
        suffix="bakery_sku",
        key_cols=[BAKERY_COL, PRODUCT_COL, CATEGORY_COL, DOW_COL],
    )
    sku_global = _rename_level(
        profiles,
        level="sku_global",
        suffix="sku_global",
        key_cols=[PRODUCT_COL, CATEGORY_COL, DOW_COL],
    )
    bakery_category = _rename_level(
        profiles,
        level="bakery_category",
        suffix="bakery_category",
        key_cols=[BAKERY_COL, CATEGORY_COL, DOW_COL],
    )
    category_global = _rename_level(
        profiles,
        level="category_global",
        suffix="category_global",
        key_cols=[CATEGORY_COL, DOW_COL],
    )

    work = bakery_sku.merge(
        sku_global,
        on=[PRODUCT_COL, CATEGORY_COL, DOW_COL],
        how="left",
    ).merge(
        bakery_category,
        on=[BAKERY_COL, CATEGORY_COL, DOW_COL],
        how="left",
    ).merge(
        category_global,
        on=[CATEGORY_COL, DOW_COL],
        how="left",
    )

    work["share_bakery_sku"] = work["mean_share_of_bakery_bakery_sku"].fillna(np.nan)
    work["share_sku_global"] = work["mean_share_of_bakery_sku_global"].fillna(np.nan)
    work["share_bakery_category"] = work["mean_share_of_category_bakery_category"].fillna(np.nan)
    work["share_category_global"] = work["mean_share_of_category_category_global"].fillna(np.nan)

    work["bakery_sku_eligible"] = (
        work["share_bakery_sku"].notna()
        & (work["n_good_days_bakery_sku"].fillna(0) >= 10)
        & (work["profile_reliability_score_bakery_sku"].fillna(0.0) >= 0.60)
    )
    work["bakery_category_eligible"] = work["share_bakery_category"].notna()
    work["sku_global_eligible"] = work["share_sku_global"].notna()
    work["category_global_eligible"] = work["share_category_global"].notna()

    work["share_source_primary"] = np.select(
        [
            work["bakery_sku_eligible"],
            work["bakery_category_eligible"],
            work["sku_global_eligible"],
            work["category_global_eligible"],
        ],
        [
            "bakery_sku",
            "bakery_category",
            "sku_global",
            "category_global",
        ],
        default="",
    )
    work["share_source_primary"] = work["share_source_primary"].replace("", np.nan)

    work["final_expected_share"] = np.select(
        [
            work["share_source_primary"] == "bakery_sku",
            work["share_source_primary"] == "bakery_category",
            work["share_source_primary"] == "sku_global",
            work["share_source_primary"] == "category_global",
        ],
        [
            work["share_bakery_sku"],
            work["share_bakery_category"],
            work["share_sku_global"],
            work["share_category_global"],
        ],
        default=np.nan,
    )

    work["w_bakery_sku"] = (work["share_source_primary"] == "bakery_sku").astype(float)
    work["w_bakery_category"] = (work["share_source_primary"] == "bakery_category").astype(float)
    work["w_sku_global"] = (work["share_source_primary"] == "sku_global").astype(float)
    work["w_category_global"] = (work["share_source_primary"] == "category_global").astype(float)

    work["blend_confidence_score"] = np.select(
        [
            work["share_source_primary"] == "bakery_sku",
            work["share_source_primary"] == "bakery_category",
            work["share_source_primary"] == "sku_global",
            work["share_source_primary"] == "category_global",
        ],
        [
            work["profile_reliability_score_bakery_sku"].fillna(0.0),
            work["profile_reliability_score_bakery_category"].fillna(0.0) * 0.85,
            work["profile_reliability_score_sku_global"].fillna(0.0) * 0.75,
            work["profile_reliability_score_category_global"].fillna(0.0) * 0.65,
        ],
        default=0.0,
    )

    return work.sort_values([BAKERY_COL, PRODUCT_COL, DOW_COL]).reset_index(drop=True)


def build_summary(blended: pd.DataFrame) -> dict:
    return {
        "rows": int(len(blended)),
        "rows_with_final_share": int(blended["final_expected_share"].notna().sum()),
        "final_share_mean": round(float(blended["final_expected_share"].mean()), 6),
        "final_share_median": round(float(blended["final_expected_share"].median()), 6),
        "blend_confidence_mean": round(float(blended["blend_confidence_score"].mean()), 6),
        "primary_source_counts": blended["share_source_primary"].value_counts().to_dict(),
    }


def save_outputs(
    output_dir: str | Path,
    blended: pd.DataFrame,
    summary: dict,
    *,
    output_suffix: str = "",
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{output_suffix}" if output_suffix else ""

    blend_path = out_dir / BLEND_OUTPUT_NAME.replace(".csv", f"{suffix}.csv")
    summary_path = out_dir / SUMMARY_OUTPUT_NAME.replace(".json", f"{suffix}.json")

    blended.to_csv(blend_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "blended": blend_path,
        "summary": summary_path,
    }


def build_and_save_profile_blending(
    profiles_path: str | Path,
    output_dir: str | Path,
    *,
    output_suffix: str = "",
) -> dict[str, Path]:
    profiles = load_profiles(profiles_path)
    blended = build_blended_profiles(profiles)
    summary = build_summary(blended)
    return save_outputs(output_dir, blended, summary, output_suffix=output_suffix)


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend hierarchical daily share profiles")
    parser.add_argument(
        "--profiles-path",
        default="data/processed/daily_share_profiles.csv",
        help="Path to daily hierarchical profiles",
    )
    parser.add_argument("--output-suffix", default="", help="Suffix for output files")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    paths = build_and_save_profile_blending(
        root / args.profiles_path,
        root / "data" / "processed",
        output_suffix=args.output_suffix,
    )

    print("=" * 72)
    print("DAILY PROFILE BLENDING")
    print("=" * 72)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

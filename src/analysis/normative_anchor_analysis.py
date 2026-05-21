from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATE_COL = "date"
BAKERY_COL = "bakery_id"
SKU_COL = "product_id"
SEGMENT_COL = "primary_segment"
SALES_COL = "observed_sales_qty"
DOW_COL = "dow"

DEFAULT_DAILY_PATH = Path("data/processed/sku_daily_research_panel.csv")
DEFAULT_SEGMENT_MAP_PATH = Path("src/experiments_v2/75_normative_demand_map/predictability_and_structure_map.csv")
DEFAULT_OUTPUT_DIR = Path("reports/normative_anchor_analysis")


def load_daily_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, BAKERY_COL, SKU_COL]).copy()
    df[SALES_COL] = pd.to_numeric(df[SALES_COL], errors="coerce").fillna(0.0)

    numeric_cols = [
        "release_qty",
        "bakery_sales_qty_total",
        "category_sales_qty_in_bakery_day",
        "sku_sales_share_in_bakery_day",
        "sku_sales_share_in_category_day",
        "weekly_seasonality_strength",
        "weekday_profile_stability",
        "release_coverage_share",
        "bakery_sales_corr",
        "release_corr_with_sales",
        "zero_share",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if DOW_COL not in df.columns:
        df[DOW_COL] = df[DATE_COL].dt.weekday
    else:
        df[DOW_COL] = pd.to_numeric(df[DOW_COL], errors="coerce").fillna(df[DATE_COL].dt.weekday).astype(int)
    return df


def load_segment_map(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    keep_cols = [
        BAKERY_COL,
        SKU_COL,
        SEGMENT_COL,
        "bakery_name",
        "product_name",
        "category_name",
        "city",
        "weekly_seasonality_strength",
        "weekday_profile_stability",
        "release_coverage_share",
        "release_corr_with_sales",
        "bakery_sales_corr",
        "zero_share",
        "predictability_score",
    ]
    keep_cols = [col for col in keep_cols if col in df.columns]
    return df[keep_cols].drop_duplicates(subset=[BAKERY_COL, SKU_COL])


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    xv = pd.to_numeric(x, errors="coerce")
    yv = pd.to_numeric(y, errors="coerce")
    valid = xv.notna() & yv.notna()
    if valid.sum() < 3:
        return np.nan
    xv = xv[valid]
    yv = yv[valid]
    if xv.nunique() < 2 or yv.nunique() < 2:
        return np.nan
    return float(np.corrcoef(xv, yv)[0, 1])


def _cv(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    mean_val = float(values.mean())
    if not np.isfinite(mean_val) or abs(mean_val) <= 1e-12:
        return np.nan
    return float(values.std(ddof=0) / mean_val)


def _weekday_share_profile(group: pd.DataFrame, value_col: str) -> pd.Series:
    profile = group.groupby(DOW_COL, observed=True)[value_col].sum().reindex(range(7), fill_value=0.0)
    total = float(profile.sum())
    if total <= 1e-12:
        return pd.Series(np.nan, index=range(7), dtype=float)
    return profile / total


def _weekday_profile_alignment(left: pd.Series, right: pd.Series) -> float:
    valid = left.notna() & right.notna()
    if valid.sum() < 2:
        return np.nan
    distance = float((left[valid] - right[valid]).abs().sum() / 2.0)
    return float(np.clip(1.0 - distance, 0.0, 1.0))


def _normalized_score(value: float, *, fill: float = 0.0) -> float:
    if pd.isna(value):
        return fill
    return float(np.clip(value, 0.0, 1.0))


def _choose_dominant_anchor(anchor_scores: dict[str, float]) -> str:
    valid_scores = {key: value for key, value in anchor_scores.items() if pd.notna(value)}
    if not valid_scores:
        return "self_pattern"

    best_anchor, best_score = max(valid_scores.items(), key=lambda item: item[1])

    # Preserve bakery/release anchors when they are effectively tied with
    # smoother structural anchors. This avoids over-assigning category_role to
    # clearly bakery-driven pairs with mechanically stable shares.
    if (
        "bakery_scale" in valid_scores
        and valid_scores["bakery_scale"] >= 0.85
        and best_score - valid_scores["bakery_scale"] <= 0.05
    ):
        return "bakery_scale"
    if (
        "release" in valid_scores
        and valid_scores["release"] >= 0.85
        and best_score - valid_scores["release"] <= 0.05
    ):
        return "release"
    return best_anchor


def build_anchor_profile(daily_df: pd.DataFrame, segment_map: pd.DataFrame) -> pd.DataFrame:
    work = daily_df.merge(segment_map, on=[BAKERY_COL, SKU_COL], how="left", suffixes=("", "_segment"))
    if "category_name_segment" in work.columns and "category_name" not in work.columns:
        work["category_name"] = work["category_name_segment"]

    product_weekday_profiles: dict[object, pd.Series] = {}
    if "category_name" in work.columns:
        for product_id, product_group in work.groupby(SKU_COL, observed=True):
            product_weekday_profiles[product_id] = _weekday_share_profile(product_group, SALES_COL)

    records: list[dict[str, object]] = []
    group_cols = [BAKERY_COL, SKU_COL]
    for (bakery_id, product_id), group in work.groupby(group_cols, observed=True, sort=False):
        group = group.sort_values(DATE_COL).copy()

        share_bakery_cv = _cv(group.get("sku_sales_share_in_bakery_day", pd.Series(np.nan, index=group.index)))
        share_category_cv = _cv(group.get("sku_sales_share_in_category_day", pd.Series(np.nan, index=group.index)))
        pair_weekday_profile = _weekday_share_profile(group, SALES_COL)
        product_weekday_profile = product_weekday_profiles.get(product_id, pd.Series(np.nan, index=range(7), dtype=float))
        product_weekday_alignment = _weekday_profile_alignment(pair_weekday_profile, product_weekday_profile)

        self_pattern_strength = np.nanmean(
            [
                group["weekly_seasonality_strength"].iloc[0] if "weekly_seasonality_strength" in group.columns else np.nan,
                group["weekday_profile_stability"].iloc[0] if "weekday_profile_stability" in group.columns else np.nan,
                _normalized_score(1.0 - min(share_bakery_cv, 1.0), fill=np.nan) if pd.notna(share_bakery_cv) else np.nan,
            ]
        )
        release_anchor_strength = np.nanmean(
            [
                group["release_coverage_share"].iloc[0] if "release_coverage_share" in group.columns else np.nan,
                _normalized_score(group["release_corr_with_sales"].iloc[0], fill=np.nan) if "release_corr_with_sales" in group.columns else np.nan,
            ]
        )
        bakery_anchor_strength = np.nanmean(
            [
                _normalized_score(group["bakery_sales_corr"].iloc[0], fill=np.nan) if "bakery_sales_corr" in group.columns else np.nan,
                _normalized_score(1.0 - min(share_bakery_cv, 1.0), fill=np.nan) if pd.notna(share_bakery_cv) else np.nan,
            ]
        )
        category_anchor_strength = np.nanmean(
            [
                _normalized_score(1.0 - min(share_category_cv, 1.0), fill=np.nan) if pd.notna(share_category_cv) else np.nan,
                product_weekday_alignment,
            ]
        )

        anchor_scores = {
            "self_pattern": self_pattern_strength,
            "release": release_anchor_strength,
            "bakery_scale": bakery_anchor_strength,
            "category_role": category_anchor_strength,
        }
        dominant_anchor = _choose_dominant_anchor(anchor_scores)

        records.append(
            {
                BAKERY_COL: bakery_id,
                SKU_COL: product_id,
                SEGMENT_COL: group[SEGMENT_COL].iloc[0] if SEGMENT_COL in group.columns else "unmapped",
                "bakery_name": group["bakery_name"].iloc[0] if "bakery_name" in group.columns else np.nan,
                "product_name": group["product_name"].iloc[0] if "product_name" in group.columns else np.nan,
                "category_name": group["category_name"].iloc[0] if "category_name" in group.columns else np.nan,
                "city": group["city"].iloc[0] if "city" in group.columns else np.nan,
                "observed_mean": float(group[SALES_COL].mean()),
                "zero_share": float((group[SALES_COL] <= 0).mean()),
                "share_in_bakery_cv": share_bakery_cv,
                "share_in_category_cv": share_category_cv,
                "product_weekday_alignment": product_weekday_alignment,
                "self_pattern_strength": self_pattern_strength,
                "release_anchor_strength": release_anchor_strength,
                "bakery_anchor_strength": bakery_anchor_strength,
                "category_anchor_strength": category_anchor_strength,
                "dominant_anchor": dominant_anchor,
            }
        )

    result = pd.DataFrame.from_records(records)
    score_cols = [
        "self_pattern_strength",
        "release_anchor_strength",
        "bakery_anchor_strength",
        "category_anchor_strength",
    ]
    for col in score_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def build_anchor_summary(anchor_profile: pd.DataFrame) -> pd.DataFrame:
    return (
        anchor_profile.groupby(SEGMENT_COL, observed=True)
        .agg(
            pairs=(SEGMENT_COL, "size"),
            observed_mean=("observed_mean", "mean"),
            zero_share=("zero_share", "mean"),
            self_pattern_strength=("self_pattern_strength", "mean"),
            release_anchor_strength=("release_anchor_strength", "mean"),
            bakery_anchor_strength=("bakery_anchor_strength", "mean"),
            category_anchor_strength=("category_anchor_strength", "mean"),
            product_weekday_alignment=("product_weekday_alignment", "mean"),
            share_in_bakery_cv=("share_in_bakery_cv", "mean"),
            share_in_category_cv=("share_in_category_cv", "mean"),
        )
        .reset_index()
        .sort_values("pairs", ascending=False)
    )


def build_anchor_dominance(anchor_profile: pd.DataFrame) -> pd.DataFrame:
    dominance = (
        anchor_profile.groupby([SEGMENT_COL, "dominant_anchor"], observed=True)
        .size()
        .rename("pairs")
        .reset_index()
    )
    totals = dominance.groupby(SEGMENT_COL, observed=True)["pairs"].transform("sum")
    dominance["share"] = dominance["pairs"] / totals
    return dominance.sort_values([SEGMENT_COL, "pairs"], ascending=[True, False]).reset_index(drop=True)


def build_metrics(anchor_profile: pd.DataFrame) -> dict[str, object]:
    score_cols = [
        "self_pattern_strength",
        "release_anchor_strength",
        "bakery_anchor_strength",
        "category_anchor_strength",
    ]
    return {
        "pairs": int(len(anchor_profile)),
        "segments": anchor_profile[SEGMENT_COL].value_counts().to_dict(),
        "dominant_anchor_counts": anchor_profile["dominant_anchor"].value_counts().to_dict(),
        "mean_scores": {
            col: round(float(anchor_profile[col].mean()), 6) for col in score_cols
        },
    }


def save_outputs(
    output_dir: str | Path,
    anchor_profile: pd.DataFrame,
    anchor_summary: pd.DataFrame,
    anchor_dominance: pd.DataFrame,
    metrics: dict[str, object],
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profile_path = out_dir / "anchor_profile_by_pair.csv"
    summary_path = out_dir / "anchor_summary_by_segment.csv"
    dominance_path = out_dir / "anchor_dominance_by_segment.csv"
    metrics_path = out_dir / "metrics.json"

    anchor_profile.to_csv(profile_path, index=False, encoding="utf-8-sig")
    anchor_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    anchor_dominance.to_csv(dominance_path, index=False, encoding="utf-8-sig")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "profile": profile_path,
        "summary": summary_path,
        "dominance": dominance_path,
        "metrics": metrics_path,
    }


def run_analysis(daily_path: str | Path, segment_map_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    daily_df = load_daily_dataset(daily_path)
    segment_map = load_segment_map(segment_map_path)
    anchor_profile = build_anchor_profile(daily_df, segment_map)
    anchor_summary = build_anchor_summary(anchor_profile)
    anchor_dominance = build_anchor_dominance(anchor_profile)
    metrics = build_metrics(anchor_profile)
    return save_outputs(output_dir, anchor_profile, anchor_summary, anchor_dominance, metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze anchor sources for normative demand construction")
    parser.add_argument("--daily-path", default=str(DEFAULT_DAILY_PATH))
    parser.add_argument("--segment-map-path", default=str(DEFAULT_SEGMENT_MAP_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    outputs = run_analysis(
        daily_path=root / args.daily_path,
        segment_map_path=root / args.segment_map_path,
        output_dir=root / args.output_dir,
    )
    print("=" * 72)
    print("NORMATIVE ANCHOR ANALYSIS")
    print("=" * 72)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

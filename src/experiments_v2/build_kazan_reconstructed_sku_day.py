from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATE_COL = "date"
BAKERY_COL = "bakery_id"
BAKERY_NAME_COL = "bakery_name"
CITY_COL = "city"
CATEGORY_COL = "category_name"
SKU_COL = "product_id"
SKU_NAME_COL = "product_name"

OUTPUT_NAME = "kazan_reconstructed_sku_day.csv"
SUMMARY_OUTPUT = "kazan_reconstructed_sku_day_summary.json"


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    return df


def _safe_mean_profile(df: pd.DataFrame, group_cols: list[str], value_col: str, out_col: str) -> pd.DataFrame:
    profile = (
        df.groupby(group_cols, as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: out_col})
    )
    return profile


def build_reconstructed_sku_day(
    *,
    sku_daily: pd.DataFrame,
    bakery_category_daily: pd.DataFrame,
    bakery_category_share_in_total_daily: pd.DataFrame,
    city_sku_day: pd.DataFrame,
    bakery_share_in_city_sku_daily: pd.DataFrame,
    bakery_sku_cluster_day: pd.DataFrame,
    sku_cluster_share_in_bakery_category_daily: pd.DataFrame,
    sku_share_in_bakery_sku_cluster_daily: pd.DataFrame,
    bakery_cluster_sku_day: pd.DataFrame,
    bakery_share_in_bakery_cluster_sku_daily: pd.DataFrame,
    path_scores: pd.DataFrame,
) -> pd.DataFrame:
    base = sku_daily[
        [
            DATE_COL,
            BAKERY_COL,
            BAKERY_NAME_COL,
            CITY_COL,
            CATEGORY_COL,
            SKU_COL,
            SKU_NAME_COL,
            "observed_sales_qty",
            "release_qty",
            "row_quality_score",
            "bakery_total_sales_qty",
        ]
    ].copy()

    # Mean shares used as stable rule-based coefficients.
    bakery_category_share_profile = _safe_mean_profile(
        bakery_category_share_in_total_daily,
        [BAKERY_COL, CATEGORY_COL],
        "bakery_category_share_in_total",
        "mean_bakery_category_share_in_total",
    )
    sku_cluster_category_share_profile = _safe_mean_profile(
        sku_cluster_share_in_bakery_category_daily,
        [BAKERY_COL, CATEGORY_COL, "sku_cluster"],
        "sku_cluster_share_in_bakery_category",
        "mean_sku_cluster_share_in_bakery_category",
    )
    sku_in_cluster_share_profile = _safe_mean_profile(
        sku_share_in_bakery_sku_cluster_daily,
        [BAKERY_COL, CATEGORY_COL, "sku_cluster", SKU_COL],
        "sku_share_in_bakery_sku_cluster",
        "mean_sku_share_in_bakery_sku_cluster",
    )
    city_sku_share_profile = _safe_mean_profile(
        bakery_share_in_city_sku_daily,
        [BAKERY_COL, CATEGORY_COL, SKU_COL],
        "bakery_share_in_city_sku",
        "mean_bakery_share_in_city_sku",
    )
    bakery_cluster_sku_share_profile = _safe_mean_profile(
        bakery_share_in_bakery_cluster_sku_daily,
        ["bakery_cluster", BAKERY_COL, CATEGORY_COL, SKU_COL],
        "bakery_share_in_bakery_cluster_sku",
        "mean_bakery_share_in_bakery_cluster_sku",
    )

    path_lookup = path_scores[
        [
            BAKERY_COL,
            BAKERY_NAME_COL,
            CITY_COL,
            CATEGORY_COL,
            SKU_COL,
            SKU_NAME_COL,
            "sku_cluster",
            "bakery_cluster",
            "best_decomposition_path",
            "best_path_score",
            "path_confidence",
        ]
    ].drop_duplicates()

    bakery_category_lookup = bakery_category_daily[
        [DATE_COL, BAKERY_COL, CATEGORY_COL, "category_sales_qty"]
    ].drop_duplicates()
    city_sku_lookup = city_sku_day[
        [DATE_COL, CITY_COL, CATEGORY_COL, SKU_COL, SKU_NAME_COL, "city_sku_sales_qty"]
    ].drop_duplicates()
    bakery_sku_cluster_lookup = bakery_sku_cluster_day[
        [DATE_COL, BAKERY_COL, CATEGORY_COL, "sku_cluster", "bakery_sku_cluster_sales_qty"]
    ].drop_duplicates()
    bakery_cluster_sku_lookup = bakery_cluster_sku_day[
        [DATE_COL, "bakery_cluster", CITY_COL, CATEGORY_COL, SKU_COL, SKU_NAME_COL, "bakery_cluster_sku_sales_qty"]
    ].drop_duplicates()

    work = base.merge(
        path_lookup,
        on=[BAKERY_COL, BAKERY_NAME_COL, CITY_COL, CATEGORY_COL, SKU_COL, SKU_NAME_COL],
        how="left",
        validate="many_to_one",
    )
    work = work.merge(
        bakery_category_lookup,
        on=[DATE_COL, BAKERY_COL, CATEGORY_COL],
        how="left",
        validate="many_to_one",
    )
    work = work.merge(
        city_sku_lookup,
        on=[DATE_COL, CITY_COL, CATEGORY_COL, SKU_COL, SKU_NAME_COL],
        how="left",
        validate="many_to_one",
    )
    work = work.merge(
        bakery_sku_cluster_lookup,
        on=[DATE_COL, BAKERY_COL, CATEGORY_COL, "sku_cluster"],
        how="left",
        validate="many_to_one",
    )
    work = work.merge(
        bakery_cluster_sku_lookup,
        on=[DATE_COL, "bakery_cluster", CITY_COL, CATEGORY_COL, SKU_COL, SKU_NAME_COL],
        how="left",
        validate="many_to_one",
    )

    work = work.merge(
        bakery_category_share_profile,
        on=[BAKERY_COL, CATEGORY_COL],
        how="left",
        validate="many_to_one",
    )
    work = work.merge(
        sku_cluster_category_share_profile,
        on=[BAKERY_COL, CATEGORY_COL, "sku_cluster"],
        how="left",
        validate="many_to_one",
    )
    work = work.merge(
        sku_in_cluster_share_profile,
        on=[BAKERY_COL, CATEGORY_COL, "sku_cluster", SKU_COL],
        how="left",
        validate="many_to_one",
    )
    work = work.merge(
        city_sku_share_profile,
        on=[BAKERY_COL, CATEGORY_COL, SKU_COL],
        how="left",
        validate="many_to_one",
    )
    work = work.merge(
        bakery_cluster_sku_share_profile,
        on=["bakery_cluster", BAKERY_COL, CATEGORY_COL, SKU_COL],
        how="left",
        validate="many_to_one",
    )

    work["recon_bakery_total_to_category_to_sku_cluster_to_sku"] = (
        work["bakery_total_sales_qty"]
        * work["mean_bakery_category_share_in_total"]
        * work["mean_sku_cluster_share_in_bakery_category"]
        * work["mean_sku_share_in_bakery_sku_cluster"]
    )
    work["recon_bakery_category_to_sku_cluster_to_sku"] = (
        work["category_sales_qty"]
        * work["mean_sku_cluster_share_in_bakery_category"]
        * work["mean_sku_share_in_bakery_sku_cluster"]
    )
    work["recon_city_sku_to_bakery"] = (
        work["city_sku_sales_qty"]
        * work["mean_bakery_share_in_city_sku"]
    )
    work["recon_bakery_cluster_sku_to_bakery"] = (
        work["bakery_cluster_sku_sales_qty"]
        * work["mean_bakery_share_in_bakery_cluster_sku"]
    )

    path_to_col = {
        "bakery_total_to_category_to_sku_cluster_to_sku": "recon_bakery_total_to_category_to_sku_cluster_to_sku",
        "bakery_category_to_sku_cluster_to_sku": "recon_bakery_category_to_sku_cluster_to_sku",
        "city_sku_to_bakery": "recon_city_sku_to_bakery",
        "bakery_cluster_sku_to_bakery": "recon_bakery_cluster_sku_to_bakery",
    }
    work["reconstructed_sales_qty"] = np.nan
    for path_name, col_name in path_to_col.items():
        mask = work["best_decomposition_path"] == path_name
        work.loc[mask, "reconstructed_sales_qty"] = work.loc[mask, col_name]

    # Fallback to category-based reconstruction if best path is missing.
    fallback = work["recon_bakery_category_to_sku_cluster_to_sku"]
    work["reconstructed_sales_qty"] = work["reconstructed_sales_qty"].fillna(fallback)
    work["reconstructed_sales_qty"] = pd.to_numeric(work["reconstructed_sales_qty"], errors="coerce").clip(lower=0.0)
    work["reconstruction_abs_gap"] = (work["reconstructed_sales_qty"] - work["observed_sales_qty"]).abs()
    work["reconstruction_bias"] = work["reconstructed_sales_qty"] - work["observed_sales_qty"]

    return work.sort_values([BAKERY_COL, SKU_COL, DATE_COL]).reset_index(drop=True)


def build_summary(reconstructed: pd.DataFrame) -> dict[str, object]:
    summary = {
        "rows": int(len(reconstructed)),
        "dates": int(reconstructed[DATE_COL].nunique()) if not reconstructed.empty else 0,
        "bakeries": int(reconstructed[BAKERY_COL].nunique()) if not reconstructed.empty else 0,
        "sku": int(reconstructed[SKU_COL].nunique()) if not reconstructed.empty else 0,
        "mean_observed_sales": round(float(pd.to_numeric(reconstructed["observed_sales_qty"], errors="coerce").mean()), 6) if not reconstructed.empty else 0.0,
        "mean_reconstructed_sales": round(float(pd.to_numeric(reconstructed["reconstructed_sales_qty"], errors="coerce").mean()), 6) if not reconstructed.empty else 0.0,
        "mean_abs_gap": round(float(pd.to_numeric(reconstructed["reconstruction_abs_gap"], errors="coerce").mean()), 6) if not reconstructed.empty else 0.0,
        "mean_bias": round(float(pd.to_numeric(reconstructed["reconstruction_bias"], errors="coerce").mean()), 6) if not reconstructed.empty else 0.0,
        "path_counts": reconstructed["best_decomposition_path"].value_counts(dropna=False).to_dict(),
    }
    return summary


def save_outputs(output_dir: str | Path, reconstructed: pd.DataFrame, summary: dict[str, object]) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / OUTPUT_NAME
    summary_path = out_dir / SUMMARY_OUTPUT
    reconstructed.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"reconstructed": csv_path, "summary": summary_path}


def build_kazan_reconstructed_sku_day(
    *,
    sku_daily_path: str | Path,
    bakery_category_daily_path: str | Path,
    bakery_category_share_in_total_daily_path: str | Path,
    city_sku_day_path: str | Path,
    bakery_share_in_city_sku_daily_path: str | Path,
    bakery_sku_cluster_day_path: str | Path,
    sku_cluster_share_in_bakery_category_daily_path: str | Path,
    sku_share_in_bakery_sku_cluster_daily_path: str | Path,
    bakery_cluster_sku_day_path: str | Path,
    bakery_share_in_bakery_cluster_sku_daily_path: str | Path,
    path_scores_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    sku_daily = load_csv(sku_daily_path)
    bakery_category_daily = load_csv(bakery_category_daily_path)
    bakery_category_share_in_total_daily = load_csv(bakery_category_share_in_total_daily_path)
    city_sku_day = load_csv(city_sku_day_path)
    bakery_share_in_city_sku_daily = load_csv(bakery_share_in_city_sku_daily_path)
    bakery_sku_cluster_day = load_csv(bakery_sku_cluster_day_path)
    sku_cluster_share_in_bakery_category_daily = load_csv(sku_cluster_share_in_bakery_category_daily_path)
    sku_share_in_bakery_sku_cluster_daily = load_csv(sku_share_in_bakery_sku_cluster_daily_path)
    bakery_cluster_sku_day = load_csv(bakery_cluster_sku_day_path)
    bakery_share_in_bakery_cluster_sku_daily = load_csv(bakery_share_in_bakery_cluster_sku_daily_path)
    path_scores = load_csv(path_scores_path)

    reconstructed = build_reconstructed_sku_day(
        sku_daily=sku_daily,
        bakery_category_daily=bakery_category_daily,
        bakery_category_share_in_total_daily=bakery_category_share_in_total_daily,
        city_sku_day=city_sku_day,
        bakery_share_in_city_sku_daily=bakery_share_in_city_sku_daily,
        bakery_sku_cluster_day=bakery_sku_cluster_day,
        sku_cluster_share_in_bakery_category_daily=sku_cluster_share_in_bakery_category_daily,
        sku_share_in_bakery_sku_cluster_daily=sku_share_in_bakery_sku_cluster_daily,
        bakery_cluster_sku_day=bakery_cluster_sku_day,
        bakery_share_in_bakery_cluster_sku_daily=bakery_share_in_bakery_cluster_sku_daily,
        path_scores=path_scores,
    )
    summary = build_summary(reconstructed)
    return save_outputs(output_dir, reconstructed, summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rule-based reconstructed sku-day series for Kazan sample")
    parser.add_argument("--sku-daily-path", default=str(ROOT / "data" / "processed" / "kazan_sitnaya_daily_sample.csv"))
    parser.add_argument("--bakery-category-daily-path", default=str(ROOT / "data" / "processed" / "kazan_sitnaya_bakery_category_daily_sample.csv"))
    parser.add_argument("--bakery-category-share-in-total-daily-path", default=str(ROOT / "data" / "processed" / "kazan_bakery_category_share_in_total_daily.csv"))
    parser.add_argument("--city-sku-day-path", default=str(ROOT / "data" / "processed" / "kazan_city_sku_day.csv"))
    parser.add_argument("--bakery-share-in-city-sku-daily-path", default=str(ROOT / "data" / "processed" / "kazan_bakery_share_in_city_sku_daily.csv"))
    parser.add_argument("--bakery-sku-cluster-day-path", default=str(ROOT / "data" / "processed" / "kazan_bakery_sku_cluster_day.csv"))
    parser.add_argument("--sku-cluster-share-in-bakery-category-daily-path", default=str(ROOT / "data" / "processed" / "kazan_sku_cluster_share_in_bakery_category_daily.csv"))
    parser.add_argument("--sku-share-in-bakery-sku-cluster-daily-path", default=str(ROOT / "data" / "processed" / "kazan_sku_share_in_bakery_sku_cluster_daily.csv"))
    parser.add_argument("--bakery-cluster-sku-day-path", default=str(ROOT / "data" / "processed" / "kazan_bakery_cluster_sku_day.csv"))
    parser.add_argument("--bakery-share-in-bakery-cluster-sku-daily-path", default=str(ROOT / "data" / "processed" / "kazan_bakery_share_in_bakery_cluster_sku_daily.csv"))
    parser.add_argument("--path-scores-path", default=str(ROOT / "data" / "processed" / "kazan_decomposition_path_scores.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "processed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_kazan_reconstructed_sku_day(
        sku_daily_path=args.sku_daily_path,
        bakery_category_daily_path=args.bakery_category_daily_path,
        bakery_category_share_in_total_daily_path=args.bakery_category_share_in_total_daily_path,
        city_sku_day_path=args.city_sku_day_path,
        bakery_share_in_city_sku_daily_path=args.bakery_share_in_city_sku_daily_path,
        bakery_sku_cluster_day_path=args.bakery_sku_cluster_day_path,
        sku_cluster_share_in_bakery_category_daily_path=args.sku_cluster_share_in_bakery_category_daily_path,
        sku_share_in_bakery_sku_cluster_daily_path=args.sku_share_in_bakery_sku_cluster_daily_path,
        bakery_cluster_sku_day_path=args.bakery_cluster_sku_day_path,
        bakery_share_in_bakery_cluster_sku_daily_path=args.bakery_share_in_bakery_cluster_sku_daily_path,
        path_scores_path=args.path_scores_path,
        output_dir=args.output_dir,
    )
    print("=" * 72)
    print("KAZAN RECONSTRUCTED SKU DAY")
    print("=" * 72)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

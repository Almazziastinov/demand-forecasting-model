from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATE_COL = "date"
CITY_COL = "city"
BAKERY_COL = "bakery_id"
BAKERY_NAME_COL = "bakery_name"
CATEGORY_COL = "category_name"
SKU_COL = "product_id"
SKU_NAME_COL = "product_name"

OUTPUT_FILES = {
    "city_category_day": "kazan_city_category_day.csv",
    "city_sku_day": "kazan_city_sku_day.csv",
    "bakery_sku_cluster_day": "kazan_bakery_sku_cluster_day.csv",
    "bakery_cluster_sku_day": "kazan_bakery_cluster_sku_day.csv",
    "bakery_cluster_category_day": "kazan_bakery_cluster_category_day.csv",
    "bakery_category_share_in_total_daily": "kazan_bakery_category_share_in_total_daily.csv",
    "sku_cluster_share_in_bakery_category_daily": "kazan_sku_cluster_share_in_bakery_category_daily.csv",
    "sku_share_in_bakery_sku_cluster_daily": "kazan_sku_share_in_bakery_sku_cluster_daily.csv",
    "bakery_share_in_city_sku_daily": "kazan_bakery_share_in_city_sku_daily.csv",
    "bakery_share_in_bakery_cluster_sku_daily": "kazan_bakery_share_in_bakery_cluster_sku_daily.csv",
}
SUMMARY_OUTPUT = "kazan_decomposition_levels_summary.json"


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    return df


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


def build_decomposition_levels(
    *,
    bakery_daily: pd.DataFrame,
    bakery_category_daily: pd.DataFrame,
    sku_daily: pd.DataFrame,
    bakery_clusters: pd.DataFrame,
    sku_clusters: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    bakery_cluster_lookup = bakery_clusters[[BAKERY_COL, "bakery_cluster"]].drop_duplicates()
    sku_cluster_lookup = sku_clusters[[BAKERY_COL, SKU_COL, "sku_cluster"]].drop_duplicates()

    sku_work = sku_daily.merge(bakery_cluster_lookup, on=BAKERY_COL, how="left", validate="many_to_one")
    sku_work = sku_work.merge(sku_cluster_lookup, on=[BAKERY_COL, SKU_COL], how="left", validate="many_to_one")

    category_work = bakery_category_daily.merge(bakery_cluster_lookup, on=BAKERY_COL, how="left", validate="many_to_one")

    city_category_day = (
        category_work.groupby([DATE_COL, CITY_COL, CATEGORY_COL], as_index=False)
        .agg(city_category_sales_qty=("category_sales_qty", "sum"))
    )
    city_sku_day = (
        sku_work.groupby([DATE_COL, CITY_COL, CATEGORY_COL, SKU_COL, SKU_NAME_COL], as_index=False)
        .agg(city_sku_sales_qty=("observed_sales_qty", "sum"))
    )
    bakery_sku_cluster_day = (
        sku_work.groupby([DATE_COL, BAKERY_COL, BAKERY_NAME_COL, CITY_COL, CATEGORY_COL, "sku_cluster"], as_index=False)
        .agg(bakery_sku_cluster_sales_qty=("observed_sales_qty", "sum"))
    )
    bakery_cluster_sku_day = (
        sku_work.groupby([DATE_COL, "bakery_cluster", CITY_COL, CATEGORY_COL, SKU_COL, SKU_NAME_COL], as_index=False)
        .agg(bakery_cluster_sku_sales_qty=("observed_sales_qty", "sum"))
    )
    bakery_cluster_category_day = (
        category_work.groupby([DATE_COL, "bakery_cluster", CITY_COL, CATEGORY_COL], as_index=False)
        .agg(bakery_cluster_category_sales_qty=("category_sales_qty", "sum"))
    )

    bakery_category_share_in_total_daily = category_work.copy()
    bakery_category_share_in_total_daily["bakery_category_share_in_total"] = _safe_divide(
        bakery_category_share_in_total_daily["category_sales_qty"],
        bakery_category_share_in_total_daily["bakery_total_sales_qty"],
    )
    bakery_category_share_in_total_daily = bakery_category_share_in_total_daily[
        [
            DATE_COL,
            BAKERY_COL,
            BAKERY_NAME_COL,
            CITY_COL,
            CATEGORY_COL,
            "category_sales_qty",
            "bakery_total_sales_qty",
            "bakery_category_share_in_total",
        ]
    ].copy()

    sku_cluster_share_in_bakery_category_daily = bakery_sku_cluster_day.merge(
        category_work[[DATE_COL, BAKERY_COL, CATEGORY_COL, "category_sales_qty"]].drop_duplicates(),
        on=[DATE_COL, BAKERY_COL, CATEGORY_COL],
        how="left",
        validate="many_to_one",
    )
    sku_cluster_share_in_bakery_category_daily["sku_cluster_share_in_bakery_category"] = _safe_divide(
        sku_cluster_share_in_bakery_category_daily["bakery_sku_cluster_sales_qty"],
        sku_cluster_share_in_bakery_category_daily["category_sales_qty"],
    )

    sku_share_in_bakery_sku_cluster_daily = sku_work.merge(
        bakery_sku_cluster_day[
            [DATE_COL, BAKERY_COL, CATEGORY_COL, "sku_cluster", "bakery_sku_cluster_sales_qty"]
        ],
        on=[DATE_COL, BAKERY_COL, CATEGORY_COL, "sku_cluster"],
        how="left",
        validate="many_to_one",
    )
    sku_share_in_bakery_sku_cluster_daily["sku_share_in_bakery_sku_cluster"] = _safe_divide(
        sku_share_in_bakery_sku_cluster_daily["observed_sales_qty"],
        sku_share_in_bakery_sku_cluster_daily["bakery_sku_cluster_sales_qty"],
    )
    sku_share_in_bakery_sku_cluster_daily = sku_share_in_bakery_sku_cluster_daily[
        [
            DATE_COL,
            BAKERY_COL,
            BAKERY_NAME_COL,
            CITY_COL,
            CATEGORY_COL,
            "sku_cluster",
            SKU_COL,
            SKU_NAME_COL,
            "observed_sales_qty",
            "bakery_sku_cluster_sales_qty",
            "sku_share_in_bakery_sku_cluster",
        ]
    ].copy()

    bakery_share_in_city_sku_daily = sku_work.merge(
        city_sku_day,
        on=[DATE_COL, CITY_COL, CATEGORY_COL, SKU_COL, SKU_NAME_COL],
        how="left",
        validate="many_to_one",
    )
    bakery_share_in_city_sku_daily["bakery_share_in_city_sku"] = _safe_divide(
        bakery_share_in_city_sku_daily["observed_sales_qty"],
        bakery_share_in_city_sku_daily["city_sku_sales_qty"],
    )
    bakery_share_in_city_sku_daily = bakery_share_in_city_sku_daily[
        [
            DATE_COL,
            BAKERY_COL,
            BAKERY_NAME_COL,
            CITY_COL,
            CATEGORY_COL,
            SKU_COL,
            SKU_NAME_COL,
            "observed_sales_qty",
            "city_sku_sales_qty",
            "bakery_share_in_city_sku",
        ]
    ].copy()

    bakery_share_in_bakery_cluster_sku_daily = sku_work.merge(
        bakery_cluster_sku_day,
        on=[DATE_COL, "bakery_cluster", CITY_COL, CATEGORY_COL, SKU_COL, SKU_NAME_COL],
        how="left",
        validate="many_to_one",
    )
    bakery_share_in_bakery_cluster_sku_daily["bakery_share_in_bakery_cluster_sku"] = _safe_divide(
        bakery_share_in_bakery_cluster_sku_daily["observed_sales_qty"],
        bakery_share_in_bakery_cluster_sku_daily["bakery_cluster_sku_sales_qty"],
    )
    bakery_share_in_bakery_cluster_sku_daily = bakery_share_in_bakery_cluster_sku_daily[
        [
            DATE_COL,
            BAKERY_COL,
            BAKERY_NAME_COL,
            CITY_COL,
            "bakery_cluster",
            CATEGORY_COL,
            SKU_COL,
            SKU_NAME_COL,
            "observed_sales_qty",
            "bakery_cluster_sku_sales_qty",
            "bakery_share_in_bakery_cluster_sku",
        ]
    ].copy()

    return {
        "city_category_day": city_category_day,
        "city_sku_day": city_sku_day,
        "bakery_sku_cluster_day": bakery_sku_cluster_day,
        "bakery_cluster_sku_day": bakery_cluster_sku_day,
        "bakery_cluster_category_day": bakery_cluster_category_day,
        "bakery_category_share_in_total_daily": bakery_category_share_in_total_daily,
        "sku_cluster_share_in_bakery_category_daily": sku_cluster_share_in_bakery_category_daily,
        "sku_share_in_bakery_sku_cluster_daily": sku_share_in_bakery_sku_cluster_daily,
        "bakery_share_in_city_sku_daily": bakery_share_in_city_sku_daily,
        "bakery_share_in_bakery_cluster_sku_daily": bakery_share_in_bakery_cluster_sku_daily,
    }


def build_summary(outputs: dict[str, pd.DataFrame]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for name, df in outputs.items():
        summary[name] = {
            "rows": int(len(df)),
            "dates": int(df[DATE_COL].nunique()) if DATE_COL in df.columns and not df.empty else 0,
        }
        share_cols = [col for col in df.columns if col.endswith("_share_in_total") or col.endswith("_share_in_bakery_category") or col.endswith("_share_in_bakery_sku_cluster") or col.endswith("_share_in_city_sku") or col.endswith("_share_in_bakery_cluster_sku")]
        if share_cols:
            for col in share_cols:
                summary[name][f"{col}_mean"] = round(float(pd.to_numeric(df[col], errors="coerce").mean()), 6) if not df.empty else 0.0
    return summary


def save_outputs(output_dir: str | Path, outputs: dict[str, pd.DataFrame], summary: dict[str, object]) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, df in outputs.items():
        path = out_dir / OUTPUT_FILES[name]
        df.to_csv(path, index=False, encoding="utf-8-sig")
        paths[name] = path
    summary_path = out_dir / SUMMARY_OUTPUT
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["summary"] = summary_path
    return paths


def build_kazan_decomposition_levels(
    *,
    bakery_daily_path: str | Path,
    bakery_category_daily_path: str | Path,
    sku_daily_path: str | Path,
    bakery_clusters_path: str | Path,
    sku_clusters_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    bakery_daily = load_csv(bakery_daily_path)
    bakery_category_daily = load_csv(bakery_category_daily_path)
    sku_daily = load_csv(sku_daily_path)
    bakery_clusters = load_csv(bakery_clusters_path)
    sku_clusters = load_csv(sku_clusters_path)

    outputs = build_decomposition_levels(
        bakery_daily=bakery_daily,
        bakery_category_daily=bakery_category_daily,
        sku_daily=sku_daily,
        bakery_clusters=bakery_clusters,
        sku_clusters=sku_clusters,
    )
    summary = build_summary(outputs)
    return save_outputs(output_dir, outputs, summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Kazan decomposition levels and share tables")
    parser.add_argument("--bakery-daily-path", default=str(ROOT / "data" / "processed" / "kazan_bakery_daily_sample.csv"))
    parser.add_argument("--bakery-category-daily-path", default=str(ROOT / "data" / "processed" / "kazan_sitnaya_bakery_category_daily_sample.csv"))
    parser.add_argument("--sku-daily-path", default=str(ROOT / "data" / "processed" / "kazan_sitnaya_daily_sample.csv"))
    parser.add_argument("--bakery-clusters-path", default=str(ROOT / "data" / "processed" / "kazan_bakery_clusters.csv"))
    parser.add_argument("--sku-clusters-path", default=str(ROOT / "data" / "processed" / "kazan_sitnaya_sku_clusters.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "processed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_kazan_decomposition_levels(
        bakery_daily_path=args.bakery_daily_path,
        bakery_category_daily_path=args.bakery_category_daily_path,
        sku_daily_path=args.sku_daily_path,
        bakery_clusters_path=args.bakery_clusters_path,
        sku_clusters_path=args.sku_clusters_path,
        output_dir=args.output_dir,
    )
    print("=" * 72)
    print("KAZAN DECOMPOSITION LEVELS")
    print("=" * 72)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

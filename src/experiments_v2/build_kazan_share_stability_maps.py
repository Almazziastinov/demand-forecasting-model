from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATE_COL = "date"
DOW_COL = "dow"

OUTPUT_FILES = {
    "bakery_share_in_city_sku": "kazan_bakery_share_in_city_sku_stability.csv",
    "sku_cluster_share_in_bakery_category": "kazan_sku_cluster_share_in_bakery_category_stability.csv",
    "sku_share_in_bakery_sku_cluster": "kazan_sku_share_in_bakery_sku_cluster_stability.csv",
    "bakery_share_in_bakery_cluster_sku": "kazan_bakery_share_in_bakery_cluster_sku_stability.csv",
}
SUMMARY_OUTPUT = "kazan_share_stability_summary.json"


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        if DOW_COL not in df.columns:
            df[DOW_COL] = df[DATE_COL].dt.dayofweek
    return df


def _weekday_profile_stability(group: pd.DataFrame, value_col: str) -> float:
    work = group[[DATE_COL, DOW_COL, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[DATE_COL, DOW_COL, value_col])
    if len(work) < 7:
        return np.nan

    work["iso_week"] = work[DATE_COL].dt.isocalendar().week.astype(int)
    work["iso_year"] = work[DATE_COL].dt.isocalendar().year.astype(int)
    weekly = (
        work.groupby(["iso_year", "iso_week", DOW_COL], observed=True)[value_col]
        .mean()
        .unstack(fill_value=0.0)
        .reindex(columns=range(7), fill_value=0.0)
    )
    if len(weekly) < 2:
        return np.nan

    global_profile = weekly.mean(axis=0).to_numpy(dtype=float)
    if np.nansum(global_profile) <= 0:
        return np.nan

    distances = []
    for _, row in weekly.iterrows():
        row_vec = row.to_numpy(dtype=float)
        dist = np.abs(row_vec - global_profile).mean()
        distances.append(dist)
    if not distances:
        return np.nan
    return float(np.clip(1.0 - float(np.mean(distances)), 0.0, 1.0))


def _trend_metrics(group: pd.DataFrame, value_col: str) -> tuple[float, float]:
    work = group[[DATE_COL, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[DATE_COL, value_col])
    if len(work) < 3:
        return np.nan, np.nan
    x = (work[DATE_COL] - work[DATE_COL].min()).dt.days.astype(float).to_numpy()
    y = work[value_col].astype(float).to_numpy()
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan, np.nan
    corr = float(np.corrcoef(x, y)[0, 1])
    slope = float(np.polyfit(x, y, deg=1)[0])
    return corr, slope


def _build_stability_map(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    share_col: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for key, group in df.groupby(group_cols, observed=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        group = group.sort_values(DATE_COL).copy()
        values = pd.to_numeric(group[share_col], errors="coerce")
        valid = values.notna()
        valid_values = values[valid]
        trend_corr, trend_slope = _trend_metrics(group, share_col)

        row = {col: val for col, val in zip(group_cols, key)}
        row.update(
            {
                "observed_days": int(valid.sum()),
                "date_min": None if group.empty else group[DATE_COL].min(),
                "date_max": None if group.empty else group[DATE_COL].max(),
                "mean_share": float(valid_values.mean()) if len(valid_values) else np.nan,
                "median_share": float(valid_values.median()) if len(valid_values) else np.nan,
                "std_share": float(valid_values.std(ddof=0)) if len(valid_values) else np.nan,
                "cv_share": (
                    float(valid_values.std(ddof=0) / valid_values.mean())
                    if len(valid_values) and abs(float(valid_values.mean())) > 1e-12
                    else np.nan
                ),
                "min_share": float(valid_values.min()) if len(valid_values) else np.nan,
                "max_share": float(valid_values.max()) if len(valid_values) else np.nan,
                "weekday_share_stability": _weekday_profile_stability(group, share_col),
                "trend_corr": trend_corr,
                "trend_slope": trend_slope,
            }
        )
        records.append(row)
    return pd.DataFrame.from_records(records)


def build_kazan_share_stability_maps(
    *,
    bakery_share_in_city_sku_path: str | Path,
    sku_cluster_share_in_bakery_category_path: str | Path,
    sku_share_in_bakery_sku_cluster_path: str | Path,
    bakery_share_in_bakery_cluster_sku_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    bakery_share_in_city_sku = load_csv(bakery_share_in_city_sku_path)
    sku_cluster_share_in_bakery_category = load_csv(sku_cluster_share_in_bakery_category_path)
    sku_share_in_bakery_sku_cluster = load_csv(sku_share_in_bakery_sku_cluster_path)
    bakery_share_in_bakery_cluster_sku = load_csv(bakery_share_in_bakery_cluster_sku_path)

    outputs = {
        "bakery_share_in_city_sku": _build_stability_map(
            bakery_share_in_city_sku,
            group_cols=["bakery_id", "bakery_name", "city", "category_name", "product_id", "product_name"],
            share_col="bakery_share_in_city_sku",
        ),
        "sku_cluster_share_in_bakery_category": _build_stability_map(
            sku_cluster_share_in_bakery_category,
            group_cols=["bakery_id", "bakery_name", "city", "category_name", "sku_cluster"],
            share_col="sku_cluster_share_in_bakery_category",
        ),
        "sku_share_in_bakery_sku_cluster": _build_stability_map(
            sku_share_in_bakery_sku_cluster,
            group_cols=["bakery_id", "bakery_name", "city", "category_name", "sku_cluster", "product_id", "product_name"],
            share_col="sku_share_in_bakery_sku_cluster",
        ),
        "bakery_share_in_bakery_cluster_sku": _build_stability_map(
            bakery_share_in_bakery_cluster_sku,
            group_cols=["bakery_cluster", "bakery_id", "bakery_name", "city", "category_name", "product_id", "product_name"],
            share_col="bakery_share_in_bakery_cluster_sku",
        ),
    }

    summary: dict[str, object] = {}
    for name, df in outputs.items():
        summary[name] = {
            "rows": int(len(df)),
            "mean_share": round(float(pd.to_numeric(df["mean_share"], errors="coerce").mean()), 6) if not df.empty else 0.0,
            "mean_cv_share": round(float(pd.to_numeric(df["cv_share"], errors="coerce").mean()), 6) if not df.empty else 0.0,
            "mean_weekday_share_stability": round(float(pd.to_numeric(df["weekday_share_stability"], errors="coerce").mean()), 6) if not df.empty else 0.0,
        }

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build share stability maps for Kazan decomposition levels")
    parser.add_argument("--bakery-share-in-city-sku-path", default=str(ROOT / "data" / "processed" / "kazan_bakery_share_in_city_sku_daily.csv"))
    parser.add_argument("--sku-cluster-share-in-bakery-category-path", default=str(ROOT / "data" / "processed" / "kazan_sku_cluster_share_in_bakery_category_daily.csv"))
    parser.add_argument("--sku-share-in-bakery-sku-cluster-path", default=str(ROOT / "data" / "processed" / "kazan_sku_share_in_bakery_sku_cluster_daily.csv"))
    parser.add_argument("--bakery-share-in-bakery-cluster-sku-path", default=str(ROOT / "data" / "processed" / "kazan_bakery_share_in_bakery_cluster_sku_daily.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "processed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_kazan_share_stability_maps(
        bakery_share_in_city_sku_path=args.bakery_share_in_city_sku_path,
        sku_cluster_share_in_bakery_category_path=args.sku_cluster_share_in_bakery_category_path,
        sku_share_in_bakery_sku_cluster_path=args.sku_share_in_bakery_sku_cluster_path,
        bakery_share_in_bakery_cluster_sku_path=args.bakery_share_in_bakery_cluster_sku_path,
        output_dir=args.output_dir,
    )
    print("=" * 72)
    print("KAZAN SHARE STABILITY MAPS")
    print("=" * 72)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

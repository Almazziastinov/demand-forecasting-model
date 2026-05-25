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
DOW_COL = "dow"
HOUR_COL = "hour"

OUTPUT_NAME = "kazan_anchor_suitability_map.csv"
SUMMARY_OUTPUT_NAME = "kazan_anchor_suitability_summary.json"


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    if DOW_COL not in df.columns and DATE_COL in df.columns:
        df[DOW_COL] = df[DATE_COL].dt.dayofweek
    return df


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    valid = x.notna() & y.notna()
    if valid.sum() < 3:
        return np.nan
    xv = x[valid]
    yv = y[valid]
    if xv.nunique() < 2 or yv.nunique() < 2:
        return np.nan
    return float(np.corrcoef(xv, yv)[0, 1])


def _corr_to_score(value: float) -> float:
    if pd.isna(value):
        return np.nan
    return float(np.clip((value + 1.0) / 2.0, 0.0, 1.0))


def _l1_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.isnan(a).all() or np.isnan(b).all():
        return np.nan
    a = np.nan_to_num(a, nan=0.0)
    b = np.nan_to_num(b, nan=0.0)
    a_sum = a.sum()
    b_sum = b.sum()
    if a_sum <= 0 or b_sum <= 0:
        return np.nan
    a = a / a_sum
    b = b / b_sum
    distance = np.abs(a - b).sum() / 2.0
    return float(np.clip(1.0 - distance, 0.0, 1.0))


def _build_share_profile(
    df: pd.DataFrame,
    *,
    key_cols: list[str],
    bucket_col: str,
    value_col: str,
    bucket_range: range,
) -> pd.DataFrame:
    grouped = (
        df.groupby(key_cols + [bucket_col], observed=True, as_index=False)[value_col]
        .sum()
    )
    pivot = (
        grouped.pivot_table(index=key_cols, columns=bucket_col, values=value_col, aggfunc="sum", fill_value=0.0)
        .reindex(columns=list(bucket_range), fill_value=0.0)
        .reset_index()
    )
    value_cols = [col for col in pivot.columns if col not in key_cols]
    total = pivot[value_cols].sum(axis=1)
    for col in value_cols:
        pivot[col] = np.where(total > 0, pivot[col] / total, np.nan)
    rename_map = {col: f"share_{bucket_col}_{col}" for col in value_cols}
    return pivot.rename(columns=rename_map)


def _extract_share_vector(row: pd.Series, prefix: str, bucket_range: range) -> np.ndarray:
    cols = [f"{prefix}_{bucket}" for bucket in bucket_range]
    return row.reindex(cols).astype(float).to_numpy()


def _mean_available(values: list[float]) -> float:
    valid = [value for value in values if pd.notna(value)]
    if not valid:
        return np.nan
    return float(np.mean(valid))


def build_anchor_suitability_map(
    *,
    bakery_daily: pd.DataFrame,
    bakery_category_daily: pd.DataFrame,
    sku_daily: pd.DataFrame,
    sku_hourly: pd.DataFrame,
    bakery_clusters: pd.DataFrame,
    sku_clusters: pd.DataFrame,
    sku_profile_map: pd.DataFrame,
) -> pd.DataFrame:
    bakery_cluster_lookup = bakery_clusters[[BAKERY_COL, "bakery_cluster"]].drop_duplicates()
    sku_cluster_lookup = sku_clusters[[BAKERY_COL, SKU_COL, "sku_cluster"]].drop_duplicates()

    sku_daily = sku_daily.merge(bakery_cluster_lookup, on=BAKERY_COL, how="left", validate="many_to_one")
    sku_daily = sku_daily.merge(sku_cluster_lookup, on=[BAKERY_COL, SKU_COL], how="left", validate="many_to_one")
    if "category_sales_qty" not in sku_daily.columns:
        sku_daily = sku_daily.merge(
            bakery_category_daily[[DATE_COL, BAKERY_COL, "category_sales_qty"]].drop_duplicates(),
            on=[DATE_COL, BAKERY_COL],
            how="left",
            validate="many_to_one",
        )
    sku_hourly = sku_hourly.merge(bakery_cluster_lookup, on=BAKERY_COL, how="left", validate="many_to_one")
    sku_hourly = sku_hourly.merge(sku_cluster_lookup, on=[BAKERY_COL, SKU_COL], how="left", validate="many_to_one")

    bakery_weekday = _build_share_profile(
        bakery_daily,
        key_cols=[BAKERY_COL],
        bucket_col=DOW_COL,
        value_col="bakery_sales",
        bucket_range=range(7),
    )
    bakery_category_weekday = _build_share_profile(
        bakery_category_daily,
        key_cols=[BAKERY_COL],
        bucket_col=DOW_COL,
        value_col="category_sales_qty",
        bucket_range=range(7),
    )
    bakery_cluster_weekday = _build_share_profile(
        sku_daily,
        key_cols=["bakery_cluster"],
        bucket_col=DOW_COL,
        value_col="observed_sales_qty",
        bucket_range=range(7),
    )
    sku_cluster_weekday = _build_share_profile(
        sku_daily,
        key_cols=["sku_cluster"],
        bucket_col=DOW_COL,
        value_col="observed_sales_qty",
        bucket_range=range(7),
    )
    local_weekday = _build_share_profile(
        sku_daily,
        key_cols=[BAKERY_COL, SKU_COL],
        bucket_col=DOW_COL,
        value_col="observed_sales_qty",
        bucket_range=range(7),
    )

    bakery_category_hour = _build_share_profile(
        sku_hourly.groupby([DATE_COL, BAKERY_COL, HOUR_COL], as_index=False)["sku_hour_sales"].sum(),
        key_cols=[BAKERY_COL],
        bucket_col=HOUR_COL,
        value_col="sku_hour_sales",
        bucket_range=range(24),
    )
    bakery_cluster_hour = _build_share_profile(
        sku_hourly,
        key_cols=["bakery_cluster"],
        bucket_col=HOUR_COL,
        value_col="sku_hour_sales",
        bucket_range=range(24),
    )
    sku_cluster_hour = _build_share_profile(
        sku_hourly,
        key_cols=["sku_cluster"],
        bucket_col=HOUR_COL,
        value_col="sku_hour_sales",
        bucket_range=range(24),
    )
    local_hour = _build_share_profile(
        sku_hourly,
        key_cols=[BAKERY_COL, SKU_COL],
        bucket_col=HOUR_COL,
        value_col="sku_hour_sales",
        bucket_range=range(24),
    )

    bakery_cluster_daily = (
        sku_daily.groupby(["bakery_cluster", DATE_COL], as_index=False)
        .agg(bakery_cluster_sales_qty=("observed_sales_qty", "mean"))
    )
    sku_cluster_daily = (
        sku_daily.groupby(["sku_cluster", DATE_COL], as_index=False)
        .agg(sku_cluster_sales_qty=("observed_sales_qty", "mean"))
    )

    records: list[dict[str, object]] = []
    profile_lookup = sku_profile_map.set_index([BAKERY_COL, SKU_COL])

    for (bakery_id, sku_id), group in sku_daily.groupby([BAKERY_COL, SKU_COL], observed=True):
        group = group.sort_values(DATE_COL).copy()
        if group.empty:
            continue
        bakery_cluster = group["bakery_cluster"].iloc[0]
        sku_cluster = group["sku_cluster"].iloc[0]

        local_daily_corr_bakery = _safe_corr(group["observed_sales_qty"], group["bakery_total_sales_qty"])
        local_daily_corr_category = _safe_corr(group["observed_sales_qty"], group["category_sales_qty"])

        group_bc = group.merge(
            bakery_cluster_daily,
            on=["bakery_cluster", DATE_COL],
            how="left",
            validate="many_to_one",
        )
        group_sc = group.merge(
            sku_cluster_daily,
            on=["sku_cluster", DATE_COL],
            how="left",
            validate="many_to_one",
        )
        bakery_cluster_corr = _safe_corr(group_bc["observed_sales_qty"], group_bc["bakery_cluster_sales_qty"])
        sku_cluster_corr = _safe_corr(group_sc["observed_sales_qty"], group_sc["sku_cluster_sales_qty"])

        local_weekday_row = local_weekday[(local_weekday[BAKERY_COL] == bakery_id) & (local_weekday[SKU_COL] == sku_id)]
        bakery_weekday_row = bakery_weekday[bakery_weekday[BAKERY_COL] == bakery_id]
        bakery_category_weekday_row = bakery_category_weekday[bakery_category_weekday[BAKERY_COL] == bakery_id]
        bakery_cluster_weekday_row = bakery_cluster_weekday[bakery_cluster_weekday["bakery_cluster"] == bakery_cluster]
        sku_cluster_weekday_row = sku_cluster_weekday[sku_cluster_weekday["sku_cluster"] == sku_cluster]

        local_hour_row = local_hour[(local_hour[BAKERY_COL] == bakery_id) & (local_hour[SKU_COL] == sku_id)]
        bakery_category_hour_row = bakery_category_hour[bakery_category_hour[BAKERY_COL] == bakery_id]
        bakery_cluster_hour_row = bakery_cluster_hour[bakery_cluster_hour["bakery_cluster"] == bakery_cluster]
        sku_cluster_hour_row = sku_cluster_hour[sku_cluster_hour["sku_cluster"] == sku_cluster]

        local_weekday_vec = _extract_share_vector(local_weekday_row.iloc[0], "share_dow", range(7)) if not local_weekday_row.empty else np.full(7, np.nan)
        bakery_weekday_vec = _extract_share_vector(bakery_weekday_row.iloc[0], "share_dow", range(7)) if not bakery_weekday_row.empty else np.full(7, np.nan)
        bakery_category_weekday_vec = _extract_share_vector(bakery_category_weekday_row.iloc[0], "share_dow", range(7)) if not bakery_category_weekday_row.empty else np.full(7, np.nan)
        bakery_cluster_weekday_vec = _extract_share_vector(bakery_cluster_weekday_row.iloc[0], "share_dow", range(7)) if not bakery_cluster_weekday_row.empty else np.full(7, np.nan)
        sku_cluster_weekday_vec = _extract_share_vector(sku_cluster_weekday_row.iloc[0], "share_dow", range(7)) if not sku_cluster_weekday_row.empty else np.full(7, np.nan)

        local_hour_vec = _extract_share_vector(local_hour_row.iloc[0], "share_hour", range(24)) if not local_hour_row.empty else np.full(24, np.nan)
        bakery_category_hour_vec = _extract_share_vector(bakery_category_hour_row.iloc[0], "share_hour", range(24)) if not bakery_category_hour_row.empty else np.full(24, np.nan)
        bakery_cluster_hour_vec = _extract_share_vector(bakery_cluster_hour_row.iloc[0], "share_hour", range(24)) if not bakery_cluster_hour_row.empty else np.full(24, np.nan)
        sku_cluster_hour_vec = _extract_share_vector(sku_cluster_hour_row.iloc[0], "share_hour", range(24)) if not sku_cluster_hour_row.empty else np.full(24, np.nan)

        match_bakery_weekday = _l1_similarity(local_weekday_vec, bakery_weekday_vec)
        match_bakery_category_weekday = _l1_similarity(local_weekday_vec, bakery_category_weekday_vec)
        match_bakery_cluster_weekday = _l1_similarity(local_weekday_vec, bakery_cluster_weekday_vec)
        match_sku_cluster_weekday = _l1_similarity(local_weekday_vec, sku_cluster_weekday_vec)

        match_bakery_category_hour = _l1_similarity(local_hour_vec, bakery_category_hour_vec)
        match_bakery_cluster_hour = _l1_similarity(local_hour_vec, bakery_cluster_hour_vec)
        match_sku_cluster_hour = _l1_similarity(local_hour_vec, sku_cluster_hour_vec)

        local_profile = profile_lookup.loc[(bakery_id, sku_id)] if (bakery_id, sku_id) in profile_lookup.index else None
        local_anchor_score = _mean_available(
            [
                float(local_profile["weekday_profile_stability"]) if local_profile is not None else np.nan,
                float(local_profile["hour_profile_stability"]) if local_profile is not None else np.nan,
            ]
        )
        bakery_total_anchor_score = _mean_available(
            [_corr_to_score(local_daily_corr_bakery), match_bakery_weekday]
        )
        bakery_category_anchor_score = _mean_available(
            [_corr_to_score(local_daily_corr_category), match_bakery_category_weekday, match_bakery_category_hour]
        )
        bakery_cluster_anchor_score = _mean_available(
            [_corr_to_score(bakery_cluster_corr), match_bakery_cluster_weekday, match_bakery_cluster_hour]
        )
        sku_cluster_anchor_score = _mean_available(
            [_corr_to_score(sku_cluster_corr), match_sku_cluster_weekday, match_sku_cluster_hour]
        )

        anchor_scores = {
            "local": local_anchor_score,
            "bakery_total": bakery_total_anchor_score,
            "bakery_category": bakery_category_anchor_score,
            "bakery_cluster": bakery_cluster_anchor_score,
            "sku_cluster": sku_cluster_anchor_score,
        }
        ranked = sorted(
            [(name, score) for name, score in anchor_scores.items() if pd.notna(score)],
            key=lambda item: item[1],
            reverse=True,
        )
        best_anchor = ranked[0][0] if ranked else np.nan
        best_anchor_score = ranked[0][1] if ranked else np.nan
        second_anchor_score = ranked[1][1] if len(ranked) > 1 else np.nan
        anchor_confidence = (
            best_anchor_score - second_anchor_score
            if pd.notna(best_anchor_score) and pd.notna(second_anchor_score)
            else np.nan
        )

        records.append(
            {
                BAKERY_COL: bakery_id,
                BAKERY_NAME_COL: group[BAKERY_NAME_COL].astype(str).mode().iloc[0],
                CITY_COL: group[CITY_COL].astype(str).mode().iloc[0],
                CATEGORY_COL: group[CATEGORY_COL].astype(str).mode().iloc[0],
                SKU_COL: sku_id,
                SKU_NAME_COL: group[SKU_NAME_COL].astype(str).mode().iloc[0],
                "bakery_cluster": bakery_cluster,
                "sku_cluster": sku_cluster,
                "corr_bakery_total_daily": local_daily_corr_bakery,
                "corr_bakery_category_daily": local_daily_corr_category,
                "corr_bakery_cluster_daily": bakery_cluster_corr,
                "corr_sku_cluster_daily": sku_cluster_corr,
                "match_bakery_weekday": match_bakery_weekday,
                "match_bakery_category_weekday": match_bakery_category_weekday,
                "match_bakery_cluster_weekday": match_bakery_cluster_weekday,
                "match_sku_cluster_weekday": match_sku_cluster_weekday,
                "match_bakery_category_hour": match_bakery_category_hour,
                "match_bakery_cluster_hour": match_bakery_cluster_hour,
                "match_sku_cluster_hour": match_sku_cluster_hour,
                "local_anchor_score": local_anchor_score,
                "bakery_total_anchor_score": bakery_total_anchor_score,
                "bakery_category_anchor_score": bakery_category_anchor_score,
                "bakery_cluster_anchor_score": bakery_cluster_anchor_score,
                "sku_cluster_anchor_score": sku_cluster_anchor_score,
                "best_anchor_level": best_anchor,
                "best_anchor_score": best_anchor_score,
                "second_anchor_score": second_anchor_score,
                "anchor_confidence": anchor_confidence,
            }
        )

    return pd.DataFrame.from_records(records).sort_values(
        ["best_anchor_level", "best_anchor_score"], ascending=[True, False]
    ).reset_index(drop=True)


def build_summary(anchor_map: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(anchor_map)),
        "best_anchor_counts": anchor_map["best_anchor_level"].value_counts(dropna=False).to_dict(),
        "mean_best_anchor_score": round(float(anchor_map["best_anchor_score"].mean()), 6) if not anchor_map.empty else 0.0,
        "mean_anchor_confidence": round(float(anchor_map["anchor_confidence"].mean()), 6) if not anchor_map.empty else 0.0,
    }


def save_outputs(output_dir: str | Path, anchor_map: pd.DataFrame, summary: dict[str, object]) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / OUTPUT_NAME
    summary_path = out_dir / SUMMARY_OUTPUT_NAME
    anchor_map.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"anchor_map": csv_path, "summary": summary_path}


def build_kazan_anchor_suitability(
    *,
    bakery_daily_path: str | Path,
    bakery_category_daily_path: str | Path,
    sku_daily_path: str | Path,
    sku_hourly_path: str | Path,
    bakery_clusters_path: str | Path,
    sku_clusters_path: str | Path,
    sku_profile_map_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    bakery_daily = load_csv(bakery_daily_path)
    bakery_category_daily = load_csv(bakery_category_daily_path)
    sku_daily = load_csv(sku_daily_path)
    sku_hourly = load_csv(sku_hourly_path)
    bakery_clusters = load_csv(bakery_clusters_path)
    sku_clusters = load_csv(sku_clusters_path)
    sku_profile_map = load_csv(sku_profile_map_path)

    anchor_map = build_anchor_suitability_map(
        bakery_daily=bakery_daily,
        bakery_category_daily=bakery_category_daily,
        sku_daily=sku_daily,
        sku_hourly=sku_hourly,
        bakery_clusters=bakery_clusters,
        sku_clusters=sku_clusters,
        sku_profile_map=sku_profile_map,
    )
    summary = build_summary(anchor_map)
    return save_outputs(output_dir, anchor_map, summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build anchor suitability map for Kazan sitnaya sample")
    parser.add_argument("--bakery-daily-path", default=str(ROOT / "data" / "processed" / "kazan_bakery_daily_sample.csv"))
    parser.add_argument("--bakery-category-daily-path", default=str(ROOT / "data" / "processed" / "kazan_sitnaya_bakery_category_daily_sample.csv"))
    parser.add_argument("--sku-daily-path", default=str(ROOT / "data" / "processed" / "kazan_sitnaya_daily_sample.csv"))
    parser.add_argument("--sku-hourly-path", default=str(ROOT / "data" / "processed" / "kazan_sitnaya_hourly_sample.csv"))
    parser.add_argument("--bakery-clusters-path", default=str(ROOT / "data" / "processed" / "kazan_bakery_clusters.csv"))
    parser.add_argument("--sku-clusters-path", default=str(ROOT / "data" / "processed" / "kazan_sitnaya_sku_clusters.csv"))
    parser.add_argument("--sku-profile-map-path", default=str(ROOT / "data" / "processed" / "kazan_sitnaya_sku_profile_map.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "processed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_kazan_anchor_suitability(
        bakery_daily_path=args.bakery_daily_path,
        bakery_category_daily_path=args.bakery_category_daily_path,
        sku_daily_path=args.sku_daily_path,
        sku_hourly_path=args.sku_hourly_path,
        bakery_clusters_path=args.bakery_clusters_path,
        sku_clusters_path=args.sku_clusters_path,
        sku_profile_map_path=args.sku_profile_map_path,
        output_dir=args.output_dir,
    )
    print("=" * 72)
    print("KAZAN ANCHOR SUITABILITY")
    print("=" * 72)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

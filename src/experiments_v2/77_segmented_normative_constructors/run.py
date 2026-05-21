"""
Experiment 77: segmented normative constructors.

Implements the first explicit segment-specific normative builders for the two
segments where the anchor analysis already gives a clear direction:

- stable: release-aware weekday normative
- bakery_driven: bakery-level normative total * stable SKU share
"""

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
TARGET_SEGMENTS = {"stable", "bakery_driven"}


def load_daily_dataset(path: str | Path) -> pd.DataFrame:
    usecols = [
        DATE_COL,
        BAKERY_COL,
        SKU_COL,
        SALES_COL,
        DOW_COL,
        "release_qty",
        "bakery_sales_qty_total",
        "sku_sales_share_in_bakery_day",
    ]
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, usecols=usecols)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, BAKERY_COL, SKU_COL]).copy()
    df[SALES_COL] = pd.to_numeric(df[SALES_COL], errors="coerce").fillna(0.0)
    if "release_qty" in df.columns:
        df["release_qty"] = pd.to_numeric(df["release_qty"], errors="coerce").fillna(0.0)
    else:
        df["release_qty"] = 0.0
    if "bakery_sales_qty_total" in df.columns:
        df["bakery_sales_qty_total"] = pd.to_numeric(df["bakery_sales_qty_total"], errors="coerce").fillna(0.0)
    else:
        bakery_total = df.groupby([DATE_COL, BAKERY_COL], observed=True)[SALES_COL].transform("sum")
        df["bakery_sales_qty_total"] = bakery_total
    if "sku_sales_share_in_bakery_day" in df.columns:
        df["sku_sales_share_in_bakery_day"] = pd.to_numeric(df["sku_sales_share_in_bakery_day"], errors="coerce")
    else:
        df["sku_sales_share_in_bakery_day"] = df[SALES_COL] / df["bakery_sales_qty_total"].replace(0, np.nan)
    if DOW_COL not in df.columns:
        df[DOW_COL] = df[DATE_COL].dt.weekday
    else:
        df[DOW_COL] = pd.to_numeric(df[DOW_COL], errors="coerce").fillna(df[DATE_COL].dt.weekday).astype(int)
    return df.sort_values([BAKERY_COL, SKU_COL, DATE_COL]).reset_index(drop=True)


def load_segment_map(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    keep_cols = [BAKERY_COL, SKU_COL, SEGMENT_COL, "bakery_name", "product_name", "category_name", "city"]
    keep_cols = [col for col in keep_cols if col in df.columns]
    segment_map = df[keep_cols].drop_duplicates(subset=[BAKERY_COL, SKU_COL])
    if SEGMENT_COL in segment_map.columns:
        segment_map = segment_map.loc[segment_map[SEGMENT_COL].isin(TARGET_SEGMENTS)].copy()
    return segment_map


def _rolling_median(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    return series.rolling(window=window, min_periods=min_periods).median().bfill().ffill().fillna(0.0)


def _weekday_factor(group: pd.DataFrame, value_col: str) -> pd.Series:
    weekday_mean = group.groupby(DOW_COL, observed=True)[value_col].mean()
    overall_mean = float(group[value_col].mean())
    if overall_mean <= 1e-12:
        factors = pd.Series(1.0, index=range(7), dtype=float)
    else:
        factors = (weekday_mean / overall_mean).reindex(range(7)).fillna(1.0)
        factor_mean = float(factors.mean())
        if factor_mean <= 1e-12:
            factors = pd.Series(1.0, index=range(7), dtype=float)
        else:
            factors = factors / factor_mean
    return factors


def _build_bakery_normative_totals(df: pd.DataFrame) -> pd.DataFrame:
    bakery_daily = (
        df.groupby([DATE_COL, BAKERY_COL, DOW_COL], observed=True, as_index=False)
        .agg(bakery_sales_qty_total=("bakery_sales_qty_total", "max"))
        .sort_values([BAKERY_COL, DATE_COL])
    )

    parts: list[pd.DataFrame] = []
    for bakery_id, group in bakery_daily.groupby(BAKERY_COL, observed=True, sort=False):
        group = group.sort_values(DATE_COL).copy()
        weekday_factor = _weekday_factor(group, "bakery_sales_qty_total")
        trend = _rolling_median(group["bakery_sales_qty_total"], window=28, min_periods=7)
        group["bakery_normative_total"] = (trend * group[DOW_COL].map(weekday_factor).astype(float)).clip(lower=0.0)
        parts.append(group[[DATE_COL, BAKERY_COL, "bakery_normative_total"]])
    return pd.concat(parts, ignore_index=True)


def build_segmented_normative(df: pd.DataFrame, segment_map: pd.DataFrame) -> pd.DataFrame:
    work = df.merge(segment_map, on=[BAKERY_COL, SKU_COL], how="inner")

    bakery_normative = _build_bakery_normative_totals(work)
    work = work.merge(bakery_normative, on=[DATE_COL, BAKERY_COL], how="left")

    groups: list[pd.DataFrame] = []
    for _, group in work.groupby([BAKERY_COL, SKU_COL], observed=True, sort=False):
        group = group.sort_values(DATE_COL).copy()
        segment = group[SEGMENT_COL].iloc[0]
        sales = group[SALES_COL].astype(float)

        if segment == "stable":
            sales_weekday_factor = _weekday_factor(group, SALES_COL)
            sales_trend = _rolling_median(sales, window=28, min_periods=7)
            release_trend = _rolling_median(group["release_qty"].astype(float), window=28, min_periods=7)
            blended_level = 0.65 * release_trend + 0.35 * sales_trend
            candidate = (blended_level * group[DOW_COL].map(sales_weekday_factor).astype(float)).clip(lower=0.0)
            constructor_name = "stable_release_weekday"
        elif segment == "bakery_driven":
            rolling_share = _rolling_median(
                group["sku_sales_share_in_bakery_day"].astype(float).fillna(0.0),
                window=28,
                min_periods=7,
            ).clip(lower=0.0)
            share_cap = rolling_share.quantile(0.95) if rolling_share.notna().any() else np.nan
            if pd.notna(share_cap):
                rolling_share = rolling_share.clip(upper=float(share_cap))
            candidate = (group["bakery_normative_total"].astype(float) * rolling_share).clip(lower=0.0)
            constructor_name = "bakery_total_x_sku_share"
        group["segment_normative_candidate"] = candidate
        group["segment_constructor_name"] = constructor_name
        groups.append(group)

    return pd.concat(groups, ignore_index=True).sort_values([BAKERY_COL, SKU_COL, DATE_COL]).reset_index(drop=True)


def build_pair_summary(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (bakery_id, product_id), group in df.groupby([BAKERY_COL, SKU_COL], observed=True):
        observed = group[SALES_COL].astype(float)
        candidate = group["segment_normative_candidate"].astype(float)
        release = group["release_qty"].astype(float)
        records.append(
            {
                BAKERY_COL: bakery_id,
                SKU_COL: product_id,
                SEGMENT_COL: group[SEGMENT_COL].iloc[0],
                "segment_constructor_name": group["segment_constructor_name"].iloc[0],
                "observed_mean": float(observed.mean()),
                "candidate_mean": float(candidate.mean()),
                "release_mean": float(release.mean()),
                "candidate_to_observed_ratio": float(candidate.mean() / observed.mean()) if float(observed.mean()) > 1e-12 else np.nan,
                "candidate_corr_with_observed": _safe_corr(candidate, observed),
                "candidate_corr_with_release": _safe_corr(candidate, release),
                "candidate_corr_with_bakery_total": _safe_corr(candidate, group["bakery_sales_qty_total"]),
            }
        )
    return pd.DataFrame.from_records(records)


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


def build_segment_summary(pair_summary: pd.DataFrame) -> pd.DataFrame:
    return (
        pair_summary.groupby([SEGMENT_COL, "segment_constructor_name"], observed=True)
        .agg(
            pairs=(SEGMENT_COL, "size"),
            observed_mean=("observed_mean", "mean"),
            candidate_mean=("candidate_mean", "mean"),
            candidate_to_observed_ratio=("candidate_to_observed_ratio", "mean"),
            candidate_corr_with_observed=("candidate_corr_with_observed", "mean"),
            candidate_corr_with_release=("candidate_corr_with_release", "mean"),
            candidate_corr_with_bakery_total=("candidate_corr_with_bakery_total", "mean"),
        )
        .reset_index()
        .sort_values(["pairs"], ascending=False)
    )


def build_metrics(result_df: pd.DataFrame, pair_summary: pd.DataFrame) -> dict[str, object]:
    return {
        "built_rows": int(len(result_df)),
        "built_pairs": int(len(pair_summary)),
        "built_segments": pair_summary[SEGMENT_COL].value_counts().to_dict(),
        "constructor_counts": result_df["segment_constructor_name"].value_counts().to_dict(),
    }


def save_outputs(
    output_dir: Path,
    result_df: pd.DataFrame,
    pair_summary: pd.DataFrame,
    segment_summary: pd.DataFrame,
    metrics: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_dir / "segmented_normative_daily.csv", index=False, encoding="utf-8-sig")
    pair_summary.to_csv(output_dir / "segmented_normative_pair_summary.csv", index=False, encoding="utf-8-sig")
    segment_summary.to_csv(output_dir / "segmented_normative_segment_summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def run_experiment(daily_path: Path, segment_map_path: Path, output_dir: Path) -> dict[str, object]:
    daily_df = load_daily_dataset(daily_path)
    segment_map = load_segment_map(segment_map_path)
    result_df = build_segmented_normative(daily_df, segment_map)
    pair_summary = build_pair_summary(result_df)
    segment_summary = build_segment_summary(pair_summary)
    metrics = build_metrics(result_df, pair_summary)
    save_outputs(output_dir, result_df, pair_summary, segment_summary, metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build first segment-specific normative constructors")
    parser.add_argument("--daily-path", default=str(DEFAULT_DAILY_PATH))
    parser.add_argument("--segment-map-path", default=str(DEFAULT_SEGMENT_MAP_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[3]
    metrics = run_experiment(
        daily_path=root / args.daily_path,
        segment_map_path=root / args.segment_map_path,
        output_dir=Path(__file__).resolve().parent,
    )
    print("=" * 72)
    print("EXPERIMENT 77: SEGMENTED NORMATIVE CONSTRUCTORS")
    print("=" * 72)
    print(f"built_rows: {metrics['built_rows']:,}")
    print(f"built_pairs: {metrics['built_pairs']:,}")
    print(f"constructor_counts: {metrics['constructor_counts']}")


if __name__ == "__main__":
    main()

"""
Experiment 76: first normative demand construction candidates.

Builds two interpretable normative-demand candidates on top of the SKU-day panel:

- normative_v1: trend + static weekday profile
- normative_v2: trend + weekday profile with adaptive amplitude

The experiment also produces a simple segment-aware candidate selection that can
be used as a baseline for later research iterations.
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
SALES_COL = "observed_sales_qty"
DOW_COL = "dow"
SEGMENT_COL = "primary_segment"
PAIR_COLS = [BAKERY_COL, SKU_COL]

DEFAULT_DAILY_PATH = Path("data/processed/sku_daily_research_panel.csv")
DEFAULT_SEGMENT_MAP_PATH = Path("src/experiments_v2/75_normative_demand_map/predictability_and_structure_map.csv")


def load_daily_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, BAKERY_COL, SKU_COL]).copy()
    df[SALES_COL] = pd.to_numeric(df[SALES_COL], errors="coerce").fillna(0.0)
    if DOW_COL not in df.columns:
        df[DOW_COL] = df[DATE_COL].dt.weekday
    else:
        df[DOW_COL] = pd.to_numeric(df[DOW_COL], errors="coerce").fillna(df[DATE_COL].dt.weekday).astype(int)
    return df.sort_values(PAIR_COLS + [DATE_COL]).reset_index(drop=True)


def load_segment_map(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    cols = [BAKERY_COL, SKU_COL, SEGMENT_COL, "predictability_score"]
    available = [col for col in cols if col in df.columns]
    return df[available].drop_duplicates(subset=PAIR_COLS)


def _compute_static_weekday_profile(group: pd.DataFrame) -> pd.Series:
    weekday_mean = group.groupby(DOW_COL, observed=True)[SALES_COL].mean()
    overall_mean = float(group[SALES_COL].mean())
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


def _rolling_trend(series: pd.Series, window: int = 28, min_periods: int = 7) -> pd.Series:
    trend = series.rolling(window=window, min_periods=min_periods).median()
    trend = trend.ewm(span=14, adjust=False, min_periods=1).mean()
    return trend.bfill().ffill().fillna(0.0)


def _adaptive_amplitude(series: pd.Series, weekday_factor: pd.Series) -> pd.Series:
    base_abs = float((weekday_factor - 1.0).abs().mean())
    if base_abs <= 1e-12:
        return pd.Series(1.0, index=series.index, dtype=float)

    rolling_mean = series.rolling(window=28, min_periods=7).mean()
    rolling_std = series.rolling(window=28, min_periods=7).std(ddof=0)
    rolling_cv = rolling_std / rolling_mean.replace(0, np.nan)
    long_mean = float(series.mean())
    long_std = float(series.std(ddof=0))
    long_cv = long_std / long_mean if long_mean > 1e-12 else np.nan
    if pd.isna(long_cv) or long_cv <= 1e-12:
        amplitude = pd.Series(1.0, index=series.index, dtype=float)
    else:
        amplitude = (rolling_cv / long_cv).clip(lower=0.5, upper=1.5)
    return amplitude.bfill().ffill().fillna(1.0)


def build_normative_candidates(df: pd.DataFrame, segment_map: pd.DataFrame) -> pd.DataFrame:
    work = df.merge(segment_map, on=PAIR_COLS, how="left")
    work[SEGMENT_COL] = work[SEGMENT_COL].fillna("unmapped")

    groups: list[pd.DataFrame] = []
    for _, group in work.groupby(PAIR_COLS, observed=True, sort=False):
        group = group.sort_values(DATE_COL).copy()
        sales = group[SALES_COL].astype(float)
        trend = _rolling_trend(sales)
        weekday_factor = _compute_static_weekday_profile(group)
        weekday_series = group[DOW_COL].map(weekday_factor).astype(float)

        normative_v1 = (trend * weekday_series).clip(lower=0.0)

        amplitude = _adaptive_amplitude(sales, weekday_factor)
        normative_v2 = (trend * (1.0 + amplitude * (weekday_series - 1.0))).clip(lower=0.0)

        segment = group[SEGMENT_COL].iloc[0]
        if segment == "stable":
            selected = normative_v1
            selected_name = "normative_v1"
        elif segment in {"amplitude_unstable", "bakery_driven", "trend_dominated", "noisy"}:
            selected = normative_v2
            selected_name = "normative_v2"
        elif segment == "intermittent":
            selected = pd.Series(np.minimum(normative_v1, trend), index=group.index).clip(lower=0.0)
            selected_name = "normative_sparse_fallback"
        elif segment == "high_censoring":
            selected = pd.Series(np.maximum.reduce([sales.to_numpy(), normative_v2.to_numpy()]), index=group.index)
            selected_name = "normative_censoring_guard"
        else:
            selected = normative_v1
            selected_name = "normative_v1"

        group["normative_v1"] = normative_v1
        group["normative_v2"] = normative_v2
        group["normative_candidate"] = selected
        group["normative_candidate_name"] = selected_name
        group["normative_v1_to_observed_ratio"] = group["normative_v1"] / group[SALES_COL].replace(0, np.nan)
        group["normative_v2_to_observed_ratio"] = group["normative_v2"] / group[SALES_COL].replace(0, np.nan)
        group["normative_candidate_to_observed_ratio"] = group["normative_candidate"] / group[SALES_COL].replace(0, np.nan)
        groups.append(group)

    result = pd.concat(groups, ignore_index=True)
    return result.sort_values(PAIR_COLS + [DATE_COL]).reset_index(drop=True)


def _wmape(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true = pd.to_numeric(y_true, errors="coerce").fillna(0.0)
    y_pred = pd.to_numeric(y_pred, errors="coerce").fillna(0.0)
    denom = float(y_true.abs().sum())
    if denom <= 1e-12:
        return 0.0
    return float((y_true.sub(y_pred).abs().sum()) / denom)


def _r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true = pd.to_numeric(y_true, errors="coerce").fillna(0.0)
    y_pred = pd.to_numeric(y_pred, errors="coerce").fillna(0.0)
    if y_true.nunique() < 2:
        return np.nan
    sst = float(((y_true - y_true.mean()) ** 2).sum())
    if sst <= 1e-12:
        return np.nan
    sse = float(((y_true - y_pred) ** 2).sum())
    return float(1.0 - sse / sst)


def build_pair_summary(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (bakery_id, product_id), group in df.groupby(PAIR_COLS, observed=True):
        record = {
            BAKERY_COL: bakery_id,
            SKU_COL: product_id,
            "primary_segment": group[SEGMENT_COL].iloc[0],
            "normative_candidate_name": group["normative_candidate_name"].iloc[0],
            "observed_mean": float(group[SALES_COL].mean()),
            "normative_v1_mean": float(group["normative_v1"].mean()),
            "normative_v2_mean": float(group["normative_v2"].mean()),
            "normative_candidate_mean": float(group["normative_candidate"].mean()),
            "observed_cv": float(group[SALES_COL].std(ddof=0) / group[SALES_COL].mean()) if float(group[SALES_COL].mean()) > 1e-12 else np.nan,
            "normative_v1_cv": float(group["normative_v1"].std(ddof=0) / group["normative_v1"].mean()) if float(group["normative_v1"].mean()) > 1e-12 else np.nan,
            "normative_v2_cv": float(group["normative_v2"].std(ddof=0) / group["normative_v2"].mean()) if float(group["normative_v2"].mean()) > 1e-12 else np.nan,
            "candidate_wmape_vs_observed": _wmape(group[SALES_COL], group["normative_candidate"]),
            "candidate_r2_vs_observed": _r2(group[SALES_COL], group["normative_candidate"]),
        }
        records.append(record)
    return pd.DataFrame.from_records(records)


def build_segment_summary(pair_summary: pd.DataFrame) -> pd.DataFrame:
    return (
        pair_summary.groupby("primary_segment", observed=True)
        .agg(
            pairs=("primary_segment", "size"),
            candidate_mean=("normative_candidate_mean", "mean"),
            observed_mean=("observed_mean", "mean"),
            observed_cv=("observed_cv", "mean"),
            normative_v1_cv=("normative_v1_cv", "mean"),
            normative_v2_cv=("normative_v2_cv", "mean"),
            candidate_wmape_vs_observed=("candidate_wmape_vs_observed", "mean"),
            candidate_r2_vs_observed=("candidate_r2_vs_observed", "mean"),
        )
        .reset_index()
        .sort_values("pairs", ascending=False)
    )


def build_metrics(df: pd.DataFrame, pair_summary: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(df)),
        "pairs": int(pair_summary.shape[0]),
        "segments": pair_summary["primary_segment"].value_counts().to_dict(),
        "candidate_name_counts": df["normative_candidate_name"].value_counts().to_dict(),
        "observed_mean": round(float(df[SALES_COL].mean()), 6),
        "normative_v1_mean": round(float(df["normative_v1"].mean()), 6),
        "normative_v2_mean": round(float(df["normative_v2"].mean()), 6),
        "normative_candidate_mean": round(float(df["normative_candidate"].mean()), 6),
        "observed_cv_mean": round(float(pair_summary["observed_cv"].mean()), 6),
        "normative_v1_cv_mean": round(float(pair_summary["normative_v1_cv"].mean()), 6),
        "normative_v2_cv_mean": round(float(pair_summary["normative_v2_cv"].mean()), 6),
        "candidate_wmape_vs_observed_mean": round(float(pair_summary["candidate_wmape_vs_observed"].mean()), 6),
        "candidate_r2_vs_observed_mean": round(float(pair_summary["candidate_r2_vs_observed"].mean()), 6),
    }


def save_outputs(
    output_dir: Path,
    daily_with_normative: pd.DataFrame,
    pair_summary: pd.DataFrame,
    segment_summary: pd.DataFrame,
    metrics: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_with_normative.to_csv(output_dir / "normative_daily_candidates.csv", index=False, encoding="utf-8-sig")
    pair_summary.to_csv(output_dir / "normative_pair_summary.csv", index=False, encoding="utf-8-sig")
    segment_summary.to_csv(output_dir / "normative_segment_summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def run_experiment(daily_path: Path, segment_map_path: Path, output_dir: Path) -> dict[str, object]:
    daily = load_daily_dataset(daily_path)
    segment_map = load_segment_map(segment_map_path)
    daily_with_normative = build_normative_candidates(daily, segment_map)
    pair_summary = build_pair_summary(daily_with_normative)
    segment_summary = build_segment_summary(pair_summary)
    metrics = build_metrics(daily_with_normative, pair_summary)
    save_outputs(output_dir, daily_with_normative, pair_summary, segment_summary, metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build first normative demand candidates")
    parser.add_argument("--daily-path", default=str(DEFAULT_DAILY_PATH), help="Path to sku daily research panel dataset")
    parser.add_argument(
        "--segment-map-path",
        default=str(DEFAULT_SEGMENT_MAP_PATH),
        help="Path to predictability and structure map",
    )
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
    print("EXPERIMENT 76: NORMATIVE V1 / V2")
    print("=" * 72)
    print(f"rows: {metrics['rows']:,}")
    print(f"pairs: {metrics['pairs']:,}")
    print(f"candidate_wmape_vs_observed_mean: {metrics['candidate_wmape_vs_observed_mean']}")
    print(f"candidate_r2_vs_observed_mean: {metrics['candidate_r2_vs_observed_mean']}")


if __name__ == "__main__":
    main()

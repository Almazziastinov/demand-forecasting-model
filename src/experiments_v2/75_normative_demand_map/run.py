"""
Experiment N1: predictability and structure map for normative demand research.

Builds a profiling table for each `bakery x SKU` pair and assigns a provisional
behavior segment. The output is intended to be the foundation for subsequent
normative-demand experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATE_COL = "date"
BAKERY_COL = "bakery_id"
BAKERY_NAME_COL = "bakery_name"
CATEGORY_COL = "category_name"
SKU_COL = "product_id"
SKU_NAME_COL = "product_name"
CITY_COL = "city"
SALES_COL = "observed_sales_qty"
DOW_COL = "dow"
MONTH_COL = "month"
LAG7_COL = "sales_lag7"

DEFAULT_DAILY_PATH = Path("data/processed/sku_daily_research_base.csv")


def load_daily_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, BAKERY_COL, SKU_COL]).copy()
    df[SALES_COL] = pd.to_numeric(df[SALES_COL], errors="coerce").fillna(0.0)
    numeric_cols = [
        "release_qty",
        "incoming_move_qty",
        "outgoing_move_qty",
        "net_move_qty",
        "available_qty_proxy",
        "available_to_sales_ratio",
        "release_to_sales_ratio",
        "row_quality_score",
        "sku_sales_share_in_bakery_day",
        "sku_sales_share_in_category_day",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    flag_cols = [
        "release_present_flag",
        "moves_present_flag",
        "release_conflict_flag",
        "moves_conflict_flag",
        "organization_conflict_flag",
    ]
    for col in flag_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    if LAG7_COL in df.columns:
        df[LAG7_COL] = pd.to_numeric(df[LAG7_COL], errors="coerce")
    return df


def _mode_or_nan(series: pd.Series) -> str | float:
    mode = series.dropna().astype(str).mode()
    if mode.empty:
        return np.nan
    return mode.iloc[0]


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


def _weekday_strength(values: pd.Series, dow: pd.Series) -> float:
    if len(values) < 7 or values.nunique() < 2:
        return np.nan
    overall_mean = float(values.mean())
    sst = float(((values - overall_mean) ** 2).sum())
    if sst <= 1e-12:
        return np.nan
    dow_means = values.groupby(dow).transform("mean")
    sse = float(((values - dow_means) ** 2).sum())
    strength = 1.0 - sse / sst
    return float(np.clip(strength, -1.0, 1.0))


def _weekday_profile_stability(group: pd.DataFrame) -> float:
    work = group[[DATE_COL, DOW_COL, SALES_COL]].copy()
    work["iso_week"] = work[DATE_COL].dt.isocalendar().week.astype(int)
    work["iso_year"] = work[DATE_COL].dt.isocalendar().year.astype(int)

    weekly = (
        work.groupby(["iso_year", "iso_week", DOW_COL], observed=True)[SALES_COL]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(columns=range(7), fill_value=0.0)
    )
    if weekly.empty or len(weekly) < 2:
        return np.nan

    weekly_sum = weekly.sum(axis=1)
    weekly = weekly.loc[weekly_sum > 0]
    if len(weekly) < 2:
        return np.nan

    weekly_share = weekly.div(weekly.sum(axis=1), axis=0)
    global_share = weekly.sum(axis=0)
    global_share = global_share / global_share.sum()

    l1_distance = (weekly_share.sub(global_share, axis=1).abs().sum(axis=1) / 2.0).mean()
    stability = 1.0 - float(l1_distance)
    return float(np.clip(stability, 0.0, 1.0))


def _weekly_amplitude_cv(group: pd.DataFrame) -> float:
    work = group[[DATE_COL, SALES_COL]].copy()
    iso = work[DATE_COL].dt.isocalendar()
    weekly = work.groupby([iso.year.astype(int), iso.week.astype(int)], observed=True)[SALES_COL].sum()
    if len(weekly) < 2:
        return np.nan
    mean_val = float(weekly.mean())
    std_val = float(weekly.std(ddof=0))
    if mean_val <= 1e-12:
        return np.nan
    return std_val / mean_val


def _trend_metrics(group: pd.DataFrame) -> tuple[float, float]:
    if len(group) < 3:
        return np.nan, np.nan
    x = (group[DATE_COL] - group[DATE_COL].min()).dt.days.astype(float).to_numpy()
    y = group[SALES_COL].astype(float).to_numpy()
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan, np.nan
    corr = float(np.corrcoef(x, y)[0, 1])
    slope = float(np.polyfit(x, y, deg=1)[0])
    scale = float(np.mean(y)) if float(np.mean(y)) != 0 else np.nan
    slope_ratio = slope / scale if pd.notna(scale) and abs(scale) > 1e-12 else np.nan
    return corr, slope_ratio


def _lag7_metrics(group: pd.DataFrame) -> tuple[float, float, float, float]:
    values = group[SALES_COL].astype(float).to_numpy()
    if len(values) < 14:
        return np.nan, np.nan, np.nan, 0.0
    y_true = values[7:]
    y_pred = values[:-7]
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if valid_mask.sum() < 7:
        return np.nan, np.nan, np.nan, float(valid_mask.sum() / max(len(group), 1))
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    abs_err = np.abs(y_true - y_pred)
    mae = float(abs_err.mean())
    wmape = float(abs_err.sum() / (y_true.sum() + 1e-8))
    if np.unique(y_true).size < 2:
        r2 = np.nan
    else:
        sst = float(((y_true - y_true.mean()) ** 2).sum())
        sse = float(((y_true - y_pred) ** 2).sum())
        r2 = 1.0 - sse / sst if sst > 1e-12 else np.nan
    coverage = float(len(y_true) / max(len(group), 1))
    return mae, wmape, r2, coverage


def _predictability_score(row: pd.Series) -> float:
    lag7_component = np.clip((row.get("lag7_r2", np.nan) + 1.0) / 2.0, 0.0, 1.0) if pd.notna(row.get("lag7_r2", np.nan)) else 0.25
    seasonality_component = np.clip(row.get("weekly_seasonality_strength", np.nan), 0.0, 1.0) if pd.notna(row.get("weekly_seasonality_strength", np.nan)) else 0.0
    stability_component = np.clip(row.get("weekday_profile_stability", np.nan), 0.0, 1.0) if pd.notna(row.get("weekday_profile_stability", np.nan)) else 0.0
    zero_component = np.clip(1.0 - row.get("zero_share", 1.0), 0.0, 1.0)
    release_component = np.clip(row.get("release_coverage_share", 0.0), 0.0, 1.0) if pd.notna(row.get("release_coverage_share", np.nan)) else 0.0
    cv = row.get("cv_sales", np.nan)
    noise_component = 1.0 / (1.0 + max(cv, 0.0)) if pd.notna(cv) else 0.0

    score = (
        0.30 * lag7_component
        + 0.20 * seasonality_component
        + 0.20 * stability_component
        + 0.10 * zero_component
        + 0.10 * release_component
        + 0.10 * noise_component
    )
    return float(np.clip(score, 0.0, 1.0))


def assign_segment(row: pd.Series) -> str:
    zero_share = row.get("zero_share", np.nan)
    release_coverage_share = row.get("release_coverage_share", np.nan)
    lag7_r2 = row.get("lag7_r2", np.nan)
    cv_sales = row.get("cv_sales", np.nan)
    bakery_corr = row.get("bakery_sales_share_corr", np.nan)
    trend_corr = row.get("trend_corr", np.nan)
    weekly_strength = row.get("weekly_seasonality_strength", np.nan)
    amplitude_cv = row.get("weekly_amplitude_cv", np.nan)
    active_days_share = row.get("active_days_share", np.nan)

    if pd.notna(zero_share) and (zero_share >= 0.60 or (pd.notna(active_days_share) and active_days_share <= 0.40)):
        return "intermittent"
    if pd.notna(release_coverage_share) and release_coverage_share >= 0.50 and pd.notna(cv_sales) and cv_sales >= 1.5:
        return "high_censoring"
    if (
        pd.notna(lag7_r2)
        and lag7_r2 >= 0.20
        and pd.notna(cv_sales)
        and cv_sales <= 1.25
        and pd.notna(zero_share)
        and zero_share < 0.25
        and (pd.isna(release_coverage_share) or release_coverage_share >= 0.10)
    ):
        return "stable"
    if pd.notna(bakery_corr) and bakery_corr >= 0.70 and pd.notna(zero_share) and zero_share < 0.40:
        return "bakery_driven"
    if pd.notna(trend_corr) and abs(trend_corr) >= 0.35 and (pd.isna(weekly_strength) or weekly_strength < 0.15):
        return "trend_dominated"
    if pd.notna(amplitude_cv) and amplitude_cv >= 0.45 and pd.notna(weekly_strength) and weekly_strength >= 0.15:
        return "amplitude_unstable"
    return "noisy"


def build_pair_profile_map(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [BAKERY_COL, SKU_COL, DATE_COL]
    df = df.sort_values(sort_cols).copy()

    group_cols = [BAKERY_COL, CATEGORY_COL, SKU_COL]
    records: list[dict[str, object]] = []
    total_date_span = df[DATE_COL].nunique()

    for (bakery, category, sku), group in df.groupby(group_cols, observed=True):
        group = group.sort_values(DATE_COL).copy()
        sales = group[SALES_COL].astype(float)

        mean_sales = float(sales.mean())
        std_sales = float(sales.std(ddof=0))
        trend_corr, trend_slope_ratio = _trend_metrics(group)
        lag7_mae, lag7_wmape, lag7_r2, lag7_coverage = _lag7_metrics(group)
        bakery_total = group["bakery_sales_qty_total"] if "bakery_sales_qty_total" in group.columns else pd.Series(np.nan, index=group.index)
        release_qty = group["release_qty"] if "release_qty" in group.columns else pd.Series(np.nan, index=group.index)
        net_move_qty = group["net_move_qty"] if "net_move_qty" in group.columns else pd.Series(np.nan, index=group.index)

        record = {
            BAKERY_COL: bakery,
            CATEGORY_COL: category,
            SKU_COL: sku,
            BAKERY_NAME_COL: group[BAKERY_NAME_COL].mode().iloc[0] if BAKERY_NAME_COL in group.columns and not group[BAKERY_NAME_COL].mode().empty else np.nan,
            SKU_NAME_COL: group[SKU_NAME_COL].mode().iloc[0] if SKU_NAME_COL in group.columns and not group[SKU_NAME_COL].mode().empty else np.nan,
            CITY_COL: group[CITY_COL].mode().iloc[0] if CITY_COL in group.columns and not group[CITY_COL].mode().empty else np.nan,
            "date_min": group[DATE_COL].min(),
            "date_max": group[DATE_COL].max(),
            "observed_days": int(len(group)),
            "history_length_days": int((group[DATE_COL].max() - group[DATE_COL].min()).days + 1),
            "dataset_date_coverage": float(len(group) / max(total_date_span, 1)),
            "active_days_share": float((sales > 0).mean()),
            "mean_sales": mean_sales,
            "median_sales": float(sales.median()),
            "std_sales": std_sales,
            "cv_sales": std_sales / mean_sales if mean_sales > 1e-12 else np.nan,
            "zero_share": float((sales <= 0).mean()),
            "weekly_seasonality_strength": _weekday_strength(sales, group[DOW_COL]),
            "weekday_profile_stability": _weekday_profile_stability(group),
            "weekly_amplitude_cv": _weekly_amplitude_cv(group),
            "trend_corr": trend_corr,
            "trend_slope_ratio": trend_slope_ratio,
            "bakery_sales_corr": _safe_corr(sales, bakery_total),
            "bakery_sales_share_corr": _safe_corr(group.get("sku_sales_share_in_bakery_day", pd.Series(np.nan, index=group.index)), bakery_total),
            "lag7_mae": lag7_mae,
            "lag7_wmape": lag7_wmape,
            "lag7_r2": lag7_r2,
            "lag7_coverage": lag7_coverage,
            "release_coverage_share": float(group["release_present_flag"].mean()) if "release_present_flag" in group.columns else np.nan,
            "moves_coverage_share": float(group["moves_present_flag"].mean()) if "moves_present_flag" in group.columns else np.nan,
            "mean_release_qty": float(release_qty.fillna(0.0).mean()) if "release_qty" in group.columns else np.nan,
            "mean_release_to_sales_ratio": float(pd.to_numeric(group.get("release_to_sales_ratio", pd.Series(np.nan, index=group.index)), errors="coerce").replace([np.inf, -np.inf], np.nan).mean()),
            "release_corr_with_sales": _safe_corr(sales, release_qty),
            "move_corr_with_sales": _safe_corr(sales, net_move_qty),
            "mean_net_move_qty": float(net_move_qty.fillna(0.0).mean()) if "net_move_qty" in group.columns else np.nan,
            "organization_id": _mode_or_nan(group["organization_id"]) if "organization_id" in group.columns else np.nan,
            "organization_name": _mode_or_nan(group["organization_name"]) if "organization_name" in group.columns else np.nan,
            "organization_conflict_share": float(group["organization_conflict_flag"].mean()) if "organization_conflict_flag" in group.columns else np.nan,
            "mean_row_quality_score": float(group["row_quality_score"].mean()) if "row_quality_score" in group.columns else np.nan,
        }

        records.append(record)

    result = pd.DataFrame.from_records(records)
    result["predictability_score"] = result.apply(_predictability_score, axis=1)
    result["primary_segment"] = result.apply(assign_segment, axis=1)
    return result.sort_values(["primary_segment", "predictability_score"], ascending=[True, False]).reset_index(drop=True)


def build_segment_summary(profile_map: pd.DataFrame) -> pd.DataFrame:
    summary = (
        profile_map.groupby("primary_segment", observed=True)
        .agg(
            pairs=("primary_segment", "size"),
            mean_predictability=("predictability_score", "mean"),
            mean_sales=("mean_sales", "mean"),
            zero_share=("zero_share", "mean"),
            release_coverage_share=("release_coverage_share", "mean"),
            moves_coverage_share=("moves_coverage_share", "mean"),
            weekly_seasonality_strength=("weekly_seasonality_strength", "mean"),
            weekday_profile_stability=("weekday_profile_stability", "mean"),
            lag7_r2=("lag7_r2", "mean"),
            bakery_sales_corr=("bakery_sales_corr", "mean"),
            release_corr_with_sales=("release_corr_with_sales", "mean"),
        )
        .reset_index()
        .sort_values("pairs", ascending=False)
    )
    return summary


def build_segment_examples(profile_map: pd.DataFrame, examples_per_segment: int = 10) -> pd.DataFrame:
    top_examples = (
        profile_map.sort_values(["primary_segment", "predictability_score"], ascending=[True, False])
        .groupby("primary_segment", observed=True)
        .head(examples_per_segment)
        .copy()
    )
    cols = [
        "primary_segment",
        BAKERY_COL,
        BAKERY_NAME_COL,
        CATEGORY_COL,
        SKU_COL,
        SKU_NAME_COL,
        CITY_COL,
        "predictability_score",
        "mean_sales",
        "zero_share",
        "release_coverage_share",
        "moves_coverage_share",
        "weekly_seasonality_strength",
        "weekday_profile_stability",
        "weekly_amplitude_cv",
        "lag7_r2",
        "bakery_sales_corr",
        "release_corr_with_sales",
        "organization_name",
    ]
    cols = [col for col in cols if col in top_examples.columns]
    return top_examples[cols].reset_index(drop=True)


def build_metrics(profile_map: pd.DataFrame) -> dict[str, object]:
    return {
        "pairs": int(len(profile_map)),
        "bakeries": int(profile_map[BAKERY_COL].nunique()),
        "sku": int(profile_map[SKU_COL].nunique()),
        "categories": int(profile_map[CATEGORY_COL].nunique()),
        "mean_predictability_score": round(float(profile_map["predictability_score"].mean()), 6),
        "median_predictability_score": round(float(profile_map["predictability_score"].median()), 6),
        "segment_counts": profile_map["primary_segment"].value_counts().to_dict(),
        "stable_share": round(float(profile_map["primary_segment"].eq("stable").mean()), 6),
        "intermittent_share": round(float(profile_map["primary_segment"].eq("intermittent").mean()), 6),
        "high_censoring_share": round(float(profile_map["primary_segment"].eq("high_censoring").mean()), 6),
    }


def save_outputs(
    output_dir: Path,
    profile_map: pd.DataFrame,
    segment_summary: pd.DataFrame,
    segment_examples: pd.DataFrame,
    metrics: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_map.to_csv(output_dir / "predictability_and_structure_map.csv", index=False, encoding="utf-8-sig")
    segment_summary.to_csv(output_dir / "segment_summary.csv", index=False, encoding="utf-8-sig")
    segment_examples.to_csv(output_dir / "segment_examples.csv", index=False, encoding="utf-8-sig")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def run_experiment(
    daily_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    df = load_daily_dataset(daily_path)
    profile_map = build_pair_profile_map(df)
    segment_summary = build_segment_summary(profile_map)
    segment_examples = build_segment_examples(profile_map)
    metrics = build_metrics(profile_map)

    save_outputs(output_dir, profile_map, segment_summary, segment_examples, metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build predictability and structure map for normative demand research")
    parser.add_argument(
        "--daily-path",
        default=str(DEFAULT_DAILY_PATH),
        help="Path to sku daily research base dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[3]
    daily_path = root / args.daily_path
    output_dir = Path(__file__).resolve().parent

    metrics = run_experiment(daily_path=daily_path, output_dir=output_dir)

    print("=" * 72)
    print("EXPERIMENT N1: PREDICTABILITY AND STRUCTURE MAP")
    print("=" * 72)
    print(f"pairs: {metrics['pairs']:,}")
    print(f"mean_predictability_score: {metrics['mean_predictability_score']}")
    print("segment_counts:")
    for segment, count in metrics["segment_counts"].items():
        print(f"  {segment}: {count:,}")


if __name__ == "__main__":
    main()

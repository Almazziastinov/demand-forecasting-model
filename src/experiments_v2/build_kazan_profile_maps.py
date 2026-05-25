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

BAKERY_PROFILE_OUTPUT_NAME = "kazan_bakery_profile_map.csv"
SKU_PROFILE_OUTPUT_NAME = "kazan_sitnaya_sku_profile_map.csv"
SUMMARY_OUTPUT_NAME = "kazan_profile_maps_summary.json"


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        if DOW_COL not in df.columns:
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


def _weekday_strength(values: pd.Series, dow: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    valid = values.notna() & dow.notna()
    if valid.sum() < 7 or values[valid].nunique() < 2:
        return np.nan
    values = values[valid]
    dow = dow[valid]
    overall_mean = float(values.mean())
    sst = float(((values - overall_mean) ** 2).sum())
    if sst <= 1e-12:
        return np.nan
    dow_means = values.groupby(dow).transform("mean")
    sse = float(((values - dow_means) ** 2).sum())
    strength = 1.0 - sse / sst
    return float(np.clip(strength, -1.0, 1.0))


def _weekday_profile_stability(group: pd.DataFrame, value_col: str) -> float:
    work = group[[DATE_COL, DOW_COL, value_col]].copy()
    work["iso_week"] = work[DATE_COL].dt.isocalendar().week.astype(int)
    work["iso_year"] = work[DATE_COL].dt.isocalendar().year.astype(int)
    weekly = (
        work.groupby(["iso_year", "iso_week", DOW_COL], observed=True)[value_col]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(columns=range(7), fill_value=0.0)
    )
    if len(weekly) < 2:
        return np.nan
    weekly = weekly.loc[weekly.sum(axis=1) > 0]
    if len(weekly) < 2:
        return np.nan
    weekly_share = weekly.div(weekly.sum(axis=1), axis=0)
    global_share = weekly.sum(axis=0)
    global_share = global_share / global_share.sum()
    l1_distance = (weekly_share.sub(global_share, axis=1).abs().sum(axis=1) / 2.0).mean()
    return float(np.clip(1.0 - float(l1_distance), 0.0, 1.0))


def _weekly_amplitude_cv(group: pd.DataFrame, value_col: str) -> float:
    work = group[[DATE_COL, value_col]].copy()
    iso = work[DATE_COL].dt.isocalendar()
    weekly = work.groupby([iso.year.astype(int), iso.week.astype(int)], observed=True)[value_col].sum()
    if len(weekly) < 2:
        return np.nan
    mean_val = float(weekly.mean())
    std_val = float(weekly.std(ddof=0))
    if mean_val <= 1e-12:
        return np.nan
    return std_val / mean_val


def _trend_metrics(group: pd.DataFrame, value_col: str) -> tuple[float, float]:
    if len(group) < 3:
        return np.nan, np.nan
    x = (group[DATE_COL] - group[DATE_COL].min()).dt.days.astype(float).to_numpy()
    y = pd.to_numeric(group[value_col], errors="coerce").astype(float).to_numpy()
    valid = ~np.isnan(x) & ~np.isnan(y)
    if valid.sum() < 3:
        return np.nan, np.nan
    x = x[valid]
    y = y[valid]
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan, np.nan
    corr = float(np.corrcoef(x, y)[0, 1])
    slope = float(np.polyfit(x, y, deg=1)[0])
    scale = float(np.mean(y)) if float(np.mean(y)) != 0 else np.nan
    slope_ratio = slope / scale if pd.notna(scale) and abs(scale) > 1e-12 else np.nan
    return corr, slope_ratio


def _hour_profile_metrics(hourly_group: pd.DataFrame) -> tuple[float, float, float]:
    if hourly_group.empty:
        return np.nan, np.nan, np.nan
    day_hour = (
        hourly_group.groupby([DATE_COL, HOUR_COL], observed=True)["sku_hour_sales"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(columns=range(24), fill_value=0.0)
    )
    if day_hour.empty:
        return np.nan, np.nan, np.nan
    day_sum = day_hour.sum(axis=1)
    positive = day_hour.loc[day_sum > 0]
    if positive.empty:
        return float((day_sum <= 0).mean()), np.nan, np.nan
    share = positive.div(positive.sum(axis=1), axis=0)
    global_share = positive.sum(axis=0)
    global_share = global_share / global_share.sum()
    l1_distance = (share.sub(global_share, axis=1).abs().sum(axis=1) / 2.0).mean()
    stability = float(np.clip(1.0 - float(l1_distance), 0.0, 1.0))
    active_hours_mean = float((positive > 0).sum(axis=1).mean())
    zero_day_share = float((day_sum <= 0).mean())
    return zero_day_share, stability, active_hours_mean


def build_bakery_profile_map(
    bakery_daily: pd.DataFrame,
    bakery_category_daily: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for bakery_id, group in bakery_daily.groupby(BAKERY_COL, observed=True):
        group = group.sort_values(DATE_COL).copy()
        sales = pd.to_numeric(group["bakery_sales"], errors="coerce").fillna(0.0)
        mean_sales = float(sales.mean())
        std_sales = float(sales.std(ddof=0))
        trend_corr, trend_slope_ratio = _trend_metrics(group, "bakery_sales")
        category_group = bakery_category_daily[bakery_category_daily[BAKERY_COL] == bakery_id].copy()
        if not category_group.empty:
            category_share_mean = float(pd.to_numeric(category_group["category_share_in_bakery_total"], errors="coerce").mean())
            category_share_std = float(pd.to_numeric(category_group["category_share_in_bakery_total"], errors="coerce").std(ddof=0))
            active_sku_mean = float(pd.to_numeric(category_group["active_sku_count"], errors="coerce").mean())
            selling_sku_mean = float(pd.to_numeric(category_group["selling_sku_count"], errors="coerce").mean())
            category_release_corr = _safe_corr(category_group["category_sales_qty"], category_group["category_release_qty"])
        else:
            category_share_mean = np.nan
            category_share_std = np.nan
            active_sku_mean = np.nan
            selling_sku_mean = np.nan
            category_release_corr = np.nan

        records.append(
            {
                BAKERY_COL: bakery_id,
                BAKERY_NAME_COL: group[BAKERY_NAME_COL].astype(str).mode().iloc[0],
                CITY_COL: group[CITY_COL].astype(str).mode().iloc[0],
                "date_min": group[DATE_COL].min(),
                "date_max": group[DATE_COL].max(),
                "history_days": int(group[DATE_COL].nunique()),
                "active_days": int((sales > 0).sum()),
                "active_days_share": float((sales > 0).mean()),
                "mean_bakery_sales": mean_sales,
                "median_bakery_sales": float(sales.median()),
                "std_bakery_sales": std_sales,
                "cv_bakery_sales": std_sales / mean_sales if mean_sales > 1e-12 else np.nan,
                "weekday_strength": _weekday_strength(sales, group[DOW_COL]),
                "weekday_profile_stability": _weekday_profile_stability(group, "bakery_sales"),
                "weekly_amplitude_cv": _weekly_amplitude_cv(group, "bakery_sales"),
                "trend_corr": trend_corr,
                "trend_slope_ratio": trend_slope_ratio,
                "mean_avg_price": float(pd.to_numeric(group["avg_price"], errors="coerce").mean()),
                "category_share_mean": category_share_mean,
                "category_share_std": category_share_std,
                "active_sku_mean": active_sku_mean,
                "selling_sku_mean": selling_sku_mean,
                "category_release_corr": category_release_corr,
            }
        )
    return pd.DataFrame.from_records(records).sort_values("mean_bakery_sales", ascending=False).reset_index(drop=True)


def build_sku_profile_map(
    sku_daily: pd.DataFrame,
    bakery_category_daily: pd.DataFrame,
    sku_hourly: pd.DataFrame,
) -> pd.DataFrame:
    category_lookup = bakery_category_daily[
        [DATE_COL, BAKERY_COL, "category_sales_qty", "category_release_qty", "category_share_in_bakery_total"]
    ].copy()
    work = sku_daily.merge(
        category_lookup,
        on=[DATE_COL, BAKERY_COL],
        how="left",
        validate="many_to_one",
    )

    records: list[dict[str, object]] = []
    for (bakery_id, sku_id), group in work.groupby([BAKERY_COL, SKU_COL], observed=True):
        group = group.sort_values(DATE_COL).copy()
        sales = pd.to_numeric(group["observed_sales_qty"], errors="coerce").fillna(0.0)
        mean_sales = float(sales.mean())
        std_sales = float(sales.std(ddof=0))
        trend_corr, trend_slope_ratio = _trend_metrics(group, "observed_sales_qty")
        hourly_group = sku_hourly[(sku_hourly[BAKERY_COL] == bakery_id) & (sku_hourly[SKU_COL] == sku_id)].copy()
        zero_day_share_from_hour, hour_profile_stability, active_hours_mean = _hour_profile_metrics(hourly_group)

        records.append(
            {
                BAKERY_COL: bakery_id,
                BAKERY_NAME_COL: group[BAKERY_NAME_COL].astype(str).mode().iloc[0],
                CITY_COL: group[CITY_COL].astype(str).mode().iloc[0],
                CATEGORY_COL: group[CATEGORY_COL].astype(str).mode().iloc[0],
                SKU_COL: sku_id,
                SKU_NAME_COL: group[SKU_NAME_COL].astype(str).mode().iloc[0],
                "date_min": group[DATE_COL].min(),
                "date_max": group[DATE_COL].max(),
                "history_days": int(group[DATE_COL].nunique()),
                "active_days": int((sales > 0).sum()),
                "active_days_share": float((sales > 0).mean()),
                "mean_sales": mean_sales,
                "median_sales": float(sales.median()),
                "std_sales": std_sales,
                "cv_sales": std_sales / mean_sales if mean_sales > 1e-12 else np.nan,
                "zero_share": float((sales <= 0).mean()),
                "weekday_strength": _weekday_strength(sales, group[DOW_COL]),
                "weekday_profile_stability": _weekday_profile_stability(group, "observed_sales_qty"),
                "weekly_amplitude_cv": _weekly_amplitude_cv(group, "observed_sales_qty"),
                "trend_corr": trend_corr,
                "trend_slope_ratio": trend_slope_ratio,
                "mean_sales_hours_count": float(pd.to_numeric(group["sales_hours_count"], errors="coerce").mean()),
                "release_present_share": float(pd.to_numeric(group["release_present_flag"], errors="coerce").fillna(0).mean()),
                "mean_release_qty": float(pd.to_numeric(group["release_qty"], errors="coerce").fillna(0.0).mean()),
                "release_sales_corr": _safe_corr(group["observed_sales_qty"], group["release_qty"]),
                "bakery_total_sales_corr": _safe_corr(group["observed_sales_qty"], group["bakery_total_sales_qty"]),
                "category_total_sales_corr": _safe_corr(group["observed_sales_qty"], group["category_sales_qty"]),
                "sku_share_in_bakery_total_mean": float(pd.to_numeric(group["sku_sales_share_in_bakery_total"], errors="coerce").mean()),
                "sku_share_in_bakery_total_std": float(pd.to_numeric(group["sku_sales_share_in_bakery_total"], errors="coerce").std(ddof=0)),
                "category_share_in_bakery_total_mean": float(pd.to_numeric(group["category_share_in_bakery_total"], errors="coerce").mean()),
                "hour_zero_day_share": zero_day_share_from_hour,
                "hour_profile_stability": hour_profile_stability,
                "active_hours_mean": active_hours_mean,
                "mean_row_quality_score": float(pd.to_numeric(group["row_quality_score"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame.from_records(records).sort_values(["bakery_id", "mean_sales"], ascending=[True, False]).reset_index(drop=True)


def build_summary(
    bakery_profile_map: pd.DataFrame,
    sku_profile_map: pd.DataFrame,
) -> dict[str, object]:
    return {
        "bakery_profiles": int(len(bakery_profile_map)),
        "sku_profiles": int(len(sku_profile_map)),
        "bakeries": int(bakery_profile_map[BAKERY_COL].nunique()) if not bakery_profile_map.empty else 0,
        "sku": int(sku_profile_map[SKU_COL].nunique()) if not sku_profile_map.empty else 0,
        "mean_bakery_sales": round(float(bakery_profile_map["mean_bakery_sales"].mean()), 6) if not bakery_profile_map.empty else 0.0,
        "mean_sku_sales": round(float(sku_profile_map["mean_sales"].mean()), 6) if not sku_profile_map.empty else 0.0,
        "mean_bakery_weekday_stability": round(float(bakery_profile_map["weekday_profile_stability"].mean()), 6) if not bakery_profile_map.empty else 0.0,
        "mean_sku_weekday_stability": round(float(sku_profile_map["weekday_profile_stability"].mean()), 6) if not sku_profile_map.empty else 0.0,
        "mean_sku_hour_stability": round(float(sku_profile_map["hour_profile_stability"].mean()), 6) if not sku_profile_map.empty else 0.0,
    }


def save_outputs(
    output_dir: str | Path,
    bakery_profile_map: pd.DataFrame,
    sku_profile_map: pd.DataFrame,
    summary: dict[str, object],
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bakery_path = out_dir / BAKERY_PROFILE_OUTPUT_NAME
    sku_path = out_dir / SKU_PROFILE_OUTPUT_NAME
    summary_path = out_dir / SUMMARY_OUTPUT_NAME
    bakery_profile_map.to_csv(bakery_path, index=False, encoding="utf-8-sig")
    sku_profile_map.to_csv(sku_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"bakery_profile_map": bakery_path, "sku_profile_map": sku_path, "summary": summary_path}


def build_kazan_profile_maps(
    *,
    bakery_daily_path: str | Path,
    bakery_category_daily_path: str | Path,
    sku_daily_path: str | Path,
    sku_hourly_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    bakery_daily = load_csv(bakery_daily_path)
    bakery_category_daily = load_csv(bakery_category_daily_path)
    sku_daily = load_csv(sku_daily_path)
    sku_hourly = load_csv(sku_hourly_path)

    bakery_profile_map = build_bakery_profile_map(bakery_daily, bakery_category_daily)
    sku_profile_map = build_sku_profile_map(sku_daily, bakery_category_daily, sku_hourly)
    summary = build_summary(bakery_profile_map, sku_profile_map)
    return save_outputs(output_dir, bakery_profile_map, sku_profile_map, summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build bakery and SKU profile maps for Kazan sitnaya sample")
    parser.add_argument("--bakery-daily-path", default=str(ROOT / "data" / "processed" / "kazan_bakery_daily_sample.csv"))
    parser.add_argument("--bakery-category-daily-path", default=str(ROOT / "data" / "processed" / "kazan_sitnaya_bakery_category_daily_sample.csv"))
    parser.add_argument("--sku-daily-path", default=str(ROOT / "data" / "processed" / "kazan_sitnaya_daily_sample.csv"))
    parser.add_argument("--sku-hourly-path", default=str(ROOT / "data" / "processed" / "kazan_sitnaya_hourly_sample.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "processed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_kazan_profile_maps(
        bakery_daily_path=args.bakery_daily_path,
        bakery_category_daily_path=args.bakery_category_daily_path,
        sku_daily_path=args.sku_daily_path,
        sku_hourly_path=args.sku_hourly_path,
        output_dir=args.output_dir,
    )
    print("=" * 72)
    print("KAZAN PROFILE MAPS")
    print("=" * 72)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

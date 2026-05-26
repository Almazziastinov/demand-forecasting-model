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

OUTPUT_NAME = "kazan_temporal_normative_sku_day.csv"
SUMMARY_OUTPUT = "kazan_temporal_normative_sku_day_summary.json"
DEFAULT_VARIANT = "reference_profile"
SHORT_SHARE_DAYS = 28
SHORT_SHARE_WEIGHT = 0.5
SHORT_SHARE_DELTA_CLIP = 0.10


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        if DOW_COL not in df.columns:
            df[DOW_COL] = df[DATE_COL].dt.dayofweek
    return df


def _robust_rolling_zscore(
    series: pd.Series,
    *,
    window: int,
    min_periods: int = 4,
) -> pd.Series:
    center = series.rolling(window=window, min_periods=min_periods).median()
    mad = series.rolling(window=window, min_periods=min_periods).apply(
        lambda values: float(np.median(np.abs(values - np.median(values)))),
        raw=True,
    )
    scale = mad * 1.4826
    zscore = (series - center) / scale.replace(0, np.nan)
    return zscore.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _zscore_to_amplitude_multiplier(
    zscore: pd.Series,
    *,
    scale: float = 0.48,
    lower: float = 0.35,
    upper: float = 2.0,
) -> pd.Series:
    amplitude = 1.0 + scale * zscore
    return amplitude.clip(lower=lower, upper=upper).fillna(1.0)


def _amplitude_to_effective_strength(
    amplitude: pd.Series,
    *,
    base_strength: float = 2.4,
    lower: float = 0.15,
    upper: float = 3.5,
) -> pd.Series:
    effective = 1.0 + base_strength * (pd.to_numeric(amplitude, errors="coerce").fillna(1.0) - 1.0)
    return effective.clip(lower=lower, upper=upper)


def _compute_share_profile(frame: pd.DataFrame, base_col: str, output_prefix: str) -> pd.DataFrame:
    weekday_profile = (
        frame.groupby(DOW_COL, as_index=False)[base_col]
        .median()
        .rename(columns={base_col: f"{output_prefix}_weekday_profile_raw"})
    )
    weekday_profile = weekday_profile.set_index(DOW_COL).reindex(range(7), fill_value=0.0).reset_index()
    raw_col = f"{output_prefix}_weekday_profile_raw"
    share_col = f"{output_prefix}_weekday_share"
    factor_col = f"{output_prefix}_weekday_factor"
    profile_sum = float(weekday_profile[raw_col].sum())
    if profile_sum <= 0:
        weekday_profile[share_col] = 1.0 / 7.0
    else:
        weekday_profile[share_col] = weekday_profile[raw_col] / profile_sum
    weekday_profile[factor_col] = weekday_profile[share_col] * 7.0
    return weekday_profile


def _compute_reference_weekday_profile(group: pd.DataFrame, base_col: str, recent_weeks: int) -> pd.DataFrame:
    long_profile = _compute_share_profile(group, base_col, "long")
    short_days = group.tail(SHORT_SHARE_DAYS).copy()
    short_profile = _compute_share_profile(short_days, base_col, "short")
    weekday_profile = long_profile.merge(short_profile[[DOW_COL, "short_weekday_share"]], on=DOW_COL, how="left")
    weekday_profile["short_weekday_share"] = weekday_profile["short_weekday_share"].fillna(weekday_profile["long_weekday_share"])
    weekday_profile["share_delta_short_vs_long"] = (
        weekday_profile["short_weekday_share"] - weekday_profile["long_weekday_share"]
    )
    weekday_profile["share_delta_short_vs_long"] = weekday_profile["share_delta_short_vs_long"].clip(
        lower=-SHORT_SHARE_DELTA_CLIP,
        upper=SHORT_SHARE_DELTA_CLIP,
    )
    weekday_profile["weekday_share_normative"] = (
        (1.0 - SHORT_SHARE_WEIGHT) * weekday_profile["long_weekday_share"]
        + SHORT_SHARE_WEIGHT * weekday_profile["short_weekday_share"]
    )
    weekday_profile["weekday_share_normative"] = weekday_profile["weekday_share_normative"].clip(lower=0.01)
    weekday_profile["weekday_share_normative"] = (
        weekday_profile["weekday_share_normative"] / weekday_profile["weekday_share_normative"].sum()
    )
    weekday_profile["weekday_factor_normative"] = weekday_profile["weekday_share_normative"] * 7.0
    return weekday_profile


def _build_weekly_strength_table(
    group: pd.DataFrame,
    weekly: pd.DataFrame,
    *,
    base_col: str,
    weekday_profile: pd.DataFrame,
) -> pd.DataFrame:
    week_day = (
        group[["iso_year", "iso_week", DOW_COL, base_col]]
        .groupby(["iso_year", "iso_week", DOW_COL], as_index=False)[base_col]
        .sum()
    )
    week_day = week_day.merge(
        weekly[["iso_year", "iso_week", "week_total", "week_mean", "week_std"]],
        on=["iso_year", "iso_week"],
        how="left",
        validate="many_to_one",
    )
    week_day["week_share"] = week_day[base_col] / week_day["week_total"].replace(0, np.nan)
    week_day["week_share"] = week_day["week_share"].fillna(0.0)

    pivot = (
        week_day.pivot_table(
            index=["iso_year", "iso_week"],
            columns=DOW_COL,
            values="week_share",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(columns=range(7), fill_value=0.0)
        .reset_index()
    )
    share_cols = list(range(7))
    flat_share = 1.0 / 7.0
    reference_shares = weekday_profile["weekday_share_normative"].to_numpy(dtype=float)
    share_matrix = pivot[share_cols].to_numpy(dtype=float)

    weekly_profile = pivot[["iso_year", "iso_week"]].copy()
    weekly_profile["weekly_cv_raw"] = weekly["week_std"] / weekly["week_mean"].replace(0, np.nan)
    weekly_profile["weekly_cv_raw"] = weekly_profile["weekly_cv_raw"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weekly_profile["flat_deviation_raw"] = np.abs(share_matrix - flat_share).sum(axis=1)
    weekly_profile["reference_distance_raw"] = np.abs(share_matrix - reference_shares).sum(axis=1)
    reference_strength = float(np.abs(reference_shares - flat_share).sum())
    weekly_profile["reference_profile_raw"] = (reference_strength - weekly_profile["reference_distance_raw"]).clip(lower=0.0)
    return weekly_profile


def build_temporal_normative(
    reconstructed: pd.DataFrame,
    *,
    recent_weeks: int = 8,
    ewma_alpha: float = 0.35,
) -> pd.DataFrame:
    work = reconstructed.copy()
    work[DATE_COL] = pd.to_datetime(work[DATE_COL], errors="coerce")
    work[DOW_COL] = work[DATE_COL].dt.dayofweek
    iso = work[DATE_COL].dt.isocalendar()
    work["iso_year"] = iso.year.astype(int)
    work["iso_week"] = iso.week.astype(int)
    work["week_id"] = work["iso_year"].astype(str) + "-" + work["iso_week"].astype(str).str.zfill(2)

    result_parts: list[pd.DataFrame] = []
    group_cols = [BAKERY_COL, BAKERY_NAME_COL, CITY_COL, CATEGORY_COL, SKU_COL, SKU_NAME_COL]

    for _, group in work.groupby(group_cols, observed=True, dropna=False):
        group = group.sort_values(DATE_COL).copy()
        base_col = "reconstructed_sales_qty"
        group[base_col] = pd.to_numeric(group[base_col], errors="coerce").fillna(0.0)
        group["observed_sales_qty"] = pd.to_numeric(group["observed_sales_qty"], errors="coerce").fillna(0.0)

        weekly = (
            group.groupby(["iso_year", "iso_week"], as_index=False)
            .agg(
                week_start=(DATE_COL, "min"),
                week_total=(base_col, "sum"),
                week_mean=(base_col, "mean"),
                week_std=(base_col, lambda values: float(np.std(values, ddof=0))),
            )
            .sort_values("week_start")
            .reset_index(drop=True)
        )
        weekly["week_total_smoothed"] = (
            weekly["week_total"]
            .ewm(alpha=ewma_alpha, adjust=False)
            .mean()
        )
        weekly["week_mean_smoothed"] = weekly["week_mean"].ewm(alpha=ewma_alpha, adjust=False).mean()

        # Slow SKU trend stays local to the bakery x SKU series.
        trend = group[base_col].rolling(window=28, min_periods=7).median()
        trend = trend.ewm(span=14, adjust=False, min_periods=1).mean()
        group["sku_trend_normative"] = trend.bfill().ffill().fillna(0.0)

        weekday_profile = _compute_reference_weekday_profile(group, base_col, recent_weeks)

        week_lookup = weekly[["iso_year", "iso_week", "week_total_smoothed"]]
        group = group.merge(week_lookup, on=["iso_year", "iso_week"], how="left", validate="many_to_one")
        group = group.merge(
            weekday_profile[
                [
                    DOW_COL,
                    "long_weekday_share",
                    "short_weekday_share",
                    "share_delta_short_vs_long",
                    "weekday_share_normative",
                    "weekday_factor_normative",
                ]
            ],
            on=DOW_COL,
            how="left",
            validate="many_to_one",
        )
        group["long_weekday_share"] = group["long_weekday_share"].fillna(1.0 / 7.0)
        group["short_weekday_share"] = group["short_weekday_share"].fillna(group["long_weekday_share"])
        group["share_delta_short_vs_long"] = group["share_delta_short_vs_long"].fillna(0.0)
        group["weekday_share_normative"] = group["weekday_share_normative"].fillna(1.0 / 7.0)
        group["weekday_factor_normative"] = group["weekday_factor_normative"].fillna(1.0)

        # Legacy decomposition: weekly total first, then weekday allocation.
        group["temporal_normative_legacy_qty"] = group["week_total_smoothed"] * group["weekday_share_normative"]
        group["temporal_normative_legacy_qty"] = pd.to_numeric(
            group["temporal_normative_legacy_qty"],
            errors="coerce",
        ).clip(lower=0.0)

        weekly_profile = _build_weekly_strength_table(
            group,
            weekly,
            base_col=base_col,
            weekday_profile=weekday_profile,
        )

        for variant in ("weekly_cv", "flat_deviation", "reference_profile"):
            raw_col = f"{variant}_raw"
            z_col = f"{variant}_zscore"
            amp_col = f"{variant}_amplitude_multiplier"
            weekly_profile[z_col] = _robust_rolling_zscore(
                pd.to_numeric(weekly_profile[raw_col], errors="coerce").fillna(0.0),
                window=max(recent_weeks, 4),
            )
            if variant == "weekly_cv":
                weekly_profile[amp_col] = _zscore_to_amplitude_multiplier(weekly_profile[z_col], scale=0.42, lower=0.45, upper=1.9)
            elif variant == "flat_deviation":
                weekly_profile[amp_col] = _zscore_to_amplitude_multiplier(weekly_profile[z_col], scale=0.52, lower=0.35, upper=2.1)
            else:
                weekly_profile[amp_col] = _zscore_to_amplitude_multiplier(weekly_profile[z_col], scale=0.6, lower=0.25, upper=2.3)

        group = group.merge(
            weekly_profile[
                [
                    "iso_year",
                    "iso_week",
                    "weekly_cv_raw",
                    "weekly_cv_zscore",
                    "weekly_cv_amplitude_multiplier",
                    "flat_deviation_raw",
                    "flat_deviation_zscore",
                    "flat_deviation_amplitude_multiplier",
                    "reference_profile_raw",
                    "reference_profile_zscore",
                    "reference_profile_amplitude_multiplier",
                ]
            ],
            on=["iso_year", "iso_week"],
            how="left",
            validate="many_to_one",
        )

        for variant in ("weekly_cv", "flat_deviation", "reference_profile"):
            amp_col = f"{variant}_amplitude_multiplier"
            effective_col = f"{variant}_effective_strength"
            qty_col = f"temporal_normative_{variant}_qty"
            group[amp_col] = pd.to_numeric(group[amp_col], errors="coerce").fillna(1.0)
            if variant == "weekly_cv":
                group[effective_col] = _amplitude_to_effective_strength(group[amp_col], base_strength=2.0, lower=0.2, upper=3.0)
            elif variant == "flat_deviation":
                group[effective_col] = _amplitude_to_effective_strength(group[amp_col], base_strength=2.6, lower=0.15, upper=3.6)
            else:
                group[effective_col] = _amplitude_to_effective_strength(group[amp_col], base_strength=3.0, lower=0.1, upper=4.0)
            group[qty_col] = (
                group["sku_trend_normative"]
                * (1.0 + group[effective_col] * (group["weekday_factor_normative"] - 1.0))
            ).clip(lower=0.0)

        # Default exported temporal series points to the reference-profile variant.
        group["temporal_normative_qty"] = group[f"temporal_normative_{DEFAULT_VARIANT}_qty"]

        long_run_week_mean = float(weekly["week_total"].mean()) if len(weekly) else np.nan
        group["long_run_week_mean"] = long_run_week_mean
        group["weekly_amplitude_factor"] = np.where(
            pd.notna(long_run_week_mean) and abs(long_run_week_mean) > 1e-12,
            group["week_total_smoothed"] / long_run_week_mean,
            np.nan,
        )
        group["temporal_normative_abs_gap"] = (group["temporal_normative_qty"] - group["observed_sales_qty"]).abs()
        group["temporal_normative_bias"] = group["temporal_normative_qty"] - group["observed_sales_qty"]

        for variant in ("legacy", "weekly_cv", "flat_deviation", "reference_profile"):
            qty_col = "temporal_normative_legacy_qty" if variant == "legacy" else f"temporal_normative_{variant}_qty"
            gap_col = f"{variant}_abs_gap"
            bias_col = f"{variant}_bias"
            group[gap_col] = (group[qty_col] - group["observed_sales_qty"]).abs()
            group[bias_col] = group[qty_col] - group["observed_sales_qty"]

        result_parts.append(group)

    result = pd.concat(result_parts, ignore_index=True)
    return result.sort_values([BAKERY_COL, SKU_COL, DATE_COL]).reset_index(drop=True)


def build_summary(df: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(df)),
        "dates": int(df[DATE_COL].nunique()) if not df.empty else 0,
        "bakeries": int(df[BAKERY_COL].nunique()) if not df.empty else 0,
        "sku": int(df[SKU_COL].nunique()) if not df.empty else 0,
        "mean_observed_sales": round(float(pd.to_numeric(df["observed_sales_qty"], errors="coerce").mean()), 6) if not df.empty else 0.0,
        "mean_reconstructed_sales": round(float(pd.to_numeric(df["reconstructed_sales_qty"], errors="coerce").mean()), 6) if not df.empty else 0.0,
        "mean_temporal_normative_legacy_sales": round(float(pd.to_numeric(df["temporal_normative_legacy_qty"], errors="coerce").mean()), 6) if not df.empty else 0.0,
        "mean_temporal_normative_weekly_cv_sales": round(float(pd.to_numeric(df["temporal_normative_weekly_cv_qty"], errors="coerce").mean()), 6) if not df.empty else 0.0,
        "mean_temporal_normative_flat_deviation_sales": round(float(pd.to_numeric(df["temporal_normative_flat_deviation_qty"], errors="coerce").mean()), 6) if not df.empty else 0.0,
        "mean_temporal_normative_reference_profile_sales": round(float(pd.to_numeric(df["temporal_normative_reference_profile_qty"], errors="coerce").mean()), 6) if not df.empty else 0.0,
        "mean_temporal_normative_sales": round(float(pd.to_numeric(df["temporal_normative_qty"], errors="coerce").mean()), 6) if not df.empty else 0.0,
        "selected_variant": DEFAULT_VARIANT,
        "short_share_days": SHORT_SHARE_DAYS,
        "short_share_weight": SHORT_SHARE_WEIGHT,
        "mean_temporal_abs_gap": round(float(pd.to_numeric(df["temporal_normative_abs_gap"], errors="coerce").mean()), 6) if not df.empty else 0.0,
        "mean_temporal_bias": round(float(pd.to_numeric(df["temporal_normative_bias"], errors="coerce").mean()), 6) if not df.empty else 0.0,
        "variant_gap_summary": {
            "legacy": round(float(pd.to_numeric(df["legacy_abs_gap"], errors="coerce").mean()), 6) if not df.empty else 0.0,
            "weekly_cv": round(float(pd.to_numeric(df["weekly_cv_abs_gap"], errors="coerce").mean()), 6) if not df.empty else 0.0,
            "flat_deviation": round(float(pd.to_numeric(df["flat_deviation_abs_gap"], errors="coerce").mean()), 6) if not df.empty else 0.0,
            "reference_profile": round(float(pd.to_numeric(df["reference_profile_abs_gap"], errors="coerce").mean()), 6) if not df.empty else 0.0,
        },
    }


def save_outputs(output_dir: str | Path, df: pd.DataFrame, summary: dict[str, object]) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / OUTPUT_NAME
    summary_path = out_dir / SUMMARY_OUTPUT
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"temporal_normative": csv_path, "summary": summary_path}


def build_kazan_temporal_normative_sku_day(
    *,
    reconstructed_path: str | Path,
    output_dir: str | Path,
    recent_weeks: int = 8,
    ewma_alpha: float = 0.35,
) -> dict[str, Path]:
    reconstructed = load_csv(reconstructed_path)
    temporal = build_temporal_normative(
        reconstructed,
        recent_weeks=recent_weeks,
        ewma_alpha=ewma_alpha,
    )
    summary = build_summary(temporal)
    return save_outputs(output_dir, temporal, summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build temporal normative sku-day series for Kazan sample")
    parser.add_argument("--reconstructed-path", default=str(ROOT / "data" / "processed" / "kazan_reconstructed_sku_day.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--recent-weeks", type=int, default=8)
    parser.add_argument("--ewma-alpha", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_kazan_temporal_normative_sku_day(
        reconstructed_path=args.reconstructed_path,
        output_dir=args.output_dir,
        recent_weeks=args.recent_weeks,
        ewma_alpha=args.ewma_alpha,
    )
    print("=" * 72)
    print("KAZAN TEMPORAL NORMATIVE SKU DAY")
    print("=" * 72)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

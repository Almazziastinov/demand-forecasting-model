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


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        if DOW_COL not in df.columns:
            df[DOW_COL] = df[DATE_COL].dt.dayofweek
    return df


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

        weekly = (
            group.groupby(["iso_year", "iso_week"], as_index=False)
            .agg(
                week_start=(DATE_COL, "min"),
                week_total=(base_col, "sum"),
            )
            .sort_values("week_start")
            .reset_index(drop=True)
        )
        weekly["week_total_smoothed"] = (
            weekly["week_total"]
            .ewm(alpha=ewma_alpha, adjust=False)
            .mean()
        )

        recent_days = group.tail(recent_weeks * 7).copy()
        weekday_profile = (
            recent_days.groupby(DOW_COL, as_index=False)[base_col]
            .median()
            .rename(columns={base_col: "weekday_profile_raw"})
        )
        weekday_profile = weekday_profile.set_index(DOW_COL).reindex(range(7), fill_value=0.0).reset_index()
        profile_sum = weekday_profile["weekday_profile_raw"].sum()
        if profile_sum <= 0:
            weekday_profile["weekday_share_normative"] = 1.0 / 7.0
        else:
            weekday_profile["weekday_share_normative"] = weekday_profile["weekday_profile_raw"] / profile_sum

        week_lookup = weekly[["iso_year", "iso_week", "week_total_smoothed"]]
        group = group.merge(week_lookup, on=["iso_year", "iso_week"], how="left", validate="many_to_one")
        group = group.merge(
            weekday_profile[[DOW_COL, "weekday_share_normative"]],
            on=DOW_COL,
            how="left",
            validate="many_to_one",
        )
        group["weekday_share_normative"] = group["weekday_share_normative"].fillna(1.0 / 7.0)

        group["temporal_normative_qty"] = group["week_total_smoothed"] * group["weekday_share_normative"]
        group["temporal_normative_qty"] = pd.to_numeric(group["temporal_normative_qty"], errors="coerce").clip(lower=0.0)

        long_run_week_mean = float(weekly["week_total"].mean()) if len(weekly) else np.nan
        group["long_run_week_mean"] = long_run_week_mean
        group["weekly_amplitude_factor"] = np.where(
            pd.notna(long_run_week_mean) and abs(long_run_week_mean) > 1e-12,
            group["week_total_smoothed"] / long_run_week_mean,
            np.nan,
        )
        group["temporal_normative_abs_gap"] = (group["temporal_normative_qty"] - group["observed_sales_qty"]).abs()
        group["temporal_normative_bias"] = group["temporal_normative_qty"] - group["observed_sales_qty"]

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
        "mean_temporal_normative_sales": round(float(pd.to_numeric(df["temporal_normative_qty"], errors="coerce").mean()), 6) if not df.empty else 0.0,
        "mean_temporal_abs_gap": round(float(pd.to_numeric(df["temporal_normative_abs_gap"], errors="coerce").mean()), 6) if not df.empty else 0.0,
        "mean_temporal_bias": round(float(pd.to_numeric(df["temporal_normative_bias"], errors="coerce").mean()), 6) if not df.empty else 0.0,
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

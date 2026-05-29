from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DAILY_PATH = ROOT / "data" / "processed" / "bakery_daily_sales.csv"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "sales_cleaning_audit"

DATE_COL = "date"
BAKERY_ID_COL = "bakery_id"
BAKERY_NAME_COL = "bakery_name"
TARGET_COL = "bakery_sales"
OBSERVED_TARGET_COL = "bakery_sales_observed"
SALES_MISSING_FLAG_COL = "sales_missing_flag"
SALES_IMPUTED_VALUE_COL = "sales_imputed_value"
CAPPED_COL = "bakery_sales_base_rolling_capped"
CAP_FLAG_COL = "rolling_base_target_capped_flag"
CAP_DELTA_COL = "rolling_base_target_cap_delta"

# Russian federal non-working holidays (fixed-date subset relevant for retail demand).
# Used purely as a sanity check for whether the cap is still absorbing event spikes.
KNOWN_HOLIDAYS_BY_YEAR: dict[int, list[str]] = {
    2025: [
        "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04",
        "2025-01-05", "2025-01-06", "2025-01-07", "2025-01-08",
        "2025-02-23", "2025-03-08", "2025-05-01", "2025-05-09",
        "2025-06-12", "2025-11-04",
    ],
    2026: [
        "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04",
        "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08",
        "2026-02-23", "2026-03-08", "2026-05-01", "2026-05-09",
        "2026-06-12", "2026-11-04",
    ],
}


def load_daily(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    return df.dropna(subset=[DATE_COL]).copy()


def _numeric_column(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def build_overall_summary(df: pd.DataFrame) -> dict[str, object]:
    observed_sales = _numeric_column(df, OBSERVED_TARGET_COL)
    if OBSERVED_TARGET_COL not in df.columns:
        observed_sales = _numeric_column(df, TARGET_COL)
    filled_sales = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0.0)
    missing_mask = (
        _numeric_column(df, SALES_MISSING_FLAG_COL).fillna(0).astype(int) == 1
    )
    imputed_sales = _numeric_column(df, SALES_IMPUTED_VALUE_COL).fillna(0.0)
    cap_delta = pd.to_numeric(df[CAP_DELTA_COL], errors="coerce").fillna(0.0)
    capped_mask = (
        pd.to_numeric(df[CAP_FLAG_COL], errors="coerce").fillna(0).astype(int) == 1
    )
    high_mask = (
        pd.to_numeric(df["sales_high_outlier_flag"], errors="coerce")
        .fillna(0)
        .astype(int)
        == 1
    )
    low_mask = (
        pd.to_numeric(df["sales_low_outlier_flag"], errors="coerce")
        .fillna(0)
        .astype(int)
        == 1
    )

    return {
        "rows": int(len(df)),
        "date_min": str(df[DATE_COL].min().date()) if len(df) else None,
        "date_max": str(df[DATE_COL].max().date()) if len(df) else None,
        "bakeries": int(df[BAKERY_ID_COL].nunique()) if len(df) else 0,
        "observed_sales_sum": round(float(observed_sales.sum()), 6),
        "filled_sales_sum": round(float(filled_sales.sum()), 6),
        "missing_sales_rows": int(missing_mask.sum()),
        "missing_sales_share": round(float(missing_mask.mean()), 6) if len(df) else 0.0,
        "imputed_sales_sum": round(float(imputed_sales.sum()), 6),
        "base_capped_sum": round(float(df[CAPPED_COL].sum()), 6),
        "cap_delta_sum": round(float(cap_delta.sum()), 6),
        "capped_rows": int(capped_mask.sum()),
        "capped_share": round(float(capped_mask.mean()), 6) if len(df) else 0.0,
        "high_outlier_rows": int(high_mask.sum()),
        "low_outlier_rows": int(low_mask.sum()),
        "negative_cap_delta_rows": int((cap_delta < 0).sum()),
        "positive_cap_delta_rows": int((cap_delta > 0).sum()),
    }


def build_bakery_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["_observed_sales"] = _numeric_column(work, OBSERVED_TARGET_COL)
    if OBSERVED_TARGET_COL not in work.columns:
        work["_observed_sales"] = _numeric_column(work, TARGET_COL)
    work["_missing"] = (
        _numeric_column(work, SALES_MISSING_FLAG_COL).fillna(0).astype(int)
    )
    work["_capped"] = (
        pd.to_numeric(work[CAP_FLAG_COL], errors="coerce").fillna(0).astype(int)
    )
    work["_high"] = (
        pd.to_numeric(work["sales_high_outlier_flag"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    work["_low"] = (
        pd.to_numeric(work["sales_low_outlier_flag"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    work[CAP_DELTA_COL] = pd.to_numeric(work[CAP_DELTA_COL], errors="coerce").fillna(
        0.0
    )

    summary = (
        work.groupby([BAKERY_ID_COL, BAKERY_NAME_COL], as_index=False)
        .agg(
            rows=(TARGET_COL, "size"),
            observed_sales_sum=("_observed_sales", "sum"),
            filled_sales_sum=(TARGET_COL, "sum"),
            capped_sales_sum=(CAPPED_COL, "sum"),
            cap_delta_sum=(CAP_DELTA_COL, "sum"),
            abs_cap_delta_sum=(CAP_DELTA_COL, lambda s: float(s.abs().sum())),
            missing_sales_rows=("_missing", "sum"),
            capped_rows=("_capped", "sum"),
            high_outlier_rows=("_high", "sum"),
            low_outlier_rows=("_low", "sum"),
        )
        .sort_values("abs_cap_delta_sum", ascending=False)
        .reset_index(drop=True)
    )
    summary["capped_share"] = summary["capped_rows"] / summary["rows"].replace(0, pd.NA)
    summary["cap_delta_pct_of_sales"] = summary["cap_delta_sum"] / summary[
        "observed_sales_sum"
    ].replace(0, pd.NA)
    return summary


def build_date_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["_observed_sales"] = _numeric_column(work, OBSERVED_TARGET_COL)
    if OBSERVED_TARGET_COL not in work.columns:
        work["_observed_sales"] = _numeric_column(work, TARGET_COL)
    work["_missing"] = (
        _numeric_column(work, SALES_MISSING_FLAG_COL).fillna(0).astype(int)
    )
    work["_capped"] = (
        pd.to_numeric(work[CAP_FLAG_COL], errors="coerce").fillna(0).astype(int)
    )
    work["_high"] = (
        pd.to_numeric(work["sales_high_outlier_flag"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    work["_low"] = (
        pd.to_numeric(work["sales_low_outlier_flag"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    work[CAP_DELTA_COL] = pd.to_numeric(work[CAP_DELTA_COL], errors="coerce").fillna(
        0.0
    )

    summary = (
        work.groupby(DATE_COL, as_index=False)
        .agg(
            rows=(TARGET_COL, "size"),
            observed_sales_sum=("_observed_sales", "sum"),
            filled_sales_sum=(TARGET_COL, "sum"),
            capped_sales_sum=(CAPPED_COL, "sum"),
            cap_delta_sum=(CAP_DELTA_COL, "sum"),
            abs_cap_delta_sum=(CAP_DELTA_COL, lambda s: float(s.abs().sum())),
            missing_sales_rows=("_missing", "sum"),
            capped_rows=("_capped", "sum"),
            high_outlier_rows=("_high", "sum"),
            low_outlier_rows=("_low", "sum"),
        )
        .sort_values("abs_cap_delta_sum", ascending=False)
        .reset_index(drop=True)
    )
    summary[DATE_COL] = summary[DATE_COL].dt.strftime("%Y-%m-%d")
    return summary


def build_dow_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["_observed_sales"] = _numeric_column(work, OBSERVED_TARGET_COL)
    if OBSERVED_TARGET_COL not in work.columns:
        work["_observed_sales"] = _numeric_column(work, TARGET_COL)
    work["_missing"] = (
        _numeric_column(work, SALES_MISSING_FLAG_COL).fillna(0).astype(int)
    )
    work["_capped"] = (
        pd.to_numeric(work[CAP_FLAG_COL], errors="coerce").fillna(0).astype(int)
    )
    work["_high"] = (
        pd.to_numeric(work["sales_high_outlier_flag"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    work["_low"] = (
        pd.to_numeric(work["sales_low_outlier_flag"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    work[CAP_DELTA_COL] = pd.to_numeric(work[CAP_DELTA_COL], errors="coerce").fillna(
        0.0
    )

    summary = (
        work.groupby("dow", as_index=False)
        .agg(
            rows=(TARGET_COL, "size"),
            observed_sales_sum=("_observed_sales", "sum"),
            filled_sales_sum=(TARGET_COL, "sum"),
            capped_sales_sum=(CAPPED_COL, "sum"),
            cap_delta_sum=(CAP_DELTA_COL, "sum"),
            abs_cap_delta_sum=(CAP_DELTA_COL, lambda s: float(s.abs().sum())),
            missing_sales_rows=("_missing", "sum"),
            capped_rows=("_capped", "sum"),
            high_outlier_rows=("_high", "sum"),
            low_outlier_rows=("_low", "sum"),
        )
        .sort_values("dow")
        .reset_index(drop=True)
    )
    summary["capped_share"] = summary["capped_rows"] / summary["rows"].replace(0, pd.NA)
    return summary


def build_holiday_hit_rate(df: pd.DataFrame) -> dict[str, object]:
    """Share of known holidays that show up as bakery-day high-outlier flags.

    Reading guide (see PRODUCTION_PREPROCESSING_PLAN.md):
      hit_rate > 0.30 -> cap is still doing event handling on its own.
      hit_rate < 0.10 -> events are absorbed into q95 norm; revisit base cap.
    """
    if DATE_COL not in df.columns or "sales_high_outlier_flag" not in df.columns:
        return {"applicable": False}

    dates = pd.to_datetime(df[DATE_COL], errors="coerce")
    high_mask = (
        pd.to_numeric(df["sales_high_outlier_flag"], errors="coerce")
        .fillna(0)
        .astype(int)
        == 1
    )
    if not len(dates):
        return {"applicable": False}

    min_year = int(dates.min().year)
    max_year = int(dates.max().year)
    holidays: list[str] = []
    for year in range(min_year, max_year + 1):
        holidays.extend(KNOWN_HOLIDAYS_BY_YEAR.get(year, []))
    holiday_set = {h for h in holidays if min_year <= int(h[:4]) <= max_year}
    if not holiday_set:
        return {"applicable": False}

    date_strings = dates.dt.strftime("%Y-%m-%d")
    in_range = holiday_set & set(date_strings.dropna().unique())
    if not in_range:
        return {"applicable": False}

    flagged_dates = set(date_strings[high_mask].dropna().unique())
    hits = in_range & flagged_dates

    bakeries_per_date = (
        df.assign(_d=date_strings, _h=high_mask.astype(int))
        .groupby("_d")["_h"]
        .agg(["sum", "size"])
    )
    bakeries_per_date["share"] = bakeries_per_date["sum"] / bakeries_per_date[
        "size"
    ].replace(0, pd.NA)
    per_holiday = []
    for h in sorted(in_range):
        if h in bakeries_per_date.index:
            row = bakeries_per_date.loc[h]
            per_holiday.append({
                "date": h,
                "high_outlier_bakeries": int(row["sum"]),
                "total_bakeries": int(row["size"]),
                "share": round(
                    float(row["share"]) if pd.notna(row["share"]) else 0.0,
                    6,
                ),
                "is_hit": h in hits,
            })

    return {
        "applicable": True,
        "holidays_in_range": sorted(in_range),
        "hits": sorted(hits),
        "hit_rate": round(len(hits) / len(in_range), 6),
        "per_holiday": per_holiday,
    }


def save_audit(df: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overall = build_overall_summary(df)
    overall["holiday_hit_rate"] = build_holiday_hit_rate(df)
    bakery_summary = build_bakery_summary(df)
    date_summary = build_date_summary(df)
    dow_summary = build_dow_summary(df)
    top_capped = df.assign(
        abs_cap_delta=pd.to_numeric(df[CAP_DELTA_COL], errors="coerce")
        .fillna(0.0)
        .abs()
    ).sort_values("abs_cap_delta", ascending=False)

    paths = {
        "summary": out_dir / "summary.json",
        "bakery_summary": out_dir / "bakery_summary.csv",
        "date_summary": out_dir / "date_summary.csv",
        "dow_summary": out_dir / "dow_summary.csv",
        "top_capped_rows": out_dir / "top_capped_rows.csv",
    }
    paths["summary"].write_text(
        json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    bakery_summary.to_csv(paths["bakery_summary"], index=False, encoding="utf-8-sig")
    date_summary.to_csv(paths["date_summary"], index=False, encoding="utf-8-sig")
    dow_summary.to_csv(paths["dow_summary"], index=False, encoding="utf-8-sig")
    top_capped.head(500).to_csv(
        paths["top_capped_rows"], index=False, encoding="utf-8-sig"
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit bakery daily sales cleaning")
    parser.add_argument("--daily-path", default=str(DEFAULT_DAILY_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_daily(args.daily_path)
    paths = save_audit(df, args.output_dir)
    print("=" * 72)
    print("SALES CLEANING AUDIT")
    print("=" * 72)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

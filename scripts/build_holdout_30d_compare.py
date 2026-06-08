"""Build a tidy bakery-level forecast-vs-fact CSV from the 30d holdout artifacts.

Source files:
  reports/bakery_day_model_holdout_predictions.csv          (base target)
  reports/bakery_day_model_uplifted_holdout_predictions.csv (uplifted target)

Window: 2026-05-02..2026-05-31 (30 days).
Output: reports/holdout_30d_bakery_compare.csv

Both source files already contain (date, bakery_id, bakery_name, city,
bakery_sales, bakery_day_forecast). `bakery_sales` is the fact for the
corresponding target (base vs uplifted), and they differ — uplifted fact is
inflated by the uplift coefficient. We keep both so charts can show:
  - line "fact (base)"        — raw daily sales
  - line "fact (uplifted)"    — fact normalized to the uplifted target space
  - line "forecast (base)"    — what the base bakery model predicted
  - line "forecast (uplifted)"— what the uplifted bakery model predicted

This script does no modeling; it only joins, validates row counts, and writes
the merged CSV.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_CSV = REPO_ROOT / "reports" / "bakery_day_model_holdout_predictions.csv"
UPLIFTED_CSV = (
    REPO_ROOT / "reports" / "bakery_day_model_uplifted_holdout_predictions.csv"
)
OUT_CSV = REPO_ROOT / "reports" / "holdout_30d_bakery_compare.csv"


def _read_holdout(path: Path, suffix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lstrip("﻿") for c in df.columns]
    expected = {"date", "bakery_id", "bakery_name", "city",
                "bakery_sales", "bakery_day_forecast"}
    missing = expected - set(df.columns)
    if missing:
        raise SystemExit(f"{path.name}: missing columns {missing}")
    df = df.rename(columns={
        "bakery_sales": f"fact_{suffix}",
        "bakery_day_forecast": f"forecast_{suffix}",
    })
    df["date"] = pd.to_datetime(df["date"])
    return df


def main() -> int:
    base = _read_holdout(BASE_CSV, "base")
    uplifted = _read_holdout(UPLIFTED_CSV, "uplifted")

    keys = ["date", "bakery_id", "bakery_name", "city"]
    merged = base.merge(uplifted, on=keys, how="outer", validate="one_to_one")

    # Errors and abs errors for both tracks.
    merged["err_base"] = merged["forecast_base"] - merged["fact_base"]
    merged["err_uplifted"] = merged["forecast_uplifted"] - merged["fact_uplifted"]
    merged["abs_err_base"] = merged["err_base"].abs()
    merged["abs_err_uplifted"] = merged["err_uplifted"].abs()

    merged = merged.sort_values(["bakery_id", "date"]).reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_CSV, index=False, encoding="utf-8")

    # Console summary (ASCII only — cp1251-safe).
    n_rows = len(merged)
    n_bakeries = merged["bakery_id"].nunique()
    date_min = merged["date"].min().date()
    date_max = merged["date"].max().date()
    mae_base = merged["abs_err_base"].mean()
    mae_uplifted = merged["abs_err_uplifted"].mean()
    wmape_base = merged["abs_err_base"].sum() / merged["fact_base"].sum() * 100
    wmape_uplifted = (
        merged["abs_err_uplifted"].sum() / merged["fact_uplifted"].sum() * 100
    )

    print(f"output: {OUT_CSV.relative_to(REPO_ROOT)}")
    print(f"rows:           {n_rows}")
    print(f"bakeries:       {n_bakeries}")
    print(f"date range:     {date_min} .. {date_max}")
    print(f"MAE   base:     {mae_base:.4f}")
    print(f"MAE   uplifted: {mae_uplifted:.4f}")
    print(f"WMAPE base:     {wmape_base:.4f} %")
    print(f"WMAPE uplifted: {wmape_uplifted:.4f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())

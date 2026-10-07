"""Extend the trusted 39-bakery/93-SKU scope to every August date."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/historical_daily_retraining_gate_loss075_full_20260910/predictions.parquet"
TRUSTED = ROOT / "reports/prod_direct_end_of_day_economics_august_folds_20260910/daily_rows.parquet"
TOTALS = ROOT / "reports/network_trained_bakery_p50_loss075_full_20260910/bakery_totals.parquet"
OUTPUT = Path(
    os.environ.get(
        "AUGUST_FIXED_SCOPE_OUTPUT",
        ROOT / "reports/august_fixed_39_bakeries_93_skus_full_month_20260911",
    )
)


def main() -> None:
    trusted = pd.read_parquet(TRUSTED, columns=["bakery_id", "product_id"])
    bakery_ids = set(trusted["bakery_id"].unique())
    product_ids = set(trusted["product_id"].unique())

    rows = pd.read_parquet(SOURCE)
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    rows = rows[
        rows["date"].between("2026-08-01", "2026-08-31")
        & rows["bakery_id"].isin(bakery_ids)
        & rows["product_id"].isin(product_ids)
    ].copy()
    if os.environ.get("AUGUST_TRUSTED_DATES_ONLY") == "1":
        trusted_dates = set(
            pd.to_datetime(
                pd.read_parquet(TRUSTED, columns=["date"])["date"]
            ).dt.normalize()
        )
        rows = rows[rows["date"].isin(trusted_dates)].copy()
    totals = pd.read_parquet(TOTALS)
    totals["date"] = pd.to_datetime(totals["date"]).dt.normalize()
    rows = rows.drop(columns=["p50_total", "p50_plan"], errors="ignore").merge(
        totals, on=["date", "bakery_id"], how="inner", validate="many_to_one"
    )
    share_sum = rows.groupby(["date", "bakery_id"])["share"].transform("sum")
    rows = rows[share_sum.gt(0)].copy()
    share_sum = rows.groupby(["date", "bakery_id"])["share"].transform("sum")
    rows["share"] = rows["share"] / share_sum
    rows["direct_plan"] = rows["direct_total"] * rows["share"]
    rows["p50_total"] = rows["p50_total_exact"]
    rows["p50_plan"] = rows["p50_total"] * rows["share"]
    rows["gate_plan"] = rows["direct_plan"]
    rows["use_p50"] = False
    rows["gate_prior_advantage"] = np.nan

    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(OUTPUT / "predictions.parquet", index=False)
    print(
        {
            "dates": rows["date"].nunique(),
            "bakeries": rows["bakery_id"].nunique(),
            "skus": rows["product_id"].nunique(),
            "rows": len(rows),
            "sales": float(rows["observed_sales_qty"].sum()),
        }
    )


if __name__ == "__main__":
    main()

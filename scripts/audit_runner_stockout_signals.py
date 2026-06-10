"""Audit stockout/lost-demand signals for problematic runner SKU.

This diagnostic is meant to answer whether low local runner sales/share may be
caused by constrained release rather than truly weak demand.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "processed" / "sku_daily_research_base.csv"
BEST_VARIANT_PATH = (
    REPO_ROOT
    / "reports"
    / "prod_holdout_sku_backtest_variants"
    / "runner_recent_heavy_weekpart_by_bakery_sku.csv"
)
RISK_PATH = (
    REPO_ROOT
    / "reports"
    / "rollout_sku_risk_audit_runner_recent_heavy_weekpart"
    / "bakery_sku_risk_summary.csv"
)
OUT_DIR = REPO_ROOT / "reports" / "runner_stockout_audit"

RECENT_START = "2026-04-02"
RECENT_END = "2026-05-01"
HOLDOUT_START = "2026-05-02"
HOLDOUT_END = "2026-05-12"

SERVICE_PATTERN = "прочие|заказ"


def _load_problem_pairs() -> pd.DataFrame:
    sku = pd.read_csv(BEST_VARIANT_PATH)
    risk = pd.read_csv(RISK_PATH)
    runner_bakeries = set(
        risk[
            risk["risk_flags"].fillna("").str.contains("runner_sku", regex=False)
        ]["bakery_id"]
    )
    service = sku["category_name"].fillna("").str.casefold().str.contains(
        SERVICE_PATTERN,
        regex=True,
    )
    runner = sku[
        (sku["fact_qty"] >= 500)
        & (sku["recent_days_sold"] >= 20)
        & ~service
    ].copy()
    runner["bias_qty"] = runner["forecast_variant"] - runner["fact_qty"]
    runner["is_severe"] = runner["bias_qty"].abs() >= np.maximum(
        100.0,
        runner["fact_qty"] * 0.35,
    )
    runner["is_runner_risk_bakery"] = runner["bakery_id"].isin(runner_bakeries)
    runner["is_underforecast_focus"] = runner["bias_qty"] <= -250
    runner["is_overforecast_focus"] = (
        runner["is_runner_risk_bakery"] & runner["is_severe"]
    )
    return runner[
        runner["is_underforecast_focus"] | runner["is_overforecast_focus"]
    ].copy()


def _load_daily_for_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    usecols = [
        "date",
        "bakery_id",
        "bakery_name",
        "city",
        "product_id",
        "product_name",
        "category_name",
        "observed_sales_qty",
        "bakery_sales_qty_total",
        "sku_sales_share_in_bakery_day",
        "release_qty",
        "release_present_flag",
        "release_to_sales_ratio",
        "available_qty_proxy",
        "first_sale_hour",
        "last_sale_hour",
        "sales_hours_count",
        "row_quality_score",
    ]
    keys = pairs[["bakery_id", "product_id"]].drop_duplicates()
    chunks = []
    for chunk in pd.read_csv(DATA_PATH, usecols=usecols, chunksize=500_000):
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        mask = (chunk["date"] >= RECENT_START) & (chunk["date"] <= HOLDOUT_END)
        if not mask.any():
            continue
        filtered = chunk.loc[mask].merge(
            keys,
            on=["bakery_id", "product_id"],
            how="inner",
        )
        if not filtered.empty:
            chunks.append(filtered)
    if chunks:
        return pd.concat(chunks, ignore_index=True)
    return pd.DataFrame(columns=usecols)


def _summarize_daily(daily: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pairs
    work = daily.copy()
    for col in [
        "observed_sales_qty",
        "bakery_sales_qty_total",
        "sku_sales_share_in_bakery_day",
        "release_qty",
        "release_to_sales_ratio",
        "available_qty_proxy",
        "first_sale_hour",
        "last_sale_hour",
        "sales_hours_count",
        "row_quality_score",
    ]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work["window"] = np.where(work["date"] <= RECENT_END, "recent", "holdout_overlap")
    work["sales_to_release_ratio"] = np.where(
        work["release_qty"] > 0,
        work["observed_sales_qty"] / work["release_qty"],
        np.nan,
    )
    work["sold_to_release_flag"] = (
        (work["release_qty"] > 0)
        & (work["observed_sales_qty"] >= work["release_qty"] * 0.9)
    )
    work["early_last_sale_flag"] = work["last_sale_hour"] <= 16
    work["hard_early_last_sale_flag"] = work["last_sale_hour"] <= 15
    work["stockout_like_day"] = (
        work["sold_to_release_flag"] & work["early_last_sale_flag"]
    )
    work["hard_stockout_like_day"] = (
        work["sold_to_release_flag"] & work["hard_early_last_sale_flag"]
    )
    work["release_missing_or_zero"] = work["release_qty"].fillna(0) <= 0

    grouped = (
        work.groupby(["bakery_id", "product_id", "window"], as_index=False)
        .agg(
            days=("date", "nunique"),
            sales_qty=("observed_sales_qty", "sum"),
            release_qty=("release_qty", "sum"),
            mean_daily_share=("sku_sales_share_in_bakery_day", "mean"),
            median_daily_share=("sku_sales_share_in_bakery_day", "median"),
            p90_daily_share=(
                "sku_sales_share_in_bakery_day",
                lambda x: x.quantile(0.90),
            ),
            mean_last_sale_hour=("last_sale_hour", "mean"),
            median_last_sale_hour=("last_sale_hour", "median"),
            early_last_sale_days=("early_last_sale_flag", "sum"),
            sold_to_release_days=("sold_to_release_flag", "sum"),
            stockout_like_days=("stockout_like_day", "sum"),
            hard_stockout_like_days=("hard_stockout_like_day", "sum"),
            release_missing_or_zero_days=("release_missing_or_zero", "sum"),
            mean_sales_to_release_ratio=("sales_to_release_ratio", "mean"),
            median_sales_to_release_ratio=("sales_to_release_ratio", "median"),
            mean_row_quality=("row_quality_score", "mean"),
        )
    )
    pivot = grouped.pivot_table(
        index=["bakery_id", "product_id"],
        columns="window",
        values=[
            "days",
            "sales_qty",
            "release_qty",
            "mean_daily_share",
            "median_daily_share",
            "p90_daily_share",
            "mean_last_sale_hour",
            "median_last_sale_hour",
            "early_last_sale_days",
            "sold_to_release_days",
            "stockout_like_days",
            "hard_stockout_like_days",
            "release_missing_or_zero_days",
            "mean_sales_to_release_ratio",
            "median_sales_to_release_ratio",
            "mean_row_quality",
        ],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{window}" for metric, window in pivot.columns]
    pivot = pivot.reset_index()

    out = pairs.merge(pivot, on=["bakery_id", "product_id"], how="left")
    for prefix in ["recent", "holdout_overlap"]:
        days = out[f"days_{prefix}"].replace(0, np.nan)
        out[f"stockout_like_share_{prefix}"] = (
            out[f"stockout_like_days_{prefix}"] / days
        )
        out[f"hard_stockout_like_share_{prefix}"] = (
            out[f"hard_stockout_like_days_{prefix}"] / days
        )
        out[f"sold_to_release_share_{prefix}"] = (
            out[f"sold_to_release_days_{prefix}"] / days
        )
        out[f"early_last_sale_share_{prefix}"] = (
            out[f"early_last_sale_days_{prefix}"] / days
        )
    out["stockout_score"] = (
        out["stockout_like_share_recent"].fillna(0) * 0.45
        + out["stockout_like_share_holdout_overlap"].fillna(0) * 0.35
        + out["hard_stockout_like_share_recent"].fillna(0) * 0.10
        + out["hard_stockout_like_share_holdout_overlap"].fillna(0) * 0.10
    )
    out["stockout_interpretation"] = np.select(
        [
            out["stockout_score"] >= 0.35,
            out["stockout_score"] >= 0.15,
        ],
        ["likely_constrained_demand", "possible_constrained_demand"],
        default="no_strong_stockout_signal",
    )
    return out


def _city_runner_prior(daily: pd.DataFrame) -> pd.DataFrame:
    work = daily.copy()
    work = work[(work["date"] >= RECENT_START) & (work["date"] <= RECENT_END)]
    if work.empty:
        return pd.DataFrame()
    city = (
        work.groupby(["city", "product_id"], as_index=False)
        .agg(city_sales_qty=("observed_sales_qty", "sum"))
    )
    totals = (
        work.groupby("city", as_index=False)["observed_sales_qty"]
        .sum()
        .rename(columns={"observed_sales_qty": "city_total_sales_qty"})
    )
    city = city.merge(totals, on="city", how="left")
    city["city_recent_share"] = city["city_sales_qty"] / city["city_total_sales_qty"]
    city["city_rank"] = city.groupby("city")["city_sales_qty"].rank(
        method="dense",
        ascending=False,
    )
    return city


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = _load_problem_pairs()
    daily = _load_daily_for_pairs(pairs)
    audit = _summarize_daily(daily, pairs)
    city_prior = _city_runner_prior(daily)
    if not city_prior.empty:
        audit = audit.merge(city_prior, on=["city", "product_id"], how="left")
    audit = audit.sort_values(
        ["stockout_score", "bias_qty"],
        ascending=[False, True],
    )
    audit.to_csv(
        OUT_DIR / "runner_stockout_problem_pairs.csv",
        index=False,
        encoding="utf-8-sig",
    )
    daily.to_csv(
        OUT_DIR / "runner_stockout_daily_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )

    counts = audit["stockout_interpretation"].value_counts().rename_axis(
        "stockout_interpretation"
    )
    print(counts.to_string())
    print(f"Wrote {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

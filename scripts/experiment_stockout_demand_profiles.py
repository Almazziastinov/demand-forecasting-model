"""Compare sales-based and reconstructed-demand hourly profiles locally.

The script only reads local CSV files and writes local report artifacts. It
has no ClickHouse client, deployment hook, or production pipeline integration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments_v2.stockout_demand_preprocessing import (  # noqa: E402
    aggregate_daily_training_target,
    build_bakery_share_reference,
    build_uncensored_hour_reference,
    mark_stockout_days,
    reconstruct_stockout_demand,
    reconstruct_stockout_demand_from_bakery_share,
)


PILOT_BAKERY_IDS = {20, 21, 22, 28, 80, 89, 107, 221, 222, 257}
BAKEABLE_CATEGORIES = {
    "Пироги сытные",
    "Пироги сладкие",
    "Выпечка сытная",
    "Выпечка сладкая",
    "Фастфуд",
}
PROFILE_KEYS = ["bakery_id", "product_id", "dow", "hour"]


def _numeric_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype("int64")


def load_sales(path: Path, *, chunksize: int = 750_000) -> pd.DataFrame:
    usecols = [
        "check_datetime",
        "check_date",
        "cash_event_type",
        "quantity",
        "bakery_id",
        "product_id",
        "category_name",
    ]
    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        chunk["bakery_id"] = _numeric_id(chunk["bakery_id"])
        chunk = chunk[
            chunk["bakery_id"].isin(PILOT_BAKERY_IDS)
            & (chunk["cash_event_type"] == "Продажа")
            & chunk["category_name"].isin(BAKEABLE_CATEGORIES)
        ].copy()
        if chunk.empty:
            continue
        chunk["product_id"] = _numeric_id(chunk["product_id"])
        chunk["date"] = pd.to_datetime(chunk["check_date"], errors="coerce")
        chunk["hour"] = (
            pd.to_datetime(chunk["check_datetime"], errors="coerce", utc=True)
            .dt.tz_convert("Europe/Moscow")
            .dt.hour
        )
        chunk["sold"] = pd.to_numeric(chunk["quantity"], errors="coerce").fillna(0.0)
        parts.append(
            chunk.groupby(
                ["date", "bakery_id", "product_id", "category_name", "hour"],
                as_index=False,
            )["sold"].sum()
        )
    if not parts:
        return pd.DataFrame()
    return (
        pd.concat(parts, ignore_index=True)
        .groupby(
            ["date", "bakery_id", "product_id", "category_name", "hour"],
            as_index=False,
        )["sold"]
        .sum()
    )


def load_production(path: Path, *, chunksize: int = 750_000) -> pd.DataFrame:
    usecols = ["release_date", "bakery_id", "product_id", "quantity"]
    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        chunk["bakery_id"] = _numeric_id(chunk["bakery_id"])
        chunk = chunk[chunk["bakery_id"].isin(PILOT_BAKERY_IDS)].copy()
        if chunk.empty:
            continue
        chunk["product_id"] = _numeric_id(chunk["product_id"])
        chunk["date"] = pd.to_datetime(chunk["release_date"], errors="coerce")
        chunk["produced"] = pd.to_numeric(chunk["quantity"], errors="coerce").fillna(
            0.0
        )
        parts.append(
            chunk.groupby(["date", "bakery_id", "product_id"], as_index=False)[
                "produced"
            ].sum()
        )
    return (
        pd.concat(parts, ignore_index=True)
        .groupby(["date", "bakery_id", "product_id"], as_index=False)["produced"]
        .sum()
    )


def complete_hourly_frame(
    sales: pd.DataFrame, *, first_hour: int = 6, last_hour: int = 23
) -> pd.DataFrame:
    """Materialize zero-sale hours for every observed SKU-day."""
    sku_days = sales[
        ["date", "bakery_id", "product_id", "category_name"]
    ].drop_duplicates()
    hours = pd.DataFrame({"hour": range(first_hour, last_hour + 1)})
    frame = sku_days.merge(hours, how="cross")
    frame = frame.merge(
        sales,
        on=["date", "bakery_id", "product_id", "category_name", "hour"],
        how="left",
    )
    frame["sold"] = frame["sold"].fillna(0.0)
    frame["dow"] = frame["date"].dt.dayofweek
    return frame


def build_profile(train: pd.DataFrame, value_col: str) -> pd.DataFrame:
    work = train.copy()
    work["bakery_hour_total"] = work.groupby(["date", "bakery_id", "hour"])[
        value_col
    ].transform("sum")
    work["share"] = work[value_col] / work["bakery_hour_total"].replace(0.0, np.nan)
    profile = work.groupby(PROFILE_KEYS, as_index=False).agg(
        profile_share=("share", "mean"),
        profile_days=("date", "nunique"),
    )
    totals = profile.groupby(["bakery_id", "dow", "hour"])["profile_share"].transform(
        "sum"
    )
    profile["profile_share"] = profile["profile_share"] / totals.replace(0.0, np.nan)
    return profile


def evaluate_profile(
    profile: pd.DataFrame, holdout: pd.DataFrame
) -> dict[str, float | int]:
    work = holdout.copy()
    work["actual_total"] = work.groupby(["date", "bakery_id", "hour"])[
        "sold"
    ].transform("sum")
    work["actual_share"] = work["sold"] / work["actual_total"].replace(0.0, np.nan)
    work = work.merge(profile, on=PROFILE_KEYS, how="left")
    valid = work[work["actual_share"].notna() & work["profile_share"].notna()].copy()
    error = (valid["profile_share"] - valid["actual_share"]).abs()
    return {
        "rows": int(len(valid)),
        "coverage": float(len(valid) / len(work)) if len(work) else 0.0,
        "share_mae": float(error.mean()) if len(valid) else float("nan"),
        "sales_weighted_share_mae": (
            float(np.average(error, weights=valid["actual_total"]))
            if len(valid)
            else float("nan")
        ),
    }


def build_case_comparison(
    marked_train: pd.DataFrame,
    good_day_reconstructed: pd.DataFrame,
    share_reconstructed: pd.DataFrame,
) -> pd.DataFrame:
    """Build an auditable stockout/non-stockout SKU-day comparison."""
    known = marked_train[marked_train["is_production_observed"].fillna(False)].copy()
    known["positive_hour"] = known["hour"].where(known["sold"] > 0)
    known["bakery_hour_sales"] = known.groupby(["date", "bakery_id", "hour"])[
        "sold"
    ].transform("sum")
    daily = known.groupby(
        ["date", "bakery_id", "product_id", "dow"], as_index=False
    ).agg(
        sold=("sold", "sum"),
        produced=("produced", "first"),
        sell_through=("sell_through", "first"),
        is_stockout_day=("is_stockout_day", "first"),
        first_sale_hour=("positive_hour", "min"),
        last_sale_hour=("positive_hour", "max"),
    )

    last_hour = daily[["date", "bakery_id", "product_id", "last_sale_hour"]].copy()
    after = known.merge(
        last_hour,
        on=["date", "bakery_id", "product_id"],
        how="left",
    )
    after["bakery_sales_after_last"] = np.where(
        after["hour"] > after["last_sale_hour"], after["bakery_hour_sales"], 0.0
    )
    after_daily = after.groupby(["date", "bakery_id", "product_id"], as_index=False)[
        "bakery_sales_after_last"
    ].sum()
    daily = daily.merge(after_daily, on=["date", "bakery_id", "product_id"])

    for label, frame in [
        ("good_day", good_day_reconstructed),
        ("bakery_share", share_reconstructed),
    ]:
        additions = frame.groupby(
            ["date", "bakery_id", "product_id"], as_index=False
        ).agg(
            **{
                f"{label}_imputed": ("imputed_demand", "sum"),
                f"{label}_censored_hours": ("is_censored_hour", "sum"),
            }
        )
        daily = daily.merge(additions, on=["date", "bakery_id", "product_id"])

    non_stockout = daily[~daily["is_stockout_day"]].copy()
    benchmark = non_stockout.groupby(
        ["bakery_id", "product_id", "dow"], as_index=False
    ).agg(
        non_stockout_median_sold=("sold", "median"),
        non_stockout_mean_sold=("sold", "mean"),
        non_stockout_days=("date", "nunique"),
        non_stockout_median_last_hour=("last_sale_hour", "median"),
    )
    daily = daily.merge(benchmark, on=["bakery_id", "product_id", "dow"], how="left")
    daily["sold_vs_non_stockout_median"] = daily["sold"] / daily[
        "non_stockout_median_sold"
    ].replace(0.0, np.nan)
    daily["last_hour_gap_vs_non_stockout"] = (
        daily["non_stockout_median_last_hour"] - daily["last_sale_hour"]
    )
    daily["good_day_adjusted_total"] = daily["sold"] + daily["good_day_imputed"]
    daily["bakery_share_adjusted_total"] = daily["sold"] + daily["bakery_share_imputed"]
    return daily.sort_values(
        ["is_stockout_day", "bakery_share_imputed"], ascending=[False, False]
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales-path", default="data/raw/sales_hrs_stg_2026.csv")
    parser.add_argument(
        "--production-path", default="data/raw/production_release_clickhouse.csv"
    )
    parser.add_argument("--holdout-days", type=int, default=7)
    parser.add_argument("--output-dir", default="reports/stockout_demand_profiles_10")
    args = parser.parse_args()

    sales = load_sales(ROOT / args.sales_path)
    production = load_production(ROOT / args.production_path)
    common_end = min(sales["date"].max(), production["date"].max())
    sales = sales[sales["date"] <= common_end].copy()
    production = production[production["date"] <= common_end].copy()
    hourly = complete_hourly_frame(sales)
    marked = mark_stockout_days(hourly, production)

    holdout_start = common_end - pd.Timedelta(days=args.holdout_days - 1)
    train = marked[marked["date"] < holdout_start].copy()
    holdout = marked[marked["date"] >= holdout_start].copy()
    if train.empty or holdout.empty:
        raise ValueError(
            "The selected local files do not provide both train and holdout rows "
            f"for the common period ending {common_end.date()}"
        )
    reference = build_uncensored_hour_reference(train)
    reconstructed_train = reconstruct_stockout_demand(train, reference)
    share_reference = build_bakery_share_reference(train)
    share_reconstructed_train = reconstruct_stockout_demand_from_bakery_share(
        train, share_reference
    )

    baseline_profile = build_profile(reconstructed_train, "sold_observed")
    good_day_profile = build_profile(reconstructed_train, "sold_demand")
    bakery_share_profile = build_profile(share_reconstructed_train, "sold_demand")

    clean_holdout_keys = holdout.loc[
        ~holdout["is_stockout_day"].fillna(False), ["date", "bakery_id", "product_id"]
    ].drop_duplicates()
    clean_holdout = holdout.merge(
        clean_holdout_keys.assign(_clean=True),
        on=["date", "bakery_id", "product_id"],
        how="inner",
    )
    metrics = {
        "baseline_all": evaluate_profile(baseline_profile, holdout),
        "good_day_all": evaluate_profile(good_day_profile, holdout),
        "bakery_share_all": evaluate_profile(bakery_share_profile, holdout),
        "baseline_clean": evaluate_profile(baseline_profile, clean_holdout),
        "good_day_clean": evaluate_profile(good_day_profile, clean_holdout),
        "bakery_share_clean": evaluate_profile(bakery_share_profile, clean_holdout),
    }

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_profile.to_csv(output_dir / "baseline_profile.csv", index=False)
    good_day_profile.to_csv(output_dir / "good_day_profile.csv", index=False)
    bakery_share_profile.to_csv(output_dir / "bakery_share_profile.csv", index=False)
    aggregate_daily_training_target(reconstructed_train).to_csv(
        output_dir / "daily_training_target.csv", index=False
    )
    audit_cols = [
        "date",
        "bakery_id",
        "product_id",
        "hour",
        "sold_observed",
        "sold_demand",
        "imputed_demand",
        "is_censored_hour",
        "expected_demand",
    ]
    reconstructed_train.loc[reconstructed_train["is_censored_hour"], audit_cols].to_csv(
        output_dir / "good_day_imputed_hours_audit.csv", index=False
    )
    share_reconstructed_train.loc[
        share_reconstructed_train["is_censored_hour"], audit_cols
    ].to_csv(output_dir / "bakery_share_imputed_hours_audit.csv", index=False)
    cases = build_case_comparison(
        train,
        reconstructed_train,
        share_reconstructed_train,
    )
    cases.to_csv(output_dir / "stockout_case_comparison.csv", index=False)
    stockout_cases = cases[cases["is_stockout_day"]].copy()
    stockout_cases.head(100).to_csv(
        output_dir / "top_100_stockout_cases.csv", index=False
    )

    summary = {
        "pilot_bakeries": sorted(PILOT_BAKERY_IDS),
        "date_min": str(hourly["date"].min().date()),
        "date_max": str(common_end.date()),
        "holdout_start": str(holdout_start.date()),
        "train_rows": int(len(train)),
        "holdout_rows": int(len(holdout)),
        "stockout_days_train": int(
            train.loc[
                train["is_stockout_day"].fillna(False),
                ["date", "bakery_id", "product_id"],
            ]
            .drop_duplicates()
            .shape[0]
        ),
        "good_day_imputed_hours_train": int(
            reconstructed_train["is_censored_hour"].sum()
        ),
        "bakery_share_imputed_hours_train": int(
            share_reconstructed_train["is_censored_hour"].sum()
        ),
        "observed_train_units": float(reconstructed_train["sold_observed"].sum()),
        "good_day_imputed_train_units": float(
            reconstructed_train["imputed_demand"].sum()
        ),
        "bakery_share_imputed_train_units": float(
            share_reconstructed_train["imputed_demand"].sum()
        ),
        "case_diagnostics": {
            "known_production_days": int(len(cases)),
            "stockout_cases": int(len(stockout_cases)),
            "stockout_with_non_stockout_benchmark": int(
                stockout_cases["non_stockout_days"].fillna(0).gt(0).sum()
            ),
            "median_stockout_sold_vs_non_stockout": float(
                stockout_cases["sold_vs_non_stockout_median"].median()
            ),
            "median_last_hour_gap": float(
                stockout_cases["last_hour_gap_vs_non_stockout"].median()
            ),
            "median_bakery_sales_after_last": float(
                stockout_cases["bakery_sales_after_last"].median()
            ),
        },
        "metrics": metrics,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

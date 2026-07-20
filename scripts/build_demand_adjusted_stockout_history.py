"""Build a conservative, offline demand-adjusted history for robust loss cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.export_clickhouse_checks import create_client  # noqa: E402
from scripts.classify_stockout_mechanisms import SALE_EVENT_HEX  # noqa: E402

DEFAULT_CASES = ROOT / "reports/stockout_mechanism_classification/classified_cases.csv"
DEFAULT_ALL_STOCKOUTS = (
    ROOT / "reports/pilot_stockout_responsibility/stockout_cases_classified.csv"
)
DEFAULT_OUTPUT = ROOT / "reports/demand_adjusted_stockout_history"


def load_hourly_sales(
    client,
    *,
    bakery_ids: list[int],
    date_from: str,
    date_to: str,
) -> pd.DataFrame:
    frame = client.query_df(
        f"""
        select
            m.check_date as sales_date,
            toInt64(m.bakery_id) as bakery_id_int,
            toInt64(m.product_id) as product_id_int,
            toHour(m.check_datetime) as sales_hour,
            any(m.product_name) as product_name,
            sum(m.quantity) as sold
        from mart_sales_60d as m
        where m.check_date between toDate(%(date_from)s) and toDate(%(date_to)s)
          and toInt64OrNull(m.bakery_id) in %(bakery_ids)s
          and m.quantity > 0
          and hex(m.cash_event_type) = '{SALE_EVENT_HEX}'
        group by sales_date, bakery_id_int, product_id_int, sales_hour
        """,
        parameters={
            "date_from": date_from,
            "date_to": date_to,
            "bakery_ids": bakery_ids,
        },
    )
    return frame.rename(
        columns={
            "sales_date": "date",
            "bakery_id_int": "bakery_id",
            "product_id_int": "product_id",
            "sales_hour": "hour",
        }
    )


def reconstruct_cases(
    hourly: pd.DataFrame,
    cases: pd.DataFrame,
    all_stockouts: pd.DataFrame,
    *,
    lookback_days: int = 42,
    min_reference_days: int = 3,
    min_bakery_hour_sales: float = 3.0,
    max_hour_rate_ratio: float = 2.0,
    max_case_uplift_ratio: float = 0.75,
    max_case_uplift_units: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = hourly.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    work["dow"] = work["date"].dt.dayofweek
    bakery_hour = (
        work.groupby(["date", "bakery_id", "hour"], as_index=False)["sold"]
        .sum()
        .rename(columns={"sold": "bakery_hour_sales"})
    )
    work = work.merge(bakery_hour, on=["date", "bakery_id", "hour"], how="left")
    work["sku_share"] = work["sold"] / work["bakery_hour_sales"].replace(0.0, np.nan)
    stockout_keys = set(
        zip(
            pd.to_datetime(all_stockouts["date"]).dt.normalize(),
            all_stockouts["bakery_id"].astype(int),
            all_stockouts["product_id"].astype(int),
            strict=True,
        )
    )

    audit_rows: list[dict[str, object]] = []
    hourly_rows: list[dict[str, object]] = []
    for case in cases.itertuples(index=False):
        case_date = pd.Timestamp(case.date).normalize()
        bakery_id = int(case.bakery_id)
        product_id = int(case.product_id)
        start = case_date - pd.Timedelta(days=lookback_days)
        history = work[
            work["bakery_id"].eq(bakery_id)
            & work["product_id"].eq(product_id)
            & work["date"].lt(case_date)
            & work["date"].ge(start)
            & work["dow"].eq(case_date.dayofweek)
        ].copy()
        if not history.empty:
            history["is_known_stockout"] = [
                (date, bakery_id, product_id) in stockout_keys for date in history["date"]
            ]
            history = history[~history["is_known_stockout"]]
        reference_days = int(history["date"].nunique())
        reference = history.groupby("hour", as_index=False).agg(
            expected_hour=("sold", "mean"),
            mean_share=("sku_share", "mean"),
        )
        current_bakery = bakery_hour[
            bakery_hour["date"].eq(case_date)
            & bakery_hour["bakery_id"].eq(bakery_id)
        ]
        current_sku = work[
            work["date"].eq(case_date)
            & work["bakery_id"].eq(bakery_id)
            & work["product_id"].eq(product_id)
        ]
        positive_rate = float(current_sku["sold"].mean()) if len(current_sku) else 0.0
        cutoff = float(case.last_sale_hour)
        candidates = current_bakery[current_bakery["hour"].gt(cutoff)].merge(
            reference, on="hour", how="left"
        )
        candidates["share_estimate"] = (
            candidates["mean_share"] * candidates["bakery_hour_sales"]
        )
        candidates["raw_imputed"] = candidates[
            ["expected_hour", "share_estimate"]
        ].max(axis=1, skipna=True)
        candidates["raw_imputed"] = candidates["raw_imputed"].fillna(0.0)
        candidates.loc[
            candidates["bakery_hour_sales"].lt(min_bakery_hour_sales), "raw_imputed"
        ] = 0.0
        if positive_rate > 0:
            candidates["raw_imputed"] = candidates["raw_imputed"].clip(
                upper=positive_rate * max_hour_rate_ratio
            )
        raw_total = float(candidates["raw_imputed"].sum())
        case_cap = min(
            max_case_uplift_units,
            max(float(case.daily_sold), 4.0) * max_case_uplift_ratio,
        )
        scale = min(1.0, case_cap / raw_total) if raw_total > 0 else 0.0
        if reference_days < min_reference_days:
            scale = 0.0
        candidates["imputed_demand"] = candidates["raw_imputed"] * scale
        imputed = float(candidates["imputed_demand"].sum())
        for row in candidates.itertuples(index=False):
            hourly_rows.append(
                {
                    "date": case_date,
                    "bakery_id": bakery_id,
                    "product_id": product_id,
                    "hour": int(row.hour),
                    "bakery_hour_sales": float(row.bakery_hour_sales),
                    "expected_hour": float(row.expected_hour)
                    if pd.notna(row.expected_hour)
                    else None,
                    "share_estimate": float(row.share_estimate)
                    if pd.notna(row.share_estimate)
                    else None,
                    "imputed_demand": float(row.imputed_demand),
                }
            )
        audit_rows.append(
            {
                "date": case_date,
                "bakery_id": bakery_id,
                "bakery_name": case.bakery_name,
                "product_id": product_id,
                "product_name": case.product_name,
                "daily_sold_observed": float(case.daily_sold),
                "last_sale_hour": cutoff,
                "reference_days": reference_days,
                "raw_imputed_demand": raw_total,
                "case_cap": case_cap,
                "imputed_demand": imputed,
                "demand_adjusted_sku": float(case.daily_sold) + imputed,
                "bakery_gap": float(case.bakery_gap),
                "bakery_ratio": float(case.bakery_ratio),
                "robust_case_type": case.robust_case_type,
            }
        )
    return pd.DataFrame(audit_rows), pd.DataFrame(hourly_rows)


def build_adjusted_history(
    hourly: pd.DataFrame, audit: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hourly = hourly.copy()
    if "product_name" not in hourly.columns:
        hourly["product_name"] = hourly["product_id"].astype(str)
    daily_sku = hourly.groupby(
        ["date", "bakery_id", "product_id"], as_index=False
    ).agg(
        product_name=("product_name", "first"),
        observed_sales=("sold", "sum"),
    )
    adjustment = audit.groupby(
        ["date", "bakery_id", "product_id"], as_index=False
    )["imputed_demand"].sum()
    daily_sku = daily_sku.merge(
        adjustment, on=["date", "bakery_id", "product_id"], how="left"
    )
    daily_sku["imputed_demand"] = daily_sku["imputed_demand"].fillna(0.0)
    daily_sku["demand_adjusted_sales"] = (
        daily_sku["observed_sales"] + daily_sku["imputed_demand"]
    )
    daily_sku["date"] = pd.to_datetime(daily_sku["date"]).dt.normalize()
    daily_sku["dow"] = daily_sku["date"].dt.dayofweek

    daily_bakery = daily_sku.groupby(["date", "bakery_id"], as_index=False).agg(
        observed_sales=("observed_sales", "sum"),
        imputed_demand=("imputed_demand", "sum"),
        demand_adjusted_sales=("demand_adjusted_sales", "sum"),
    )
    daily_bakery = daily_bakery.sort_values(["bakery_id", "date"])
    for column in ["observed_sales", "demand_adjusted_sales"]:
        daily_bakery[f"{column}_lag7"] = daily_bakery.groupby("bakery_id")[
            column
        ].shift(7)
        daily_bakery[f"{column}_mean28"] = daily_bakery.groupby("bakery_id")[
            column
        ].transform(lambda values: values.shift(1).rolling(28, min_periods=7).mean())

    totals = daily_bakery[
        ["date", "bakery_id", "observed_sales", "demand_adjusted_sales"]
    ].rename(
        columns={
            "observed_sales": "bakery_observed_sales",
            "demand_adjusted_sales": "bakery_adjusted_sales",
        }
    )
    profile_rows = daily_sku.merge(totals, on=["date", "bakery_id"], how="left")
    profile_rows["observed_share"] = profile_rows["observed_sales"] / profile_rows[
        "bakery_observed_sales"
    ].replace(0.0, np.nan)
    profile_rows["adjusted_share"] = profile_rows["demand_adjusted_sales"] / profile_rows[
        "bakery_adjusted_sales"
    ].replace(0.0, np.nan)
    profile = profile_rows.groupby(
        ["bakery_id", "product_id", "product_name", "dow"], as_index=False
    ).agg(
        observed_share=("observed_share", "mean"),
        adjusted_share=("adjusted_share", "mean"),
        history_days=("date", "nunique"),
        total_imputed_demand=("imputed_demand", "sum"),
    )
    profile["share_delta"] = profile["adjusted_share"] - profile["observed_share"]
    return daily_sku, daily_bakery, profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--all-stockouts", default=str(DEFAULT_ALL_STOCKOUTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    classified = pd.read_csv(args.cases, encoding="utf-8-sig")
    classified["date"] = pd.to_datetime(classified["date"]).dt.normalize()
    cases = classified[classified["robust_case_type"].eq("demand_loss")].copy()
    all_stockouts = pd.read_csv(args.all_stockouts, encoding="utf-8-sig")
    client = create_client(args.env_file)
    date_from = max(cases["date"].min() - pd.Timedelta(days=42), pd.Timestamp("2026-05-03"))
    hourly = load_hourly_sales(
        client,
        bakery_ids=sorted(classified["bakery_id"].unique().tolist()),
        date_from=str(date_from.date()),
        date_to=str(cases["date"].max().date()),
    )
    audit, hourly_audit = reconstruct_cases(hourly, cases, all_stockouts)
    daily_sku, daily_bakery, adjusted_profile = build_adjusted_history(hourly, audit)
    bakery_adjustments = audit.groupby(["date", "bakery_id"], as_index=False).agg(
        imputed_demand=("imputed_demand", "sum")
    )
    observed_lookup = classified.groupby(["date", "bakery_id"], as_index=False)[
        "actual_bakery"
    ].first()
    bakery_adjustments = bakery_adjustments.merge(
        observed_lookup, on=["date", "bakery_id"], how="left"
    )
    bakery_adjustments["demand_adjusted_bakery"] = (
        bakery_adjustments["actual_bakery"] + bakery_adjustments["imputed_demand"]
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output / "case_adjustments.csv", index=False, encoding="utf-8-sig")
    hourly_audit.to_csv(output / "hourly_adjustments.csv", index=False, encoding="utf-8-sig")
    bakery_adjustments.to_csv(
        output / "bakery_day_adjustments.csv", index=False, encoding="utf-8-sig"
    )
    daily_sku.to_csv(
        output / "demand_adjusted_sku_day.csv", index=False, encoding="utf-8-sig"
    )
    daily_bakery.to_csv(
        output / "demand_adjusted_bakery_day.csv", index=False, encoding="utf-8-sig"
    )
    adjusted_profile.to_csv(
        output / "demand_adjusted_share_profile.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "robust_demand_loss_cases": int(len(cases)),
        "cases_with_adjustment": int(audit["imputed_demand"].gt(0).sum()),
        "imputed_demand_units": float(audit["imputed_demand"].sum()),
        "median_case_imputation": float(audit["imputed_demand"].median()),
        "max_case_imputation": float(audit["imputed_demand"].max()),
        "reference_coverage": float(audit["reference_days"].ge(3).mean()),
        "adjusted_sku_day_rows": int(len(daily_sku)),
        "adjusted_bakery_day_rows": int(len(daily_bakery)),
        "profile_rows": int(len(adjusted_profile)),
        "max_profile_share_delta_pp": float(adjusted_profile["share_delta"].max() * 100),
        "production_write": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

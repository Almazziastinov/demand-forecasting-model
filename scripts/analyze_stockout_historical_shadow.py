"""Summarize the stockout direction over the full leakage-safe history.

The script consumes local artifacts produced by the read-only stockout shadow
pipeline. It never connects to ClickHouse and never writes production state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "reports/stockout_mechanism_classification/classified_cases.csv"
DEFAULT_ADJUSTMENTS = (
    ROOT / "reports/demand_adjusted_stockout_history/case_adjustments.csv"
)
DEFAULT_DAILY_SKU = (
    ROOT / "reports/demand_adjusted_stockout_history/demand_adjusted_sku_day.csv"
)
DEFAULT_OUTPUT = ROOT / "reports/stockout_historical_shadow"
CASE_KEYS = ["date", "bakery_id", "product_id"]


def _require_columns(frame: pd.DataFrame, columns: set[str], *, name: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def build_case_replay(
    cases: pd.DataFrame,
    adjustments: pd.DataFrame,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build a demand-only replay for confirmed misses within a date cutoff."""
    _require_columns(
        cases,
        {
            *CASE_KEYS,
            "bakery_name",
            "product_name",
            "daily_sold",
            "forecast_qty",
            "confirmed_model_shortfall_qty",
            "robust_case_type",
            "bakery_ratio",
            "reference_days_actual_bakery",
            "reference_days_sold",
        },
        name="cases",
    )
    _require_columns(
        adjustments,
        {*CASE_KEYS, "imputed_demand", "reference_days"},
        name="adjustments",
    )

    work = cases.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    adjustment = adjustments.copy()
    adjustment["date"] = pd.to_datetime(adjustment["date"]).dt.normalize()
    if adjustment.duplicated(CASE_KEYS).any():
        raise ValueError("adjustments must contain at most one row per case")

    effective_start = (
        pd.Timestamp(start).normalize() if start is not None else work["date"].min()
    )
    effective_end = (
        pd.Timestamp(end).normalize() if end is not None else work["date"].max()
    )
    if effective_start > effective_end:
        raise ValueError("start must be less than or equal to end")
    work = work[work["date"].between(effective_start, effective_end)].copy()
    adjustment = adjustment[
        adjustment["date"].between(effective_start, effective_end)
    ].copy()

    work = work.merge(
        adjustment[[*CASE_KEYS, "imputed_demand", "reference_days"]],
        on=CASE_KEYS,
        how="left",
        validate="one_to_one",
    )
    work["imputed_demand"] = work["imputed_demand"].fillna(0.0)
    work["reference_days"] = work["reference_days"].fillna(0).astype(int)
    invalid_adjustment = work["imputed_demand"].gt(0) & ~work["robust_case_type"].eq(
        "demand_loss"
    )
    if invalid_adjustment.any():
        raise ValueError(
            "positive demand adjustments are allowed only for robust demand-loss cases"
        )

    work["baseline_shortfall"] = (work["daily_sold"] - work["forecast_qty"]).clip(
        lower=0.0
    )
    work["shadow_forecast_qty"] = work["forecast_qty"] + work["imputed_demand"]
    work["shadow_shortfall"] = (work["daily_sold"] - work["shadow_forecast_qty"]).clip(
        lower=0.0
    )
    work["shortfall_reduction"] = work["baseline_shortfall"] - work["shadow_shortfall"]
    work["case_improved"] = work["shadow_shortfall"].lt(
        work["baseline_shortfall"] - 0.01
    )
    work["case_worsened"] = work["shadow_shortfall"].gt(
        work["baseline_shortfall"] + 0.01
    )
    work["case_fixed"] = work["shadow_shortfall"].le(0.5)
    work["week_start"] = work["date"] - pd.to_timedelta(
        work["date"].dt.dayofweek, unit="D"
    )
    return work.sort_values(CASE_KEYS).reset_index(drop=True)


def summarize_periods(
    replay: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    frequency: str,
) -> pd.DataFrame:
    """Return daily or weekly stability rows, including periods without cases."""
    if frequency not in {"D", "W-MON"}:
        raise ValueError("frequency must be D or W-MON")
    work = replay.copy()
    period_column = "date" if frequency == "D" else "week_start"
    period_start = (
        start if frequency == "D" else start - pd.Timedelta(days=start.dayofweek)
    )
    periods = pd.DataFrame(
        {period_column: pd.date_range(period_start, end, freq=frequency)}
    )

    def aggregate(group: pd.DataFrame) -> pd.Series:
        counts = group["robust_case_type"].value_counts()
        baseline = float(group["baseline_shortfall"].sum())
        reduction = float(group["shortfall_reduction"].sum())
        return pd.Series(
            {
                "cases": len(group),
                "allocation_cases": int(counts.get("allocation", 0)),
                "demand_loss_cases": int(counts.get("demand_loss", 0)),
                "uncertain_cases": int(counts.get("uncertain", 0)),
                "adjusted_cases": int(group["imputed_demand"].gt(0).sum()),
                "imputed_demand": float(group["imputed_demand"].sum()),
                "baseline_shortfall": baseline,
                "shadow_shortfall": float(group["shadow_shortfall"].sum()),
                "shortfall_reduction": reduction,
                "shortfall_reduction_ratio": reduction / baseline
                if baseline > 0
                else np.nan,
                "cases_fixed": int(group["case_fixed"].sum()),
                "cases_improved": int(group["case_improved"].sum()),
                "cases_worsened": int(group["case_worsened"].sum()),
                "median_bakery_ratio": float(group["bakery_ratio"].median()),
            }
        )

    if work.empty:
        aggregated = pd.DataFrame(columns=[period_column])
    else:
        aggregated = work.groupby(period_column, as_index=False).apply(
            aggregate, include_groups=False
        )
    result = periods.merge(aggregated, on=period_column, how="left")
    count_columns = [
        "cases",
        "allocation_cases",
        "demand_loss_cases",
        "uncertain_cases",
        "adjusted_cases",
        "cases_fixed",
        "cases_improved",
        "cases_worsened",
    ]
    quantity_columns = [
        "imputed_demand",
        "baseline_shortfall",
        "shadow_shortfall",
        "shortfall_reduction",
    ]
    for column in count_columns:
        result[column] = result[column].fillna(0).astype(int)
    for column in quantity_columns:
        result[column] = result[column].fillna(0.0)
    return result


def build_entity_stability(
    replay: pd.DataFrame,
    *,
    keys: list[str],
    minimum_recurring_cases: int = 2,
    minimum_recurring_weeks: int = 2,
) -> pd.DataFrame:
    """Aggregate recurrence and outcome metrics for bakeries or SKU entities."""
    if replay.empty:
        return pd.DataFrame()

    def aggregate(group: pd.DataFrame) -> pd.Series:
        counts = group["robust_case_type"].value_counts()
        demand_loss = group[group["robust_case_type"].eq("demand_loss")]
        allocation = group[group["robust_case_type"].eq("allocation")]
        return pd.Series(
            {
                "cases": len(group),
                "case_weeks": int(group["week_start"].nunique()),
                "allocation_cases": int(counts.get("allocation", 0)),
                "allocation_weeks": int(allocation["week_start"].nunique()),
                "demand_loss_cases": int(counts.get("demand_loss", 0)),
                "demand_loss_weeks": int(demand_loss["week_start"].nunique()),
                "uncertain_cases": int(counts.get("uncertain", 0)),
                "imputed_demand": float(group["imputed_demand"].sum()),
                "baseline_shortfall": float(group["baseline_shortfall"].sum()),
                "shortfall_reduction": float(group["shortfall_reduction"].sum()),
                "cases_improved": int(group["case_improved"].sum()),
                "cases_worsened": int(group["case_worsened"].sum()),
                "median_bakery_ratio": float(group["bakery_ratio"].median()),
            }
        )

    result = replay.groupby(keys, as_index=False, dropna=False).apply(
        aggregate, include_groups=False
    )
    result["recurrent_demand_loss"] = result["demand_loss_cases"].ge(
        minimum_recurring_cases
    ) & result["demand_loss_weeks"].ge(minimum_recurring_weeks)
    result["recurrent_allocation"] = result["allocation_cases"].ge(
        minimum_recurring_cases
    ) & result["allocation_weeks"].ge(minimum_recurring_weeks)
    return result.sort_values(
        ["demand_loss_cases", "allocation_cases", "baseline_shortfall"],
        ascending=False,
    ).reset_index(drop=True)


def add_sales_ranks(
    stability: pd.DataFrame,
    daily_sku: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Add global and within-bakery sales ranks without using case-only volume."""
    if stability.empty:
        return stability
    _require_columns(
        daily_sku,
        {"date", "bakery_id", "product_id", "observed_sales"},
        name="daily_sku",
    )
    sales = daily_sku.copy()
    sales["date"] = pd.to_datetime(sales["date"]).dt.normalize()
    sales = sales[sales["date"].between(start, end)]
    pair_sales = sales.groupby(["bakery_id", "product_id"], as_index=False)[
        "observed_sales"
    ].sum()
    pair_sales["bakery_sales_rank"] = pair_sales.groupby("bakery_id")[
        "observed_sales"
    ].rank(method="dense", ascending=False)
    result = stability.merge(pair_sales, on=["bakery_id", "product_id"], how="left")
    result["is_bakery_top5_by_sales"] = result["bakery_sales_rank"].le(5)
    result["is_potentially_problematic"] = (
        result["recurrent_demand_loss"] | result["recurrent_allocation"]
    )
    return result


def build_summary(
    replay: pd.DataFrame,
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    bakery_stability: pd.DataFrame,
    sku_stability: pd.DataFrame,
    bakery_sku_stability: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    requested_days: int,
    minimum_prospective_days: int,
) -> dict[str, object]:
    baseline = float(replay["baseline_shortfall"].sum())
    reduction = float(replay["shortfall_reduction"].sum())
    available_days = int((end - start).days + 1)
    return {
        "mode": "historical_walk_forward_read_only",
        "production_write": False,
        "leakage_contract": {
            "classification_counterfactual": "prior_same_weekdays_only",
            "demand_references": "dates_strictly_before_case_date_only",
            "current_profile_diagnostic_included": False,
        },
        "coverage": {
            "date_from": str(start.date()),
            "date_to": str(end.date()),
            "calendar_days": available_days,
            "weeks": int(weekly["week_start"].nunique()),
            "requested_days": requested_days,
            "requested_window_available": available_days >= requested_days,
            "days_with_cases": int(daily["cases"].gt(0).sum()),
        },
        "classification": {
            "cases": int(len(replay)),
            "allocation_cases": int(replay["robust_case_type"].eq("allocation").sum()),
            "demand_loss_cases": int(
                replay["robust_case_type"].eq("demand_loss").sum()
            ),
            "uncertain_cases": int(replay["robust_case_type"].eq("uncertain").sum()),
            "weeks_with_demand_loss": int(weekly["demand_loss_cases"].gt(0).sum()),
        },
        "demand_shadow": {
            "adjusted_cases": int(replay["imputed_demand"].gt(0).sum()),
            "imputed_demand": float(replay["imputed_demand"].sum()),
            "baseline_shortfall": baseline,
            "shadow_shortfall": float(replay["shadow_shortfall"].sum()),
            "shortfall_reduction": reduction,
            "shortfall_reduction_ratio": reduction / baseline if baseline > 0 else None,
            "cases_fixed": int(replay["case_fixed"].sum()),
            "cases_improved": int(replay["case_improved"].sum()),
            "cases_worsened": int(replay["case_worsened"].sum()),
        },
        "recurrence": {
            "bakeries_with_recurrent_demand_loss": int(
                bakery_stability["recurrent_demand_loss"].sum()
            ),
            "skus_with_recurrent_demand_loss": int(
                sku_stability["recurrent_demand_loss"].sum()
            ),
            "bakery_sku_pairs_with_recurrent_demand_loss": int(
                bakery_sku_stability["recurrent_demand_loss"].sum()
            ),
            "bakery_sku_pairs_with_recurrent_allocation": int(
                bakery_sku_stability["recurrent_allocation"].sum()
            ),
            "problem_pairs_in_bakery_top5": int(
                (
                    bakery_sku_stability["is_potentially_problematic"]
                    & bakery_sku_stability["is_bakery_top5_by_sales"]
                ).sum()
            ),
        },
        "promotion_gate": {
            "historical_days_count_as_prospective_days": False,
            "prospective_days_observed": 0,
            "minimum_prospective_days": minimum_prospective_days,
            "ready_for_production_proposal": False,
            "reason": (
                "historical replay supports development but does not replace "
                "prospective shadow"
            ),
        },
        "limitations": [
            "confirmed-miss cases cannot measure false uplift on normal days",
            (
                "true censored demand is unavailable and requires manual or "
                "synthetic validation"
            ),
            "available confirmed-case history is shorter than the requested window",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--adjustments", default=str(DEFAULT_ADJUSTMENTS))
    parser.add_argument("--daily-sku", default=str(DEFAULT_DAILY_SKU))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--requested-days", type=int, default=84)
    parser.add_argument("--minimum-prospective-days", type=int, default=21)
    args = parser.parse_args()

    cases = pd.read_csv(args.cases, encoding="utf-8-sig")
    adjustments = pd.read_csv(args.adjustments, encoding="utf-8-sig")
    daily_sku = pd.read_csv(args.daily_sku, encoding="utf-8-sig")
    available_dates = pd.to_datetime(cases["date"]).dt.normalize()
    start = (
        pd.Timestamp(args.start).normalize() if args.start else available_dates.min()
    )
    end = pd.Timestamp(args.end).normalize() if args.end else available_dates.max()

    replay = build_case_replay(cases, adjustments, start=start, end=end)
    daily = summarize_periods(replay, start=start, end=end, frequency="D")
    weekly = summarize_periods(replay, start=start, end=end, frequency="W-MON")
    bakery = build_entity_stability(replay, keys=["bakery_id", "bakery_name"])
    sku = build_entity_stability(replay, keys=["product_id", "product_name"])
    bakery_sku = build_entity_stability(
        replay,
        keys=["bakery_id", "bakery_name", "product_id", "product_name"],
    )
    bakery_sku = add_sales_ranks(bakery_sku, daily_sku, start=start, end=end)
    summary = build_summary(
        replay,
        daily,
        weekly,
        bakery,
        sku,
        bakery_sku,
        start=start,
        end=end,
        requested_days=args.requested_days,
        minimum_prospective_days=args.minimum_prospective_days,
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    replay.to_csv(output / "case_replay.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(output / "daily_stability.csv", index=False)
    weekly.to_csv(output / "weekly_stability.csv", index=False)
    bakery.to_csv(output / "bakery_stability.csv", index=False, encoding="utf-8-sig")
    sku.to_csv(output / "sku_stability.csv", index=False, encoding="utf-8-sig")
    bakery_sku.to_csv(
        output / "bakery_sku_stability.csv", index=False, encoding="utf-8-sig"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

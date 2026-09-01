"""Recalculate 2026-08-24 research artifacts on observable bakery-days.

A bakery-day is evaluable only when its aggregated observed sales/target is
strictly positive. Forecast-only bakery-days are reported as a separate data
quality population and are never silently interpreted as zero demand.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DAILY_DIR = ROOT / "reports" / "daily_sku_allocation_backtest_20260824"
BASE_DIR = ROOT / "reports" / "base_norm_recent_vs_mean7_20260824"


def metric(rows: pd.DataFrame, forecast: str, target: str) -> dict[str, float | int]:
    actual = float(rows[target].sum())
    error = rows[forecast] - rows[target]
    return {
        "rows": int(len(rows)),
        "actual": actual,
        "forecast": float(rows[forecast].sum()),
        "wape_pct": float(100 * error.abs().sum() / actual),
        "bias_pct": float(100 * error.sum() / actual),
    }


def allocation_metric(
    rows: pd.DataFrame, forecast: str, target: str
) -> dict[str, float | int]:
    keys = ["date", "bakery_id", "category"]
    forecast_total = rows.groupby(keys)[forecast].transform("sum")
    actual_total = rows.groupby(keys)[target].transform("sum")
    oracle = (
        rows[forecast] / forecast_total.replace(0, pd.NA) * actual_total
    ).fillna(0)
    actual = float(rows[target].sum())
    return {
        "rows": int(len(rows)),
        "wape_pct": float(100 * (oracle - rows[target]).abs().sum() / actual),
    }


def bakery_concentration(rows: pd.DataFrame, forecast: str) -> dict[str, float | int]:
    keys = ["date", "bakery_id"]
    total = rows.groupby(keys)[forecast].transform("sum")
    top = (
        (rows[forecast] / total.replace(0, pd.NA))
        .groupby([rows[k] for k in keys])
        .max()
        .dropna()
    )
    return {
        "groups": int(len(top)),
        "p95": float(top.quantile(0.95)),
        "max": float(top.max()),
        "ge20": int(top.ge(0.20).sum()),
        "ge30": int(top.ge(0.30).sum()),
        "ge40": int(top.ge(0.40).sum()),
    }


def recalculate_daily_allocation() -> None:
    source = pd.read_parquet(DAILY_DIR / "sku_day_predictions.parquet")
    bakery_actual = source.groupby(["date", "bakery_id"])["demand_mid"].sum()
    active_index = bakery_actual[bakery_actual.gt(0)].index
    indexed = source.set_index(["date", "bakery_id"])
    active = indexed[indexed.index.isin(active_index)].reset_index()
    inactive = indexed[~indexed.index.isin(active_index)].reset_index()
    methods = [
        "incumbent_sku_forecast",
        "daily_profile_forecast",
        "predictive_forecast",
    ]

    summary = {
        "scope": {
            "dates": int(active["date"].nunique()),
            "bakeries": int(active["bakery_id"].nunique()),
            "bakery_days": int(len(active_index)),
            "sku_rows": int(len(active)),
            "rule": "aggregate demand_mid > 0 for date x bakery",
        },
        "dq_excluded": {
            "bakery_days": int(len(bakery_actual) - len(active_index)),
            "bakeries": int(inactive["bakery_id"].nunique()),
            "incumbent_forecast": float(inactive["incumbent_sku_forecast"].sum()),
        },
        "end_to_end": {
            method: metric(active, method, "demand_mid") for method in methods
        },
        "allocation": {
            method: allocation_metric(active, method, "demand_mid")
            for method in methods
        },
        "concentration": {
            method: bakery_concentration(active, method) for method in methods
        },
        "production_write": False,
    }
    (DAILY_DIR / "corrected_active_universe_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def recalculate_base_vs_mean() -> None:
    source = pd.read_parquet(BASE_DIR / "sku_day_comparison.parquet")
    bakery = source.groupby(["date", "bakery_id"], as_index=False)[
        ["incumbent", "sold", "strict_demand", "mean7_sold", "mean7_strict_demand"]
    ].sum()
    active = bakery[bakery["sold"].gt(0)].copy()
    inactive = bakery[bakery["sold"].le(0)].copy()

    methods = {
        "sold": ["incumbent", "mean7_sold"],
        "strict_demand": ["incumbent", "mean7_strict_demand"],
    }
    summary = {
        "scope": {
            "dates": int(active["date"].nunique()),
            "bakeries": int(active["bakery_id"].nunique()),
            "bakery_days": int(len(active)),
            "rule": "aggregated observed sold > 0 for date x bakery",
        },
        "dq_excluded": {
            "bakery_days": int(len(inactive)),
            "bakeries": int(inactive["bakery_id"].nunique()),
            "incumbent_forecast": float(inactive["incumbent"].sum()),
        },
        "bakery_level": {
            target: {
                method: metric(active, method, target) for method in target_methods
            }
            for target, target_methods in methods.items()
        },
        "production_write": False,
    }
    (BASE_DIR / "corrected_active_universe_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    records = []
    for date, rows in active.groupby("date", sort=True):
        for target, target_methods in methods.items():
            record = {"date": date, "target": target}
            for method in target_methods:
                record[f"{method}_wape_pct"] = metric(rows, method, target)["wape_pct"]
            records.append(record)
    pd.DataFrame(records).to_csv(
        BASE_DIR / "corrected_active_metrics_by_date.csv", index=False
    )


def main() -> None:
    recalculate_daily_allocation()
    recalculate_base_vs_mean()
    print("ACTIVE-UNIVERSE RECALCULATION COMPLETE")


if __name__ == "__main__":
    main()

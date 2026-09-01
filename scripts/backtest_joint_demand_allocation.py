"""Evaluate bakery-volume uplift together with predictive SKU allocation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "reports/rebuilt_predictive_choice_20260825/predictions.parquet"
FACTS = ROOT / "reports/base_norm_recent_vs_mean7_20260824/sku_day_comparison.parquet"
BAKERY_FORECASTS = (
    ROOT / ".codex_tmp/joint_demand_allocation_20260825/bakery_forecasts.parquet"
)
OUTPUT = ROOT / "reports/joint_demand_allocation_20260825"
KEYS = ["date", "bakery_id", "product_id"]
BAKERY_KEYS = ["date", "bakery_id"]
UPLIFT_FACTORS = (1.02, 1.04, 1.06, 1.08, 1.10, 1.12)


def score(rows: pd.DataFrame, forecast_col: str, level: str) -> dict[str, float | int]:
    if level == "bakery":
        rows = rows.groupby(BAKERY_KEYS, as_index=False)[
            [forecast_col, "sold", "strict_demand"]
        ].sum()
    forecast = rows[forecast_col].clip(lower=0.0)
    sold = rows["sold"].clip(lower=0.0)
    demand = rows["strict_demand"].clip(lower=0.0)
    lost = (demand - sold).clip(lower=0.0)
    recognized = np.minimum(lost, (forecast - sold).clip(lower=0.0))
    error = forecast - demand
    actual = float(demand.sum())
    return {
        "rows": int(len(rows)),
        "forecast_qty": float(forecast.sum()),
        "sold_qty": float(sold.sum()),
        "demand_qty": actual,
        "lost_qty": float(lost.sum()),
        "recognized_lost_qty": float(recognized.sum()),
        "recognized_lost_pct": float(100 * recognized.sum() / lost.sum())
        if lost.sum() > 0
        else 0.0,
        "wape_pct": float(100 * error.abs().sum() / actual),
        "bias_pct": float(100 * error.sum() / actual),
        "underforecast_qty": float((-error).clip(lower=0.0).sum()),
        "true_overforecast_qty": float(error.clip(lower=0.0).sum()),
        "underforecast_rows": int(error.lt(0).sum()),
        "true_overforecast_rows": int(error.gt(0).sum()),
    }


def main() -> None:
    predictions = pd.read_parquet(PREDICTIONS)
    predictions = predictions[predictions["fold"].eq("current")].copy()
    facts = pd.read_parquet(FACTS)[KEYS + ["sold", "strict_demand"]]
    bakery = pd.read_parquet(BAKERY_FORECASTS)
    for frame in (predictions, facts, bakery):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()

    run_dates = predictions[["date", "source_run_id"]].drop_duplicates()
    bakery = bakery.merge(run_dates, on=["date", "source_run_id"], how="inner")
    if bakery.duplicated(BAKERY_KEYS).any():
        raise ValueError("Bakery forecasts are not unique for the selected run/date")

    rows = predictions.merge(facts, on=KEYS, how="left", validate="one_to_one")
    rows[["sold", "strict_demand"]] = rows[["sold", "strict_demand"]].fillna(0.0)
    bakery_sales = rows.groupby(BAKERY_KEYS)["sold"].transform("sum")
    rows = rows[bakery_sales.gt(0)].copy()
    rows = rows.merge(
        bakery[BAKERY_KEYS + ["forecast_base", "forecast_final"]],
        on=BAKERY_KEYS,
        how="left",
        validate="many_to_one",
    )
    if rows[["forecast_base", "forecast_final"]].isna().any().any():
        raise ValueError("Missing raw/final bakery forecast for observable rows")

    incumbent_total = rows.groupby(BAKERY_KEYS)["incumbent_sku_forecast"].transform(
        "sum"
    )
    rows["raw_base_incumbent_mix"] = rows["incumbent_sku_forecast"] * (
        rows["forecast_base"] / incumbent_total.replace(0.0, np.nan)
    )
    rows["base_recent_incumbent"] = rows["incumbent_sku_forecast"]
    rows["base_recent_predictive"] = rows["predictive_forecast"]

    variants = [
        "raw_base_incumbent_mix",
        "base_recent_incumbent",
        "base_recent_predictive",
    ]
    for factor in UPLIFT_FACTORS:
        name = f"demand_uplift_{int(round((factor - 1) * 100)):02d}_predictive"
        rows[name] = rows["predictive_forecast"] * factor
        variants.append(name)

    metrics = []
    for variant in variants:
        for level in ("bakery", "sku"):
            metrics.append(
                {"variant": variant, "level": level, **score(rows, variant, level)}
            )
    metrics = pd.DataFrame(metrics)

    baseline = metrics[metrics["variant"].eq("base_recent_incumbent")].set_index(
        "level"
    )
    for metric in [
        "forecast_qty",
        "recognized_lost_qty",
        "true_overforecast_qty",
        "true_overforecast_rows",
        "wape_pct",
    ]:
        metrics[f"{metric}_delta_vs_current"] = metrics.apply(
            lambda row: row[metric] - baseline.loc[row["level"], metric], axis=1
        )

    sku = metrics[metrics["level"].eq("sku")]
    eligible = sku[
        sku["variant"].str.startswith("demand_uplift")
        & sku["forecast_qty_delta_vs_current"].gt(0)
        & sku["recognized_lost_qty_delta_vs_current"].gt(0)
        & sku["true_overforecast_qty_delta_vs_current"].lt(0)
        & sku["true_overforecast_rows_delta_vs_current"].lt(0)
        & sku["wape_pct_delta_vs_current"].lt(0)
    ].sort_values(["wape_pct", "recognized_lost_qty"], ascending=[True, False])
    selected = eligible.iloc[0]["variant"] if not eligible.empty else None

    ordered_dates = sorted(rows["date"].unique())
    train_dates = ordered_dates[:4]
    test_dates = ordered_dates[4:]
    train_rows = rows[rows["date"].isin(train_dates)]
    test_rows = rows[rows["date"].isin(test_dates)].copy()
    train_bakery = train_rows.groupby(BAKERY_KEYS, as_index=False)[
        ["base_recent_incumbent", "strict_demand"]
    ].sum()
    calibration = train_bakery.groupby("bakery_id", as_index=False).agg(
        forecast_qty=("base_recent_incumbent", "sum"),
        demand_qty=("strict_demand", "sum"),
        observed_days=("date", "nunique"),
    )
    calibration["raw_factor"] = calibration["demand_qty"] / calibration[
        "forecast_qty"
    ].replace(0.0, np.nan)
    calibration["selective_factor"] = (
        1.0
        + calibration["observed_days"]
        / (calibration["observed_days"] + 7.0)
        * (calibration["raw_factor"] - 1.0)
    ).clip(0.95, 1.08)
    test_rows = test_rows.merge(
        calibration[["bakery_id", "selective_factor"]],
        on="bakery_id",
        how="left",
        validate="many_to_one",
    )
    test_rows["selective_factor"] = test_rows["selective_factor"].fillna(1.0)
    test_rows["blocked_uniform_02_predictive"] = 1.02 * test_rows["predictive_forecast"]
    test_rows["blocked_selective_predictive"] = (
        test_rows["selective_factor"] * test_rows["predictive_forecast"]
    )
    blocked_metrics = []
    for variant in [
        "base_recent_incumbent",
        "base_recent_predictive",
        "blocked_uniform_02_predictive",
        "blocked_selective_predictive",
    ]:
        for level in ("bakery", "sku"):
            blocked_metrics.append(
                {"variant": variant, "level": level, **score(test_rows, variant, level)}
            )
    blocked_metrics = pd.DataFrame(blocked_metrics)

    summary = {
        "scope": {
            "dates": int(rows["date"].nunique()),
            "bakeries": int(rows["bakery_id"].nunique()),
            "bakery_days": int(rows.groupby(BAKERY_KEYS).ngroups),
            "sku_rows": int(len(rows)),
        },
        "definitions": {
            "demand": "sold + conservative lost demand (strict_demand)",
            "recognized_lost": "min(lost, max(forecast - sold, 0))",
            "true_overforecast": "max(forecast - strict_demand, 0)",
        },
        "selected_diagnostic_candidate": selected,
        "selection_note": (
            "Diagnostic frontier on the current completed dates; "
            "not a frozen validation."
        ),
        "blocked_calibration": {
            "train_dates": [str(pd.Timestamp(value).date()) for value in train_dates],
            "test_dates": [str(pd.Timestamp(value).date()) for value in test_dates],
            "status": (
                "selective residual calibration did not pass bakery overforecast gate"
            ),
        },
        "production_write": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(OUTPUT / "predictions.parquet", index=False)
    metrics.to_csv(OUTPUT / "metrics.csv", index=False, encoding="utf-8-sig")
    blocked_metrics.to_csv(
        OUTPUT / "blocked_metrics.csv", index=False, encoding="utf-8-sig"
    )
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()

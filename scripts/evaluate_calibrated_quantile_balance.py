"""Operational SKU balance for calibrated post-last-sale demand quantiles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "reports/network_quantile_operational_balance_20260826/rows.parquet"
PREDICTIONS = ROOT / "reports/calibrated_network_quantiles_20260826/predictions.parquet"
LABELS = ROOT / "reports/calibrated_stockout_network_20260826/sku_day_demand.csv"
OUTPUT = ROOT / "reports/calibrated_quantile_operational_balance_20260826"
KEYS = ["date", "bakery_id", "product_id"]


def score(name: str, plan: pd.Series, demand: pd.Series) -> dict[str, float | str]:
    error = plan - demand
    return {
        "variant": name,
        "volume": float(plan.sum()),
        "surplus": float(error.clip(lower=0).sum()),
        "underbake": float((-error).clip(lower=0).sum()),
        "imbalance": float(error.abs().sum()),
    }


def main() -> None:
    rows = pd.read_parquet(ROWS)
    labels = pd.read_csv(
        LABELS,
        usecols=[*KEYS, "imputed_demand"],
        encoding="utf-8-sig",
    )
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    rows = rows.drop(columns=["relaxed_lost", "demand"], errors="ignore").merge(
        labels, on=KEYS, how="left", validate="one_to_one"
    )
    rows["calibrated_lost"] = rows.pop("imputed_demand").fillna(0.0)
    rows["demand"] = rows["sold"] + rows["calibrated_lost"]

    old_p50 = rows.groupby(["date", "bakery_id"], as_index=False)["p50"].first()
    old_predictions = pd.read_parquet(
        ROOT / "reports/network_quantiles_high_20260826/predictions.parquet"
    )
    old_predictions = old_predictions[old_predictions["variant"].eq("p50")][
        ["date", "bakery_id", "prediction"]
    ].rename(columns={"prediction": "old_p50_prediction"})
    denominator = old_p50.merge(old_predictions, on=["date", "bakery_id"], validate="one_to_one")
    denominator["base_bakery_prediction"] = denominator["old_p50_prediction"] / denominator["p50"]

    predictions = pd.read_parquet(PREDICTIONS).merge(
        denominator[["date", "bakery_id", "base_bakery_prediction"]],
        on=["date", "bakery_id"],
        validate="many_to_one",
    )
    predictions["factor"] = predictions["prediction"] / predictions["base_bakery_prediction"]
    factors = predictions.pivot(index=["date", "bakery_id"], columns="variant", values="factor").reset_index()
    rows = rows.merge(factors, on=["date", "bakery_id"], validate="many_to_one", suffixes=("", "_new"))

    metrics = []
    actual_error = rows["available_to_sell"] - rows["demand"]
    metrics.append(
        {
            "variant": "actual_state",
            "volume": float(rows["available_to_sell"].sum()),
            "surplus": float(actual_error.clip(lower=0).sum()),
            "underbake": float((-actual_error).clip(lower=0).sum()),
            "imbalance": float(actual_error.abs().sum()),
        }
    )
    metrics.append(score("current", rows["incumbent_sku_forecast"], rows["demand"]))
    metrics.append(score("predictive", rows["predictive_forecast"], rows["demand"]))
    metrics.append(score("predictive_+2%", 1.02 * rows["predictive_forecast"], rows["demand"]))
    for quantile in [50, 55, 60, 67, 75, 80, 85, 90, 95]:
        column = f"p{quantile:02d}_new" if f"p{quantile:02d}_new" in rows else f"p{quantile:02d}"
        plan = rows["predictive_forecast"] * rows[column]
        metrics.append(score(f"P{quantile}", plan, rows["demand"]))
        metrics.append(score(f"P{quantile}_+2%", 1.02 * plan, rows["demand"]))

    result = pd.DataFrame(metrics)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT / "metrics.csv", index=False)
    rows.to_parquet(OUTPUT / "rows.parquet", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

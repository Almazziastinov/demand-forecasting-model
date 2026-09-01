"""Canonical ML metrics and recognized lost demand for rolling candidates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "reports/rolling_actual_state_comparison_20260826/rows.parquet"
OUTPUT = ROOT / "reports/candidate_canonical_metrics_20260826"
VARIANTS = {
    "current": "incumbent_sku_forecast",
    "p50_predictive": "p50_predictive",
    "p50_predictive_simple_floor": "p50_simple_floor",
}


def canonical(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    absolute = np.abs(error)
    squared = error**2
    denominator = np.abs(actual).sum()
    smape_denominator = np.abs(actual) + np.abs(predicted)
    nonzero_smape = smape_denominator > 0
    centered = ((actual - actual.mean()) ** 2).sum()
    return {
        "actual_qty": float(actual.sum()),
        "forecast_qty": float(predicted.sum()),
        "wape_pct": float(100 * absolute.sum() / denominator),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(squared.mean())),
        "bias_qty": float(error.sum()),
        "bias_pct": float(100 * error.sum() / denominator),
        "smape_pct": float(100 * np.mean(2 * absolute[nonzero_smape] / smape_denominator[nonzero_smape])),
        "r2": float(1 - squared.sum() / centered) if centered > 0 else float("nan"),
    }


def recognized_loss(frame: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    sold = pd.to_numeric(frame["actual_sold"], errors="coerce").fillna(0.0).clip(lower=0.0)
    lost = (frame["demand"] - sold).clip(lower=0.0)
    plan_above_sales = (frame[prediction_column] - sold).clip(lower=0.0)
    recognized = np.minimum(plan_above_sales, lost)
    lost_rows = lost > 0
    return {
        "reconstructed_lost": float(lost.sum()),
        "recognized_lost": float(recognized.sum()),
        "recognized_lost_pct": float(100 * recognized.sum() / lost.sum()),
        "lost_rows": int(lost_rows.sum()),
        "recognized_lost_rows": int(((recognized > 0) & lost_rows).sum()),
        "recognized_lost_rows_pct": float(100 * ((recognized > 0) & lost_rows).sum() / lost_rows.sum()),
    }


def main() -> None:
    rows = pd.read_parquet(ROWS)
    sku_rows = []
    bakery_rows = []
    fold_rows = []
    for variant, column in VARIANTS.items():
        sku = canonical(rows["demand"].to_numpy(), rows[column].to_numpy())
        sku.update(recognized_loss(rows, column))
        sku.update({"level": "sku_day", "variant": variant, "rows": len(rows)})
        sku_rows.append(sku)

        aggregation = {"demand": "sum", "actual_sold": "sum", column: "sum"}
        bakery = rows.groupby(["date", "bakery_id"], as_index=False).agg(aggregation)
        bakery_metric = canonical(bakery["demand"].to_numpy(), bakery[column].to_numpy())
        bakery_metric.update(recognized_loss(bakery, column))
        bakery_metric.update({"level": "bakery_day", "variant": variant, "rows": len(bakery)})
        bakery_rows.append(bakery_metric)

        for fold, part in rows.groupby("rolling_fold"):
            metric = canonical(part["demand"].to_numpy(), part[column].to_numpy())
            metric.update(recognized_loss(part, column))
            metric.update({"fold": str(fold), "variant": variant, "rows": len(part)})
            fold_rows.append(metric)

    sku_frame = pd.DataFrame(sku_rows)
    bakery_frame = pd.DataFrame(bakery_rows)
    fold_frame = pd.DataFrame(fold_rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sku_frame.to_csv(OUTPUT / "sku_day_metrics.csv", index=False)
    bakery_frame.to_csv(OUTPUT / "bakery_day_metrics.csv", index=False)
    fold_frame.to_csv(OUTPUT / "sku_day_fold_metrics.csv", index=False)
    print("SKU-day metrics")
    print(sku_frame.to_string(index=False))
    print("\nBakery-day metrics")
    print(bakery_frame.to_string(index=False))
    print("\nSKU-day fold metrics")
    print(fold_frame.to_string(index=False))


if __name__ == "__main__":
    main()

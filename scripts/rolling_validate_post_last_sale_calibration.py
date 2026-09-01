"""Rolling weekly validation of post-last-sale same-day-rate coefficients."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.calibrate_post_last_sale_demand import build_cases

HOURLY = ROOT / ".codex_tmp/rolling_hourly_sales_20260601_20260823.parquet"
OUTPUT = ROOT / "reports/rolling_post_last_sale_calibration_20260826"


def main() -> None:
    cases = build_cases(pd.read_parquet(HOURLY), list(range(7, 19)))
    fold_starts = pd.date_range("2026-06-22", "2026-08-17", freq="7D")
    metrics = []
    coefficients = []
    for test_start in fold_starts:
        calibration_start = test_start - pd.Timedelta(days=21)
        calibration_end = test_start - pd.Timedelta(days=1)
        test_end = min(test_start + pd.Timedelta(days=6), cases["date"].max())
        calibration = cases[cases["date"].between(calibration_start, calibration_end)]
        test = cases[cases["date"].between(test_start, test_end)].copy()
        fitted = calibration.groupby("cutoff", as_index=False).agg(
            calibration_true=("true_hidden", "sum"),
            calibration_raw=("raw_prediction", "sum"),
            calibration_cases=("date", "size"),
        )
        fitted["multiplier"] = fitted["calibration_true"] / fitted["calibration_raw"]
        fitted["fold"] = str(test_start.date())
        coefficients.append(fitted)
        test = test.merge(fitted[["cutoff", "multiplier"]], on="cutoff", how="left")
        test["prediction"] = test["raw_prediction"] * test["multiplier"]
        for cutoff, part in test.groupby("cutoff"):
            true = part["true_hidden"].sum()
            error = part["prediction"] - part["true_hidden"]
            metrics.append(
                {
                    "fold": str(test_start.date()),
                    "test_end": str(test_end.date()),
                    "cutoff": int(cutoff),
                    "cases": int(len(part)),
                    "true_hidden": float(true),
                    "predicted": float(part["prediction"].sum()),
                    "recovery_ratio": float(part["prediction"].sum() / true),
                    "bias_pct": float(100 * error.sum() / true),
                    "wape_pct": float(100 * error.abs().sum() / true),
                }
            )

    metric_frame = pd.DataFrame(metrics)
    coefficient_frame = pd.concat(coefficients, ignore_index=True)
    fold_summary = metric_frame.groupby("fold", as_index=False).agg(
        cases=("cases", "sum"),
        true_hidden=("true_hidden", "sum"),
        predicted=("predicted", "sum"),
    )
    fold_summary["recovery_ratio"] = fold_summary["predicted"] / fold_summary["true_hidden"]
    hour_summary = metric_frame.groupby("cutoff", as_index=False).agg(
        folds=("fold", "nunique"),
        mean_recovery=("recovery_ratio", "mean"),
        min_recovery=("recovery_ratio", "min"),
        max_recovery=("recovery_ratio", "max"),
        mean_wape=("wape_pct", "mean"),
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    metric_frame.to_csv(OUTPUT / "fold_hour_metrics.csv", index=False)
    coefficient_frame.to_csv(OUTPUT / "fold_coefficients.csv", index=False)
    fold_summary.to_csv(OUTPUT / "fold_summary.csv", index=False)
    hour_summary.to_csv(OUTPUT / "hour_summary.csv", index=False)
    print("Fold summary")
    print(fold_summary.to_string(index=False))
    print("\nHour stability")
    print(hour_summary.to_string(index=False))


if __name__ == "__main__":
    main()

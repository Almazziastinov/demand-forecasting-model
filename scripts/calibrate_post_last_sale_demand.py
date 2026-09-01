"""Calibrate same-day-rate lost demand after the last sale on a frozen holdout."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["date", "bakery_id", "product_id"]


def build_cases(hourly: pd.DataFrame, cutoffs: list[int]) -> pd.DataFrame:
    hourly = hourly.copy()
    hourly["date"] = pd.to_datetime(hourly["date"])
    positive = hourly[hourly["sold"] > 0]
    last_sale = positive.groupby(KEYS, as_index=False)["hour"].max().rename(
        columns={"hour": "actual_last_sale_hour"}
    )
    eligible = last_sale[last_sale["actual_last_sale_hour"] >= 21][KEYS]
    work = hourly.merge(eligible, on=KEYS, how="inner")

    rows: list[pd.DataFrame] = []
    for cutoff in cutoffs:
        part = work.assign(
            observed=lambda frame: np.where(
                (frame["hour"] >= 7) & (frame["hour"] <= cutoff), frame["sold"], 0.0
            ),
            hidden=lambda frame: np.where(
                (frame["hour"] > cutoff) & (frame["hour"] <= 23), frame["sold"], 0.0
            ),
        )
        cases = part.groupby(KEYS, as_index=False).agg(
            observed=("observed", "sum"), true_hidden=("hidden", "sum")
        )
        cases = cases[cases["observed"] > 0].copy()
        # Match the research label formula exactly: elapsed time is measured
        # from opening to the last-sale cutoff, not as a count of hourly bins.
        elapsed_hours = max(cutoff - 7, 0.25)
        missing_hours = 23 - cutoff
        cases["cutoff"] = cutoff
        cases["raw_prediction"] = cases["observed"] / elapsed_hours * missing_hours
        cases["current_prediction"] = np.minimum.reduce(
            [
                cases["raw_prediction"].to_numpy(),
                np.full(len(cases), 10.0),
                (0.5 * cases["observed"]).to_numpy(),
            ]
        )
        rows.append(cases)
    return pd.concat(rows, ignore_index=True)


def fit_calibration(cases: pd.DataFrame, calibration_end: pd.Timestamp) -> pd.DataFrame:
    calibration = cases[cases["date"] <= calibration_end]
    fitted = (
        calibration.groupby("cutoff", as_index=False)
        .agg(true_hidden=("true_hidden", "sum"), raw_prediction=("raw_prediction", "sum"), cases=("date", "size"))
    )
    fitted["rate_multiplier"] = fitted["true_hidden"] / fitted["raw_prediction"]
    return fitted


def summarize(cases: pd.DataFrame, calibration_end: pd.Timestamp) -> pd.DataFrame:
    holdout = cases[cases["date"] > calibration_end].copy()
    rows = []
    for cutoff, group in holdout.groupby("cutoff"):
        true_total = group["true_hidden"].sum()
        for method, column in [
            ("current_cap", "current_prediction"),
            ("raw_rate", "raw_prediction"),
            ("calibrated_rate", "calibrated_prediction"),
        ]:
            error = group[column] - group["true_hidden"]
            predicted = group[column].sum()
            rows.append(
                {
                    "cutoff": int(cutoff),
                    "method": method,
                    "cases": int(len(group)),
                    "true_hidden": float(true_total),
                    "predicted": float(predicted),
                    "recovery_ratio": float(predicted / true_total),
                    "bias_pct": float(100 * error.sum() / true_total),
                    "wape_pct": float(100 * error.abs().sum() / true_total),
                    "underpredict_share": float((error < 0).mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["cutoff", "method"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hourly",
        type=Path,
        default=Path(".codex_tmp/pseudo_stockout_network_hourly_20260826.parquet"),
    )
    parser.add_argument("--calibration-end", type=pd.Timestamp, default=pd.Timestamp("2026-08-11"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/post_last_sale_calibration_20260826")
    )
    args = parser.parse_args()

    cases = build_cases(pd.read_parquet(args.hourly), list(range(7, 19)))
    coefficients = fit_calibration(cases, args.calibration_end)
    cases = cases.merge(coefficients[["cutoff", "rate_multiplier"]], on="cutoff", how="left")
    cases["calibrated_prediction"] = cases["raw_prediction"] * cases["rate_multiplier"]
    metrics = summarize(cases, args.calibration_end)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    coefficients.to_csv(args.output_dir / "calibration_coefficients.csv", index=False)
    metrics.to_csv(args.output_dir / "holdout_metrics.csv", index=False)
    cases.to_parquet(args.output_dir / "cases.parquet", index=False)
    print("Calibration coefficients")
    print(coefficients.to_string(index=False))
    print("\nFrozen holdout metrics")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()

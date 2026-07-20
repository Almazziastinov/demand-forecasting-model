"""Walk-forward experiment for a leakage-free SKU-day stockout correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "reports" / "pilot_stockout_forecast_bias" / "sku_day_comparison.csv"
)
DEFAULT_OUTPUT = ROOT / "reports" / "daily_stockout_correction_experiment"


def apply_daily_correction(
    frame: pd.DataFrame,
    *,
    lookback_days: int,
    min_history_days: int,
    min_stockouts: int,
    max_factor: float,
) -> pd.DataFrame:
    """Apply a pair-level factor calculated strictly from earlier dates."""
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    work = work.sort_values(["bakery_id", "product_id", "date"]).reset_index(drop=True)
    work["confirmed_shortfall"] = np.where(
        work["stockout_group"].eq("clear_stockout"),
        (work["daily_sold"] - work["forecast_qty"]).clip(lower=0.0),
        0.0,
    )
    work["daily_correction_factor"] = 1.0
    work["correction_history_days"] = 0
    work["correction_history_stockouts"] = 0

    for _, indexes in work.groupby(
        ["bakery_id", "product_id"], sort=False
    ).groups.items():
        group = work.loc[indexes].sort_values("date")
        for index, row in group.iterrows():
            start = row["date"] - pd.Timedelta(days=lookback_days)
            history = group[(group["date"] < row["date"]) & (group["date"] >= start)]
            stockout_history = history[history["stockout_group"].eq("clear_stockout")]
            work.at[index, "correction_history_days"] = history["date"].nunique()
            work.at[index, "correction_history_stockouts"] = len(stockout_history)
            if (
                history["date"].nunique() < min_history_days
                or len(stockout_history) < min_stockouts
            ):
                continue
            denominator = float(history["forecast_qty"].sum())
            if denominator <= 0:
                continue
            factor = (
                1.0 + float(stockout_history["confirmed_shortfall"].sum()) / denominator
            )
            work.at[index, "daily_correction_factor"] = min(
                max(factor, 1.0), max_factor
            )

    work["adjusted_forecast_qty"] = (
        work["forecast_qty"] * work["daily_correction_factor"]
    )
    return work


def evaluate(frame: pd.DataFrame) -> dict[str, float | int]:
    stockout = frame[frame["stockout_group"].eq("clear_stockout")]
    normal = frame[frame["stockout_group"].eq("confirmed_non_stockout")]
    baseline_shortfall = (stockout["daily_sold"] - stockout["forecast_qty"]).clip(
        lower=0.0
    )
    adjusted_shortfall = (
        stockout["daily_sold"] - stockout["adjusted_forecast_qty"]
    ).clip(lower=0.0)
    normal_sales = float(normal["daily_sold"].sum())
    return {
        "rows_corrected": int((frame["daily_correction_factor"] > 1.0).sum()),
        "stockout_rows_corrected": int(
            (stockout["daily_correction_factor"] > 1.0).sum()
        ),
        "baseline_stockout_underforecast_cases": int((baseline_shortfall > 0.5).sum()),
        "adjusted_stockout_underforecast_cases": int((adjusted_shortfall > 0.5).sum()),
        "underforecast_cases_removed": int(
            ((baseline_shortfall > 0.5) & (adjusted_shortfall <= 0.5)).sum()
        ),
        "baseline_confirmed_shortfall_qty": float(baseline_shortfall.sum()),
        "adjusted_confirmed_shortfall_qty": float(adjusted_shortfall.sum()),
        "normal_baseline_forecast_to_sales": (
            float(normal["forecast_qty"].sum()) / normal_sales
            if normal_sales
            else np.nan
        ),
        "normal_adjusted_forecast_to_sales": (
            float(normal["adjusted_forecast_qty"].sum()) / normal_sales
            if normal_sales
            else np.nan
        ),
        "normal_extra_forecast_qty": float(
            (normal["adjusted_forecast_qty"] - normal["forecast_qty"]).sum()
        ),
    }


def run_grid(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    results = []
    details = {}
    for lookback in [28, 56]:
        for minimum in [2, 3]:
            for cap in [1.2, 1.3, 1.5]:
                name = f"lb{lookback}_min{minimum}_cap{cap:.1f}"
                adjusted = apply_daily_correction(
                    frame,
                    lookback_days=lookback,
                    min_history_days=7,
                    min_stockouts=minimum,
                    max_factor=cap,
                )
                result = {
                    "scenario": name,
                    "lookback_days": lookback,
                    "min_stockouts": minimum,
                    "max_factor": cap,
                }
                result.update(evaluate(adjusted))
                results.append(result)
                details[name] = adjusted
    return pd.DataFrame(results), details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test a walk-forward daily stockout correction"
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input, encoding="utf-8-sig")
    frame = frame[
        frame["stockout_group"].isin(["clear_stockout", "confirmed_non_stockout"])
    ]
    grid, details = run_grid(frame)
    grid["shortfall_reduction_qty"] = (
        grid["baseline_confirmed_shortfall_qty"]
        - grid["adjusted_confirmed_shortfall_qty"]
    )
    grid["normal_ratio_increase"] = (
        grid["normal_adjusted_forecast_to_sales"]
        - grid["normal_baseline_forecast_to_sales"]
    )
    grid = grid.sort_values(
        ["underforecast_cases_removed", "normal_ratio_increase"],
        ascending=[False, True],
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output / "scenario_comparison.csv", index=False, encoding="utf-8-sig")
    best_name = str(grid.iloc[0]["scenario"])
    details[best_name].to_csv(
        output / "best_scenario_rows.csv", index=False, encoding="utf-8-sig"
    )
    summary = {"best_scenario": best_name, "scenarios": grid.to_dict(orient="records")}
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

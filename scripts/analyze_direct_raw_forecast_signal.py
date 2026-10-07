"""Evaluate the raw Direct forecast as a lost-demand direction signal."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/three_prod_models_august_holdout_20260911"
KEYS = ["date", "bakery_id", "product_id"]
UNIT_THRESHOLD = 1.0


def main() -> None:
    actual = pd.read_parquet(
        OUTPUT / "economics_actual_state.parquet",
        columns=KEYS + ["demand", "served", "lost"],
    )
    forecast = pd.read_parquet(
        OUTPUT / "direct_plan.parquet",
        columns=KEYS + ["product_name", "category_name", "forecast_qty"],
    )
    direct = pd.read_parquet(
        OUTPUT / "economics_direct_alpha_025_v1.parquet",
        columns=KEYS + ["lost"],
    ).rename(columns={"lost": "direct_final_lost"})
    rows = actual.merge(forecast, on=KEYS, validate="one_to_one").merge(
        direct, on=KEYS, validate="one_to_one"
    )

    rows["raw_forecast_implied_lost"] = (
        rows["demand"] - rows["forecast_qty"]
    ).clip(lower=0.0)
    rows["lost_reduction_vs_actual"] = (
        rows["lost"] - rows["raw_forecast_implied_lost"]
    )
    rows["actual_lost_case"] = rows["lost"].ge(UNIT_THRESHOLD)
    rows["helpful_signal"] = rows["actual_lost_case"] & rows[
        "lost_reduction_vs_actual"
    ].ge(UNIT_THRESHOLD)
    rows["fully_covers_lost"] = rows["helpful_signal"] & rows[
        "raw_forecast_implied_lost"
    ].lt(UNIT_THRESHOLD)
    rows["partially_reduces_lost"] = (
        rows["helpful_signal"] & ~rows["fully_covers_lost"]
    )
    rows["missed_actual_lost"] = rows["actual_lost_case"] & ~rows[
        "helpful_signal"
    ]
    rows["potential_new_lost_case"] = ~rows["actual_lost_case"] & rows[
        "raw_forecast_implied_lost"
    ].ge(UNIT_THRESHOLD)
    rows["false_upward_signal"] = ~rows["actual_lost_case"] & (
        rows["forecast_qty"] - rows["served"]
    ).ge(UNIT_THRESHOLD)
    rows["final_policy_helpful"] = rows["actual_lost_case"] & (
        rows["lost"] - rows["direct_final_lost"]
    ).ge(UNIT_THRESHOLD)

    actual_cases = int(rows["actual_lost_case"].sum())
    helpful = int(rows["helpful_signal"].sum())
    summary = pd.DataFrame(
        [
            {
                "rows": len(rows),
                "actual_lost_cases": actual_cases,
                "helpful_signal_cases": helpful,
                "helpful_signal_share_pct": 100.0 * helpful / actual_cases,
                "partial_improvement_cases": int(
                    rows["partially_reduces_lost"].sum()
                ),
                "fully_covered_cases": int(rows["fully_covers_lost"].sum()),
                "missed_actual_lost_cases": int(rows["missed_actual_lost"].sum()),
                "potential_new_lost_cases": int(
                    rows["potential_new_lost_case"].sum()
                ),
                "false_upward_signal_cases": int(
                    rows["false_upward_signal"].sum()
                ),
                "raw_and_final_helpful_cases": int(
                    (rows["helpful_signal"] & rows["final_policy_helpful"]).sum()
                ),
                "raw_helpful_lost_downstream_cases": int(
                    (rows["helpful_signal"] & ~rows["final_policy_helpful"]).sum()
                ),
                "final_helpful_without_raw_signal_cases": int(
                    (~rows["helpful_signal"] & rows["final_policy_helpful"]).sum()
                ),
                "potential_recovered_units": float(
                    rows["lost_reduction_vs_actual"].clip(lower=0.0).sum()
                ),
                "potential_added_lost_units": float(
                    (-rows["lost_reduction_vs_actual"]).clip(lower=0.0).sum()
                ),
                "net_potential_recovered_units": float(
                    rows["lost_reduction_vs_actual"].sum()
                ),
                "raw_forecast_excess_units_over_demand": float(
                    (rows["forecast_qty"] - rows["demand"]).clip(lower=0.0).sum()
                ),
            }
        ]
    )

    rows.to_parquet(OUTPUT / "direct_raw_forecast_signal_rows.parquet", index=False)
    summary.to_csv(
        OUTPUT / "direct_raw_forecast_signal_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

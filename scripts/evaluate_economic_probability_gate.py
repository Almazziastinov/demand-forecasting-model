"""Evaluate hard probability gates for the Direct lost-demand uplift."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_weighted_loss_direct import KEYS, RUN, SOURCE, summarize  # noqa: E402
from scripts.run_freshness_calibrated_economics import simulate_group  # noqa: E402
from src.experiments_v2.direct_alpha_allocation import (  # noqa: E402
    DirectAlphaAllocationConfig,
    build_selected_direct_plan,
)


THRESHOLDS = (0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.40, 0.50)


def build_gate_plan(threshold: float) -> pd.DataFrame:
    parts = []
    for input_path in sorted((RUN / "direct_days").glob("*/shadow_input.parquet")):
        rows = pd.read_parquet(input_path)
        rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
        rows["predictive_uplift"] = rows["predicted_lost_if_stockout"].where(
            rows["predicted_stockout_probability"].ge(threshold), 0.0
        )
        plan = build_selected_direct_plan(rows, DirectAlphaAllocationConfig())
        parts.append(plan[KEYS + ["selected_sku_forecast"]])
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    rows = pd.read_parquet(SOURCE / "evaluation_input.parquet")
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    actual = pd.read_parquet(SOURCE / "economics_actual_state.parquet")
    current = pd.read_parquet(SOURCE / "economics_direct_alpha_025_v1.parquet")
    actual_gp = float(actual["gross_profit"].sum())
    current_gp = float(current["gross_profit"].sum())
    summaries = [summarize("direct_alpha_025_v1", current, actual_gp)]

    for threshold in THRESHOLDS:
        label = f"economic_gate_{int(threshold * 100):02d}_volume_neutral"
        simulation_path = RUN / f"economics_{label}.parquet"
        if simulation_path.exists():
            simulation = pd.read_parquet(simulation_path)
            summaries.append(summarize(label, simulation, actual_gp))
            continue

        plan_column = f"gate_{int(threshold * 100):02d}_plan"
        gate_plan = build_gate_plan(threshold).rename(
            columns={"selected_sku_forecast": plan_column}
        )
        candidate_rows = rows.merge(gate_plan, on=KEYS, how="left", validate="one_to_one")
        candidate_rows[plan_column] = candidate_rows[plan_column].fillna(0.0).clip(lower=0.0)

        group_keys = [candidate_rows["date"], candidate_rows["bakery_id"]]
        current_total = candidate_rows["plan_1"].groupby(group_keys).transform("sum")
        gate_total = candidate_rows[plan_column].groupby(group_keys).transform("sum")
        neutral_column = f"{plan_column}_neutral"
        candidate_rows[neutral_column] = candidate_rows[plan_column] * (
            current_total / gate_total.replace(0.0, pd.NA)
        ).fillna(1.0)

        simulation = pd.concat(
            [
                simulate_group(group, neutral_column)
                for _, group in candidate_rows.groupby(
                    ["bakery_id", "product_id"], sort=False
                )
            ],
            ignore_index=True,
        )
        summaries.append(summarize(label, simulation, actual_gp))
        simulation.to_parquet(simulation_path, index=False)

    summary = pd.DataFrame(summaries)
    summary["gp_delta_vs_current_direct"] = summary["gross_profit"] - current_gp
    summary.to_csv(RUN / "economic_gate_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

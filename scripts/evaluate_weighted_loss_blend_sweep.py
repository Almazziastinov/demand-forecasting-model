"""Sweep blends of current Direct and volume-neutral weighted-loss plans."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_weighted_loss_direct import KEYS, RUN, SOURCE, summarize  # noqa: E402
from scripts.run_freshness_calibrated_economics import simulate_group  # noqa: E402


BLEND_WEIGHTS = (0.10, 0.25, 0.50, 0.75, 0.90)


def main() -> None:
    rows = pd.read_parquet(SOURCE / "evaluation_input.parquet")
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()

    plan_parts = []
    for path in sorted((RUN / "direct_days").glob("*/shadow_rows.parquet")):
        plan_parts.append(pd.read_parquet(path, columns=KEYS + ["selected_sku_forecast"]))
    weighted_plan = pd.concat(plan_parts, ignore_index=True)
    weighted_plan["date"] = pd.to_datetime(weighted_plan["date"]).dt.normalize()
    weighted_plan = weighted_plan.rename(columns={"selected_sku_forecast": "weighted_plan"})
    rows = rows.merge(weighted_plan, on=KEYS, how="left", validate="one_to_one")
    rows["weighted_plan"] = rows["weighted_plan"].fillna(0.0).clip(lower=0.0)

    group_keys = [rows["date"], rows["bakery_id"]]
    current_total = rows["plan_1"].groupby(group_keys).transform("sum")
    weighted_total = rows["weighted_plan"].groupby(group_keys).transform("sum")
    rows["weighted_neutral"] = rows["weighted_plan"] * (
        current_total / weighted_total.replace(0.0, pd.NA)
    ).fillna(1.0)

    actual = pd.read_parquet(SOURCE / "economics_actual_state.parquet")
    current = pd.read_parquet(SOURCE / "economics_direct_alpha_025_v1.parquet")
    actual_gp = float(actual["gross_profit"].sum())
    current_gp = float(current["gross_profit"].sum())
    summaries = [summarize("direct_alpha_025_v1", current, actual_gp)]

    for weight in BLEND_WEIGHTS:
        weight_pct = int(weight * 100)
        plan_column = f"blend_{weight_pct:02d}_plan"
        rows[plan_column] = (1.0 - weight) * rows["plan_1"] + weight * rows["weighted_neutral"]
        simulation = pd.concat(
            [
                simulate_group(group, plan_column)
                for _, group in rows.groupby(["bakery_id", "product_id"], sort=False)
            ],
            ignore_index=True,
        )
        label = f"weighted_loss_blend_{weight_pct:02d}pct"
        summaries.append(summarize(label, simulation, actual_gp))
        simulation.to_parquet(RUN / f"economics_{label}.parquet", index=False)

    neutral = pd.read_parquet(RUN / "economics_direct_weighted_loss_volume_neutral.parquet")
    summaries.append(summarize("weighted_loss_blend_100pct", neutral, actual_gp))
    summary = pd.DataFrame(summaries)
    summary["gp_delta_vs_current_direct"] = summary["gross_profit"] - current_gp
    summary.to_csv(RUN / "blend_sweep_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

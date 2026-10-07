"""Evaluate positive corrected residual overlays on the frozen August holdout."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_corrected_direct_ablation import (  # noqa: E402
    CORRECTED,
    CURRENT,
    build_plan,
)
from scripts.evaluate_weighted_loss_direct import KEYS, SOURCE, summarize  # noqa: E402
from scripts.run_freshness_calibrated_economics import simulate_group  # noqa: E402


OUTPUT = ROOT / "reports/corrected_direct_additive_august_20260914"
BUDGETS = (0.005, 0.01, 0.02)


def simulate(rows: pd.DataFrame, plan_column: str) -> pd.DataFrame:
    return pd.concat(
        [
            simulate_group(group, plan_column)
            for _, group in rows.groupby(["bakery_id", "product_id"], sort=False)
        ],
        ignore_index=True,
    )


def main() -> None:
    rows = pd.read_parquet(SOURCE / "evaluation_input.parquet")
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    corrected = build_plan(CORRECTED, CURRENT, use_uplift=True, use_floor=False).rename(
        columns={"selected_sku_forecast": "corrected_plan"}
    )
    current_no_loss = build_plan(
        CURRENT, CURRENT, use_uplift=False, use_floor=False
    ).rename(columns={"selected_sku_forecast": "current_no_loss_plan"})
    rows = rows.merge(corrected, on=KEYS, how="left", validate="one_to_one").merge(
        current_no_loss, on=KEYS, how="left", validate="one_to_one"
    )
    rows["corrected_plan"] = rows["corrected_plan"].fillna(0.0).clip(lower=0.0)
    rows["current_no_loss_plan"] = (
        rows["current_no_loss_plan"].fillna(0.0).clip(lower=0.0)
    )
    groups = [rows["date"], rows["bakery_id"]]
    current_total = rows["plan_1"].groupby(groups).transform("sum")
    corrected_total = rows["corrected_plan"].groupby(groups).transform("sum")
    corrected_neutral = rows["corrected_plan"] * (
        current_total / corrected_total.replace(0.0, pd.NA)
    ).fillna(1.0)
    positive_residual = (corrected_neutral - rows["plan_1"]).clip(lower=0.0)
    residual_total = positive_residual.groupby(groups).transform("sum")
    no_loss_total = rows["current_no_loss_plan"].groupby(groups).transform("sum")
    no_loss_neutral = rows["current_no_loss_plan"] * (
        current_total / no_loss_total.replace(0.0, pd.NA)
    ).fillna(1.0)
    old_loss_residual = (rows["plan_1"] - no_loss_neutral).clip(lower=0.0)
    old_loss_total = old_loss_residual.groupby(groups).transform("sum")

    actual = pd.read_parquet(SOURCE / "economics_actual_state.parquet")
    current = pd.read_parquet(SOURCE / "economics_direct_alpha_025_v1.parquet")
    actual_gp = float(actual["gross_profit"].sum())
    current_gp = float(current["gross_profit"].sum())
    summaries = [summarize("direct_alpha_025_v1", current, actual_gp)]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for budget_share in BUDGETS:
        budget = budget_share * current_total
        scale = (budget / residual_total.replace(0.0, pd.NA)).clip(upper=1.0).fillna(0.0)
        column = f"additive_{budget_share * 100:.1f}pct_plan".replace(".", "p")
        rows[column] = rows["plan_1"] + positive_residual * scale
        simulation = simulate(rows, column)
        label = column.removesuffix("_plan")
        summaries.append(summarize(label, simulation, actual_gp))
        simulation.to_parquet(OUTPUT / f"economics_{label}.parquet", index=False)
        uniform_column = f"uniform_{budget_share * 100:.1f}pct_plan".replace(".", "p")
        rows[uniform_column] = rows["plan_1"] * (1.0 + budget_share)
        uniform_simulation = simulate(rows, uniform_column)
        uniform_label = uniform_column.removesuffix("_plan")
        summaries.append(summarize(uniform_label, uniform_simulation, actual_gp))
        uniform_simulation.to_parquet(
            OUTPUT / f"economics_{uniform_label}.parquet", index=False
        )
        old_scale = (
            budget / old_loss_total.replace(0.0, pd.NA)
        ).clip(upper=1.0).fillna(0.0)
        old_column = f"old_loss_{budget_share * 100:.1f}pct_plan".replace(".", "p")
        rows[old_column] = rows["plan_1"] + old_loss_residual * old_scale
        old_simulation = simulate(rows, old_column)
        old_label = old_column.removesuffix("_plan")
        summaries.append(summarize(old_label, old_simulation, actual_gp))
        old_simulation.to_parquet(OUTPUT / f"economics_{old_label}.parquet", index=False)
    summary = pd.DataFrame(summaries)
    summary["gp_delta_vs_current_direct"] = summary["gross_profit"] - current_gp
    summary.to_csv(OUTPUT / "summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

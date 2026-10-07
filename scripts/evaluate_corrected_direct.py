"""Evaluate fully retrained corrected-demand Direct on frozen August economics."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_weighted_loss_direct import KEYS, SOURCE, summarize  # noqa: E402
from scripts.run_freshness_calibrated_economics import simulate_group  # noqa: E402


SOURCE = ROOT / "reports/three_prod_models_august_holdout_20260911"
RUN = ROOT / "reports/corrected_direct_august_holdout_20260914"


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
    plan = pd.read_parquet(RUN / "plan.parquet").rename(
        columns={"selected_sku_forecast": "corrected_plan"}
    )
    plan["date"] = pd.to_datetime(plan["date"]).dt.normalize()
    rows = rows.merge(plan, on=KEYS, how="left", validate="one_to_one")
    rows["corrected_plan"] = rows["corrected_plan"].fillna(0.0).clip(lower=0.0)

    group_keys = [rows["date"], rows["bakery_id"]]
    current_total = rows["plan_1"].groupby(group_keys).transform("sum")
    corrected_total = rows["corrected_plan"].groupby(group_keys).transform("sum")
    rows["corrected_volume_neutral_plan"] = rows["corrected_plan"] * (
        current_total / corrected_total.replace(0.0, pd.NA)
    ).fillna(1.0)

    raw = simulate(rows, "corrected_plan")
    neutral = simulate(rows, "corrected_volume_neutral_plan")
    actual = pd.read_parquet(SOURCE / "economics_actual_state.parquet")
    current = pd.read_parquet(SOURCE / "economics_direct_alpha_025_v1.parquet")
    actual_gp = float(actual["gross_profit"].sum())
    current_gp = float(current["gross_profit"].sum())
    summary = pd.DataFrame(
        [
            summarize("actual_state", actual, actual_gp),
            summarize("direct_alpha_025_v1", current, actual_gp),
            summarize("corrected_direct_gate30", raw, actual_gp),
            summarize("corrected_direct_gate30_volume_neutral", neutral, actual_gp),
        ]
    )
    summary["gp_delta_vs_current_direct"] = summary["gross_profit"] - current_gp
    summary.to_csv(RUN / "evaluation_summary.csv", index=False, encoding="utf-8-sig")
    raw.to_parquet(RUN / "economics_corrected_direct_gate30.parquet", index=False)
    neutral.to_parquet(
        RUN / "economics_corrected_direct_gate30_volume_neutral.parquet", index=False
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

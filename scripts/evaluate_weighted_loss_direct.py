"""Evaluate probability-weighted Direct on the frozen August economics."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_freshness_calibrated_economics import simulate_group  # noqa: E402


SOURCE = ROOT / "reports/three_prod_models_august_holdout_20260911"
RUN = ROOT / "reports/weighted_loss_direct_august_holdout_20260914"
KEYS = ["date", "bakery_id", "product_id"]


def summarize(
    label: str, simulation: pd.DataFrame, actual_gp: float
) -> dict[str, float | int | str]:
    gross_profit = float(simulation["gross_profit"].sum())
    return {
        "variant": label,
        "production": float(simulation["production"].sum()),
        "served": float(simulation["served"].sum()),
        "lost": float(simulation["lost"].sum()),
        "lost_cases_ge_1": int(simulation["lost"].ge(1.0).sum()),
        "writeoff": float(simulation["writeoff"].sum()),
        "revenue": float(simulation["revenue"].sum()),
        "production_cost": float(simulation["production_cost"].sum()),
        "gross_profit": gross_profit,
        "gp_delta_vs_actual": gross_profit - actual_gp,
        "service_pct": 100
        * float(simulation["served"].sum())
        / float(simulation["demand"].sum()),
    }


def main() -> None:
    rows = pd.read_parquet(SOURCE / "evaluation_input.parquet")
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    plans = []
    for path in sorted((RUN / "direct_days").glob("*/shadow_rows.parquet")):
        plans.append(
            pd.read_parquet(path, columns=KEYS + ["selected_sku_forecast"])
        )
    plan = pd.concat(plans, ignore_index=True)
    plan["date"] = pd.to_datetime(plan["date"]).dt.normalize()
    plan = plan.rename(columns={"selected_sku_forecast": "weighted_loss_plan"})
    rows = rows.merge(plan, on=KEYS, how="left", validate="one_to_one")
    rows["weighted_loss_plan"] = rows["weighted_loss_plan"].fillna(0.0).clip(lower=0.0)
    group_keys = [rows["date"], rows["bakery_id"]]
    current_total = rows["plan_1"].groupby(group_keys).transform("sum")
    weighted_total = rows["weighted_loss_plan"].groupby(group_keys).transform("sum")
    rows["weighted_loss_volume_neutral_plan"] = rows["weighted_loss_plan"] * (
        current_total / weighted_total.replace(0.0, pd.NA)
    ).fillna(1.0)

    weighted = pd.concat(
        [
            simulate_group(group, "weighted_loss_plan")
            for _, group in rows.groupby(["bakery_id", "product_id"], sort=False)
        ],
        ignore_index=True,
    )
    volume_neutral = pd.concat(
        [
            simulate_group(group, "weighted_loss_volume_neutral_plan")
            for _, group in rows.groupby(["bakery_id", "product_id"], sort=False)
        ],
        ignore_index=True,
    )
    current = pd.read_parquet(SOURCE / "economics_direct_alpha_025_v1.parquet")
    actual = pd.read_parquet(SOURCE / "economics_actual_state.parquet")
    actual_gp = float(actual["gross_profit"].sum())
    summary = pd.DataFrame(
        [
            summarize("actual_state", actual, actual_gp),
            summarize("direct_alpha_025_v1", current, actual_gp),
            summarize("direct_weighted_loss_v1", weighted, actual_gp),
            summarize("direct_weighted_loss_volume_neutral", volume_neutral, actual_gp),
        ]
    )
    current_gp = float(current["gross_profit"].sum())
    summary["gp_delta_vs_current_direct"] = summary["gross_profit"] - current_gp
    summary.to_csv(RUN / "evaluation_summary.csv", index=False, encoding="utf-8-sig")

    transitions_records = []
    for label, candidate in [
        ("weighted_loss", weighted),
        ("weighted_loss_volume_neutral", volume_neutral),
    ]:
        paired = current[KEYS + ["lost"]].merge(
            candidate[KEYS + ["lost"]],
            on=KEYS,
            suffixes=("_current", "_candidate"),
            validate="one_to_one",
        ).merge(
            actual[KEYS + ["lost"]].rename(columns={"lost": "lost_actual"}),
            on=KEYS,
            validate="one_to_one",
        )
        current_case = paired["lost_current"].ge(1.0)
        candidate_case = paired["lost_candidate"].ge(1.0)
        actual_case = paired["lost_actual"].ge(1.0)
        transitions_records.append(
            {
                "variant": label,
                "current_cases": int(current_case.sum()),
                "candidate_cases": int(candidate_case.sum()),
                "case_delta": int(candidate_case.sum() - current_case.sum()),
                "current_fixed_by_candidate": int((current_case & ~candidate_case).sum()),
                "new_vs_current": int((~current_case & candidate_case).sum()),
                "actual_cases_improved": int(
                    (actual_case & paired["lost_candidate"].lt(paired["lost_current"] - 1e-9)).sum()
                ),
                "actual_cases_worsened": int(
                    (actual_case & paired["lost_candidate"].gt(paired["lost_current"] + 1e-9)).sum()
                ),
            }
        )
    transitions = pd.DataFrame(transitions_records)
    transitions.to_csv(RUN / "case_transitions.csv", index=False, encoding="utf-8-sig")

    by_day = (
        pd.concat(
            [
                current.assign(variant="current_direct"),
                weighted.assign(variant="weighted_loss_direct"),
                volume_neutral.assign(variant="weighted_loss_volume_neutral"),
            ]
        )
        .groupby(["date", "variant"], as_index=False)
        .agg(
            production=("production", "sum"),
            served=("served", "sum"),
            lost=("lost", "sum"),
            writeoff=("writeoff", "sum"),
            gross_profit=("gross_profit", "sum"),
        )
    )
    by_day.to_csv(RUN / "by_day.csv", index=False, encoding="utf-8-sig")
    weighted.to_parquet(RUN / "economics_direct_weighted_loss_v1.parquet", index=False)
    volume_neutral.to_parquet(
        RUN / "economics_direct_weighted_loss_volume_neutral.parquet", index=False
    )
    print(summary.to_string(index=False))
    print("\nTransitions")
    print(transitions.to_string(index=False))


if __name__ == "__main__":
    main()

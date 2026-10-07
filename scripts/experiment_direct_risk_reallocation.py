"""Test causal, volume-neutral risk reallocation on top of Direct alpha=.25."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_freshness_calibrated_economics import simulate_group  # noqa: E402


OUTPUT = ROOT / "reports/three_prod_models_august_holdout_20260911"
INPUT = OUTPUT / "evaluation_input.parquet"
DIRECT_DAYS = OUTPUT / "direct_days"
KEYS = ["date", "bakery_id", "product_id"]


@dataclass(frozen=True)
class Candidate:
    name: str
    recipient_probability: float
    donor_probability: float
    recipient_cap: float
    donor_fraction: float


CANDIDATES = [
    Candidate("shift_conservative", 0.70, 0.20, 3.0, 0.10),
    Candidate("shift_p60_d20_cap3", 0.60, 0.20, 3.0, 0.15),
    Candidate("shift_p60_d30_cap5", 0.60, 0.30, 5.0, 0.20),
    Candidate("shift_p50_d30_cap5", 0.50, 0.30, 5.0, 0.20),
    Candidate("shift_p60_d40_cap5", 0.60, 0.40, 5.0, 0.20),
    Candidate("shift_aggressive", 0.50, 0.40, 5.0, 0.30),
]


def load_features(universe: pd.DataFrame) -> pd.DataFrame:
    columns = KEYS + [
        "selected_sku_forecast",
        "direct_p50",
        "floor_demand_p67",
        "predicted_stockout_probability",
        "predicted_lost_if_stockout",
        "predictive_uplift",
    ]
    frames = [
        pd.read_parquet(path, columns=columns)
        for path in sorted(DIRECT_DAYS.glob("*/shadow_rows.parquet"))
    ]
    features = pd.concat(frames, ignore_index=True)
    features["date"] = pd.to_datetime(features["date"]).dt.normalize()
    return universe.merge(features, on=KEYS, how="left", validate="one_to_one")


def reallocate_group(group: pd.DataFrame, candidate: Candidate) -> pd.Series:
    plan = group["selected_sku_forecast"].clip(lower=0.0).copy()
    probability = group["predicted_stockout_probability"].fillna(0.0)
    recipient_mask = probability.ge(candidate.recipient_probability)
    donor_mask = probability.le(candidate.donor_probability)

    recipient_need = group["predicted_lost_if_stockout"].clip(lower=0.0)
    recipient_need = recipient_need.clip(upper=candidate.recipient_cap)
    recipient_need = recipient_need.where(recipient_mask, 0.0)

    # Two independent causal anchors exist in the frozen Direct artifact.  A
    # donor is never reduced below the smaller of them, and at most by the
    # configured fraction of its current plan.
    anchor = group[["direct_p50", "floor_demand_p67"]].min(axis=1).clip(lower=0.0)
    donor_capacity = (plan - anchor).clip(lower=0.0)
    donor_capacity = pd.concat(
        [donor_capacity, plan * candidate.donor_fraction], axis=1
    ).min(axis=1)
    donor_capacity = donor_capacity.where(donor_mask, 0.0)

    transfer = min(float(recipient_need.sum()), float(donor_capacity.sum()))
    if transfer <= 0.0:
        return plan

    remaining = transfer
    donor_order = group.assign(_capacity=donor_capacity).sort_values(
        ["predicted_stockout_probability", "_capacity"], ascending=[True, False]
    )
    for index, row in donor_order.iterrows():
        amount = min(float(row["_capacity"]), remaining)
        if amount <= 0.0:
            continue
        plan.loc[index] -= amount
        remaining -= amount
        if remaining <= 1e-9:
            break

    remaining = transfer
    recipient_order = group.assign(_need=recipient_need).sort_values(
        ["predicted_stockout_probability", "predictive_uplift"],
        ascending=[False, False],
    )
    for index, row in recipient_order.iterrows():
        amount = min(float(row["_need"]), remaining)
        if amount <= 0.0:
            continue
        plan.loc[index] += amount
        remaining -= amount
        if remaining <= 1e-9:
            break
    return plan


def build_candidate(features: pd.DataFrame, candidate: Candidate) -> pd.Series:
    pieces = []
    for _, group in features.groupby(["date", "bakery_id"], sort=False):
        plan = reallocate_group(group, candidate)
        pieces.append(plan)
    result = pd.concat(pieces).sort_index()
    if abs(float(result.sum() - features["selected_sku_forecast"].sum())) > 1e-6:
        raise RuntimeError(f"Volume conservation failed for {candidate.name}")
    return result


def evaluate(
    rows: pd.DataFrame,
    plan_column: str,
    label: str,
    actual_cases: pd.Series,
    actual_gp: float,
    transferred: float,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    simulation = pd.concat(
        [
            simulate_group(group, plan_column)
            for _, group in rows.groupby(["bakery_id", "product_id"], sort=False)
        ],
        ignore_index=True,
    )
    model_cases = simulation["lost"].ge(1.0)
    return (
        {
            "variant": label,
            "transferred_qty": transferred,
            "lost_cases": int(model_cases.sum()),
            "case_delta_vs_actual": int(model_cases.sum() - actual_cases.sum()),
            "lost_qty": float(simulation["lost"].sum()),
            "writeoff_qty": float(simulation["writeoff"].sum()),
            "service_pct": (
                100 * simulation["served"].sum() / simulation["demand"].sum()
            ),
            "gross_profit": float(simulation["gross_profit"].sum()),
            "gp_delta_vs_actual": float(simulation["gross_profit"].sum() - actual_gp),
        },
        simulation,
    )


def main() -> None:
    rows = pd.read_parquet(INPUT)
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    features = load_features(rows[KEYS])
    actual = pd.read_parquet(OUTPUT / "economics_actual_state.parquet")
    actual_cases = actual["lost"].ge(1.0)
    actual_gp = float(actual["gross_profit"].sum())
    direct = pd.read_parquet(OUTPUT / "economics_direct_alpha_025_v1.parquet")
    records = [
        {
            "variant": "direct_alpha_025_v1",
            "transferred_qty": 0.0,
            "lost_cases": int(direct["lost"].ge(1.0).sum()),
            "case_delta_vs_actual": int(
                direct["lost"].ge(1.0).sum() - actual_cases.sum()
            ),
            "lost_qty": float(direct["lost"].sum()),
            "writeoff_qty": float(direct["writeoff"].sum()),
            "service_pct": 100 * direct["served"].sum() / direct["demand"].sum(),
            "gross_profit": float(direct["gross_profit"].sum()),
            "gp_delta_vs_actual": float(direct["gross_profit"].sum() - actual_gp),
        }
    ]

    for index, candidate in enumerate(CANDIDATES):
        plan_column = f"candidate_plan_{index}"
        candidate_plan = build_candidate(features, candidate)
        rows[plan_column] = candidate_plan
        transferred = 0.5 * float(
            (candidate_plan - features["selected_sku_forecast"]).abs().sum()
        )
        record, simulation = evaluate(
            rows,
            plan_column,
            candidate.name,
            actual_cases,
            actual_gp,
            transferred,
        )
        records.append(record)
        simulation.to_parquet(OUTPUT / f"risk_reallocation_{candidate.name}.parquet")

    summary = pd.DataFrame(records).sort_values(
        ["lost_cases", "gross_profit"], ascending=[True, False]
    )
    summary.to_csv(
        OUTPUT / "risk_reallocation_summary.csv", index=False, encoding="utf-8-sig"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

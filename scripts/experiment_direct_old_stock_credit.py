"""Test causal old-stock credit rules in the Direct production policy."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_three_model_august_holdout import (  # noqa: E402
    add_causal_freshness_priors,
    add_plan_columns,
    add_reconciliation,
    load_base_rows,
)


OUTPUT = ROOT / "reports/direct_old_stock_credit_20260911"
SOURCE_OUTPUT = ROOT / "reports/three_prod_models_august_holdout_20260911"
KEYS = ["date", "bakery_id", "product_id"]
RULES = {
    "current_full_credit": ("full", 1.0),
    "old_share_model_q50": ("model", 0.50),
    "old_share_model_q50_q75_b25": ("blend", 0.25),
    "old_share_model_q50_q75_b50": ("blend", 0.50),
    "old_share_model_q50_q75_b75": ("blend", 0.75),
    "old_share_model_q75": ("model", 0.75),
    "sellable_share_x3": ("sellable", 3.0),
    "sellable_share_x2_5": ("sellable", 2.5),
    "sellable_share_x2": ("sellable", 2.0),
    "sellable_share_x1_5": ("sellable", 1.5),
    "sellable_share_x1_25": ("sellable", 1.25),
    "sellable_share_cap": ("sellable", 1.0),
    "half_sellable_credit": ("sellable", 0.5),
    "no_old_stock_credit": ("none", 0.0),
}


def simulate_group(
    group: pd.DataFrame,
    plan_column: str,
    credit_mode: str,
    credit_scale: float,
) -> pd.DataFrame:
    carry = 0.0
    output = []
    for row in group.sort_values("date").itertuples(index=False):
        old_opening = max(float(carry + row.old_reconciliation_in), 0.0)
        received_fresh = max(
            float(row.incoming_move_qty + row.fresh_reconciliation_in), 0.0
        )
        sent = max(float(row.outgoing_move_qty), 0.0)
        plan = max(float(getattr(row, plan_column)), 0.0)

        if credit_mode == "full":
            old_credit = old_opening
        elif credit_mode == "sellable":
            expected_old_sales = plan * float(row.old_share_prior) * credit_scale
            old_credit = min(old_opening, expected_old_sales)
        elif credit_mode == "model":
            prediction_column = f"predicted_old_share_q{int(credit_scale * 100)}"
            expected_old_sales = plan * float(getattr(row, prediction_column))
            old_credit = min(old_opening, expected_old_sales)
        elif credit_mode == "blend":
            predicted_share = float(row.predicted_old_share_q50) + credit_scale * (
                float(row.predicted_old_share_q75)
                - float(row.predicted_old_share_q50)
            )
            old_credit = min(old_opening, plan * predicted_share)
        else:
            old_credit = 0.0

        production = max(plan + sent - received_fresh - old_credit, 0.0)
        old_after_out = max(old_opening - sent, 0.0)
        fresh_after_out = max(
            production + received_fresh - max(sent - old_opening, 0.0), 0.0
        )
        demand = max(float(row.demand), 0.0)
        old_target = demand * float(row.old_share_prior)
        sold_old = min(old_after_out, old_target, demand)
        sold_fresh = min(fresh_after_out, demand - sold_old)
        expired = old_after_out - sold_old
        carry = fresh_after_out - sold_fresh

        fresh_price = float(row.fresh_price_prior)
        old_price = float(row.old_price_prior)
        unit_cost = float(row.unit_cost)
        revenue = sold_fresh * fresh_price + sold_old * old_price
        production_cost = production * unit_cost
        output.append(
            {
                "date": row.date,
                "bakery_id": row.bakery_id,
                "product_id": row.product_id,
                "demand": demand,
                "plan": plan,
                "old_opening": old_opening,
                "old_credit": old_credit,
                "production": production,
                "served": sold_old + sold_fresh,
                "sold_fresh": sold_fresh,
                "sold_old": sold_old,
                "lost": demand - sold_old - sold_fresh,
                "writeoff": expired,
                "ending_fresh_carry": carry,
                "revenue": revenue,
                "production_cost": production_cost,
                "gross_profit": revenue - production_cost,
            }
        )
    return pd.DataFrame(output)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, _ = load_base_rows()
    rows = add_plan_columns(rows)
    plan_labels = rows.attrs["plan_labels"]
    direct_plan_column = next(
        column
        for column, label in plan_labels.items()
        if label == "direct_alpha_025_v1"
    )
    rows = add_causal_freshness_priors(rows)
    rows = add_reconciliation(rows)
    predictions = pd.read_parquet(
        ROOT
        / "reports/old_stock_sellability_model_20260912"
        / "august_predictions.parquet",
        columns=KEYS + ["quantile_0.50", "quantile_0.75"],
    ).rename(
        columns={
            "quantile_0.50": "predicted_old_share_q50",
            "quantile_0.75": "predicted_old_share_q75",
        }
    )
    rows = rows.merge(predictions, on=KEYS, how="left", validate="one_to_one")
    missing_predictions = rows["predicted_old_share_q50"].isna()
    if missing_predictions.any():
        fallback = (rows["old_share_prior"] * 2.5).clip(0.0, 1.0)
        for column in ["predicted_old_share_q50", "predicted_old_share_q75"]:
            rows[column] = rows[column].fillna(fallback)

    actual = pd.read_parquet(
        SOURCE_OUTPUT / "economics_actual_state.parquet",
        columns=KEYS + ["lost"],
    ).rename(columns={"lost": "actual_lost"})
    raw_signal = pd.read_parquet(
        SOURCE_OUTPUT / "direct_raw_forecast_signal_rows.parquet",
        columns=KEYS + ["potential_new_lost_case"],
    )
    comparison = actual.merge(raw_signal, on=KEYS, validate="one_to_one")
    actual_case = comparison["actual_lost"].ge(1.0)
    current_downstream_new: pd.Series | None = None
    summaries = []

    for label, (mode, scale) in RULES.items():
        simulation = pd.concat(
            [
                simulate_group(group, direct_plan_column, mode, scale)
                for _, group in rows.groupby(
                    ["bakery_id", "product_id"], sort=False
                )
            ],
            ignore_index=True,
        )
        simulation.to_parquet(OUTPUT / f"economics_{label}.parquet", index=False)
        paired = comparison.merge(
            simulation[KEYS + ["lost"]], on=KEYS, validate="one_to_one"
        )
        delta = paired["actual_lost"] - paired["lost"]
        improved = actual_case & delta.ge(1.0)
        fully_closed = improved & paired["lost"].lt(1.0)
        new_case = ~actual_case & paired["lost"].ge(1.0)
        downstream_new = new_case & ~paired["potential_new_lost_case"]
        if label == "current_full_credit":
            current_downstream_new = downstream_new
            fixed_current_downstream = 0
        else:
            if current_downstream_new is None:
                raise RuntimeError("Current rule must be evaluated first")
            fixed_current_downstream = int(
                (current_downstream_new & ~new_case).sum()
            )
        summaries.append(
            {
                "variant": label,
                "production": float(simulation["production"].sum()),
                "production_delta_vs_current": 0.0,
                "served": float(simulation["served"].sum()),
                "lost": float(simulation["lost"].sum()),
                "writeoff": float(simulation["writeoff"].sum()),
                "gross_profit": float(simulation["gross_profit"].sum()),
                "improved_cases": int(improved.sum()),
                "partial_improvements": int((improved & ~fully_closed).sum()),
                "fully_closed": int(fully_closed.sum()),
                "new_lost_cases": int(new_case.sum()),
                "downstream_created_cases": int(downstream_new.sum()),
                "fixed_current_downstream_cases": fixed_current_downstream,
                "worsened_rows": int(delta.le(-1.0).sum()),
                "net_recovered_units_vs_actual": float(delta.sum()),
            }
        )

    summary = pd.DataFrame(summaries)
    current = summary.iloc[0]
    actual_gp = float(
        pd.read_parquet(
            SOURCE_OUTPUT / "economics_actual_state.parquet",
            columns=["gross_profit"],
        )["gross_profit"].sum()
    )
    for column in ["production", "lost", "writeoff", "gross_profit"]:
        summary[f"{column}_delta_vs_current"] = summary[column] - current[column]
    summary["gross_profit_delta_vs_actual"] = summary["gross_profit"] - actual_gp
    summary.to_csv(OUTPUT / "summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

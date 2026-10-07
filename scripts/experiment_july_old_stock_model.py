"""Causal July gate for learned yesterday-stock credit policies."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_historical_old_stock_credit import load_rows  # noqa: E402


OUTPUT = ROOT / "reports/old_stock_sellability_model_20260912"
KEYS = ["date", "bakery_id", "product_id"]
JULY_START = pd.Timestamp("2026-07-01")
JULY_END = pd.Timestamp("2026-07-31")
RULES = {
    "current_full_credit": ("full", 1.0),
    "sellable_x2_50": ("sellable", 2.5),
    "old_share_model_q50": ("model", 0.50),
    "old_share_model_q50_q75_b25": ("blend", 0.25),
    "old_share_model_q50_q75_b50": ("blend", 0.50),
    "old_share_model_q50_q75_b75": ("blend", 0.75),
    "old_share_model_q75": ("model", 0.75),
}


def simulate_group(
    group: pd.DataFrame,
    credit_mode: str,
    credit_scale: float,
) -> pd.DataFrame:
    carry = 0.0
    previous_date: pd.Timestamp | None = None
    output = []
    for row in group.sort_values("date").itertuples(index=False):
        if previous_date is None or row.date != previous_date + pd.Timedelta(days=1):
            carry = 0.0
        old_opening = carry
        received = float(row.incoming_move_qty)
        sent = float(row.outgoing_move_qty)
        plan = float(row.plan)

        # Before July every candidate follows the incumbent policy. This makes
        # the 1 July opening stock identical and keeps the evaluation causal.
        mode = "full" if row.date < JULY_START else credit_mode
        if mode == "full":
            old_credit = old_opening
        elif mode == "sellable":
            expected_old_sales = plan * float(row.old_share_prior) * credit_scale
            old_credit = min(old_opening, expected_old_sales)
        elif mode == "model":
            column = f"predicted_old_share_q{int(credit_scale * 100)}"
            old_credit = min(old_opening, plan * float(getattr(row, column)))
        elif mode == "blend":
            predicted_share = float(row.predicted_old_share_q50) + credit_scale * (
                float(row.predicted_old_share_q75)
                - float(row.predicted_old_share_q50)
            )
            old_credit = min(old_opening, plan * predicted_share)
        else:
            raise ValueError(f"Unknown credit mode: {mode}")

        production = max(plan + sent - received - old_credit, 0.0)
        old_after_out = max(old_opening - sent, 0.0)
        fresh_after_out = max(
            production + received - max(sent - old_opening, 0.0), 0.0
        )
        demand = float(row.demand)
        sold_old = min(old_after_out, demand * float(row.old_share_prior), demand)
        sold_fresh = min(fresh_after_out, demand - sold_old)
        served = sold_old + sold_fresh
        writeoff = old_after_out - sold_old
        carry = fresh_after_out - sold_fresh
        revenue = (
            sold_fresh * float(row.sale_price)
            + sold_old * float(row.sale_price) * 0.70
        )
        output.append(
            {
                "date": row.date,
                "bakery_id": row.bakery_id,
                "product_id": row.product_id,
                "production": production,
                "served": served,
                "lost": demand - served,
                "writeoff": writeoff,
                "gross_profit": revenue - production * float(row.unit_cost),
            }
        )
        previous_date = row.date
    return pd.DataFrame(output)


def main() -> None:
    rows = load_rows()
    rows = rows[rows["date"].le(JULY_END)].copy()
    predictions = pd.read_parquet(
        OUTPUT / "july_predictions.parquet",
        columns=KEYS + ["quantile_0.50", "quantile_0.75"],
    ).rename(
        columns={
            "quantile_0.50": "predicted_old_share_q50",
            "quantile_0.75": "predicted_old_share_q75",
        }
    )
    rows = rows.merge(predictions, on=KEYS, how="left", validate="one_to_one")
    july = rows["date"].between(JULY_START, JULY_END)
    if rows.loc[july, "predicted_old_share_q50"].isna().any():
        raise RuntimeError("July model predictions are incomplete")

    actual = rows.loc[
        july,
        KEYS
        + [
            "demand",
            "observed_sales_qty",
            "observed_sales_amount",
            "release_qty",
            "unit_cost",
        ],
    ].copy()
    actual["actual_lost"] = (
        actual["demand"] - actual["observed_sales_qty"]
    ).clip(lower=0.0)
    actual["actual_gp"] = (
        actual["observed_sales_amount"]
        - actual["release_qty"] * actual["unit_cost"]
    )
    summaries = []
    simulations: dict[str, pd.DataFrame] = {}
    groups = list(rows.groupby(["bakery_id", "product_id"], sort=False))

    for label, (mode, scale) in RULES.items():
        simulation = pd.concat(
            [simulate_group(group, mode, scale) for _, group in groups],
            ignore_index=True,
        )
        simulation = simulation[simulation["date"].between(JULY_START, JULY_END)]
        simulations[label] = simulation
        paired = actual.merge(simulation, on=KEYS, validate="one_to_one")
        actual_case = paired["actual_lost"].ge(1.0)
        delta = paired["actual_lost"] - paired["lost"]
        improved = actual_case & delta.ge(1.0)
        fully_closed = improved & paired["lost"].lt(1.0)
        new_case = ~actual_case & paired["lost"].ge(1.0)
        summaries.append(
            {
                "variant": label,
                "production": simulation["production"].sum(),
                "served": simulation["served"].sum(),
                "lost": simulation["lost"].sum(),
                "writeoff": simulation["writeoff"].sum(),
                "gross_profit": simulation["gross_profit"].sum(),
                "actual_gp": actual["actual_gp"].sum(),
                "improved_cases": int(improved.sum()),
                "partial_improvements": int((improved & ~fully_closed).sum()),
                "fully_closed": int(fully_closed.sum()),
                "new_lost_cases": int(new_case.sum()),
                "worsened_rows": int(delta.le(-1.0).sum()),
            }
        )

    summary = pd.DataFrame(summaries)
    current = summary.loc[summary["variant"].eq("current_full_credit")].iloc[0]
    for column in ["production", "served", "lost", "writeoff", "gross_profit"]:
        summary[f"{column}_delta_vs_current"] = summary[column] - current[column]
    summary["gross_profit_delta_vs_actual"] = summary["gross_profit"] - summary["actual_gp"]
    summary.to_csv(OUTPUT / "july_downstream_gate.csv", index=False, encoding="utf-8-sig")
    for label, simulation in simulations.items():
        simulation.to_parquet(OUTPUT / f"july_economics_{label}.parquet", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

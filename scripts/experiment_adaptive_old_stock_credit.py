"""Test causal SKU-adaptive old-stock credit policies."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_historical_old_stock_credit import (  # noqa: E402
    KEYS,
    load_rows,
)


OUTPUT = ROOT / "reports/adaptive_old_stock_credit_20260912"
DISCOUNT = 0.30


def add_causal_risk_features(rows: pd.DataFrame) -> pd.DataFrame:
    result = rows.sort_values(["bakery_id", "product_id", "date"]).copy()
    result["under_event"] = (
        result["observed_sales_qty"] - result["plan"]
    ).ge(1.0).astype(float)
    result["positive_gap"] = (
        result["observed_sales_qty"] - result["plan"]
    ).clip(lower=0.0)
    grouped = result.groupby(["bakery_id", "product_id"], sort=False)
    result["under_rate_28"] = grouped["under_event"].transform(
        lambda value: value.shift(1).rolling(28, min_periods=5).mean()
    )
    prior_gap = grouped["positive_gap"].transform(
        lambda value: value.shift(1).rolling(28, min_periods=5).sum()
    )
    prior_plan = grouped["plan"].transform(
        lambda value: value.shift(1).rolling(28, min_periods=5).sum()
    )
    result["positive_gap_ratio_28"] = (
        prior_gap / prior_plan.replace(0.0, np.nan)
    ).clip(0.0, 1.0)
    result["under_rate_28"] = result["under_rate_28"].fillna(0.15)
    result["positive_gap_ratio_28"] = result["positive_gap_ratio_28"].fillna(0.0)
    result["margin_ratio"] = (
        (result["sale_price"] - result["unit_cost"])
        / result["sale_price"].replace(0.0, np.nan)
    ).fillna(0.0).clip(-1.0, 1.0)
    return result.sort_index()


def add_policy_multipliers(rows: pd.DataFrame) -> pd.DataFrame:
    result = rows.copy()
    risk = result["under_rate_28"]
    gap = result["positive_gap_ratio_28"]
    margin = result["margin_ratio"]
    result["credit_fixed_x2_5"] = 2.5
    result["credit_risk_a"] = np.select(
        [risk.ge(0.30), risk.le(0.10)], [1.5, 4.0], default=2.5
    )
    result["credit_risk_b"] = np.select(
        [risk.ge(0.30), risk.le(0.10)], [2.0, 4.0], default=3.0
    )
    result["credit_risk_margin"] = np.select(
        [risk.ge(0.25) & margin.ge(0.50), risk.ge(0.15) & margin.ge(0.40)],
        [1.5, 2.5],
        default=4.0,
    )
    result["credit_continuous"] = (
        4.25 - 3.0 * risk - 1.5 * gap + 1.5 * (0.50 - margin)
    ).clip(1.25, 5.0)
    result["credit_continuous_conservative"] = (
        4.75 - 2.5 * risk - 1.0 * gap + 1.0 * (0.50 - margin)
    ).clip(1.75, 5.5)
    return result


def simulate_group(group: pd.DataFrame, multiplier_column: str) -> pd.DataFrame:
    carry = 0.0
    previous_date: pd.Timestamp | None = None
    output = []
    for row in group.sort_values("date").itertuples(index=False):
        if previous_date is None or row.date != previous_date + pd.Timedelta(days=1):
            carry = 0.0
        old_opening = carry
        plan = float(row.plan)
        multiplier = float(getattr(row, multiplier_column))
        old_credit = min(
            old_opening,
            plan * float(row.old_share_prior) * multiplier,
        )
        received = float(row.incoming_move_qty)
        sent = float(row.outgoing_move_qty)
        production = max(plan + sent - received - old_credit, 0.0)
        old_after_out = max(old_opening - sent, 0.0)
        fresh_after_out = max(
            production + received - max(sent - old_opening, 0.0), 0.0
        )
        demand = float(row.demand)
        sold_old = min(
            old_after_out,
            demand * float(row.old_share_prior),
            demand,
        )
        sold_fresh = min(fresh_after_out, demand - sold_old)
        served = sold_old + sold_fresh
        expired = old_after_out - sold_old
        carry = fresh_after_out - sold_fresh
        revenue = sold_fresh * float(row.sale_price) + sold_old * float(
            row.sale_price
        ) * (1.0 - DISCOUNT)
        output.append(
            {
                "date": row.date,
                "bakery_id": row.bakery_id,
                "product_id": row.product_id,
                "production": production,
                "served": served,
                "lost": demand - served,
                "writeoff": expired,
                "gross_profit": revenue - production * float(row.unit_cost),
            }
        )
        previous_date = row.date
    return pd.DataFrame(output)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = add_policy_multipliers(add_causal_risk_features(load_rows()))
    policy_columns = [column for column in rows if column.startswith("credit_")]
    actual = rows[
        KEYS
        + [
            "demand",
            "observed_sales_qty",
            "observed_sales_amount",
            "release_qty",
            "unit_cost",
        ]
    ].copy()
    actual["actual_lost"] = (
        actual["demand"] - actual["observed_sales_qty"]
    ).clip(lower=0.0)
    actual["actual_gp"] = (
        actual["observed_sales_amount"]
        - actual["release_qty"] * actual["unit_cost"]
    )
    actual["month"] = actual["date"].dt.to_period("M").astype(str)
    groups = list(rows.groupby(["bakery_id", "product_id"], sort=False))
    monthly_parts = []

    for policy in policy_columns:
        simulation = pd.concat(
            [simulate_group(group, policy) for _, group in groups],
            ignore_index=True,
        )
        paired = actual.merge(simulation, on=KEYS, validate="one_to_one")
        delta = paired["actual_lost"] - paired["lost"]
        actual_case = paired["actual_lost"].ge(1.0)
        paired["variant"] = policy.removeprefix("credit_")
        paired["improved_case"] = actual_case & delta.ge(1.0)
        paired["new_lost_case"] = ~actual_case & paired["lost"].ge(1.0)
        monthly_parts.append(
            paired.groupby(["month", "variant"], as_index=False).agg(
                production=("production", "sum"),
                lost=("lost", "sum"),
                writeoff=("writeoff", "sum"),
                gross_profit=("gross_profit", "sum"),
                actual_gp=("actual_gp", "sum"),
                improved_cases=("improved_case", "sum"),
                new_lost_cases=("new_lost_case", "sum"),
            )
        )
        print(f"simulated {policy}", flush=True)

    monthly = pd.concat(monthly_parts, ignore_index=True)
    baseline = monthly[monthly["variant"].eq("fixed_x2_5")][
        ["month", "production", "lost", "writeoff", "gross_profit"]
    ].rename(
        columns={
            column: f"fixed_{column}"
            for column in ["production", "lost", "writeoff", "gross_profit"]
        }
    )
    monthly = monthly.merge(baseline, on="month", validate="many_to_one")
    for column in ["production", "lost", "writeoff", "gross_profit"]:
        monthly[f"{column}_delta_vs_fixed"] = (
            monthly[column] - monthly[f"fixed_{column}"]
        )
    monthly["gp_delta_vs_actual"] = monthly["gross_profit"] - monthly["actual_gp"]
    monthly.to_csv(OUTPUT / "monthly.csv", index=False, encoding="utf-8-sig")

    monthly["split"] = np.select(
        [monthly["month"].le("2026-06"), monthly["month"].eq("2026-07")],
        ["train", "validation"],
        default="test",
    )
    split = monthly.groupby(["variant", "split"], as_index=False).agg(
        gross_profit=("gross_profit", "sum"),
        gp_delta_vs_fixed=("gross_profit_delta_vs_fixed", "sum"),
        production_delta_vs_fixed=("production_delta_vs_fixed", "sum"),
        lost_delta_vs_fixed=("lost_delta_vs_fixed", "sum"),
        writeoff_delta_vs_fixed=("writeoff_delta_vs_fixed", "sum"),
        improved_cases=("improved_cases", "sum"),
        new_lost_cases=("new_lost_cases", "sum"),
    )
    split.to_csv(OUTPUT / "split_summary.csv", index=False, encoding="utf-8-sig")
    print(split.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

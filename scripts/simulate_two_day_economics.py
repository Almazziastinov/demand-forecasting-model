"""Two-day FIFO inventory and relative-profit simulation for forecast plans."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "reports/rolling_actual_state_comparison_20260826/rows.parquet"
OUTPUT = ROOT / "reports/two_day_economics_20260826"
VARIANTS = {
    "actual_state": None,
    "current": "incumbent_sku_forecast",
    "p50_predictive": "p50_predictive",
    "p50_predictive_simple_floor": "p50_simple_floor",
}
COST_RATIOS = [0.20, 0.35, 0.50, 0.65, 0.80]
DISPOSAL_COST = 0.05


def simulate_group(group: pd.DataFrame, plan_column: str | None) -> pd.DataFrame:
    group = group.sort_values("date")
    rows = []
    carry = 0.0
    carry_is_initial = False
    previous_date = None
    segment_id = 0
    segment_day = 0
    for record in group.itertuples(index=False):
        if previous_date is None or record.date != previous_date + pd.Timedelta(days=1):
            carry = max(float(record.opening_stock), 0.0)
            carry_is_initial = True
            segment_id += 1
            segment_day = 0
        segment_day += 1
        received = max(float(record.received), 0.0)
        transfer_out = max(float(record.sent), 0.0)
        target_stock = None if plan_column is None else max(float(getattr(record, plan_column)), 0.0)
        production = (
            max(float(record.produced), 0.0)
            if target_stock is None
            else max(target_stock + transfer_out - carry - received, 0.0)
        )
        fresh = production + received
        from_old_for_transfer = min(carry, transfer_out)
        carry_after_transfer = carry - from_old_for_transfer
        fresh_after_transfer = max(fresh - (transfer_out - from_old_for_transfer), 0.0)
        demand = max(float(record.demand), 0.0)
        sold_old = min(carry_after_transfer, demand)
        remaining_demand = demand - sold_old
        sold_fresh = min(fresh_after_transfer, remaining_demand)
        served = sold_old + sold_fresh
        expired = carry_after_transfer - sold_old
        next_carry = fresh_after_transfer - sold_fresh
        rows.append(
            {
                "date": record.date,
                "bakery_id": record.bakery_id,
                "product_id": record.product_id,
                "demand": demand,
                "production": production,
                "target_stock": production + carry + received - transfer_out,
                "sold_fresh": sold_fresh,
                "sold_yesterday": sold_old,
                "sold_yesterday_initial_stock": sold_old if carry_is_initial else 0.0,
                "sold_yesterday_strategy_stock": 0.0 if carry_is_initial else sold_old,
                "served": served,
                "lost": demand - served,
                "expired": expired,
                "expired_initial_stock": expired if carry_is_initial else 0.0,
                "expired_strategy_stock": 0.0 if carry_is_initial else expired,
                "ending_carry": next_carry,
                "segment_id": segment_id,
                "segment_day": segment_day,
            }
        )
        carry = next_carry
        carry_is_initial = False
        previous_date = record.date
    return pd.DataFrame(rows)


def main() -> None:
    source = pd.read_parquet(ROWS)
    source["date"] = pd.to_datetime(source["date"]).dt.normalize()
    simulation_parts = []
    for variant, plan_column in VARIANTS.items():
        parts = [
            simulate_group(group, plan_column)
            for _, group in source.groupby(["bakery_id", "product_id"], sort=False)
        ]
        simulated = pd.concat(parts, ignore_index=True)
        simulated["variant"] = variant
        simulation_parts.append(simulated)
    simulations = pd.concat(simulation_parts, ignore_index=True)

    operational = simulations.groupby("variant", as_index=False).agg(
        demand=("demand", "sum"),
        production=("production", "sum"),
        served=("served", "sum"),
        lost=("lost", "sum"),
        expired=("expired", "sum"),
        terminal_carry=("ending_carry", "sum"),
    )
    operational["service_level_pct"] = 100 * operational["served"] / operational["demand"]
    operational["sell_through_pct"] = 100 * operational["served"] / operational["production"]

    economics = []
    for cost_ratio in COST_RATIOS:
        for row in operational.itertuples(index=False):
            profit = row.served - cost_ratio * row.production - DISPOSAL_COST * row.expired
            economics.append(
                {
                    "variant": row.variant,
                    "production_cost_ratio": cost_ratio,
                    "sale_price": 1.0,
                    "disposal_cost": DISPOSAL_COST,
                    "revenue": row.served,
                    "production_cost": cost_ratio * row.production,
                    "disposal_cost_total": DISPOSAL_COST * row.expired,
                    "relative_profit": profit,
                }
            )
    economics = pd.DataFrame(economics)
    actual_profit = economics[economics["variant"].eq("actual_state")][
        ["production_cost_ratio", "relative_profit"]
    ].rename(columns={"relative_profit": "actual_profit"})
    economics = economics.merge(actual_profit, on="production_cost_ratio", validate="many_to_one")
    economics["profit_delta_vs_actual"] = economics["relative_profit"] - economics["actual_profit"]
    economics["profit_delta_vs_actual_pct"] = (
        100 * economics["profit_delta_vs_actual"] / economics["actual_profit"].replace(0.0, np.nan)
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    simulations.to_parquet(OUTPUT / "daily_rows.parquet", index=False)
    operational.to_csv(OUTPUT / "operational_summary.csv", index=False)
    economics.to_csv(OUTPUT / "economic_sensitivity.csv", index=False)
    print("Two-day operational simulation")
    print(operational.to_string(index=False))
    print("\nEconomic sensitivity")
    print(economics.to_string(index=False))


if __name__ == "__main__":
    main()

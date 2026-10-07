"""Evaluate a causal bakery gate using ruble gross profit.

The two candidate histories are simulated independently.  The decision for day D
uses only their realized gross-profit difference through D-1.  A final stateful
simulation then evaluates the selected plan, including two-day FIFO carry.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from simulate_two_day_economics import simulate_group  # noqa: E402


SOURCE = Path(
    os.environ.get(
        "HISTORICAL_ECONOMIC_SOURCE",
        ROOT
        / "reports/historical_daily_retraining_gate_loss075_20260910/predictions.parquet",
    )
)
PANEL = Path(
    os.environ.get(
        "HISTORICAL_ECONOMIC_PANEL",
        ROOT / ".codex_tmp/historical_gate_backtest/causal_mature_panel.parquet",
    )
)
MAPPING = ROOT / "reports/markup_price_mapping_20260826/mapped_products.csv"
OUTPUT = Path(
    os.environ.get(
        "HISTORICAL_ECONOMIC_OUTPUT",
        ROOT / "reports/historical_economic_gate_loss075_20260910",
    )
)
DISCOUNT = 0.30
WINDOW = 14


def prepare_source() -> tuple[pd.DataFrame, dict[str, float]]:
    rows = pd.read_parquet(SOURCE)
    rows = rows.drop(
        columns=["use_p50", "gate_prior_advantage", "gate_plan"], errors="ignore"
    )
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    flows = pd.read_parquet(
        PANEL,
        columns=[
            "date",
            "bakery_id",
            "product_id",
            "incoming_move_qty",
            "outgoing_move_qty",
            "release_qty",
        ],
    )
    flows["date"] = pd.to_datetime(flows["date"]).dt.normalize()
    rows = rows.merge(
        flows,
        on=["date", "bakery_id", "product_id"],
        how="left",
        validate="one_to_one",
    )
    mapping = pd.read_csv(MAPPING, encoding="utf-8-sig")
    mapping = mapping[mapping["valid_economics"].astype(bool)].copy()
    mapping["product_id"] = pd.to_numeric(mapping["product_id"], errors="coerce")
    mapping = mapping.dropna(subset=["product_id"])
    mapping["product_id"] = mapping["product_id"].astype(int)
    mapping = mapping.sort_values("unit_price").drop_duplicates(
        "product_id", keep="last"
    )
    total_sales = float(rows["observed_sales_qty"].sum())
    rows = rows.merge(
        mapping[["product_id", "unit_cost", "unit_price"]],
        on="product_id",
        how="left",
        validate="many_to_one",
    )
    covered = rows["unit_cost"].notna()
    coverage = {
        "sku_rows_pct": float(100 * covered.mean()),
        "sales_units_pct": float(
            100 * rows.loc[covered, "observed_sales_qty"].sum() / total_sales
        ),
    }
    rows = rows[covered].copy()
    rows["sale_price"] = rows["avg_sales_price"].where(
        rows["avg_sales_price"].gt(0), rows["unit_price"]
    )
    rows["opening_stock"] = 0.0
    rows["received"] = rows["incoming_move_qty"].fillna(0).clip(lower=0)
    rows["sent"] = rows["outgoing_move_qty"].fillna(0).clip(lower=0)
    rows["produced"] = rows["release_qty"].fillna(0).clip(lower=0)
    return rows, coverage


def simulate(rows: pd.DataFrame, plan_column: str, variant: str) -> pd.DataFrame:
    parts = [
        simulate_group(group, plan_column)
        for _, group in rows.groupby(["bakery_id", "product_id"], sort=False)
    ]
    result = pd.concat(parts, ignore_index=True)
    attributes = rows[["date", "bakery_id", "product_id", "sale_price", "unit_cost"]]
    result = result.merge(
        attributes,
        on=["date", "bakery_id", "product_id"],
        how="left",
        validate="one_to_one",
    )
    result["revenue"] = result["sold_fresh"] * result["sale_price"] + result[
        "sold_yesterday"
    ] * result["sale_price"] * (1 - DISCOUNT)
    result["production_cost"] = result["production"] * result["unit_cost"]
    result["gross_profit"] = result["revenue"] - result["production_cost"]
    result["variant"] = variant
    return result


def actual_checkout(rows: pd.DataFrame) -> pd.DataFrame:
    """Observed baseline: checkout revenue and actual production, without invented stock."""
    result = rows[["date", "bakery_id", "product_id", "demand"]].copy()
    result["production"] = rows["produced"]
    result["served"] = rows["observed_sales_qty"].clip(lower=0)
    result["lost"] = (result["demand"] - result["served"]).clip(lower=0)
    result["expired_strategy_stock"] = 0.0
    result["revenue"] = rows["observed_sales_amount"].clip(lower=0)
    result["production_cost"] = result["production"] * rows["unit_cost"]
    result["gross_profit"] = result["revenue"] - result["production_cost"]
    result["variant"] = "actual_state"
    return result


def make_gate(direct: pd.DataFrame, p50: pd.DataFrame) -> pd.DataFrame:
    keys = ["date", "bakery_id"]
    left = (
        direct.groupby(keys, as_index=False)["gross_profit"]
        .sum()
        .rename(columns={"gross_profit": "direct_gp"})
    )
    right = (
        p50.groupby(keys, as_index=False)["gross_profit"]
        .sum()
        .rename(columns={"gross_profit": "p50_gp"})
    )
    gate = left.merge(right, on=keys, validate="one_to_one")
    gate["p50_advantage"] = gate["p50_gp"] - gate["direct_gp"]
    gate = gate.sort_values(["bakery_id", "date"])
    gate["prior_advantage_14"] = gate.groupby("bakery_id")["p50_advantage"].transform(
        lambda values: values.shift(1).rolling(WINDOW, min_periods=1).mean()
    )
    gate["use_p50"] = gate["prior_advantage_14"].gt(0)
    return gate


def summarize(simulations: pd.DataFrame) -> pd.DataFrame:
    result = simulations.groupby(
        [simulations["date"].dt.to_period("M").astype(str), "variant"], as_index=False
    ).agg(
        demand=("demand", "sum"),
        production=("production", "sum"),
        served=("served", "sum"),
        lost=("lost", "sum"),
        expired=("expired_strategy_stock", "sum"),
        revenue=("revenue", "sum"),
        production_cost=("production_cost", "sum"),
        gross_profit=("gross_profit", "sum"),
    )
    result = result.rename(columns={"date": "period"})
    result["service_pct"] = 100 * result["served"] / result["demand"]
    direct = result[result["variant"].eq("direct")][["period", "gross_profit"]].rename(
        columns={"gross_profit": "direct_gp"}
    )
    result = result.merge(direct, on="period", validate="many_to_one")
    result["gp_delta_vs_direct"] = result["gross_profit"] - result["direct_gp"]
    return result


def main() -> None:
    rows, coverage = prepare_source()
    direct = simulate(rows, "direct_plan", "direct")
    p50 = simulate(rows, "p50_plan", "p50_loss075")
    actual = actual_checkout(rows)
    decisions = make_gate(direct, p50)
    rows = rows.merge(
        decisions[["date", "bakery_id", "use_p50", "prior_advantage_14"]],
        on=["date", "bakery_id"],
        validate="many_to_one",
    )
    rows["economic_gate_plan"] = np.where(
        rows["use_p50"], rows["p50_plan"], rows["direct_plan"]
    )
    gate = simulate(rows, "economic_gate_plan", "economic_gate")
    simulations = pd.concat([actual, direct, p50, gate], ignore_index=True)
    monthly = summarize(simulations)
    total = simulations.groupby("variant", as_index=False).agg(
        demand=("demand", "sum"),
        production=("production", "sum"),
        served=("served", "sum"),
        lost=("lost", "sum"),
        expired=("expired_strategy_stock", "sum"),
        revenue=("revenue", "sum"),
        production_cost=("production_cost", "sum"),
        gross_profit=("gross_profit", "sum"),
    )
    direct_gp = float(total.loc[total["variant"].eq("direct"), "gross_profit"].iloc[0])
    actual_gp = float(
        total.loc[total["variant"].eq("actual_state"), "gross_profit"].iloc[0]
    )
    total["gp_delta_vs_direct"] = total["gross_profit"] - direct_gp
    total["gp_delta_vs_direct_pct"] = 100 * total["gp_delta_vs_direct"] / direct_gp
    total["gp_delta_vs_actual"] = total["gross_profit"] - actual_gp
    total["gp_delta_vs_actual_pct"] = 100 * total["gp_delta_vs_actual"] / actual_gp
    total["service_pct"] = 100 * total["served"] / total["demand"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(OUTPUT / "gate_decisions.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(OUTPUT / "monthly_summary.csv", index=False, encoding="utf-8-sig")
    total.to_csv(OUTPUT / "total_summary.csv", index=False, encoding="utf-8-sig")
    gate.to_parquet(OUTPUT / "gate_daily_rows.parquet", index=False)
    metadata = {
        "production_write": False,
        "decision_cutoff": "D-1",
        "window_days": WINDOW,
        "yesterday_product_discount": DISCOUNT,
        "coverage": coverage,
        "initial_opening_stock": 0.0,
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print("\nTOTAL")
    print(total.to_string(index=False))
    print("\nMONTHLY")
    print(monthly.to_string(index=False))


if __name__ == "__main__":
    main()

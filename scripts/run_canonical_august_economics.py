"""Canonical FIFO economics for the fixed August 39-bakery/93-SKU scope.

The factual ledger is reconstructed continuously before August. Any quantity
needed to reconcile observed checkout sales with recorded stock flows is made
explicit and then supplied identically to every simulated strategy.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/historical_daily_retraining_gate_loss075_full_20260910/predictions.parquet"
PANEL = ROOT / ".codex_tmp/historical_clickhouse_panel/causal_panel.parquet"
TRUSTED = ROOT / "reports/prod_direct_end_of_day_economics_august_folds_20260910/daily_rows.parquet"
TOTALS = ROOT / "reports/network_trained_bakery_p50_loss075_full_20260910/bakery_totals.parquet"
MAPPING = ROOT / "reports/markup_price_mapping_20260826/mapped_products.csv"
OUTPUT = Path(
    os.environ.get(
        "CANONICAL_AUGUST_OUTPUT",
        ROOT / "reports/canonical_august_economics_20260911",
    )
)
DYNAMIC_SCOPE = ROOT / "reports/comparable_produced_scope_p50_loss075_20260911/predictions.parquet"
DISCOUNT = 0.30
KEYS = ["date", "bakery_id", "product_id"]
DIAGNOSTIC_PRODUCTS = {1071, 10667, 4423, 108, 10346, 10760, 4629, 57, 4424, 10347}


def simulate(group: pd.DataFrame, plan: str | None) -> pd.DataFrame:
    carry = 0.0
    output = []
    for row in group.sort_values("date").itertuples(index=False):
        received = row.incoming_move_qty + row.reconciliation_in
        sent = row.outgoing_move_qty
        written = row.written_off_qty
        if plan is None:
            production = row.release_qty
            sale_demand = row.observed_sales_qty
            reported_demand = row.demand
        else:
            target = max(float(getattr(row, plan)), 0.0)
            production = max(target + sent + written - carry - received, 0.0)
            sale_demand = row.demand
            reported_demand = row.demand
        old_for_sent = min(carry, sent)
        old = carry - old_for_sent
        fresh = max(production + received - (sent - old_for_sent), 0.0)
        sold_old = min(old, sale_demand)
        sold_fresh = min(fresh, sale_demand - sold_old)
        old_after_sale = old - sold_old
        fresh_after_sale = fresh - sold_fresh
        write_old = min(old_after_sale, written)
        write_fresh = min(fresh_after_sale, written - write_old)
        expired = old_after_sale - write_old
        carry = fresh_after_sale - write_fresh
        output.append(
            {
                "date": row.date,
                "bakery_id": row.bakery_id,
                "product_id": row.product_id,
                "demand": reported_demand,
                "production": production,
                "served": sold_old + sold_fresh,
                "sold_old": sold_old,
                "sold_fresh": sold_fresh,
                "lost": reported_demand - sold_old - sold_fresh,
                "expired": expired,
                "ending_carry": carry,
                "reconciliation_in": row.reconciliation_in,
                "sale_price": row.sale_price,
                "unit_cost": row.unit_cost,
            }
        )
    return pd.DataFrame(output)


def summarize(rows: pd.DataFrame, label: str, dates: set[pd.Timestamp]) -> dict[str, float | str]:
    part = rows[rows["date"].isin(dates)].copy()
    part["revenue"] = part["sold_fresh"] * part["sale_price"] + part["sold_old"] * part["sale_price"] * (1 - DISCOUNT)
    part["production_cost"] = part["production"] * part["unit_cost"]
    return {
        "variant": label,
        "demand": part["demand"].sum(),
        "production": part["production"].sum(),
        "served": part["served"].sum(),
        "lost": part["lost"].sum(),
        "expired": part["expired"].sum(),
        "reconciliation_in": part["reconciliation_in"].sum(),
        "revenue": part["revenue"].sum(),
        "production_cost": part["production_cost"].sum(),
        "gross_profit": (part["revenue"] - part["production_cost"]).sum(),
    }


def add_economics(rows: pd.DataFrame) -> pd.DataFrame:
    result = rows.copy()
    result["revenue"] = result["sold_fresh"] * result["sale_price"] + result[
        "sold_old"
    ] * result["sale_price"] * (1 - DISCOUNT)
    result["production_cost"] = result["production"] * result["unit_cost"]
    result["gross_profit"] = result["revenue"] - result["production_cost"]
    return result


def main() -> None:
    trusted = pd.read_parquet(TRUSTED, columns=["date", "bakery_id", "product_id"])
    trusted["date"] = pd.to_datetime(trusted["date"]).dt.normalize()
    bakery_ids = set(trusted["bakery_id"].unique())
    product_ids = set(trusted["product_id"].unique())
    trusted_dates = set(trusted["date"].unique())

    if os.environ.get("CANONICAL_ALL_BAKERIES") == "1":
        total_scope = pd.read_parquet(TOTALS, columns=["date", "bakery_id"])
        total_scope["date"] = pd.to_datetime(total_scope["date"]).dt.normalize()
        bakery_ids = set(
            total_scope.loc[
                total_scope["date"].between("2026-08-01", "2026-08-31"),
                "bakery_id",
            ].unique()
        )

    rows = pd.read_parquet(SOURCE)
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    rows = rows[rows["bakery_id"].isin(bakery_ids)].copy()
    if os.environ.get("CANONICAL_DYNAMIC_PRODUCTS") == "1":
        dynamic_keys = pd.read_parquet(DYNAMIC_SCOPE, columns=KEYS).drop_duplicates(KEYS)
        dynamic_keys["date"] = pd.to_datetime(dynamic_keys["date"]).dt.normalize()
        rows = rows.merge(dynamic_keys.assign(dynamic_scope=True), on=KEYS, how="inner")
    else:
        rows = rows[rows["product_id"].isin(product_ids)].copy()
    flows = pd.read_parquet(PANEL, columns=KEYS + ["release_qty", "incoming_move_qty", "outgoing_move_qty", "written_off_qty"])
    flows["date"] = pd.to_datetime(flows["date"]).dt.normalize()
    rows = rows.drop(columns=["p50_total", "p50_plan"], errors="ignore").merge(flows, on=KEYS, how="left", validate="one_to_one")
    for column in ["release_qty", "incoming_move_qty", "outgoing_move_qty", "written_off_qty"]:
        rows[column] = pd.to_numeric(rows[column], errors="coerce").fillna(0).clip(lower=0)
    mapping = pd.read_csv(MAPPING, encoding="utf-8-sig")
    mapping = mapping[mapping["valid_economics"].astype(bool)].sort_values("unit_price").drop_duplicates("product_id", keep="last")
    rows = rows.merge(mapping[["product_id", "unit_cost", "unit_price"]], on="product_id", how="inner", validate="many_to_one")
    rows["sale_price"] = rows["avg_sales_price"].where(rows["avg_sales_price"].gt(0), rows["unit_price"])
    totals = pd.read_parquet(TOTALS)
    totals["date"] = pd.to_datetime(totals["date"]).dt.normalize()
    rows = rows.merge(totals, on=["date", "bakery_id"], how="left", validate="many_to_one")
    scope_sum = rows.groupby(["date", "bakery_id"])["share"].transform("sum")
    rows["scope_share"] = rows["share"] / scope_sum.where(scope_sum.gt(0))
    rows["direct_plan_canonical"] = rows["direct_total"] * rows["scope_share"]
    rows["p50_plan_canonical"] = rows["p50_total_exact"] * rows["scope_share"]
    for name, cap in (("p50_cap_105", 1.05), ("p50_cap_110", 1.10), ("p50_cap_115", 1.15)):
        rows[name] = np.minimum(
            rows["p50_plan_canonical"], rows["direct_plan_canonical"] * cap
        )

    # Continuous factual ledger exposes rather than hides missing opening stock.
    reconciled = []
    for _, group in rows.groupby(["bakery_id", "product_id"], sort=False):
        carry = 0.0
        for idx, row in group.sort_values("date").iterrows():
            required = row["observed_sales_qty"] + row["outgoing_move_qty"] + row["written_off_qty"]
            known = carry + row["release_qty"] + row["incoming_move_qty"]
            adjustment = max(required - known, 0.0)
            reconciled.append((idx, adjustment))
            old = carry
            fresh = row["release_qty"] + row["incoming_move_qty"] + adjustment
            for quantity in [row["outgoing_move_qty"], row["observed_sales_qty"], row["written_off_qty"]]:
                from_old = min(old, quantity)
                old -= from_old
                fresh = max(fresh - (quantity - from_old), 0.0)
            # Remaining old stock expires; only today's fresh stock carries.
            carry = fresh
    reconciliation = pd.Series(dict(reconciled))
    rows["reconciliation_in"] = rows.index.to_series().map(reconciliation).fillna(0)

    simulations = {}
    variants = [
        ("actual_state", None),
        ("direct", "direct_plan_canonical"),
        ("p50_loss075", "p50_plan_canonical"),
        ("p50_cap_105", "p50_cap_105"),
        ("p50_cap_110", "p50_cap_110"),
        ("p50_cap_115", "p50_cap_115"),
    ]
    for label, plan in variants:
        simulations[label] = pd.concat(
            [simulate(group, plan) for _, group in rows.groupby(["bakery_id", "product_id"], sort=False)],
            ignore_index=True,
        )
    direct_daily = add_economics(simulations["direct"]).groupby(
        ["date", "bakery_id"], as_index=False
    )["gross_profit"].sum().rename(columns={"gross_profit": "direct_gp"})
    p50_daily = add_economics(simulations["p50_loss075"]).groupby(
        ["date", "bakery_id"], as_index=False
    )["gross_profit"].sum().rename(columns={"gross_profit": "p50_gp"})
    decisions = direct_daily.merge(p50_daily, on=["date", "bakery_id"], validate="one_to_one")
    decisions["advantage"] = decisions["p50_gp"] - decisions["direct_gp"]
    decisions = decisions.sort_values(["bakery_id", "date"])
    decisions["prior_advantage_14"] = decisions.groupby("bakery_id")["advantage"].transform(
        lambda value: value.shift(1).rolling(14, min_periods=1).mean()
    )
    decisions["use_p50"] = decisions["prior_advantage_14"].gt(0)
    rows = rows.drop(columns=["use_p50"], errors="ignore").merge(
        decisions[["date", "bakery_id", "use_p50"]],
        on=["date", "bakery_id"],
        validate="many_to_one",
    )
    rows["causal_gate_plan"] = np.where(rows["use_p50"], rows["p50_plan_canonical"], rows["direct_plan_canonical"])
    simulations["causal_gate"] = pd.concat(
        [simulate(group, "causal_gate_plan") for _, group in rows.groupby(["bakery_id", "product_id"], sort=False)],
        ignore_index=True,
    )
    attributes = rows[KEYS + ["category_name"]]
    direct_group_daily = add_economics(simulations["direct"]).merge(
        attributes, on=KEYS, validate="one_to_one"
    ).groupby(["date", "bakery_id", "category_name"], as_index=False)["gross_profit"].sum().rename(columns={"gross_profit": "direct_gp"})
    p50_group_daily = add_economics(simulations["p50_loss075"]).merge(
        attributes, on=KEYS, validate="one_to_one"
    ).groupby(["date", "bakery_id", "category_name"], as_index=False)["gross_profit"].sum().rename(columns={"gross_profit": "p50_gp"})
    group_decisions = direct_group_daily.merge(
        p50_group_daily, on=["date", "bakery_id", "category_name"], validate="one_to_one"
    )
    group_decisions["advantage"] = group_decisions["p50_gp"] - group_decisions["direct_gp"]
    group_decisions = group_decisions.sort_values(["bakery_id", "category_name", "date"])
    group_decisions["prior_advantage_14"] = group_decisions.groupby(
        ["bakery_id", "category_name"]
    )["advantage"].transform(lambda value: value.shift(1).rolling(14, min_periods=1).mean())
    group_decisions["use_p50"] = group_decisions["prior_advantage_14"].gt(0)
    rows = rows.merge(
        group_decisions[["date", "bakery_id", "category_name", "use_p50"]].rename(
            columns={"use_p50": "use_p50_group"}
        ),
        on=["date", "bakery_id", "category_name"],
        how="left",
        validate="many_to_one",
    )
    rows["causal_group_gate_plan"] = np.where(
        rows["use_p50_group"].fillna(False),
        rows["p50_plan_canonical"],
        rows["direct_plan_canonical"],
    )
    simulations["causal_group_gate"] = pd.concat(
        [simulate(group, "causal_group_gate_plan") for _, group in rows.groupby(["bakery_id", "product_id"], sort=False)],
        ignore_index=True,
    )
    august_dates = set(pd.date_range("2026-08-01", "2026-08-31"))
    records = []
    for period, dates in [("trusted_10_dates", trusted_dates), ("full_august", august_dates)]:
        for label, simulation in simulations.items():
            record = summarize(simulation, label, dates)
            record["period"] = period
            records.append(record)
    summary = pd.DataFrame(records)
    actual_gp = summary[summary["variant"].eq("actual_state")][["period", "gross_profit"]].rename(columns={"gross_profit": "actual_gp"})
    summary = summary.merge(actual_gp, on="period", validate="many_to_one")
    summary["gp_delta_vs_actual"] = summary["gross_profit"] - summary["actual_gp"]
    summary["gp_delta_vs_actual_pct"] = 100 * summary["gp_delta_vs_actual"] / summary["actual_gp"]
    summary["service_pct"] = 100 * summary["served"] / summary["demand"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT / "summary.csv", index=False, encoding="utf-8-sig")
    monthly_records = []
    for month, month_rows in rows.groupby(rows["date"].dt.to_period("M")):
        month_dates = set(month_rows["date"].unique())
        for label, simulation in simulations.items():
            record = summarize(simulation, label, month_dates)
            record["period"] = str(month)
            monthly_records.append(record)
    monthly = pd.DataFrame(monthly_records)
    actual_monthly = monthly[monthly["variant"].eq("actual_state")][["period", "gross_profit"]].rename(columns={"gross_profit": "actual_gp"})
    monthly = monthly.merge(actual_monthly, on="period", validate="many_to_one")
    monthly["gp_delta_vs_actual_pct"] = 100 * (monthly["gross_profit"] - monthly["actual_gp"]) / monthly["actual_gp"]
    monthly["service_pct"] = 100 * monthly["served"] / monthly["demand"]
    monthly.to_csv(OUTPUT / "monthly_summary.csv", index=False, encoding="utf-8-sig")
    decisions.to_csv(OUTPUT / "gate_decisions.csv", index=False, encoding="utf-8-sig")
    group_decisions.to_csv(OUTPUT / "group_gate_decisions.csv", index=False, encoding="utf-8-sig")
    sku_monthly_parts = []
    for label, simulation in simulations.items():
        enriched = add_economics(simulation)
        enriched["period"] = enriched["date"].dt.to_period("M").astype(str)
        part = enriched.groupby(["period", "product_id"], as_index=False).agg(
            demand=("demand", "sum"),
            production=("production", "sum"),
            sold_fresh=("sold_fresh", "sum"),
            sold_old=("sold_old", "sum"),
            served=("served", "sum"),
            expired=("expired", "sum"),
            revenue=("revenue", "sum"),
            production_cost=("production_cost", "sum"),
            gross_profit=("gross_profit", "sum"),
        )
        part["variant"] = label
        sku_monthly_parts.append(part)
    pd.concat(sku_monthly_parts, ignore_index=True).to_parquet(
        OUTPUT / "sku_monthly.parquet", index=False
    )
    diagnostic_parts = []
    for label, simulation in simulations.items():
        part = add_economics(
            simulation[simulation["product_id"].isin(DIAGNOSTIC_PRODUCTS)]
        )
        part["variant"] = label
        diagnostic_parts.append(part)
    pd.concat(diagnostic_parts, ignore_index=True).to_parquet(
        OUTPUT / "diagnostic_daily.parquet", index=False
    )
    rows[KEYS + ["reconciliation_in"]].to_parquet(OUTPUT / "reconciliation.parquet", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

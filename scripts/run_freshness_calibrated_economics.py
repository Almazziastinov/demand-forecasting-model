"""Causal economics using recorded checkout freshness and calibrated old-stock sales."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "reports/historical_daily_retraining_gate_loss075_full_20260910"
    / "predictions.parquet"
)
PANEL = ROOT / ".codex_tmp/historical_clickhouse_panel/causal_panel.parquet"
FRESHNESS = ROOT / ".codex_tmp/pilot_freshness_history_20260911.parquet"
TRUSTED = (
    ROOT
    / "reports/prod_direct_end_of_day_economics_august_folds_20260910"
    / "daily_rows.parquet"
)
TOTALS = (
    ROOT
    / "reports/network_trained_bakery_p50_loss075_full_20260910/bakery_totals.parquet"
)
MAPPING = ROOT / "reports/markup_price_mapping_20260826/mapped_products.csv"
OUTPUT = ROOT / "reports/freshness_endogenous_writeoff_economics_20260911"
KEYS = ["date", "bakery_id", "product_id"]


def freshness_kind(value: str) -> str:
    value = str(value)
    if value.startswith("Вч"):
        return "old"
    if value.startswith("Св"):
        return "fresh"
    return "unknown"


def simulate_group(group: pd.DataFrame, plan_column: str) -> pd.DataFrame:
    carry = 0.0
    output = []
    for row in group.sort_values("date").itertuples(index=False):
        # The source stock ledger is incomplete (most visibly for opening stock).
        # Supply the same factual reconciliation to every counterfactual so that
        # a strategy is judged on its plan, not on differences in data coverage.
        old_opening = max(float(carry + row.old_reconciliation_in), 0.0)
        received_fresh = max(
            float(row.incoming_move_qty + row.fresh_reconciliation_in), 0.0
        )
        sent = max(float(row.outgoing_move_qty), 0.0)
        # Factual write-offs are endogenous to the factual production policy.
        # A counterfactual strategy must generate its own write-offs from stock
        # ageing rather than inherit the factual quantity.
        written = 0.0
        plan = max(float(getattr(row, plan_column)), 0.0)
        production = max(plan + sent + written - received_fresh - old_opening, 0.0)
        old_after_out = max(old_opening - sent, 0.0)
        fresh_after_out = max(
            production + received_fresh - max(sent - old_opening, 0.0), 0.0
        )
        demand = max(float(row.demand), 0.0)
        # Historical freshness share represents the merchandising/customer
        # capacity for yesterday stock, rather than assuming old-first FIFO.
        old_target = demand * row.old_share_prior
        sold_old = min(old_after_out, old_target, demand)
        sold_fresh = min(fresh_after_out, demand - sold_old)
        old_remaining = old_after_out - sold_old
        fresh_remaining = fresh_after_out - sold_fresh
        write_old = min(old_remaining, written)
        write_fresh = min(fresh_remaining, written - write_old)
        expired = old_remaining - write_old
        carry = fresh_remaining - write_fresh
        output.append(
            {
                "date": row.date,
                "bakery_id": row.bakery_id,
                "product_id": row.product_id,
                "demand": demand,
                "production": production,
                "served": sold_old + sold_fresh,
                "sold_fresh": sold_fresh,
                "sold_old": sold_old,
                "lost": demand - sold_old - sold_fresh,
                "expired": expired,
                "writeoff": expired,
                "ending_fresh_carry": carry,
                "fresh_price": row.fresh_price_prior,
                "old_price": row.old_price_prior,
                "unit_cost": row.unit_cost,
                "reconciliation_in": row.reconciliation_in,
                "old_reconciliation_in": row.old_reconciliation_in,
                "fresh_reconciliation_in": row.fresh_reconciliation_in,
            }
        )
    result = pd.DataFrame(output)
    result["revenue"] = (
        result["sold_fresh"] * result["fresh_price"]
        + result["sold_old"] * result["old_price"]
    )
    result["production_cost"] = result["production"] * result["unit_cost"]
    result["gross_profit"] = result["revenue"] - result["production_cost"]
    return result


def main() -> None:
    trusted = pd.read_parquet(TRUSTED, columns=["product_id"])
    product_ids = set(trusted["product_id"].astype(int).unique())
    totals = pd.read_parquet(TOTALS)
    totals["date"] = pd.to_datetime(totals["date"]).dt.normalize()
    bakery_ids = set(totals.loc[totals["date"].dt.month.eq(8), "bakery_id"].unique())
    rows = pd.read_parquet(SOURCE)
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    rows = rows[
        rows["bakery_id"].isin(bakery_ids) & rows["product_id"].isin(product_ids)
    ].copy()
    flows = pd.read_parquet(
        PANEL,
        columns=KEYS
        + ["release_qty", "incoming_move_qty", "outgoing_move_qty", "written_off_qty"],
    )
    flows["date"] = pd.to_datetime(flows["date"]).dt.normalize()
    rows = rows.merge(flows, on=KEYS, how="left", validate="one_to_one")
    for column in [
        "release_qty",
        "incoming_move_qty",
        "outgoing_move_qty",
        "written_off_qty",
    ]:
        rows[column] = (
            pd.to_numeric(rows[column], errors="coerce").fillna(0).clip(lower=0)
        )
    mapping = pd.read_csv(MAPPING, encoding="utf-8-sig")
    mapping = (
        mapping[mapping["valid_economics"].astype(bool)]
        .sort_values("unit_price")
        .drop_duplicates("product_id", keep="last")
    )
    rows = rows.merge(
        mapping[["product_id", "unit_cost", "unit_price"]],
        on="product_id",
        how="inner",
        validate="many_to_one",
    )
    rows = rows.merge(
        totals, on=["date", "bakery_id"], how="left", validate="many_to_one"
    )
    share_sum = rows.groupby(["date", "bakery_id"])["share"].transform("sum")
    rows["scope_share"] = rows["share"] / share_sum.where(share_sum.gt(0))
    rows["direct_plan_calibrated"] = rows["direct_total"] * rows["scope_share"]
    rows["p50_plan_calibrated"] = rows["p50_total_exact"] * rows["scope_share"]

    freshness = pd.read_parquet(FRESHNESS).rename(columns={"check_date": "date"})
    freshness["date"] = pd.to_datetime(freshness["date"]).dt.normalize()
    freshness["kind"] = freshness["freshness"].map(freshness_kind)
    daily_quantity = freshness.pivot_table(
        index=KEYS, columns="kind", values="quantity", aggfunc="sum", fill_value=0
    ).reset_index()
    daily_quantity = daily_quantity.rename(
        columns={kind: f"recorded_{kind}_qty" for kind in ("fresh", "old", "unknown")}
    )
    for column in ["recorded_fresh_qty", "recorded_old_qty", "recorded_unknown_qty"]:
        if column not in daily_quantity:
            daily_quantity[column] = 0.0

    priced_freshness = freshness[freshness["kind"].isin(["fresh", "old"])].copy()
    daily = priced_freshness.pivot_table(
        index=KEYS,
        columns="kind",
        values=["quantity", "amount"],
        aggfunc="sum",
        fill_value=0,
    )
    daily.columns = [f"{metric}_{kind}" for metric, kind in daily.columns]
    daily = daily.reset_index().sort_values(["product_id", "date"])
    product_day = daily.groupby(["date", "product_id"], as_index=False).agg(
        fresh_qty=("quantity_fresh", "sum"),
        old_qty=("quantity_old", "sum"),
        fresh_amount=("amount_fresh", "sum"),
        old_amount=("amount_old", "sum"),
    )
    grid = pd.MultiIndex.from_product(
        [pd.date_range(rows["date"].min(), rows["date"].max()), sorted(product_ids)],
        names=["date", "product_id"],
    ).to_frame(index=False)
    product_day = grid.merge(product_day, on=["date", "product_id"], how="left")
    for column in ["fresh_qty", "old_qty", "fresh_amount", "old_amount"]:
        product_day[column] = product_day[column].fillna(0)
    product_day = product_day.sort_values(["product_id", "date"])
    for column in ["fresh_qty", "old_qty", "fresh_amount", "old_amount"]:
        product_day[f"prior_{column}"] = product_day.groupby("product_id")[
            column
        ].transform(lambda value: value.shift(1).rolling(56, min_periods=7).sum())
    denom = product_day["prior_fresh_qty"] + product_day["prior_old_qty"]
    product_day["old_share_prior"] = (product_day["prior_old_qty"] / denom).clip(0, 0.5)
    product_day["fresh_price_prior"] = (
        product_day["prior_fresh_amount"] / product_day["prior_fresh_qty"]
    )
    product_day["old_price_prior"] = (
        product_day["prior_old_amount"] / product_day["prior_old_qty"]
    )
    rates = product_day[
        [
            "date",
            "product_id",
            "old_share_prior",
            "fresh_price_prior",
            "old_price_prior",
        ]
    ]
    rows = rows.merge(
        rates, on=["date", "product_id"], how="left", validate="many_to_one"
    )
    rows["old_share_prior"] = rows["old_share_prior"].fillna(0.05).clip(0, 0.5)
    rows["fresh_price_prior"] = rows["fresh_price_prior"].fillna(rows["unit_price"])
    rows["old_price_prior"] = rows["old_price_prior"].fillna(
        rows["fresh_price_prior"] * 0.7
    )

    rows = rows.merge(daily_quantity, on=KEYS, how="left", validate="one_to_one")
    recorded_columns = [
        "recorded_fresh_qty",
        "recorded_old_qty",
        "recorded_unknown_qty",
    ]
    rows[recorded_columns] = rows[recorded_columns].fillna(0.0).clip(lower=0.0)
    rows["recorded_freshness_qty"] = rows[recorded_columns].sum(axis=1)
    rows["has_recorded_freshness"] = rows["recorded_freshness_qty"].gt(0)
    scale = rows["observed_sales_qty"] / rows["recorded_freshness_qty"].replace(
        0.0, np.nan
    )
    recorded_old = rows["recorded_old_qty"] * scale
    # An unspecified label is not evidence of yesterday stock. Treat it as
    # fresh for inventory purposes; this convention reproduces the independent
    # March closing-stock report far better than old-first FIFO.
    recorded_fresh = (rows["recorded_fresh_qty"] + rows["recorded_unknown_qty"]) * scale
    estimated_old = rows["observed_sales_qty"] * rows["old_share_prior"]
    rows["actual_sold_old"] = np.where(
        rows["has_recorded_freshness"], recorded_old, estimated_old
    )
    rows["actual_sold_old"] = rows["actual_sold_old"].clip(
        lower=0.0, upper=rows["observed_sales_qty"]
    )
    rows["actual_sold_fresh"] = np.where(
        rows["has_recorded_freshness"],
        recorded_fresh,
        rows["observed_sales_qty"] - estimated_old,
    )
    rows["actual_sold_fresh"] = (
        rows["observed_sales_qty"] - rows["actual_sold_old"]
    ).clip(lower=0.0)

    # Reconstruct missing inbound/opening stock once from the factual ledger.
    # This is deliberately computed before running any strategy and then held
    # fixed across Direct and p50.
    reconciled: list[tuple[int, float, float]] = []
    for _, group in rows.groupby(["bakery_id", "product_id"], sort=False):
        carry = 0.0
        for idx, row in group.sort_values("date").iterrows():
            old = carry
            old_reconciliation = max(float(row["actual_sold_old"] - old), 0.0)
            old += old_reconciliation
            old -= min(old, row["actual_sold_old"])

            other_out = row["outgoing_move_qty"] + row["written_off_qty"]
            from_old = min(old, other_out)
            fresh_required = row["actual_sold_fresh"] + other_out - from_old
            fresh_known = row["release_qty"] + row["incoming_move_qty"]
            fresh_reconciliation = max(float(fresh_required - fresh_known), 0.0)
            carry = max(float(fresh_known + fresh_reconciliation - fresh_required), 0.0)
            reconciled.append((idx, old_reconciliation, fresh_reconciliation))
    reconciliation = pd.DataFrame(
        reconciled,
        columns=["row_index", "old_reconciliation_in", "fresh_reconciliation_in"],
    ).set_index("row_index")
    rows = rows.join(reconciliation, how="left")
    rows[["old_reconciliation_in", "fresh_reconciliation_in"]] = rows[
        ["old_reconciliation_in", "fresh_reconciliation_in"]
    ].fillna(0.0)
    rows["reconciliation_in"] = (
        rows["old_reconciliation_in"] + rows["fresh_reconciliation_in"]
    )

    simulations = {
        "direct": pd.concat(
            [
                simulate_group(g, "direct_plan_calibrated")
                for _, g in rows.groupby(["bakery_id", "product_id"], sort=False)
            ],
            ignore_index=True,
        ),
        "p50_loss075": pd.concat(
            [
                simulate_group(g, "p50_plan_calibrated")
                for _, g in rows.groupby(["bakery_id", "product_id"], sort=False)
            ],
            ignore_index=True,
        ),
    }
    direct_daily = (
        simulations["direct"]
        .groupby(["date", "bakery_id"], as_index=False)["gross_profit"]
        .sum()
        .rename(columns={"gross_profit": "direct_gp"})
    )
    p50_daily = (
        simulations["p50_loss075"]
        .groupby(["date", "bakery_id"], as_index=False)["gross_profit"]
        .sum()
        .rename(columns={"gross_profit": "p50_gp"})
    )
    decisions = direct_daily.merge(
        p50_daily, on=["date", "bakery_id"], validate="one_to_one"
    )
    decisions["advantage"] = decisions["p50_gp"] - decisions["direct_gp"]
    decisions = decisions.sort_values(["bakery_id", "date"])
    decisions["prior_advantage_14"] = decisions.groupby("bakery_id")[
        "advantage"
    ].transform(lambda value: value.shift(1).rolling(14, min_periods=1).mean())
    decisions["use_p50"] = decisions["prior_advantage_14"].gt(0)
    rows = rows.drop(columns=["use_p50"], errors="ignore").merge(
        decisions[["date", "bakery_id", "use_p50"]],
        on=["date", "bakery_id"],
        validate="many_to_one",
    )
    rows["gate_plan_calibrated"] = np.where(
        rows["use_p50"], rows["p50_plan_calibrated"], rows["direct_plan_calibrated"]
    )
    simulations["causal_gate"] = pd.concat(
        [
            simulate_group(g, "gate_plan_calibrated")
            for _, g in rows.groupby(["bakery_id", "product_id"], sort=False)
        ],
        ignore_index=True,
    )
    # Actual economics uses recorded checkout revenue/freshness, never inferred FIFO.
    actual = rows[
        KEYS
        + [
            "demand",
            "release_qty",
            "written_off_qty",
            "observed_sales_qty",
            "observed_sales_amount",
            "unit_cost",
        ]
    ].copy()
    actual["production"] = actual["release_qty"]
    actual["served"] = actual["observed_sales_qty"]
    actual["lost"] = (actual["demand"] - actual["served"]).clip(lower=0)
    actual["writeoff"] = actual["written_off_qty"]
    actual["ending_fresh_carry"] = np.nan
    actual["revenue"] = actual["observed_sales_amount"]
    actual["production_cost"] = actual["production"] * actual["unit_cost"]
    actual["gross_profit"] = actual["revenue"] - actual["production_cost"]
    simulations["actual_state"] = actual

    records = []
    for label, simulation in simulations.items():
        simulation["period"] = simulation["date"].dt.to_period("M").astype(str)
        part = simulation.groupby("period", as_index=False).agg(
            demand=("demand", "sum"),
            production=("production", "sum"),
            served=("served", "sum"),
            lost=("lost", "sum"),
            writeoff=("writeoff", "sum"),
            ending_fresh_carry=("ending_fresh_carry", "sum"),
            revenue=("revenue", "sum"),
            production_cost=("production_cost", "sum"),
            gross_profit=("gross_profit", "sum"),
        )
        part["variant"] = label
        records.append(part)
    result = pd.concat(records, ignore_index=True)
    actual_gp = result[result["variant"].eq("actual_state")][
        ["period", "gross_profit"]
    ].rename(columns={"gross_profit": "actual_gp"})
    result = result.merge(actual_gp, on="period", validate="many_to_one")
    result["gp_delta_vs_actual_pct"] = (
        100 * (result["gross_profit"] - result["actual_gp"]) / result["actual_gp"]
    )
    result["service_pct"] = 100 * result["served"] / result["demand"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT / "monthly_summary.csv", index=False, encoding="utf-8-sig")
    total = result.groupby("variant", as_index=False).agg(
        demand=("demand", "sum"),
        production=("production", "sum"),
        served=("served", "sum"),
        lost=("lost", "sum"),
        writeoff=("writeoff", "sum"),
        revenue=("revenue", "sum"),
        production_cost=("production_cost", "sum"),
        gross_profit=("gross_profit", "sum"),
    )
    actual_total_gp = float(
        total.loc[total["variant"].eq("actual_state"), "gross_profit"].iloc[0]
    )
    total["gp_delta_vs_actual"] = total["gross_profit"] - actual_total_gp
    total["service_pct"] = 100 * total["served"] / total["demand"]
    total.to_csv(OUTPUT / "total_summary.csv", index=False, encoding="utf-8-sig")
    decisions.to_csv(OUTPUT / "gate_decisions.csv", index=False, encoding="utf-8-sig")
    rows[
        KEYS
        + [
            "has_recorded_freshness",
            "actual_sold_old",
            "actual_sold_fresh",
            "old_reconciliation_in",
            "fresh_reconciliation_in",
            "reconciliation_in",
        ]
    ].to_parquet(OUTPUT / "reconciliation.parquet", index=False)
    coverage = (
        rows.assign(period=rows["date"].dt.to_period("M").astype(str))
        .groupby("period", as_index=False)
        .agg(
            observed_sales_qty=("observed_sales_qty", "sum"),
            recorded_freshness_sales_qty=(
                "observed_sales_qty",
                lambda value: value[
                    rows.loc[value.index, "has_recorded_freshness"]
                ].sum(),
            ),
            rows=("date", "size"),
            rows_with_recorded_freshness=("has_recorded_freshness", "sum"),
        )
    )
    coverage["recorded_sales_coverage_pct"] = (
        100 * coverage["recorded_freshness_sales_qty"] / coverage["observed_sales_qty"]
    )
    coverage.to_csv(
        OUTPUT / "freshness_coverage.csv", index=False, encoding="utf-8-sig"
    )
    for label, simulation in simulations.items():
        simulation.to_parquet(OUTPUT / f"{label}_daily_rows.parquet", index=False)
    print(result.sort_values(["period", "variant"]).to_string(index=False))


if __name__ == "__main__":
    main()

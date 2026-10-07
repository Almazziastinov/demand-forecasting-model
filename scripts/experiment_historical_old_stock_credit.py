"""Historical causal sensitivity of old-stock credit for Direct plans."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCOPE = ROOT / "reports/comparable_produced_scope_p50_loss075_20260911/predictions.parquet"
POISSON = ROOT / "reports/direct_poisson_allocation_full_20260910/predictions.parquet"
DEMAND = ROOT / "reports/network_causal_expanded_demand_20260911/network_daily_demand.parquet"
PANEL = ROOT / ".codex_tmp/historical_clickhouse_panel/causal_panel.parquet"
FRESHNESS = ROOT / ".codex_tmp/pilot_freshness_history_20260911.parquet"
MAPPING = ROOT / "reports/markup_price_mapping_20260826/mapped_products.csv"
OUTPUT = ROOT / "reports/historical_old_stock_credit_20260911"
KEYS = ["date", "bakery_id", "product_id"]
DAY = ["date", "bakery_id"]
DISCOUNT = 0.30
RULES = {
    "current_full_credit": None,
    "sellable_x4_00": 4.00,
    "sellable_x3_00": 3.00,
    "sellable_x2_50": 2.50,
    "sellable_x2_00": 2.00,
    "sellable_x1_50": 1.50,
    "sellable_x1_25": 1.25,
    "sellable_x1_00": 1.00,
    "sellable_x0_75": 0.75,
    "sellable_x0_50": 0.50,
    "sellable_x0_25": 0.25,
    "no_old_stock_credit": 0.00,
}


def load_rows() -> pd.DataFrame:
    rows = pd.read_parquet(
        SCOPE,
        columns=KEYS
        + ["direct_total", "observed_sales_qty", "observed_sales_amount", "avg_sales_price"],
    )
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    demand = pd.read_parquet(DEMAND)
    demand["date"] = pd.to_datetime(demand["date"]).dt.normalize()
    rows = rows.drop(columns="observed_sales_qty").merge(
        demand[KEYS + ["observed_sales_qty", "lost_expanded"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    rows[["observed_sales_qty", "lost_expanded"]] = rows[
        ["observed_sales_qty", "lost_expanded"]
    ].fillna(0.0)
    rows["demand"] = rows["observed_sales_qty"] + rows["lost_expanded"]

    poisson = pd.read_parquet(POISSON, columns=KEYS + ["direct_raw_demand"])
    poisson["date"] = pd.to_datetime(poisson["date"]).dt.normalize()
    rows = rows.merge(poisson, on=KEYS, how="left", validate="one_to_one")
    raw = rows["direct_raw_demand"].fillna(0.0).clip(lower=0.0)
    raw_total = raw.groupby([rows[key] for key in DAY]).transform("sum")
    fallback = 1.0 / rows.groupby(DAY)["product_id"].transform("count")
    share = (raw / raw_total.replace(0.0, np.nan)).fillna(fallback)
    rows["plan"] = rows["direct_total"].clip(lower=0.0) * share

    panel = pd.read_parquet(
        PANEL,
        columns=KEYS
        + ["release_qty", "incoming_move_qty", "outgoing_move_qty"],
    )
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    rows = rows.merge(panel, on=KEYS, how="left", validate="one_to_one")
    for column in ["release_qty", "incoming_move_qty", "outgoing_move_qty"]:
        rows[column] = pd.to_numeric(rows[column], errors="coerce").fillna(0.0).clip(lower=0.0)

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
    rows["sale_price"] = rows["avg_sales_price"].where(
        rows["avg_sales_price"].gt(0.0), rows["unit_price"]
    )
    return add_old_share_prior(rows)


def add_old_share_prior(rows: pd.DataFrame) -> pd.DataFrame:
    freshness = pd.read_parquet(FRESHNESS)
    freshness["date"] = pd.to_datetime(freshness["check_date"]).dt.normalize()
    label = freshness["freshness"].astype(str)
    freshness["kind"] = np.where(
        label.str.startswith("Вч"), "old", np.where(label.str.startswith("Св"), "fresh", "unknown")
    )
    daily = (
        freshness[freshness["kind"].isin(["old", "fresh"])]
        .pivot_table(index=["date", "product_id"], columns="kind", values="quantity", aggfunc="sum", fill_value=0.0)
        .reset_index()
    )
    for column in ["old", "fresh"]:
        if column not in daily:
            daily[column] = 0.0
    product_ids = sorted(rows["product_id"].unique())
    grid = pd.MultiIndex.from_product(
        [pd.date_range(daily["date"].min(), rows["date"].max()), product_ids],
        names=["date", "product_id"],
    ).to_frame(index=False)
    daily = grid.merge(daily, on=["date", "product_id"], how="left")
    daily[["old", "fresh"]] = daily[["old", "fresh"]].fillna(0.0)
    daily = daily.sort_values(["product_id", "date"])
    for column in ["old", "fresh"]:
        daily[f"prior_{column}"] = daily.groupby("product_id")[column].transform(
            lambda value: value.shift(1).rolling(56, min_periods=7).sum()
        )
    denominator = daily["prior_old"] + daily["prior_fresh"]
    daily["old_share_prior"] = (daily["prior_old"] / denominator).clip(0.0, 0.5)
    result = rows.merge(
        daily[["date", "product_id", "old_share_prior"]],
        on=["date", "product_id"],
        how="left",
        validate="many_to_one",
    )
    result["old_share_prior"] = result["old_share_prior"].fillna(0.05).clip(0.0, 0.5)
    return result


def simulate_group(group: pd.DataFrame, credit_multiplier: float | None) -> pd.DataFrame:
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
        if credit_multiplier is None:
            old_credit = old_opening
        else:
            old_credit = min(old_opening, plan * float(row.old_share_prior) * credit_multiplier)
        production = max(plan + sent - received - old_credit, 0.0)
        old_after_out = max(old_opening - sent, 0.0)
        fresh_after_out = max(production + received - max(sent - old_opening, 0.0), 0.0)
        demand = float(row.demand)
        sold_old = min(old_after_out, demand * float(row.old_share_prior), demand)
        sold_fresh = min(fresh_after_out, demand - sold_old)
        served = sold_old + sold_fresh
        expired = old_after_out - sold_old
        carry = fresh_after_out - sold_fresh
        revenue = sold_fresh * float(row.sale_price) + sold_old * float(row.sale_price) * (1.0 - DISCOUNT)
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
    rows = load_rows()
    actual = rows[KEYS + ["demand", "observed_sales_qty", "observed_sales_amount", "release_qty", "unit_cost"]].copy()
    actual["actual_lost"] = (actual["demand"] - actual["observed_sales_qty"]).clip(lower=0.0)
    actual["actual_gp"] = actual["observed_sales_amount"] - actual["release_qty"] * actual["unit_cost"]
    actual["month"] = actual["date"].dt.to_period("M").astype(str)
    actual_case = actual["actual_lost"].ge(1.0)
    groups = list(rows.groupby(["bakery_id", "product_id"], sort=False))
    monthly_parts = []

    for label, multiplier in RULES.items():
        simulation = pd.concat(
            [simulate_group(group, multiplier) for _, group in groups],
            ignore_index=True,
        )
        paired = actual.merge(simulation, on=KEYS, validate="one_to_one")
        delta = paired["actual_lost"] - paired["lost"]
        paired["variant"] = label
        paired["improved_case"] = actual_case & delta.ge(1.0)
        paired["new_lost_case"] = ~actual_case & paired["lost"].ge(1.0)
        paired["worsened_row"] = delta.le(-1.0)
        monthly_parts.append(
            paired.groupby(["month", "variant"], as_index=False).agg(
                days=("date", "nunique"),
                production=("production", "sum"),
                served=("served", "sum"),
                lost=("lost", "sum"),
                writeoff=("writeoff", "sum"),
                gross_profit=("gross_profit", "sum"),
                actual_gp=("actual_gp", "sum"),
                improved_cases=("improved_case", "sum"),
                new_lost_cases=("new_lost_case", "sum"),
                worsened_rows=("worsened_row", "sum"),
            )
        )
        print(f"simulated {label}", flush=True)

    monthly = pd.concat(monthly_parts, ignore_index=True)
    current = monthly[monthly["variant"].eq("current_full_credit")][
        ["month", "production", "lost", "writeoff", "gross_profit"]
    ].rename(columns={column: f"current_{column}" for column in ["production", "lost", "writeoff", "gross_profit"]})
    monthly = monthly.merge(current, on="month", validate="many_to_one")
    for column in ["production", "lost", "writeoff", "gross_profit"]:
        monthly[f"{column}_delta_vs_current"] = monthly[column] - monthly[f"current_{column}"]
    monthly["gross_profit_delta_vs_actual"] = monthly["gross_profit"] - monthly["actual_gp"]
    monthly.to_csv(OUTPUT / "monthly.csv", index=False, encoding="utf-8-sig")

    total = monthly.groupby("variant", as_index=False).agg(
        production=("production", "sum"),
        lost=("lost", "sum"),
        writeoff=("writeoff", "sum"),
        gross_profit=("gross_profit", "sum"),
        actual_gp=("actual_gp", "sum"),
        improved_cases=("improved_cases", "sum"),
        new_lost_cases=("new_lost_cases", "sum"),
        worsened_rows=("worsened_rows", "sum"),
        gp_delta_vs_current=("gross_profit_delta_vs_current", "sum"),
        lost_delta_vs_current=("lost_delta_vs_current", "sum"),
        writeoff_delta_vs_current=("writeoff_delta_vs_current", "sum"),
        positive_gp_months_vs_current=("gross_profit_delta_vs_current", lambda value: int(value.gt(0.0).sum())),
    )
    total["gp_delta_vs_actual"] = total["gross_profit"] - total["actual_gp"]
    total = total.sort_values("gross_profit", ascending=False)
    total.to_csv(OUTPUT / "leaderboard.csv", index=False, encoding="utf-8-sig")
    print(total.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

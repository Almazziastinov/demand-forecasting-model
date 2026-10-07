"""Evaluate the three forecast configurations actually served during the pilot."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_freshness_calibrated_economics import (  # noqa: E402
    freshness_kind,
    simulate_group,
)


DETAIL = ROOT / ".codex_tmp/pilot_model_version_eval_source/detail.csv"
HISTORICAL_DEMAND = (
    ROOT
    / "reports/historical_daily_retraining_gate_loss075_full_20260910"
    / "predictions.parquet"
)
SEPTEMBER_DEMAND = (
    ROOT
    / "reports/prod_direct_end_of_day_economics_sep01_09_20260910"
    / "daily_rows.parquet"
)
FRESHNESS = ROOT / ".codex_tmp/pilot_freshness_history_20260911.parquet"
MAPPING = ROOT / "reports/markup_price_mapping_20260826/mapped_products.csv"
OUTPUT = ROOT / "reports/live_pilot_model_versions_20260911"
KEYS = ["date", "bakery_id", "product_id"]


def classify_version(run_id: str) -> str | None:
    value = str(run_id)
    if "direct_alpha_025" in value:
        return "direct_alpha_025_v1"
    if "base_bakery_norm_recent" in value:
        return "base_bakery_norm_recent"
    if "base_bakery_raw_uplift_sku" in value:
        return "base_raw_uplift"
    return None


def build_freshness_priors(rows: pd.DataFrame) -> pd.DataFrame:
    freshness = pd.read_parquet(FRESHNESS).rename(columns={"check_date": "date"})
    freshness["date"] = pd.to_datetime(freshness["date"]).dt.normalize()
    freshness["kind"] = freshness["freshness"].map(freshness_kind)
    freshness = freshness[freshness["kind"].isin(["fresh", "old"])].copy()
    daily = freshness.pivot_table(
        index=["date", "product_id"],
        columns="kind",
        values=["quantity", "amount"],
        aggfunc="sum",
        fill_value=0,
    )
    daily.columns = [f"{metric}_{kind}" for metric, kind in daily.columns]
    daily = daily.reset_index()
    for column in ["quantity_fresh", "quantity_old", "amount_fresh", "amount_old"]:
        if column not in daily:
            daily[column] = 0.0

    product_ids = sorted(rows["product_id"].unique())
    date_min = min(freshness["date"].min(), rows["date"].min() - pd.Timedelta(days=56))
    grid = pd.MultiIndex.from_product(
        [pd.date_range(date_min, rows["date"].max()), product_ids],
        names=["date", "product_id"],
    ).to_frame(index=False)
    daily = grid.merge(daily, on=["date", "product_id"], how="left")
    quantity_columns = ["quantity_fresh", "quantity_old", "amount_fresh", "amount_old"]
    daily[quantity_columns] = daily[quantity_columns].fillna(0.0)
    daily = daily.sort_values(["product_id", "date"])
    for column in quantity_columns:
        daily[f"prior_{column}"] = daily.groupby("product_id")[column].transform(
            lambda value: value.shift(1).rolling(56, min_periods=7).sum()
        )
    denominator = daily["prior_quantity_fresh"] + daily["prior_quantity_old"]
    daily["old_share_prior"] = (daily["prior_quantity_old"] / denominator).clip(0, 0.5)
    daily["fresh_price_prior"] = (
        daily["prior_amount_fresh"] / daily["prior_quantity_fresh"]
    )
    daily["old_price_prior"] = daily["prior_amount_old"] / daily["prior_quantity_old"]
    return daily[
        [
            "date",
            "product_id",
            "old_share_prior",
            "fresh_price_prior",
            "old_price_prior",
        ]
    ]


def reconstruct_inventory(rows: pd.DataFrame) -> pd.DataFrame:
    # Checkout freshness is not available during the pilot itself, so the causal
    # product-level prior estimates the old/fresh split of factual sales.
    rows["actual_sold_old"] = np.minimum(
        rows["observed_sales_qty"] * rows["old_share_prior"], rows["observed_sales_qty"]
    )
    rows["actual_sold_fresh"] = rows["observed_sales_qty"] - rows["actual_sold_old"]
    reconciled: list[tuple[int, float, float]] = []
    for _, group in rows.groupby(["bakery_id", "product_id"], sort=False):
        carry = 0.0
        previous_date: pd.Timestamp | None = None
        for idx, row in group.sort_values("date").iterrows():
            if previous_date is not None and (row["date"] - previous_date).days > 1:
                carry = 0.0
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
            previous_date = row["date"]
    reconciliation = pd.DataFrame(
        reconciled,
        columns=["row_index", "old_reconciliation_in", "fresh_reconciliation_in"],
    ).set_index("row_index")
    rows = rows.join(reconciliation, how="left")
    rows["reconciliation_in"] = (
        rows["old_reconciliation_in"] + rows["fresh_reconciliation_in"]
    )
    return rows


def main() -> None:
    detail_columns = [
        "business_date",
        "bakery_id",
        "product_id",
        "forecast_run_id",
        "forecast_qty",
        "produced_qty",
        "received_qty",
        "sent_qty",
    ]
    rows = pd.read_csv(DETAIL, usecols=detail_columns, low_memory=False).rename(
        columns={
            "business_date": "date",
            "produced_qty": "release_qty",
            "received_qty": "incoming_move_qty",
            "sent_qty": "outgoing_move_qty",
        }
    )
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    rows["variant"] = rows["forecast_run_id"].map(classify_version)
    rows = rows[rows["variant"].notna() & rows["date"].le("2026-09-09")].copy()
    numeric = [
        "forecast_qty",
        "release_qty",
        "incoming_move_qty",
        "outgoing_move_qty",
    ]
    rows[numeric] = rows[numeric].apply(pd.to_numeric, errors="coerce")
    rows = rows[rows["forecast_qty"].notna()].copy()
    rows[numeric] = rows[numeric].fillna(0.0).clip(lower=0.0)

    fact_columns = ["demand", "observed_sales_qty", "observed_sales_amount"]
    historical = pd.read_parquet(HISTORICAL_DEMAND, columns=KEYS + fact_columns)
    historical["date"] = pd.to_datetime(historical["date"]).dt.normalize()
    september = pd.read_parquet(SEPTEMBER_DEMAND)
    september = september[september["variant"].eq("actual_state")][
        KEYS + ["demand", "sold_qty", "revenue"]
    ].rename(
        columns={
            "sold_qty": "observed_sales_qty",
            "revenue": "observed_sales_amount",
        }
    )
    september["date"] = pd.to_datetime(september["date"]).dt.normalize()
    demand = pd.concat(
        [historical[KEYS + fact_columns], september], ignore_index=True
    ).drop_duplicates(KEYS, keep="last")
    rows = rows.merge(demand, on=KEYS, how="inner", validate="one_to_one")

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
    rows["written_off_qty"] = 0.0
    # The actual write-off is not retained in the report detail export. Merge it
    # from the causal factual ledger for Jul-Aug; September factual write-offs do
    # not affect the counterfactual model stock loop.
    factual = pd.read_parquet(
        ROOT / ".codex_tmp/historical_clickhouse_panel/causal_panel.parquet",
        columns=KEYS + ["written_off_qty"],
    )
    factual["date"] = pd.to_datetime(factual["date"]).dt.normalize()
    rows = rows.drop(columns="written_off_qty").merge(
        factual, on=KEYS, how="left", validate="one_to_one"
    )
    rows["written_off_qty"] = rows["written_off_qty"].fillna(0.0).clip(lower=0.0)

    rows = rows.merge(
        build_freshness_priors(rows),
        on=["date", "product_id"],
        how="left",
        validate="many_to_one",
    )
    rows["old_share_prior"] = rows["old_share_prior"].fillna(0.05).clip(0, 0.5)
    rows["fresh_price_prior"] = rows["fresh_price_prior"].fillna(rows["unit_price"])
    rows["old_price_prior"] = rows["old_price_prior"].fillna(
        rows["fresh_price_prior"] * 0.7
    )
    rows = reconstruct_inventory(rows)

    model_parts = []
    actual_parts = []
    for variant, version_rows in rows.groupby("variant", sort=False):
        model = pd.concat(
            [
                simulate_group(group, "forecast_qty")
                for _, group in version_rows.groupby(
                    ["bakery_id", "product_id"], sort=False
                )
            ],
            ignore_index=True,
        )
        model["variant"] = variant
        model_parts.append(model)
        actual_columns = KEYS + [
            "demand",
            "release_qty",
            "observed_sales_qty",
            "observed_sales_amount",
            "written_off_qty",
            "unit_cost",
        ]
        actual = version_rows[actual_columns].copy()
        actual["production"] = actual["release_qty"]
        actual["served"] = actual["observed_sales_qty"]
        actual["lost"] = (actual["demand"] - actual["served"]).clip(lower=0.0)
        actual["writeoff"] = actual["written_off_qty"]
        actual["revenue"] = actual["observed_sales_amount"]
        actual["production_cost"] = actual["production"] * actual["unit_cost"]
        actual["gross_profit"] = actual["revenue"] - actual["production_cost"]
        actual["variant"] = variant
        actual_parts.append(actual)

    model_rows = pd.concat(model_parts, ignore_index=True)
    actual_rows = pd.concat(actual_parts, ignore_index=True)
    model_summary = model_rows.groupby("variant", as_index=False).agg(
        model_production=("production", "sum"),
        model_served=("served", "sum"),
        model_lost=("lost", "sum"),
        model_writeoff=("writeoff", "sum"),
        model_gp=("gross_profit", "sum"),
    )
    actual_summary = actual_rows.groupby("variant", as_index=False).agg(
        actual_production=("production", "sum"),
        actual_served=("served", "sum"),
        demand=("demand", "sum"),
        actual_lost=("lost", "sum"),
        actual_writeoff=("writeoff", "sum"),
        actual_gp=("gross_profit", "sum"),
    )
    forecast_summary = rows.groupby("variant", as_index=False).agg(
        date_from=("date", "min"),
        date_to=("date", "max"),
        days=("date", "nunique"),
        bakeries=("bakery_id", "nunique"),
        sku_days=("date", "size"),
        forecast=("forecast_qty", "sum"),
        absolute_error=("forecast_qty", lambda value: np.nan),
    )
    error_rows = rows.assign(
        absolute_error=(rows["forecast_qty"] - rows["demand"]).abs(),
        signed_error=rows["forecast_qty"] - rows["demand"],
        underforecast=(rows["demand"] - rows["forecast_qty"]).clip(lower=0.0),
        overforecast=(rows["forecast_qty"] - rows["demand"]).clip(lower=0.0),
    )
    errors = error_rows.groupby("variant", as_index=False).agg(
        absolute_error=("absolute_error", "sum"),
        signed_error=("signed_error", "sum"),
        underforecast=("underforecast", "sum"),
        overforecast=("overforecast", "sum"),
    )
    forecast_summary = forecast_summary.drop(columns="absolute_error").merge(
        errors, on="variant", validate="one_to_one"
    )
    summary = forecast_summary.merge(actual_summary, on="variant").merge(
        model_summary, on="variant"
    )
    summary["wape_pct"] = 100 * summary["absolute_error"] / summary["demand"]
    summary["bias_pct"] = 100 * summary["signed_error"] / summary["demand"]
    summary["model_service_pct"] = 100 * summary["model_served"] / summary["demand"]
    summary["actual_service_pct"] = 100 * summary["actual_served"] / summary["demand"]
    summary["gp_delta_vs_actual"] = summary["model_gp"] - summary["actual_gp"]
    summary["gp_delta_vs_actual_pct"] = (
        100 * summary["gp_delta_vs_actual"] / summary["actual_gp"]
    )
    summary["production_execution_pct"] = (
        100 * summary["actual_production"] / summary["forecast"]
    )
    order = ["base_raw_uplift", "base_bakery_norm_recent", "direct_alpha_025_v1"]
    summary["sort_order"] = summary["variant"].map(
        {value: idx for idx, value in enumerate(order)}
    )
    summary = summary.sort_values("sort_order").drop(columns="sort_order")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT / "summary.csv", index=False, encoding="utf-8-sig")
    error_rows.to_parquet(OUTPUT / "forecast_rows.parquet", index=False)
    model_rows.to_parquet(OUTPUT / "model_economics_rows.parquet", index=False)
    actual_rows.to_parquet(OUTPUT / "actual_economics_rows.parquet", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

"""Evaluate three frozen August SKU plans with one demand/economics contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_freshness_calibrated_economics import (  # noqa: E402
    freshness_kind,
    simulate_group,
)


OUTPUT = ROOT / "reports/three_prod_models_august_holdout_20260911"
DEMAND = (
    ROOT
    / "reports/historical_daily_retraining_gate_loss075_full_20260910"
    / "predictions.parquet"
)
PANEL = ROOT / ".codex_tmp/historical_clickhouse_panel/causal_panel.parquet"
FRESHNESS = ROOT / ".codex_tmp/pilot_freshness_history_20260911.parquet"
MAPPING = ROOT / "reports/markup_price_mapping_20260826/mapped_products.csv"
START = pd.Timestamp("2026-08-24")
END = pd.Timestamp("2026-08-31")
KEYS = ["date", "bakery_id", "product_id"]
PLAN_FILES = {
    "base_bakery_norm_recent": OUTPUT / "norm_plan.parquet",
    "direct_alpha_025_v1": OUTPUT / "direct_plan.parquet",
    "p50_loss_075_causal_daily": OUTPUT / "p50_loss075_plan.parquet",
    "base_raw_uplift_reconstructed": OUTPUT / "raw_plan.parquet",
}


def export_p50_plan() -> None:
    """Export the strict D-1 daily-retrained loss-0.75 plan."""
    plan = pd.read_parquet(DEMAND, columns=KEYS + ["p50_plan"])
    plan["date"] = pd.to_datetime(plan["date"]).dt.normalize()
    plan = plan[plan["date"].between(START, END)].rename(
        columns={"p50_plan": "forecast_qty"}
    )
    plan.to_parquet(PLAN_FILES["p50_loss_075_causal_daily"], index=False)


def load_base_rows() -> tuple[pd.DataFrame, dict[str, float | int]]:
    norm = pd.read_parquet(PLAN_FILES["base_bakery_norm_recent"])
    norm["date"] = pd.to_datetime(norm["date"]).dt.normalize()
    universe = norm[KEYS].drop_duplicates()

    demand = pd.read_parquet(
        DEMAND,
        columns=KEYS
        + ["demand", "observed_sales_qty", "observed_sales_amount"],
    )
    demand["date"] = pd.to_datetime(demand["date"]).dt.normalize()
    demand = demand[demand["date"].between(START, END)]
    rows = universe.merge(demand, on=KEYS, how="left", validate="one_to_one")

    flows = pd.read_parquet(
        PANEL,
        columns=KEYS
        + ["release_qty", "incoming_move_qty", "outgoing_move_qty", "written_off_qty"],
    )
    flows["date"] = pd.to_datetime(flows["date"]).dt.normalize()
    flows = flows[flows["date"].between(START, END)]
    rows = rows.merge(flows, on=KEYS, how="left", validate="one_to_one")

    numeric = [
        "demand",
        "observed_sales_qty",
        "observed_sales_amount",
        "release_qty",
        "incoming_move_qty",
        "outgoing_move_qty",
        "written_off_qty",
    ]
    for column in numeric:
        rows[column] = pd.to_numeric(rows[column], errors="coerce").fillna(0.0)
    for column in numeric:
        rows[column] = rows[column].clip(lower=0.0)

    mapping = pd.read_csv(MAPPING, encoding="utf-8-sig")
    mapping = (
        mapping[mapping["valid_economics"].astype(bool)]
        .sort_values("unit_price")
        .drop_duplicates("product_id", keep="last")
    )
    before = len(rows)
    rows = rows.merge(
        mapping[["product_id", "unit_cost", "unit_price"]],
        on="product_id",
        how="inner",
        validate="many_to_one",
    )
    coverage = {
        "source_universe_rows": before,
        "priced_universe_rows": int(len(rows)),
        "priced_row_coverage_pct": 100 * len(rows) / before,
        "demand_qty": float(rows["demand"].sum()),
        "observed_sales_qty": float(rows["observed_sales_qty"].sum()),
    }
    return rows, coverage


def add_plan_columns(rows: pd.DataFrame) -> pd.DataFrame:
    result = rows.copy()
    plan_labels: dict[str, str] = {}
    for index, (label, path) in enumerate(PLAN_FILES.items()):
        plan = pd.read_parquet(path, columns=KEYS + ["forecast_qty"])
        plan["date"] = pd.to_datetime(plan["date"]).dt.normalize()
        column = f"plan_{index}"
        plan = plan.rename(columns={"forecast_qty": column})
        result = result.merge(plan, on=KEYS, how="left", validate="one_to_one")
        result[column] = result[column].fillna(0.0).clip(lower=0.0)
        plan_labels[column] = label
    result.attrs["plan_labels"] = plan_labels
    return result


def add_causal_freshness_priors(rows: pd.DataFrame) -> pd.DataFrame:
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

    priced = freshness[freshness["kind"].isin(["fresh", "old"])]
    daily = priced.pivot_table(
        index=KEYS,
        columns="kind",
        values=["quantity", "amount"],
        aggfunc="sum",
        fill_value=0,
    )
    daily.columns = [f"{metric}_{kind}" for metric, kind in daily.columns]
    daily = daily.reset_index()
    product_day = daily.groupby(["date", "product_id"], as_index=False).agg(
        fresh_qty=("quantity_fresh", "sum"),
        old_qty=("quantity_old", "sum"),
        fresh_amount=("amount_fresh", "sum"),
        old_amount=("amount_old", "sum"),
    )
    product_ids = sorted(rows["product_id"].unique())
    grid = pd.MultiIndex.from_product(
        [
            pd.date_range(freshness["date"].min(), END),
            product_ids,
        ],
        names=["date", "product_id"],
    ).to_frame(index=False)
    product_day = grid.merge(product_day, on=["date", "product_id"], how="left")
    measures = ["fresh_qty", "old_qty", "fresh_amount", "old_amount"]
    product_day[measures] = product_day[measures].fillna(0.0)
    product_day = product_day.sort_values(["product_id", "date"])
    for column in measures:
        product_day[f"prior_{column}"] = product_day.groupby("product_id")[
            column
        ].transform(lambda value: value.shift(1).rolling(56, min_periods=7).sum())
    denominator = product_day["prior_fresh_qty"] + product_day["prior_old_qty"]
    product_day["old_share_prior"] = (
        product_day["prior_old_qty"] / denominator
    ).clip(0, 0.5)
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
    result = rows.merge(rates, on=["date", "product_id"], how="left")
    result["old_share_prior"] = result["old_share_prior"].fillna(0.05).clip(0, 0.5)
    result["fresh_price_prior"] = result["fresh_price_prior"].fillna(
        result["unit_price"]
    )
    result["old_price_prior"] = result["old_price_prior"].fillna(
        result["fresh_price_prior"] * 0.7
    )

    result = result.merge(daily_quantity, on=KEYS, how="left", validate="one_to_one")
    recorded_columns = [
        "recorded_fresh_qty",
        "recorded_old_qty",
        "recorded_unknown_qty",
    ]
    result[recorded_columns] = result[recorded_columns].fillna(0.0).clip(lower=0.0)
    result["recorded_freshness_qty"] = result[recorded_columns].sum(axis=1)
    result["has_recorded_freshness"] = result["recorded_freshness_qty"].gt(0)
    scale = result["observed_sales_qty"] / result["recorded_freshness_qty"].replace(
        0.0, np.nan
    )
    recorded_old = result["recorded_old_qty"] * scale
    recorded_fresh = (
        result["recorded_fresh_qty"] + result["recorded_unknown_qty"]
    ) * scale
    estimated_old = result["observed_sales_qty"] * result["old_share_prior"]
    result["actual_sold_old"] = np.where(
        result["has_recorded_freshness"], recorded_old, estimated_old
    )
    result["actual_sold_old"] = result["actual_sold_old"].clip(
        lower=0.0, upper=result["observed_sales_qty"]
    )
    result["actual_sold_fresh"] = np.where(
        result["has_recorded_freshness"],
        recorded_fresh,
        result["observed_sales_qty"] - estimated_old,
    )
    result["actual_sold_fresh"] = (
        result["observed_sales_qty"] - result["actual_sold_old"]
    ).clip(lower=0.0)
    return result


def add_reconciliation(rows: pd.DataFrame) -> pd.DataFrame:
    reconciled: list[tuple[int, float, float]] = []
    for _, group in rows.groupby(["bakery_id", "product_id"], sort=False):
        carry = 0.0
        for index, row in group.sort_values("date").iterrows():
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
            reconciled.append((index, old_reconciliation, fresh_reconciliation))
    values = pd.DataFrame(
        reconciled,
        columns=["row_index", "old_reconciliation_in", "fresh_reconciliation_in"],
    ).set_index("row_index")
    result = rows.join(values, how="left")
    reconciliation_columns = ["old_reconciliation_in", "fresh_reconciliation_in"]
    result[reconciliation_columns] = result[reconciliation_columns].fillna(0.0)
    result["reconciliation_in"] = result[reconciliation_columns].sum(axis=1)
    return result


def run_simulations(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    simulations: dict[str, pd.DataFrame] = {}
    for plan_column, label in rows.attrs["plan_labels"].items():
        simulations[label] = pd.concat(
            [
                simulate_group(group, plan_column)
                for _, group in rows.groupby(["bakery_id", "product_id"], sort=False)
            ],
            ignore_index=True,
        )
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
    actual["lost"] = (actual["demand"] - actual["served"]).clip(lower=0.0)
    actual["writeoff"] = actual["written_off_qty"]
    actual["ending_fresh_carry"] = np.nan
    actual["revenue"] = actual["observed_sales_amount"]
    actual["production_cost"] = actual["production"] * actual["unit_cost"]
    actual["gross_profit"] = actual["revenue"] - actual["production_cost"]
    simulations["actual_state"] = actual
    return simulations


def summarize(
    rows: pd.DataFrame,
    simulations: dict[str, pd.DataFrame],
    coverage: dict[str, float | int],
) -> None:
    records = []
    daily_frames = []
    demand_total = float(rows["demand"].sum())
    for label, simulation in simulations.items():
        plan_column = next(
            (
                column
                for column, name in rows.attrs["plan_labels"].items()
                if name == label
            ),
            None,
        )
        forecast = rows[plan_column] if plan_column else rows["observed_sales_qty"]
        record = {
            "variant": label,
            "forecast_or_actual_sales_qty": float(forecast.sum()),
            "demand": demand_total,
            "wape_vs_demand_pct": (
                100 * float((forecast - rows["demand"]).abs().sum()) / demand_total
            ),
            "bias_vs_demand_pct": (
                100 * float((forecast - rows["demand"]).sum()) / demand_total
            ),
            "production": float(simulation["production"].sum()),
            "served": float(simulation["served"].sum()),
            "lost": float(simulation["lost"].sum()),
            "writeoff": float(simulation["writeoff"].sum()),
            "ending_fresh_carry": float(simulation["ending_fresh_carry"].sum()),
            "revenue": float(simulation["revenue"].sum()),
            "production_cost": float(simulation["production_cost"].sum()),
            "gross_profit": float(simulation["gross_profit"].sum()),
        }
        record["service_pct"] = 100 * record["served"] / record["demand"]
        records.append(record)
        daily = simulation.groupby("date", as_index=False).agg(
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
        daily["variant"] = label
        daily_frames.append(daily)
        simulation.to_parquet(OUTPUT / f"economics_{label}.parquet", index=False)

    summary = pd.DataFrame(records)
    actual_gp = float(
        summary.loc[summary["variant"].eq("actual_state"), "gross_profit"].iloc[0]
    )
    summary["gp_delta_vs_actual"] = summary["gross_profit"] - actual_gp
    summary["gp_delta_vs_actual_pct"] = 100 * summary["gp_delta_vs_actual"] / actual_gp
    summary.to_csv(OUTPUT / "evaluation_summary.csv", index=False, encoding="utf-8-sig")

    daily = pd.concat(daily_frames, ignore_index=True)
    actual_daily = daily[daily["variant"].eq("actual_state")][
        ["date", "gross_profit"]
    ].rename(columns={"gross_profit": "actual_gp"})
    daily = daily.merge(actual_daily, on="date", validate="many_to_one")
    daily["gp_delta_vs_actual"] = daily["gross_profit"] - daily["actual_gp"]
    daily["service_pct"] = 100 * daily["served"] / daily["demand"]
    daily.to_csv(OUTPUT / "evaluation_by_day.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "window": {"start": str(START.date()), "end": str(END.date())},
        "demand_contract": "sales + corrected reconstructed lost demand",
        "inventory_contract": "endogenous model inventory, ageing and write-off",
        "actual_gp_contract": "checkout revenue - factual production * unit cost",
        "model_gp_contract": "simulated fresh/old revenue - own production * unit cost",
        "scope_coverage": coverage,
        "warnings": [
            "Raw uplift is a causal reconstruction; see build_metadata.json.",
            (
                "Freshness priors are causal and use checkout labels only through "
                "2026-07-19."
            ),
        ],
    }
    (OUTPUT / "evaluation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


def main() -> None:
    export_p50_plan()
    rows, coverage = load_base_rows()
    rows = add_plan_columns(rows)
    plan_labels = rows.attrs["plan_labels"].copy()
    rows = add_causal_freshness_priors(rows)
    rows.attrs["plan_labels"] = plan_labels
    rows = add_reconciliation(rows)
    rows.attrs["plan_labels"] = plan_labels
    rows.to_parquet(OUTPUT / "evaluation_input.parquet", index=False)
    simulations = run_simulations(rows)
    summarize(rows, simulations, coverage)


if __name__ == "__main__":
    main()

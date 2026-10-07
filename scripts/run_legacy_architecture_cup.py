"""Re-score preserved legacy allocation artifacts on one corrected economic scope."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from simulate_two_day_economics import simulate_group


ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "reports/direct_bakery_sku_allocation_20260827/predictions.parquet"
FULL_DIRECT = (
    ROOT / "reports/direct_poisson_allocation_full_20260910/predictions.parquet"
)
SCOPE = (
    ROOT / "reports/comparable_produced_scope_p50_loss075_20260911/predictions.parquet"
)
DEMAND = (
    ROOT
    / "reports/network_causal_expanded_demand_20260911/network_daily_demand.parquet"
)
PANEL = ROOT / ".codex_tmp/historical_clickhouse_panel/causal_panel.parquet"
MAPPING = ROOT / "reports/markup_price_mapping_20260826/mapped_products.csv"
OUTPUT = ROOT / "reports/legacy_architecture_cup_20260911"
KEYS = ["date", "bakery_id", "product_id"]
DISCOUNT = 0.30


def main() -> None:
    legacy = pd.read_parquet(
        LEGACY,
        columns=KEYS
        + [
            "incumbent_sku_forecast",
            "predictive_forecast",
            "direct_forecast",
            "fold",
        ],
    )
    legacy["date"] = pd.to_datetime(legacy["date"]).dt.normalize()
    dates = set(legacy["date"].unique())
    scope = pd.read_parquet(
        SCOPE,
        columns=KEYS + ["observed_sales_qty", "avg_sales_price"],
    )
    scope["date"] = pd.to_datetime(scope["date"]).dt.normalize()
    scope = scope[scope["date"].isin(dates)].copy()
    rows = scope.merge(legacy, on=KEYS, how="inner", validate="one_to_one")

    corrected = pd.read_parquet(DEMAND, columns=KEYS + ["lost_expanded"])
    corrected["date"] = pd.to_datetime(corrected["date"]).dt.normalize()
    rows = rows.merge(corrected, on=KEYS, how="left", validate="one_to_one")
    rows["lost_expanded"] = rows["lost_expanded"].fillna(0.0).clip(lower=0.0)
    rows["demand"] = rows["observed_sales_qty"].clip(lower=0.0) + rows["lost_expanded"]

    full_direct = pd.read_parquet(FULL_DIRECT, columns=KEYS + ["direct_plan_poisson"])
    full_direct["date"] = pd.to_datetime(full_direct["date"]).dt.normalize()
    rows = rows.merge(full_direct, on=KEYS, how="left", validate="one_to_one")

    flows = pd.read_parquet(
        PANEL, columns=KEYS + ["release_qty", "incoming_move_qty", "outgoing_move_qty"]
    )
    flows["date"] = pd.to_datetime(flows["date"]).dt.normalize()
    rows = rows.merge(flows, on=KEYS, how="left", validate="one_to_one")
    rows["opening_stock"] = 0.0
    rows["received"] = rows["incoming_move_qty"].fillna(0.0).clip(lower=0.0)
    rows["sent"] = rows["outgoing_move_qty"].fillna(0.0).clip(lower=0.0)
    rows["produced"] = rows["release_qty"].fillna(0.0).clip(lower=0.0)

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
    required = [
        "incumbent_sku_forecast",
        "predictive_forecast",
        "direct_forecast",
        "direct_plan_poisson",
    ]
    rows = rows.dropna(subset=required).copy()

    variants = {
        "actual_supply": None,
        "legacy_incumbent": "incumbent_sku_forecast",
        "legacy_predictive": "predictive_forecast",
        "original_direct": "direct_forecast",
        "monthly_direct_poisson": "direct_plan_poisson",
    }
    parts = []
    for variant, plan in variants.items():
        simulated = pd.concat(
            [
                simulate_group(group, plan)
                for _, group in rows.groupby(["bakery_id", "product_id"], sort=False)
            ],
            ignore_index=True,
        )
        simulated = simulated.merge(
            rows[KEYS + ["sale_price", "unit_cost", "fold"]],
            on=KEYS,
            validate="one_to_one",
        )
        simulated["revenue"] = simulated["sold_fresh"] * simulated[
            "sale_price"
        ] + simulated["sold_yesterday"] * simulated["sale_price"] * (1.0 - DISCOUNT)
        simulated["gross_profit"] = simulated["revenue"] - (
            simulated["production"] * simulated["unit_cost"]
        )
        simulated["variant"] = variant
        parts.append(simulated)
    all_rows = pd.concat(parts, ignore_index=True)
    summary = all_rows.groupby("variant", as_index=False).agg(
        demand=("demand", "sum"),
        production=("production", "sum"),
        served=("served", "sum"),
        under=("lost", "sum"),
        expired=("expired_strategy_stock", "sum"),
        gross_profit=("gross_profit", "sum"),
    )
    actual_gp = float(
        summary.loc[summary["variant"].eq("actual_supply"), "gross_profit"].iloc[0]
    )
    summary["gp_delta_vs_actual"] = summary["gross_profit"] - actual_gp
    summary["service_pct"] = 100.0 * summary["served"] / summary["demand"]
    by_fold = all_rows.groupby(["fold", "variant"], as_index=False).agg(
        gross_profit=("gross_profit", "sum"), under=("lost", "sum")
    )
    fold_actual = by_fold[by_fold["variant"].eq("actual_supply")][
        ["fold", "gross_profit"]
    ].rename(columns={"gross_profit": "actual_gross_profit"})
    by_fold = by_fold.merge(fold_actual, on="fold", validate="many_to_one")
    by_fold["gp_delta_vs_actual"] = (
        by_fold["gross_profit"] - by_fold["actual_gross_profit"]
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.sort_values("gross_profit", ascending=False).to_csv(
        OUTPUT / "leaderboard.csv", index=False, encoding="utf-8-sig"
    )
    by_fold.to_csv(OUTPUT / "by_fold.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "production_write": False,
        "dates": sorted(str(pd.Timestamp(date).date()) for date in dates),
        "common_rows": int(len(rows)),
        "bakeries": int(rows["bakery_id"].nunique()),
        "products": int(rows["product_id"].nunique()),
        "note": "secondary legacy comparison; only the 20 preserved artifact dates",
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(summary.sort_values("gross_profit", ascending=False).to_string(index=False))
    print(by_fold.to_string(index=False))


if __name__ == "__main__":
    main()

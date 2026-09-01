"""Compare plans with availability including opening stock and movements."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "reports/rebuilt_predictive_choice_20260825/predictions.parquet"
FACTS = ROOT / "reports/base_norm_recent_vs_mean7_20260824/sku_day_comparison.parquet"
PRODUCTION_SCOPE = ROOT / ".codex_tmp/base_norm_recent_eval_20260824/production.csv.gz"
INVENTORY = (
    ROOT / ".codex_tmp/base_norm_recent_eval_20260824/fct_inventory_components.parquet"
)
OUTPUT = ROOT / "reports/available_to_sell_balance_20260826"
KEYS = ["date", "bakery_id", "product_id"]


def candidate_metrics(rows: pd.DataFrame, plan_col: str) -> dict[str, float | int]:
    plan = rows[plan_col].clip(lower=0.0)
    demand = rows["strict_demand"].clip(lower=0.0)
    surplus = (plan - demand).clip(lower=0.0)
    underbake = (demand - plan).clip(lower=0.0)
    return {
        "volume_qty": float(plan.sum()),
        "surplus_qty": float(surplus.sum()),
        "underbake_qty": float(underbake.sum()),
        "total_imbalance_qty": float(surplus.sum() + underbake.sum()),
        "surplus_rows": int(surplus.gt(0).sum()),
        "underbake_rows": int(underbake.gt(0).sum()),
    }


def main() -> None:
    predictions = pd.read_parquet(PREDICTIONS)
    predictions = predictions[predictions["fold"].eq("current")].copy()
    facts = pd.read_parquet(FACTS)[KEYS + ["sold", "strict_demand"]]
    production_scope = pd.read_csv(PRODUCTION_SCOPE)
    inventory = pd.read_parquet(INVENTORY)
    for frame in (predictions, facts, production_scope, inventory):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()

    # Yesterday's verified closing balance becomes today's opening stock.
    opening = inventory[KEYS + ["closing"]].copy()
    opening["date"] += pd.Timedelta(days=1)
    opening = opening.rename(columns={"closing": "opening_stock"})
    today = inventory.drop(columns=["closing", "sold"]).merge(
        opening, on=KEYS, how="left", validate="one_to_one"
    )
    today["opening_stock"] = today["opening_stock"].fillna(0.0).clip(lower=0.0)

    production_products = set(production_scope["product_id"].unique())
    rows = predictions[predictions["product_id"].isin(production_products)].merge(
        facts, on=KEYS, how="left", validate="one_to_one"
    )
    rows = rows.merge(today, on=KEYS, how="left", validate="one_to_one")
    numeric = [
        "sold",
        "strict_demand",
        "produced",
        "received",
        "sent",
        "opening_stock",
    ]
    rows[numeric] = rows[numeric].fillna(0.0)

    # Retain exactly the observable bakery-day universe used by the forecast
    # evaluation. Missing component rows mean zero activity for that SKU/day.
    eligible = rows.groupby(["date", "bakery_id"])["sold"].transform("sum").gt(0)
    rows = rows[eligible].copy()
    rows["predictive_uplift_02"] = 1.02 * rows["predictive_forecast"]
    rows["available_to_sell"] = (
        rows["produced"] + rows["opening_stock"] + rows["received"] - rows["sent"]
    ).clip(lower=0.0)
    rows["actual_surplus"] = (rows["available_to_sell"] - rows["sold"]).clip(
        lower=0.0
    )
    rows["actual_underbake"] = (rows["strict_demand"] - rows["sold"]).clip(lower=0.0)

    records = [{
        "variant": "actual_available_to_sell",
        "volume_qty": float(rows["available_to_sell"].sum()),
        "surplus_qty": float(rows["actual_surplus"].sum()),
        "underbake_qty": float(rows["actual_underbake"].sum()),
        "total_imbalance_qty": float(
            rows["actual_surplus"].sum() + rows["actual_underbake"].sum()
        ),
        "surplus_rows": int(rows["actual_surplus"].gt(0).sum()),
        "underbake_rows": int(rows["actual_underbake"].gt(0).sum()),
    }]
    for variant, column in [
        ("incumbent_plan", "incumbent_sku_forecast"),
        ("predictive_plan", "predictive_forecast"),
        ("predictive_uplift_02", "predictive_uplift_02"),
    ]:
        records.append({"variant": variant, **candidate_metrics(rows, column)})

    metrics = pd.DataFrame(records)
    observed = metrics.iloc[0]
    metrics["imbalance_delta_vs_actual"] = (
        metrics["total_imbalance_qty"] - observed["total_imbalance_qty"]
    )
    metrics["underbake_delta_vs_actual"] = (
        metrics["underbake_qty"] - observed["underbake_qty"]
    )
    summary = {
        "scope": {
            "dates": int(rows["date"].nunique()),
            "bakeries": int(rows["bakery_id"].nunique()),
            "products": int(rows["product_id"].nunique()),
            "rows": int(len(rows)),
        },
        "balance_contract": (
            "available_to_sell=production+positive prior-day closing stock+"
            "received-sent; actual surplus=max(available_to_sell-sales,0); "
            "underbake=recognized lost demand"
        ),
        "fact_source": "deduplicated fct_production_release/fct_moves/fct_check_lines",
        "production_write": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT / "metrics.csv", index=False, encoding="utf-8-sig")
    rows.to_parquet(OUTPUT / "rows.parquet", index=False)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

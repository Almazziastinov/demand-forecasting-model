"""Compare forecast plans with the observed production balance proxy."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "reports/rebuilt_predictive_choice_20260825/predictions.parquet"
FACTS = ROOT / "reports/base_norm_recent_vs_mean7_20260824/sku_day_comparison.parquet"
PRODUCTION = ROOT / ".codex_tmp/base_norm_recent_eval_20260824/production.csv.gz"
OUTPUT = ROOT / "reports/operational_balance_20260825"
KEYS = ["date", "bakery_id", "product_id"]


def candidate_metrics(rows: pd.DataFrame, plan_col: str) -> dict[str, float | int]:
    plan = rows[plan_col].clip(lower=0.0)
    demand = rows["strict_demand"].clip(lower=0.0)
    surplus = (plan - demand).clip(lower=0.0)
    underbake = (demand - plan).clip(lower=0.0)
    return {
        "plan_qty": float(plan.sum()),
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
    production = pd.read_csv(PRODUCTION)
    for frame in (predictions, facts, production):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()

    production_products = set(production["product_id"].unique())
    rows = predictions[predictions["product_id"].isin(production_products)].merge(
        facts, on=KEYS, how="left", validate="one_to_one"
    )
    rows = rows.merge(production, on=KEYS, how="left", validate="one_to_one")
    rows[["sold", "strict_demand", "produced"]] = rows[
        ["sold", "strict_demand", "produced"]
    ].fillna(0.0)
    observable = rows.groupby(["date", "bakery_id"])["sold"].transform("sum").gt(0)
    rows = rows[observable].copy()
    rows["predictive_uplift_02"] = 1.02 * rows["predictive_forecast"]

    actual_surplus = (rows["produced"] - rows["sold"]).clip(lower=0.0)
    actual_underbake = (rows["strict_demand"] - rows["sold"]).clip(lower=0.0)
    records = [
        {
            "variant": "observed_production_state",
            "plan_qty": float(rows["produced"].sum()),
            "surplus_qty": float(actual_surplus.sum()),
            "underbake_qty": float(actual_underbake.sum()),
            "total_imbalance_qty": float(actual_surplus.sum() + actual_underbake.sum()),
            "surplus_rows": int(actual_surplus.gt(0).sum()),
            "underbake_rows": int(actual_underbake.gt(0).sum()),
        }
    ]
    for variant, column in [
        ("incumbent_plan", "incumbent_sku_forecast"),
        ("predictive_plan", "predictive_forecast"),
        ("predictive_uplift_02", "predictive_uplift_02"),
    ]:
        records.append({"variant": variant, **candidate_metrics(rows, column)})
    metrics = pd.DataFrame(records)
    baseline_imbalance = metrics.iloc[0]["total_imbalance_qty"]
    metrics["imbalance_delta_vs_observed"] = (
        metrics["total_imbalance_qty"] - baseline_imbalance
    )
    observed_underbake = metrics.iloc[0]["underbake_qty"]
    metrics["underbake_delta_vs_observed"] = (
        metrics["underbake_qty"] - observed_underbake
    )
    for weight in (1.0, 1.5, 2.0, 3.0):
        metrics[f"cost_under_weight_{weight:g}"] = (
            metrics["surplus_qty"] + weight * metrics["underbake_qty"]
        )
    summary = {
        "scope": {
            "dates": int(rows["date"].nunique()),
            "bakeries": int(rows["bakery_id"].nunique()),
            "products": int(rows["product_id"].nunique()),
            "rows": int(len(rows)),
        },
        "balance_contract": (
            "Observed surplus proxy=max(same-day production-sales,0); "
            "underbake=conservative lost demand. Initial stock is unavailable."
        ),
        "primary_underbake_gate_pass": {
            row.variant: bool(row.underbake_delta_vs_observed < 0)
            for row in metrics.itertuples()
            if row.variant != "observed_production_state"
        },
        "predictive_to_uplift_02_break_even_underbake_weight": float(
            (
                metrics.loc[
                    metrics["variant"].eq("predictive_uplift_02"), "surplus_qty"
                ].iloc[0]
                - metrics.loc[
                    metrics["variant"].eq("predictive_plan"), "surplus_qty"
                ].iloc[0]
            )
            / (
                metrics.loc[
                    metrics["variant"].eq("predictive_plan"), "underbake_qty"
                ].iloc[0]
                - metrics.loc[
                    metrics["variant"].eq("predictive_uplift_02"), "underbake_qty"
                ].iloc[0]
            )
        ),
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

"""January/August counterfactual decomposition on the fixed 93-SKU scope."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/canonical_gate_caps_fixed93_full_20260911/sku_monthly.parquet"
OUTPUT = ROOT / "reports/counterfactual_january_august_20260911"


def main() -> None:
    rows = pd.read_parquet(SOURCE)
    rows = rows[rows["variant"].isin(["actual_state", "p50_loss075"])].copy()
    actual = rows[rows["variant"].eq("actual_state")].copy()
    actual["price"] = actual["revenue"] / (
        actual["sold_fresh"] + 0.7 * actual["sold_old"]
    ).replace(0, pd.NA)
    actual["cost"] = actual["production_cost"] / actual["production"].replace(0, pd.NA)
    rates = actual[["period", "product_id", "price", "cost"]]

    records = []
    for quantity_month in ["2026-01", "2026-08"]:
        quantities = rows[rows["period"].eq(quantity_month)].copy()
        for economics_month in ["2026-01", "2026-08"]:
            econ = rates[rates["period"].eq(economics_month)].drop(columns="period")
            part = quantities.merge(econ, on="product_id", how="inner", validate="many_to_one")
            part["cf_revenue"] = (part["sold_fresh"] + 0.7 * part["sold_old"]) * part["price"]
            part["cf_cost"] = part["production"] * part["cost"]
            grouped = part.groupby("variant", as_index=False).agg(
                demand=("demand", "sum"), revenue=("cf_revenue", "sum"), production_cost=("cf_cost", "sum")
            )
            grouped["gross_profit"] = grouped["revenue"] - grouped["production_cost"]
            factual = grouped.loc[grouped["variant"].eq("actual_state"), "gross_profit"].iloc[0]
            p50 = grouped.loc[grouped["variant"].eq("p50_loss075"), "gross_profit"].iloc[0]
            records.append(
                {
                    "quantity_month": quantity_month,
                    "economics_month": economics_month,
                    "actual_gp": factual,
                    "p50_gp": p50,
                    "p50_delta": p50 - factual,
                    "p50_delta_pct": 100 * (p50 - factual) / factual,
                }
            )
    result = pd.DataFrame(records)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT / "counterfactual.csv", index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

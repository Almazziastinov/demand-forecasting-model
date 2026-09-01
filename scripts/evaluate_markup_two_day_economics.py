"""Evaluate two-day candidates using workbook prices/costs and 30% day-two discount."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SIMULATION = ROOT / "reports/two_day_economics_20260826/daily_rows.parquet"
MAPPING = ROOT / "reports/markup_price_mapping_20260826/mapped_products.csv"
OUTPUT = ROOT / "reports/markup_two_day_economics_20260826"
YESTERDAY_DISCOUNT = 0.30


def main() -> None:
    simulation = pd.read_parquet(SIMULATION)
    mapping = pd.read_csv(MAPPING, encoding="utf-8-sig")
    mapping = mapping[mapping["valid_economics"].astype(bool)].copy()
    mapping["product_id"] = mapping["product_id"].astype(int)
    mapping = mapping.sort_values("unit_price").drop_duplicates("product_id", keep="last")
    rows = simulation.merge(
        mapping[
            [
                "product_id",
                "workbook_product_name",
                "workbook_category",
                "unit_price",
                "unit_cost",
            ]
        ],
        on="product_id",
        how="inner",
        validate="many_to_one",
    )
    rows["fresh_revenue"] = rows["sold_fresh"] * rows["unit_price"]
    rows["yesterday_revenue"] = (
        rows["sold_yesterday"] * rows["unit_price"] * (1 - YESTERDAY_DISCOUNT)
    )
    rows["revenue"] = rows["fresh_revenue"] + rows["yesterday_revenue"]
    rows["production_cost"] = rows["production"] * rows["unit_cost"]
    rows["gross_profit"] = rows["revenue"] - rows["production_cost"]
    rows["full_price_revenue_if_fresh"] = rows["served"] * rows["unit_price"]
    rows["discount_loss"] = rows["full_price_revenue_if_fresh"] - rows["revenue"]

    summary = rows.groupby("variant", as_index=False).agg(
        products=("product_id", "nunique"),
        demand=("demand", "sum"),
        production=("production", "sum"),
        sold_fresh=("sold_fresh", "sum"),
        sold_yesterday=("sold_yesterday", "sum"),
        served=("served", "sum"),
        lost=("lost", "sum"),
        expired=("expired", "sum"),
        revenue=("revenue", "sum"),
        production_cost=("production_cost", "sum"),
        gross_profit=("gross_profit", "sum"),
        discount_loss=("discount_loss", "sum"),
    )
    summary["gross_margin_pct"] = 100 * summary["gross_profit"] / summary["revenue"]
    summary["service_level_pct"] = 100 * summary["served"] / summary["demand"]
    actual = summary[summary["variant"].eq("actual_state")].iloc[0]
    for column in ["revenue", "production_cost", "gross_profit", "served", "lost", "expired"]:
        summary[f"{column}_delta_vs_actual"] = summary[column] - actual[column]
    summary["gross_profit_delta_vs_actual_pct"] = (
        100 * summary["gross_profit_delta_vs_actual"] / actual["gross_profit"]
    )

    aggregation = {
        "demand": ("demand", "sum"),
        "production": ("production", "sum"),
        "sold_fresh": ("sold_fresh", "sum"),
        "sold_yesterday": ("sold_yesterday", "sum"),
        "served": ("served", "sum"),
        "lost": ("lost", "sum"),
        "expired": ("expired", "sum"),
        "revenue": ("revenue", "sum"),
        "production_cost": ("production_cost", "sum"),
        "gross_profit": ("gross_profit", "sum"),
        "discount_loss": ("discount_loss", "sum"),
    }
    product_summary = rows.groupby(
        ["variant", "workbook_category", "product_id", "workbook_product_name"], as_index=False
    ).agg(
        **aggregation
    )
    actual_product = product_summary[product_summary["variant"].eq("actual_state")][
        ["product_id", "gross_profit"]
    ].rename(columns={"gross_profit": "actual_gross_profit"})
    product_summary = product_summary.merge(actual_product, on="product_id", how="left")
    product_summary["gross_profit_delta_vs_actual"] = (
        product_summary["gross_profit"] - product_summary["actual_gross_profit"]
    )
    category_summary = rows.groupby(["variant", "workbook_category"], as_index=False).agg(**aggregation)
    category_actual = category_summary[category_summary["variant"].eq("actual_state")][
        ["workbook_category", "gross_profit"]
    ].rename(columns={"gross_profit": "actual_gross_profit"})
    category_summary = category_summary.merge(category_actual, on="workbook_category", how="left")
    category_summary["gross_profit_delta_vs_actual"] = (
        category_summary["gross_profit"] - category_summary["actual_gross_profit"]
    )
    category_summary["service_level_pct"] = 100 * category_summary["served"] / category_summary["demand"]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(OUTPUT / "priced_daily_rows.parquet", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False, encoding="utf-8-sig")
    product_summary.to_csv(OUTPUT / "by_product.csv", index=False, encoding="utf-8-sig")
    category_summary.to_csv(OUTPUT / "by_category.csv", index=False, encoding="utf-8-sig")
    print(
        f"mapped_products={rows['product_id'].nunique()} "
        f"demand_coverage={rows[rows['variant'].eq('actual_state')]['demand'].sum() / simulation[simulation['variant'].eq('actual_state')]['demand'].sum():.4%}"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

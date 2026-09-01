"""Kazan two-day FIFO economics for direct bakery-to-SKU allocation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluate_clean_kazan_two_day_economics import add_global_segments, aggregate
from simulate_two_day_economics import simulate_group


ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "reports/rolling_actual_state_comparison_20260826/rows.parquet"
DIRECT = ROOT / "reports/direct_bakery_sku_allocation_20260827/predictions.parquet"
BAKERY_MAP = ROOT / "reports/bakery_day_model_bias_by_bakery.csv"
MAPPING = ROOT / "reports/markup_price_mapping_20260826/mapped_products.csv"
OUTPUT = ROOT / "reports/direct_bakery_sku_allocation_20260827/economics"
DISCOUNT = 0.30
VARIANTS = {
    "actual_state": None,
    "current": "incumbent_sku_forecast",
    "previous_predictive_p50": "p50_predictive",
    "previous_predictive_floor": "p50_simple_floor",
    "direct_same_volume": "direct_forecast",
    "direct_plus_2pct": "direct_plus_2pct",
    "direct_with_p50_volume": "direct_with_p50_volume",
}


def main() -> None:
    source = pd.read_parquet(ROWS)
    direct = pd.read_parquet(DIRECT)[
        ["date", "bakery_id", "product_id", "direct_forecast"]
    ]
    for frame in (source, direct):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["product_id"] = pd.to_numeric(frame["product_id"]).astype("int64")
    source = source.merge(
        direct,
        on=["date", "bakery_id", "product_id"],
        how="inner",
        validate="one_to_one",
    )
    source["direct_plus_2pct"] = 1.02 * source["direct_forecast"]
    source["direct_with_p50_volume"] = source["direct_forecast"] * source["p50_factor"]

    bakeries = pd.read_csv(BAKERY_MAP)
    kazan_ids = set(
        bakeries.loc[bakeries["city"].eq("Казань"), "bakery_id"].astype(int)
    )
    source = add_global_segments(source[source["bakery_id"].isin(kazan_ids)].copy())
    source = source[source["segment_length"].ge(2)].copy()

    mapping = pd.read_csv(MAPPING, encoding="utf-8-sig")
    mapping = mapping[mapping["valid_economics"].astype(bool)].copy()
    mapping["product_id"] = mapping["product_id"].astype(int)
    mapping = mapping.sort_values("unit_price").drop_duplicates(
        "product_id", keep="last"
    )
    source = source[source["product_id"].isin(set(mapping["product_id"]))].copy()

    parts = []
    for variant, column in VARIANTS.items():
        simulated = pd.concat(
            [
                simulate_group(group, column)
                for _, group in source.groupby(
                    ["bakery_id", "product_id"], sort=False
                )
            ],
            ignore_index=True,
        )
        simulated["variant"] = variant
        parts.append(simulated)
    rows = pd.concat(parts, ignore_index=True)
    rows = rows.merge(
        source[
            [
                "date",
                "bakery_id",
                "product_id",
                "global_segment",
                "segment_start",
                "segment_end",
            ]
        ],
        on=["date", "bakery_id", "product_id"],
        how="left",
        validate="many_to_one",
    )
    rows["terminal_carry"] = rows["ending_carry"].where(
        rows["date"].eq(rows["segment_end"]), 0.0
    )
    rows = rows.merge(
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
    rows["revenue"] = (
        rows["sold_fresh"] * rows["unit_price"]
        + rows["sold_yesterday"] * rows["unit_price"] * (1 - DISCOUNT)
    )
    rows["production_cost"] = rows["production"] * rows["unit_cost"]
    rows["gross_profit"] = rows["revenue"] - rows["production_cost"]
    rows["discount_loss"] = (
        rows["sold_yesterday"] * rows["unit_price"] * DISCOUNT
    )

    summary = aggregate(rows, ["variant"])
    actual_gp = float(
        summary.loc[summary["variant"].eq("actual_state"), "gross_profit"].iloc[0]
    )
    summary["gross_profit_delta_vs_actual"] = summary["gross_profit"] - actual_gp
    summary["gross_profit_delta_vs_actual_pct"] = (
        100 * summary["gross_profit_delta_vs_actual"] / actual_gp
    )
    by_category = aggregate(rows, ["variant", "workbook_category"])

    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(OUTPUT / "daily_rows.parquet", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False, encoding="utf-8-sig")
    by_category.to_csv(
        OUTPUT / "by_category.csv", index=False, encoding="utf-8-sig"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

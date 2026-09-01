"""Clean Kazan-only two-day FIFO economics with inventory provenance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from simulate_two_day_economics import VARIANTS, simulate_group

ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "reports/rolling_actual_state_comparison_20260826/rows.parquet"
BAKERY_MAP = ROOT / "reports/bakery_day_model_bias_by_bakery.csv"
MAPPING = ROOT / "reports/markup_price_mapping_20260826/mapped_products.csv"
OUTPUT = ROOT / "reports/clean_kazan_two_day_economics_20260826"
DISCOUNT = 0.30


def add_global_segments(rows: pd.DataFrame) -> pd.DataFrame:
    dates = pd.Series(sorted(rows["date"].unique()))
    segment = dates.diff().ne(pd.Timedelta(days=1)).cumsum()
    lookup = pd.DataFrame({"date": dates, "global_segment": segment})
    lengths = lookup.groupby("global_segment")["date"].transform("size")
    lookup["segment_length"] = lengths
    lookup["segment_start"] = lookup.groupby("global_segment")["date"].transform("min")
    lookup["segment_end"] = lookup.groupby("global_segment")["date"].transform("max")
    return rows.merge(lookup, on="date", how="left", validate="many_to_one")


def aggregate(rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    result = rows.groupby(keys, as_index=False).agg(
        demand=("demand", "sum"),
        production=("production", "sum"),
        sold_fresh=("sold_fresh", "sum"),
        sold_yesterday=("sold_yesterday", "sum"),
        served=("served", "sum"),
        lost=("lost", "sum"),
        expired_initial_stock=("expired_initial_stock", "sum"),
        expired_strategy_stock=("expired_strategy_stock", "sum"),
        terminal_carry=("terminal_carry", "sum"),
        revenue=("revenue", "sum"),
        production_cost=("production_cost", "sum"),
        gross_profit=("gross_profit", "sum"),
        discount_loss=("discount_loss", "sum"),
    )
    result["service_level_pct"] = 100 * result["served"] / result["demand"]
    return result


def main() -> None:
    source = pd.read_parquet(ROWS)
    source["date"] = pd.to_datetime(source["date"]).dt.normalize()
    bakeries = pd.read_csv(BAKERY_MAP)
    kazan_ids = set(bakeries.loc[bakeries["city"].eq("Казань"), "bakery_id"].astype(int))
    source = source[source["bakery_id"].isin(kazan_ids)].copy()
    source = add_global_segments(source)
    source = source[source["segment_length"].ge(2)].copy()

    simulations = []
    for variant, plan_column in VARIANTS.items():
        parts = [
            simulate_group(group, plan_column)
            for _, group in source.groupby(["bakery_id", "product_id"], sort=False)
        ]
        frame = pd.concat(parts, ignore_index=True)
        frame["variant"] = variant
        simulations.append(frame)
    rows = pd.concat(simulations, ignore_index=True)
    rows = rows.merge(
        source[["date", "bakery_id", "product_id", "global_segment", "segment_start", "segment_end"]],
        on=["date", "bakery_id", "product_id"],
        how="left",
        validate="many_to_one",
    )
    rows["terminal_carry"] = rows["ending_carry"].where(rows["date"].eq(rows["segment_end"]), 0.0)

    mapping = pd.read_csv(MAPPING, encoding="utf-8-sig")
    mapping = mapping[mapping["valid_economics"].astype(bool)].copy()
    mapping["product_id"] = mapping["product_id"].astype(int)
    mapping = mapping.sort_values("unit_price").drop_duplicates("product_id", keep="last")
    rows = rows.merge(
        mapping[["product_id", "workbook_product_name", "workbook_category", "unit_price", "unit_cost"]],
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
    rows["discount_loss"] = rows["sold_yesterday"] * rows["unit_price"] * DISCOUNT

    summary = aggregate(rows, ["variant"])
    actual_gp = float(summary.loc[summary["variant"].eq("actual_state"), "gross_profit"].iloc[0])
    summary["gross_profit_delta_vs_actual"] = summary["gross_profit"] - actual_gp
    summary["gross_profit_delta_vs_actual_pct"] = 100 * summary["gross_profit_delta_vs_actual"] / actual_gp
    by_segment = aggregate(rows, ["variant", "global_segment", "segment_start", "segment_end"])
    by_category = aggregate(rows, ["variant", "workbook_category"])
    by_product = aggregate(rows, ["variant", "workbook_category", "product_id", "workbook_product_name"])

    for frame, join_keys in [
        (by_category, ["workbook_category"]),
        (by_product, ["product_id"]),
    ]:
        actual = frame[frame["variant"].eq("actual_state")][join_keys + ["gross_profit"]].rename(
            columns={"gross_profit": "actual_gross_profit"}
        )
        merged = frame.merge(actual, on=join_keys, how="left")
        merged["gross_profit_delta_vs_actual"] = merged["gross_profit"] - merged["actual_gross_profit"]
        if frame is by_category:
            by_category = merged
        else:
            by_product = merged

    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(OUTPUT / "daily_rows.parquet", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False, encoding="utf-8-sig")
    by_segment.to_csv(OUTPUT / "by_segment.csv", index=False, encoding="utf-8-sig")
    by_category.to_csv(OUTPUT / "by_category.csv", index=False, encoding="utf-8-sig")
    by_product.to_csv(OUTPUT / "by_product.csv", index=False, encoding="utf-8-sig")
    print(f"kazan_bakeries={source['bakery_id'].nunique()} rows={len(source)} products={rows['product_id'].nunique()}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

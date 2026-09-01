"""Kazan FIFO economics for weighted Direct soft-normalization candidates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluate_clean_kazan_two_day_economics import add_global_segments, aggregate
from simulate_two_day_economics import simulate_group


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "reports/weighted_direct_soft_normalization_20260827/rows.parquet"
BASE = ROOT / "reports/rolling_actual_state_comparison_20260826/rows.parquet"
BAKERY_MAP = ROOT / "reports/bakery_day_model_bias_by_bakery.csv"
MAPPING = ROOT / "reports/markup_price_mapping_20260826/mapped_products.csv"
OUTPUT = ROOT / "reports/weighted_direct_soft_normalization_20260827/economics"
KEYS = ["date", "bakery_id", "product_id"]
EVALUATION_FOLDS = {"2026-07-27", "2026-08-10", "2026-08-17"}
DISCOUNT = 0.30
VARIANTS = {
    "actual_state": None,
    "current": "incumbent_sku_forecast",
    "direct_p50": "direct_p50",
    "previous_final": "direct_uplift_adaptive_floor",
    "original_alpha_025_floor": "original_alpha_025_floor",
    "original_alpha_050_floor": "original_alpha_050_floor",
}


def main() -> None:
    candidates = pd.read_parquet(CANDIDATES)
    base = pd.read_parquet(BASE)
    for frame in (candidates, base):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["product_id"] = pd.to_numeric(frame["product_id"]).astype("int64")
    candidates = candidates[candidates["rolling_fold"].isin(EVALUATION_FOLDS)]
    source = candidates.drop(columns=["incumbent_sku_forecast"], errors="ignore").merge(
        base[
            KEYS
            + [
                "incumbent_sku_forecast",
                "produced",
                "received",
                "sent",
                "opening_stock",
            ]
        ],
        on=KEYS,
        how="inner",
        validate="many_to_one",
    )
    source["demand"] = source["scenario_demand"]
    bakeries = pd.read_csv(BAKERY_MAP)
    kazan_ids = set(
        bakeries.loc[bakeries["city"].eq("Казань"), "bakery_id"].astype(int)
    )
    source = source[source["bakery_id"].isin(kazan_ids)].copy()
    mapping = pd.read_csv(MAPPING, encoding="utf-8-sig")
    mapping = mapping[mapping["valid_economics"].astype(bool)].copy()
    mapping["product_id"] = mapping["product_id"].astype(int)
    mapping = mapping.sort_values("unit_price").drop_duplicates(
        "product_id", keep="last"
    )
    source = source[source["product_id"].isin(set(mapping["product_id"]))].copy()

    outputs = []
    for scenario, scenario_rows in source.groupby("scenario"):
        scenario_rows = add_global_segments(scenario_rows.copy())
        scenario_rows = scenario_rows[scenario_rows["segment_length"].ge(2)]
        parts = []
        for variant, column in VARIANTS.items():
            simulated = pd.concat(
                [
                    simulate_group(group, column)
                    for _, group in scenario_rows.groupby(
                        ["bakery_id", "product_id"], sort=False
                    )
                ],
                ignore_index=True,
            )
            simulated["variant"] = variant
            parts.append(simulated)
        rows = pd.concat(parts, ignore_index=True)
        rows["scenario"] = scenario
        rows = rows.merge(
            scenario_rows[KEYS + ["segment_end"]],
            on=KEYS,
            how="left",
            validate="many_to_one",
        )
        rows["terminal_carry"] = rows["ending_carry"].where(
            rows["date"].eq(rows["segment_end"]), 0.0
        )
        rows = rows.merge(
            mapping[["product_id", "unit_price", "unit_cost"]],
            on="product_id",
            how="inner",
            validate="many_to_one",
        )
        rows["revenue"] = rows["sold_fresh"] * rows["unit_price"] + rows[
            "sold_yesterday"
        ] * rows["unit_price"] * (1 - DISCOUNT)
        rows["production_cost"] = rows["production"] * rows["unit_cost"]
        rows["gross_profit"] = rows["revenue"] - rows["production_cost"]
        rows["discount_loss"] = rows["sold_yesterday"] * rows["unit_price"] * DISCOUNT
        outputs.append(rows)
    daily = pd.concat(outputs, ignore_index=True)
    summary = aggregate(daily, ["scenario", "variant"])
    actual = summary[summary["variant"].eq("actual_state")][
        ["scenario", "gross_profit"]
    ].rename(columns={"gross_profit": "actual_gross_profit"})
    summary = summary.merge(actual, on="scenario", validate="many_to_one")
    summary["gross_profit_delta_vs_actual"] = (
        summary["gross_profit"] - summary["actual_gross_profit"]
    )
    summary["gross_profit_delta_vs_actual_pct"] = (
        100 * summary["gross_profit_delta_vs_actual"] / summary["actual_gross_profit"]
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(OUTPUT / "daily_rows.parquet", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

"""Economic and tail robustness for rolling direct allocation candidates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluate_clean_kazan_two_day_economics import add_global_segments, aggregate
from simulate_two_day_economics import simulate_group


ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "reports/rolling_direct_uplift_floor_20260827/rows.parquet"
BASE = ROOT / "reports/rolling_actual_state_comparison_20260826/rows.parquet"
BAKERY_MAP = ROOT / "reports/bakery_day_model_bias_by_bakery.csv"
MAPPING = ROOT / "reports/markup_price_mapping_20260826/mapped_products.csv"
OUTPUT = ROOT / "reports/rolling_direct_uplift_floor_20260827"
KEYS = ["date", "bakery_id", "product_id"]
EVALUATION_FOLDS = {"2026-07-27", "2026-08-10", "2026-08-17"}
DISCOUNT = 0.30
VARIANTS = {
    "actual_state": None,
    "current": "incumbent_sku_forecast",
    "direct_p50": "direct_p50",
    "direct_uplift_p50": "direct_uplift_p50",
    "direct_uplift_adaptive_floor": "direct_uplift_adaptive_floor",
}


def add_economics(rows: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    result = rows.merge(
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
    result["revenue"] = result["sold_fresh"] * result["unit_price"] + result[
        "sold_yesterday"
    ] * result["unit_price"] * (1 - DISCOUNT)
    result["production_cost"] = result["production"] * result["unit_cost"]
    result["gross_profit"] = result["revenue"] - result["production_cost"]
    result["discount_loss"] = result["sold_yesterday"] * result["unit_price"] * DISCOUNT
    return result


def economic_simulation(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bakeries = pd.read_csv(BAKERY_MAP)
    kazan_ids = set(
        bakeries.loc[bakeries["city"].eq("Казань"), "bakery_id"].astype(int)
    )
    rows = rows[rows["bakery_id"].isin(kazan_ids)].copy()
    mapping = pd.read_csv(MAPPING, encoding="utf-8-sig")
    mapping = mapping[mapping["valid_economics"].astype(bool)].copy()
    mapping["product_id"] = mapping["product_id"].astype(int)
    mapping = mapping.sort_values("unit_price").drop_duplicates(
        "product_id", keep="last"
    )
    rows = rows[rows["product_id"].isin(set(mapping["product_id"]))].copy()
    outputs = []
    for scenario, scenario_rows in rows.groupby("scenario"):
        scenario_rows = add_global_segments(scenario_rows.copy())
        scenario_rows = scenario_rows[scenario_rows["segment_length"].ge(2)].copy()
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
        simulated_rows = pd.concat(parts, ignore_index=True)
        simulated_rows["scenario"] = scenario
        simulated_rows = simulated_rows.merge(
            scenario_rows[KEYS + ["global_segment", "segment_end"]],
            on=KEYS,
            how="left",
            validate="many_to_one",
        )
        simulated_rows["terminal_carry"] = simulated_rows["ending_carry"].where(
            simulated_rows["date"].eq(simulated_rows["segment_end"]), 0.0
        )
        outputs.append(add_economics(simulated_rows, mapping))
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
    return daily, summary


def tail_audit(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calibrated = rows[rows["scenario"].eq("calibrated")].copy()
    calibrated["direct_error"] = (
        calibrated["direct_p50"] - calibrated["scenario_demand"]
    ).abs()
    calibrated["final_error"] = (
        calibrated["direct_uplift_adaptive_floor"] - calibrated["scenario_demand"]
    ).abs()
    calibrated["error_delta"] = calibrated["final_error"] - calibrated["direct_error"]
    bakery = calibrated.groupby("bakery_id", as_index=False).agg(
        direct_error=("direct_error", "sum"),
        final_error=("final_error", "sum"),
        error_delta=("error_delta", "sum"),
        demand=("scenario_demand", "sum"),
    )
    category = calibrated.groupby("category", as_index=False).agg(
        direct_error=("direct_error", "sum"),
        final_error=("final_error", "sum"),
        error_delta=("error_delta", "sum"),
        demand=("scenario_demand", "sum"),
    )
    product = calibrated.groupby("product_id", as_index=False).agg(
        direct_error=("direct_error", "sum"),
        final_error=("final_error", "sum"),
        error_delta=("error_delta", "sum"),
        demand=("scenario_demand", "sum"),
    )
    for frame in (bakery, category, product):
        frame["improved"] = frame["error_delta"].lt(0)
        frame["direct_wape_pct"] = 100 * frame["direct_error"] / frame["demand"]
        frame["final_wape_pct"] = 100 * frame["final_error"] / frame["demand"]
    return bakery, category, product


def main() -> None:
    candidates = pd.read_parquet(ROWS)
    base = pd.read_parquet(BASE)
    for frame in (candidates, base):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["product_id"] = pd.to_numeric(frame["product_id"]).astype("int64")
    candidates = candidates[candidates["rolling_fold"].isin(EVALUATION_FOLDS)]
    base_columns = KEYS + [
        "incumbent_sku_forecast",
        "produced",
        "received",
        "sent",
        "opening_stock",
    ]
    source = candidates.drop(columns=["incumbent_sku_forecast"], errors="ignore").merge(
        base[base_columns], on=KEYS, how="inner", validate="many_to_one"
    )
    source["demand"] = source["scenario_demand"]
    daily, economics = economic_simulation(source)
    bakery, category, product = tail_audit(candidates)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(OUTPUT / "economic_daily_rows.parquet", index=False)
    economics.to_csv(OUTPUT / "economic_summary.csv", index=False, encoding="utf-8-sig")
    bakery.to_csv(OUTPUT / "tail_by_bakery.csv", index=False, encoding="utf-8-sig")
    category.to_csv(OUTPUT / "tail_by_category.csv", index=False, encoding="utf-8-sig")
    product.to_csv(OUTPUT / "tail_by_product.csv", index=False, encoding="utf-8-sig")
    print(economics.to_string(index=False))
    print(
        "tail_improvement_shares",
        {
            "bakery": float(bakery["improved"].mean()),
            "category": float(category["improved"].mean()),
            "product": float(product["improved"].mean()),
        },
    )


if __name__ == "__main__":
    main()

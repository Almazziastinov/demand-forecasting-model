"""Walk-forward SKU/category economic gate over the simple floor candidate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from simulate_two_day_economics import simulate_group

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROWS = ROOT / "reports/rolling_actual_state_comparison_20260826/rows.parquet"
CLEAN_ROWS = ROOT / "reports/clean_kazan_two_day_economics_20260826/daily_rows.parquet"
BAKERY_MAP = ROOT / "reports/bakery_day_model_bias_by_bakery.csv"
MAPPING = ROOT / "reports/markup_price_mapping_20260826/mapped_products.csv"
OUTPUT = ROOT / "reports/causal_economic_floor_gate_20260827"

DISCOUNT = 0.30
SKU_PRIOR_UNITS = 50.0
CATEGORY_PRIOR_UNITS = 200.0


def add_segments(rows: pd.DataFrame) -> pd.DataFrame:
    dates = pd.Series(sorted(rows["date"].unique()))
    lookup = pd.DataFrame({"date": dates})
    lookup["global_segment"] = dates.diff().ne(pd.Timedelta(days=1)).cumsum().astype(int)
    lookup["segment_start"] = lookup.groupby("global_segment")["date"].transform("min")
    lookup["segment_end"] = lookup.groupby("global_segment")["date"].transform("max")
    return rows.merge(lookup, on="date", how="left", validate="many_to_one")


def economic_frame(simulation: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    rows = simulation.merge(
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
    return rows


def build_gate(
    historical: pd.DataFrame,
    source_history: pd.DataFrame,
    mapping: pd.DataFrame,
    target_segment: int,
) -> pd.DataFrame:
    paired = historical[historical["variant"].isin(["p50_predictive", "p50_predictive_simple_floor"])].copy()
    profit = paired.groupby(["product_id", "variant"], as_index=False)["gross_profit"].sum().pivot(
        index="product_id", columns="variant", values="gross_profit"
    ).fillna(0.0)
    profit["profit_delta"] = (
        profit.get("p50_predictive_simple_floor", 0.0) - profit.get("p50_predictive", 0.0)
    )
    extra = source_history.assign(
        extra_units=(source_history["p50_simple_floor"] - source_history["p50_predictive"]).clip(lower=0.0)
    ).groupby("product_id", as_index=True)["extra_units"].sum()
    evidence = profit[["profit_delta"]].join(extra, how="outer").fillna(0.0).reset_index()
    evidence = evidence.merge(
        mapping[["product_id", "workbook_product_name", "workbook_category"]],
        on="product_id",
        how="right",
    ).fillna({"profit_delta": 0.0, "extra_units": 0.0})

    total_extra = float(evidence["extra_units"].sum())
    global_rate = float(evidence["profit_delta"].sum() / total_extra) if total_extra > 0 else 0.0
    category = evidence.groupby("workbook_category", as_index=False).agg(
        category_profit_delta=("profit_delta", "sum"),
        category_extra_units=("extra_units", "sum"),
    )
    category["category_rate"] = (
        category["category_profit_delta"] + CATEGORY_PRIOR_UNITS * global_rate
    ) / (category["category_extra_units"] + CATEGORY_PRIOR_UNITS)
    evidence = evidence.merge(category, on="workbook_category", how="left", validate="many_to_one")
    evidence["gate_score_per_extra_unit"] = (
        evidence["profit_delta"] + SKU_PRIOR_UNITS * evidence["category_rate"]
    ) / (evidence["extra_units"] + SKU_PRIOR_UNITS)
    evidence["use_floor"] = evidence["gate_score_per_extra_unit"].gt(0.0)
    evidence["target_segment"] = target_segment
    evidence["global_rate"] = global_rate
    return evidence


def summarize(rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    result = rows.groupby(keys, as_index=False).agg(
        demand=("demand", "sum"),
        production=("production", "sum"),
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
    source = pd.read_parquet(SOURCE_ROWS)
    source["date"] = pd.to_datetime(source["date"]).dt.normalize()
    bakeries = pd.read_csv(BAKERY_MAP)
    kazan_ids = set(bakeries.loc[bakeries["city"].eq("Казань"), "bakery_id"].astype(int))
    source = add_segments(source[source["bakery_id"].isin(kazan_ids)].copy())

    mapping = pd.read_csv(MAPPING, encoding="utf-8-sig")
    mapping = mapping[mapping["valid_economics"].astype(bool)].copy()
    mapping["product_id"] = mapping["product_id"].astype(int)
    mapping = mapping.sort_values("unit_price").drop_duplicates("product_id", keep="last")
    mapped_ids = set(mapping["product_id"])
    source = source[source["product_id"].isin(mapped_ids)].copy()

    clean = pd.read_parquet(CLEAN_ROWS)
    clean["date"] = pd.to_datetime(clean["date"]).dt.normalize()
    segments = sorted(source["global_segment"].unique())
    source["causal_economic_gate"] = source["p50_predictive"]
    gate_parts = []
    for segment in segments[1:]:
        history_segments = [value for value in segments if value < segment]
        gate = build_gate(
            clean[clean["global_segment"].isin(history_segments)],
            source[source["global_segment"].isin(history_segments)],
            mapping,
            int(segment),
        )
        gate_parts.append(gate)
        decision = gate.set_index("product_id")["use_floor"]
        mask = source["global_segment"].eq(segment)
        use_floor = source.loc[mask, "product_id"].map(decision).fillna(False).astype(bool)
        source.loc[mask, "causal_economic_gate"] = np.where(
            use_floor,
            source.loc[mask, "p50_simple_floor"],
            source.loc[mask, "p50_predictive"],
        )
    gates = pd.concat(gate_parts, ignore_index=True)

    simulated = pd.concat(
        [
            simulate_group(group, "causal_economic_gate")
            for _, group in source.groupby(["bakery_id", "product_id"], sort=False)
        ],
        ignore_index=True,
    )
    simulated["variant"] = "causal_economic_gate"
    simulated = simulated.merge(
        source[["date", "bakery_id", "product_id", "global_segment", "segment_start", "segment_end"]],
        on=["date", "bakery_id", "product_id"],
        how="left",
        validate="many_to_one",
    )
    simulated["terminal_carry"] = simulated["ending_carry"].where(
        simulated["date"].eq(simulated["segment_end"]), 0.0
    )
    hybrid = economic_frame(simulated, mapping)

    comparison = clean.copy()
    comparison = comparison[comparison["variant"].isin(
        ["actual_state", "current", "p50_predictive", "p50_predictive_simple_floor"]
    )]
    comparison = pd.concat([comparison, hybrid], ignore_index=True, sort=False)
    full_summary = summarize(comparison, ["variant"])
    evaluation = comparison[comparison["global_segment"].isin(segments[1:])]
    evaluation_summary = summarize(evaluation, ["variant"])
    actual_gp = float(evaluation_summary.loc[evaluation_summary["variant"].eq("actual_state"), "gross_profit"].iloc[0])
    evaluation_summary["gross_profit_delta_vs_actual"] = evaluation_summary["gross_profit"] - actual_gp
    evaluation_summary["gross_profit_delta_vs_actual_pct"] = (
        100 * evaluation_summary["gross_profit_delta_vs_actual"] / actual_gp
    )
    by_segment = summarize(comparison, ["variant", "global_segment", "segment_start", "segment_end"])

    OUTPUT.mkdir(parents=True, exist_ok=True)
    gates.to_csv(OUTPUT / "gate_decisions.csv", index=False, encoding="utf-8-sig")
    hybrid.to_parquet(OUTPUT / "hybrid_daily_rows.parquet", index=False)
    full_summary.to_csv(OUTPUT / "full_summary.csv", index=False, encoding="utf-8-sig")
    evaluation_summary.to_csv(OUTPUT / "walk_forward_summary.csv", index=False, encoding="utf-8-sig")
    by_segment.to_csv(OUTPUT / "by_segment.csv", index=False, encoding="utf-8-sig")
    print(
        gates.groupby("target_segment").agg(
            products=("product_id", "nunique"),
            floor_products=("use_floor", "sum"),
            global_rate=("global_rate", "first"),
        ).to_string()
    )
    print("\nWalk-forward evaluation (segments 2-4)")
    print(evaluation_summary.to_string(index=False))


if __name__ == "__main__":
    main()

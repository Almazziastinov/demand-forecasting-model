"""Stateful two-day replay with SKU multiples and shared bakery capacity."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/forecast_embedded"))

from app.db import get_client  # noqa: E402
from evaluate_clean_kazan_two_day_economics import add_global_segments  # noqa: E402


CANDIDATES = ROOT / "reports/alpha25_tail_cap_20260827/rows.parquet"
ACTUAL = ROOT / "reports/rolling_actual_state_comparison_20260826/rows.parquet"
BAKERY_MAP = ROOT / "reports/bakery_day_model_bias_by_bakery.csv"
MAPPING = ROOT / "reports/markup_price_mapping_20260826/mapped_products.csv"
OUTPUT = ROOT / "reports/alpha25_stateful_operations_no_capacity_20260831"
META_SNAPSHOT = (
    ROOT
    / "reports/alpha25_operational_constraints_20260827/baking_sku_meta_snapshot.csv"
)
OPS_SUMMARY = ROOT / "reports/alpha25_operational_constraints_20260827/summary.json"
KEYS = ["date", "bakery_id", "product_id"]
EVALUATION_FOLDS = {"2026-07-27", "2026-08-10", "2026-08-17"}
VARIANTS = {
    "actual_state": None,
    "current": "incumbent_sku_forecast",
    "direct_p50": "direct_p50",
    "previous_final": "direct_uplift_adaptive_floor",
    "alpha00_floor": "original_alpha_000_floor",
    "alpha25_floor": "original_alpha_025_floor",
    "alpha25_tail_cap": "alpha25_tail_capped",
    "alpha50_floor": "original_alpha_050_floor",
    "alpha75_floor": "original_alpha_075_floor",
    "alpha100_floor": "original_alpha_100_floor",
}
CORE_CATEGORIES = {
    "Выпечка сытная",
    "Выпечка сладкая",
    "Пироги сытные",
    "Пироги сладкие",
}
PIE_CATEGORIES = {"Пироги сытные", "Пироги сладкие"}
DISCOUNT = 0.30


def load_metadata() -> tuple[pd.DataFrame, int, str]:
    try:
        client = get_client()
        meta = client.query_df(
            """
            select product_id, kratnost
            from baking_sku_meta final
            where is_active = 1 and scope = 'base'
            """
        )
        capacity = client.query_df(
            """
            select bakers_count
            from baking_capacity_config final
            where is_active = 1 and bakery_id is null
            order by valid_from desc
            limit 1
            """
        )
        daily_cap = int(capacity.iloc[0, 0]) * 600
        metadata_source = "clickhouse_read_only"
    except Exception:
        if not META_SNAPSHOT.exists() or not OPS_SUMMARY.exists():
            raise
        meta = pd.read_csv(META_SNAPSHOT)
        daily_cap = int(
            json.loads(OPS_SUMMARY.read_text(encoding="utf-8"))["daily_core_cap"]
        )
        metadata_source = "local_snapshot_20260827"
    meta["product_id"] = pd.to_numeric(meta["product_id"]).astype("int64")
    meta["kratnost"] = (
        pd.to_numeric(meta["kratnost"]).fillna(1).clip(lower=1).astype(int)
    )
    return meta.drop_duplicates("product_id", keep="last"), daily_cap, metadata_source


def round_up(value: float, multiple: int) -> float:
    if value <= 0:
        return 0.0
    return float(math.ceil(value / multiple - 1e-12) * multiple)


def apply_capacity(day: pd.DataFrame, cap: int | None) -> pd.DataFrame:
    result = day.copy()
    result["capacity_reduction"] = 0.0
    result["capacity_binding"] = False
    if cap is None:
        return result
    for _, indexes in result.groupby("bakery_id", sort=False).groups.items():
        bakery = result.loc[indexes]
        core = bakery["has_meta"] & bakery["category"].isin(CORE_CATEGORIES)
        excess = float(bakery.loc[core, "production"].sum()) - cap
        if excess <= 1e-9:
            continue
        result.loc[indexes, "capacity_binding"] = True
        candidates = bakery.loc[core].copy()
        candidates["priority_group"] = candidates["is_core_sku"].astype(int)
        candidates["priority_ratio"] = candidates["broad_56_mean"] / candidates[
            "production"
        ].replace(0, np.nan)
        candidates = candidates.sort_values(
            ["priority_group", "priority_ratio", "broad_56_mean"],
            ascending=[True, True, True],
        )
        for index, row in candidates.iterrows():
            if excess <= 1e-9:
                break
            multiple = int(row["effective_multiple"])
            steps = min(
                int(result.at[index, "production"] // multiple),
                math.ceil(excess / multiple),
            )
            reduction = float(steps * multiple)
            result.at[index, "production"] -= reduction
            result.at[index, "capacity_reduction"] = reduction
            excess -= reduction
    return result


def simulate_variant(
    source: pd.DataFrame, variant: str, forecast_column: str | None, cap: int | None
) -> pd.DataFrame:
    output: list[pd.DataFrame] = []
    for _, segment in source.groupby("global_segment", sort=True):
        carry: dict[tuple[int, int], float] = {}
        segment_start = segment["date"].min()
        initial = segment[segment["date"].eq(segment_start)]
        carry.update(
            {
                (int(row.bakery_id), int(row.product_id)): max(
                    float(row.opening_stock), 0.0
                )
                for row in initial.itertuples(index=False)
            }
        )
        for date, raw_day in segment.groupby("date", sort=True):
            day = raw_day.copy()
            day["opening_carry"] = [
                carry.get((int(bakery), int(product)), 0.0)
                for bakery, product in zip(
                    day["bakery_id"], day["product_id"], strict=True
                )
            ]
            if forecast_column is None:
                day["forecast_target"] = np.nan
                day["production"] = day["produced"].clip(lower=0.0)
                day["capacity_reduction"] = 0.0
                day["capacity_binding"] = False
            else:
                day["forecast_target"] = day[forecast_column].clip(lower=0.0)
                net_need = (
                    day["forecast_target"]
                    + day["sent"]
                    - day["opening_carry"]
                    - day["received"]
                ).clip(lower=0.0)
                day["production"] = net_need
                covered = day["has_meta"]
                day.loc[covered, "production"] = [
                    round_up(value, int(multiple))
                    for value, multiple in zip(
                        net_need.loc[covered],
                        day.loc[covered, "effective_multiple"],
                        strict=True,
                    )
                ]
                day = apply_capacity(day, cap)

            old_for_transfer = np.minimum(day["opening_carry"], day["sent"])
            old_after_transfer = day["opening_carry"] - old_for_transfer
            fresh = day["production"] + day["received"]
            fresh_after_transfer = (fresh - (day["sent"] - old_for_transfer)).clip(
                lower=0.0
            )
            day["available_to_sell"] = old_after_transfer + fresh_after_transfer
            day["sold_yesterday"] = np.minimum(
                old_after_transfer, day["scenario_demand"]
            )
            remaining_demand = day["scenario_demand"] - day["sold_yesterday"]
            day["sold_fresh"] = np.minimum(fresh_after_transfer, remaining_demand)
            day["served"] = day["sold_yesterday"] + day["sold_fresh"]
            day["underbake"] = day["scenario_demand"] - day["served"]
            day["surplus"] = (day["available_to_sell"] - day["scenario_demand"]).clip(
                lower=0.0
            )
            day["expired"] = old_after_transfer - day["sold_yesterday"]
            day["ending_carry"] = fresh_after_transfer - day["sold_fresh"]
            day["variant"] = variant
            day["is_segment_end"] = date == segment["date"].max()
            output.append(day)
            carry = {
                (int(row.bakery_id), int(row.product_id)): max(
                    float(row.ending_carry), 0.0
                )
                for row in day.itertuples(index=False)
            }
    return pd.concat(output, ignore_index=True)


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    summary = rows.groupby(["scenario", "variant"], as_index=False).agg(
        demand=("scenario_demand", "sum"),
        forecast=("forecast_target", "sum"),
        production=("production", "sum"),
        available_to_sell=("available_to_sell", "sum"),
        served=("served", "sum"),
        surplus=("surplus", "sum"),
        underbake=("underbake", "sum"),
        expired=("expired", "sum"),
        capacity_reduction=("capacity_reduction", "sum"),
    )
    summary["imbalance"] = summary["surplus"] + summary["underbake"]
    summary["service_level_pct"] = 100 * summary["served"] / summary["demand"]
    terminal = (
        rows.loc[rows["is_segment_end"]]
        .groupby(["scenario", "variant"])["ending_carry"]
        .sum()
        .rename("terminal_carry")
        .reset_index()
    )
    return summary.merge(terminal, on=["scenario", "variant"], validate="one_to_one")


def add_economics(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = pd.read_csv(MAPPING, encoding="utf-8-sig")
    mapping = mapping[mapping["valid_economics"].astype(bool)].copy()
    mapping["product_id"] = mapping["product_id"].astype(int)
    mapping = mapping.sort_values("unit_price").drop_duplicates(
        "product_id", keep="last"
    )
    bakeries = pd.read_csv(BAKERY_MAP)
    kazan = set(bakeries.loc[bakeries["city"].eq("Казань"), "bakery_id"].astype(int))
    economic = rows[
        rows["bakery_id"].isin(kazan)
        & rows["product_id"].isin(set(mapping["product_id"]))
    ].merge(
        mapping[["product_id", "unit_price", "unit_cost"]],
        on="product_id",
        how="inner",
        validate="many_to_one",
    )
    economic["revenue"] = economic["sold_fresh"] * economic["unit_price"] + economic[
        "sold_yesterday"
    ] * economic["unit_price"] * (1 - DISCOUNT)
    economic["production_cost"] = economic["production"] * economic["unit_cost"]
    economic["gross_profit"] = economic["revenue"] - economic["production_cost"]
    summary = economic.groupby(["scenario", "variant"], as_index=False).agg(
        revenue=("revenue", "sum"),
        production_cost=("production_cost", "sum"),
        gross_profit=("gross_profit", "sum"),
        served=("served", "sum"),
        lost=("underbake", "sum"),
        expired=("expired", "sum"),
    )
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
    return economic, summary


def main() -> None:
    candidates = pd.read_parquet(CANDIDATES)
    actual = pd.read_parquet(ACTUAL)
    for frame in (candidates, actual):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    candidates = candidates[candidates["rolling_fold"].isin(EVALUATION_FOLDS)].copy()
    facts = actual[
        KEYS + ["produced", "received", "sent", "opening_stock"]
    ].drop_duplicates(KEYS)
    rows = candidates.merge(facts, on=KEYS, how="left", validate="many_to_one")
    for column in ["produced", "received", "sent", "opening_stock"]:
        rows[column] = rows[column].fillna(0.0).clip(lower=0.0)
    meta, configured_stress_cap, metadata_source = load_metadata()
    cap = None
    rows = rows.merge(meta, on="product_id", how="left", validate="many_to_one")
    rows["has_meta"] = rows["kratnost"].notna()
    rows["effective_multiple"] = rows["kratnost"].fillna(1).astype(int)
    rows.loc[rows["category"].isin(PIE_CATEGORIES), "effective_multiple"] = 4

    outputs = []
    for scenario, scenario_rows in rows.groupby("scenario", sort=False):
        scenario_rows = add_global_segments(scenario_rows)
        scenario_rows = scenario_rows[scenario_rows["segment_length"].ge(2)].copy()
        for variant, forecast_column in VARIANTS.items():
            outputs.append(
                simulate_variant(scenario_rows, variant, forecast_column, cap)
            )
    daily = pd.concat(outputs, ignore_index=True)
    summary = aggregate(daily)
    economic_rows, economic_summary = add_economics(daily)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(OUTPUT / "daily_rows.parquet", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False, encoding="utf-8-sig")
    economic_rows.to_parquet(OUTPUT / "economic_rows.parquet", index=False)
    economic_summary.to_csv(
        OUTPUT / "economic_summary.csv", index=False, encoding="utf-8-sig"
    )
    (OUTPUT / "metadata.json").write_text(
        json.dumps(
            {
                "daily_core_cap": cap,
                "configured_stress_cap_not_applied": configured_stress_cap,
                "capacity_constraint_applied": False,
                "metadata_source": metadata_source,
                "production_write": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print("\nECONOMICS\n")
    print(economic_summary.to_string(index=False))


if __name__ == "__main__":
    main()

"""Operational September comparison: Direct, +1% old-loss overlay, and same-weekday Avg2.

The comparison is retrospective and never writes to ClickHouse.  Every strategy
uses the published assortment, observed previous-day stock, and production
multiples from the same publisher contract.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "apps/forecast_embedded"))

from app.db import get_client  # noqa: E402


DETAIL = ROOT / ".codex_tmp/pilot_management_sep01_09/detail.csv"
SALE_TIMES = ROOT / ".codex_tmp/sep01_09_sale_times.csv"
PANEL = ROOT / ".codex_tmp/historical_clickhouse_panel/causal_panel.parquet"
SHADOW_ROOT = ROOT / ".codex_tmp/operational_compare_sep"
MAPPING = ROOT / "reports/markup_price_mapping_20260826/mapped_products.csv"
OUTPUT = ROOT / "reports/operational_direct_old_loss_avg2_sep01_09_20260914"
KEYS = ["date", "bakery_id", "product_id"]
DISCOUNT = 0.30


def round_up(value: float, multiple: int) -> int:
    if value <= 0:
        return 0
    return int(math.ceil(value / multiple - 1e-9) * multiple)


def load_multiples(product_ids: list[int]) -> tuple[dict[int, int], dict[tuple[int, int], int]]:
    client = get_client()
    meta = client.query_df(
        """
        select product_id, bakery_id, kratnost, scope
        from baking_sku_meta final
        where is_active = 1 and product_id in %(product_ids)s
        """,
        parameters={"product_ids": [f"{product_id:09d}" for product_id in product_ids]},
    )
    base: dict[int, int] = {}
    bakery: dict[tuple[int, int], int] = {}
    for row in meta.to_dict("records"):
        product_id = int(row["product_id"])
        multiple = max(int(row.get("kratnost") or 1), 1)
        if row.get("scope") == "bakery" and pd.notna(row.get("bakery_id")):
            bakery[(int(row["bakery_id"]), product_id)] = multiple
        else:
            base[product_id] = multiple
    return base, bakery


def add_corrected_end_day_demand(detail: pd.DataFrame) -> pd.DataFrame:
    result = detail.copy()
    last_sale = pd.to_datetime(result["last_sale_time"], errors="coerce", utc=True)
    bakery_end = pd.to_datetime(result["bakery_last_sale_time"], errors="coerce", utc=True)
    opening = result["date"].dt.tz_localize("Europe/Moscow").dt.tz_convert("UTC") + pd.Timedelta(hours=7.5)
    elapsed = (last_sale - opening).dt.total_seconds() / 3600
    remaining = (bakery_end - last_sale).dt.total_seconds().clip(lower=0) / 3600
    eligible = (
        result["eligible_lost_demand"].fillna(False).astype(bool)
        & elapsed.ge(2.0)
        & remaining.gt(0.0)
        & result["sold_qty"].gt(0.0)
    )
    raw_lost = result["sold_qty"] / elapsed * remaining
    cap = pd.concat([result["sold_qty"] * 1.5, pd.Series(15.0, index=result.index)], axis=1).max(axis=1)
    result["restored_lost"] = raw_lost.where(eligible, 0.0).clip(lower=0.0).clip(upper=cap)
    result["demand"] = result["sold_qty"] + result["restored_lost"]
    return result


def load_avg2(detail: pd.DataFrame) -> pd.DataFrame:
    panel = pd.read_parquet(PANEL, columns=KEYS + ["observed_sales_qty"])
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    in_period = detail[KEYS + ["sold_qty"]].rename(columns={"sold_qty": "observed_sales_qty"})
    # Earlier September observations are valid history for later September dates.
    panel = pd.concat([panel, in_period], ignore_index=True).drop_duplicates(KEYS, keep="last")
    needed = pd.concat(
        [
            detail[KEYS].assign(history_date=detail["date"] - pd.Timedelta(days=7), lag="lag7"),
            detail[KEYS].assign(history_date=detail["date"] - pd.Timedelta(days=14), lag="lag14"),
        ],
        ignore_index=True,
    )
    history = panel.rename(columns={"date": "history_date"})
    matched = needed.merge(history, on=["history_date", "bakery_id", "product_id"], how="left")
    matched["history_present"] = matched["observed_sales_qty"].notna()
    matched["observed_sales_qty"] = matched["observed_sales_qty"].fillna(0.0)
    values = matched.pivot_table(index=KEYS, columns="lag", values="observed_sales_qty", aggfunc="first").reset_index()
    presence = matched.pivot_table(index=KEYS, columns="lag", values="history_present", aggfunc="max").reset_index()
    presence = presence.rename(columns={"lag7": "lag7_present", "lag14": "lag14_present"})
    result = values.merge(presence, on=KEYS, validate="one_to_one")
    for column in ["lag7", "lag14"]:
        result[column] = result.get(column, 0.0).fillna(0.0)
    for column in ["lag7_present", "lag14_present"]:
        result[column] = result.get(column, False).fillna(False).astype(bool)
    result["avg2_target"] = (result["lag7"] + result["lag14"]) / 2.0
    result["avg2_history_points"] = result[["lag7_present", "lag14_present"]].sum(axis=1)
    return result


def load_old_loss_signal() -> pd.DataFrame:
    parts = []
    for path in sorted(SHADOW_ROOT.glob("*/shadow_rows.parquet")):
        part = pd.read_parquet(path, columns=KEYS + ["predictive_uplift"])
        part["date"] = pd.to_datetime(part["date"]).dt.normalize()
        parts.append(part)
    if not parts:
        raise RuntimeError(f"No historical shadow rows under {SHADOW_ROOT}")
    return pd.concat(parts, ignore_index=True).drop_duplicates(KEYS)


def simulate_variant(source: pd.DataFrame, variant: str, target_column: str) -> pd.DataFrame:
    rows = source.copy()
    rows["variant"] = variant
    net_need = (rows[target_column] - rows["opening_stock"]).clip(lower=0.0)
    rows["production"] = [round_up(value, multiple) for value, multiple in zip(net_need, rows["kratnost"])]
    old_after_transfer = (rows["opening_stock"] - rows["sent_qty"]).clip(lower=0.0)
    remaining_transfer = (rows["sent_qty"] - rows["opening_stock"]).clip(lower=0.0)
    fresh_available = (rows["production"] + rows["received_qty"] - remaining_transfer).clip(lower=0.0)
    rows["sold_old"] = np.minimum(old_after_transfer, rows["demand"])
    remaining_demand = (rows["demand"] - rows["sold_old"]).clip(lower=0.0)
    rows["sold_fresh"] = np.minimum(fresh_available, remaining_demand)
    rows["served"] = rows["sold_old"] + rows["sold_fresh"]
    rows["lost"] = (rows["demand"] - rows["served"]).clip(lower=0.0)
    rows["lost_case"] = rows["lost"].gt(1e-9)
    rows["expired_old"] = (old_after_transfer - rows["sold_old"]).clip(lower=0.0)
    rows["ending_fresh"] = (fresh_available - rows["sold_fresh"]).clip(lower=0.0)
    rows["ending_stock"] = rows["expired_old"] + rows["ending_fresh"]
    rows["revenue_model"] = rows["sold_fresh"] * rows["unit_price"] + rows["sold_old"] * rows["unit_price"] * (1 - DISCOUNT)
    rows["production_cost"] = rows["production"] * rows["unit_cost"]
    rows["gross_profit"] = rows["revenue_model"] - rows["production_cost"]
    return rows


def build_actual_state(source: pd.DataFrame) -> pd.DataFrame:
    rows = source.copy()
    rows["variant"] = "actual_state"
    rows["target"] = rows["available_to_sell_qty"].fillna(rows["produced_qty"])
    rows["production"] = rows["produced_qty"].clip(lower=0.0)
    rows["served"] = rows["sold_qty"].clip(lower=0.0)
    rows["sold_old"] = 0.0
    rows["sold_fresh"] = rows["served"]
    rows["lost"] = (rows["demand"] - rows["served"]).clip(lower=0.0)
    rows["lost_case"] = rows["lost"].gt(1e-9)
    rows["ending_stock"] = rows["closing_stock_qty"].fillna(0.0).clip(lower=0.0)
    rows["expired_old"] = rows["written_off_qty"].fillna(0.0).clip(lower=0.0)
    rows["ending_fresh"] = rows["ending_stock"]
    rows["revenue_model"] = rows["revenue"].fillna(rows["served"] * rows["unit_price"])
    rows["production_cost"] = rows["production"] * rows["unit_cost"]
    rows["gross_profit"] = rows["revenue_model"] - rows["production_cost"]
    return rows


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    result = rows.groupby("variant", as_index=False).agg(
        target=("target", "sum"),
        production=("production", "sum"),
        demand=("demand", "sum"),
        served=("served", "sum"),
        lost=("lost", "sum"),
        lost_cases=("lost_case", "sum"),
        ending_stock=("ending_stock", "sum"),
        expired_old=("expired_old", "sum"),
        gross_profit=("gross_profit", "sum"),
    )
    result["service_level_pct"] = 100 * result["served"] / result["demand"]
    base = result[result["variant"].eq("direct")].iloc[0]
    for metric in ["production", "lost", "lost_cases", "ending_stock", "gross_profit"]:
        result[f"{metric}_delta_vs_direct"] = result[metric] - base[metric]
    return result


def main() -> None:
    detail = pd.read_csv(DETAIL, encoding="utf-8-sig", low_memory=False)
    detail["date"] = pd.to_datetime(detail["business_date"]).dt.normalize()
    detail = detail.drop_duplicates(KEYS)
    sale_times = pd.read_csv(SALE_TIMES)
    sale_times["date"] = pd.to_datetime(sale_times["date"]).dt.normalize()
    detail = detail.merge(sale_times, on=KEYS, how="left", validate="one_to_one")
    numeric = [
        "forecast_qty", "sold_qty", "produced_qty", "received_qty", "sent_qty",
        "issued_yesterday_stock", "issued_plan_qty", "price", "available_to_sell_qty",
        "closing_stock_qty", "revenue",
    ]
    for column in numeric:
        detail[column] = pd.to_numeric(detail[column], errors="coerce")
    detail["written_off_qty"] = 0.0
    # The exact stock-aware publisher contract is only observable on these rows.
    detail = detail.dropna(subset=["forecast_qty", "issued_yesterday_stock", "issued_plan_qty", "sold_qty"]).copy()
    detail["opening_stock"] = detail["issued_yesterday_stock"].clip(lower=0.0)
    detail["received_qty"] = detail["received_qty"].fillna(0.0).clip(lower=0.0)
    detail["sent_qty"] = detail["sent_qty"].fillna(0.0).clip(lower=0.0)
    detail = add_corrected_end_day_demand(detail)

    avg2 = load_avg2(detail)
    detail = detail.merge(avg2, on=KEYS, how="left", validate="one_to_one")
    old_loss = load_old_loss_signal()
    detail = detail.merge(old_loss, on=KEYS, how="left", validate="one_to_one")
    detail["predictive_uplift"] = detail["predictive_uplift"].fillna(0.0).clip(lower=0.0)

    groups = [detail["date"], detail["bakery_id"]]
    direct_total = detail["forecast_qty"].groupby(groups).transform("sum")
    signal_total = detail["predictive_uplift"].groupby(groups).transform("sum")
    detail["old_loss_plus1_target"] = detail["forecast_qty"] + detail["predictive_uplift"] * (
        0.01 * direct_total / signal_total.replace(0.0, np.nan)
    ).fillna(0.0)
    detail["avg2_cold_start_target"] = detail["avg2_target"].where(
        detail["avg2_history_points"].gt(0), detail["forecast_qty"]
    )

    base_multiple, bakery_multiple = load_multiples(sorted(detail["product_id"].astype(int).unique()))
    detail["kratnost"] = [
        bakery_multiple.get((int(bakery_id), int(product_id)), base_multiple.get(int(product_id), 1))
        for bakery_id, product_id in zip(detail["bakery_id"], detail["product_id"])
    ]

    mapping = pd.read_csv(MAPPING, encoding="utf-8-sig")
    mapping = mapping[mapping["valid_economics"].fillna(False).astype(bool)].copy()
    mapping["product_id"] = mapping["product_id"].astype(int)
    mapping = mapping.sort_values("unit_price").drop_duplicates("product_id", keep="last")
    detail = detail.merge(mapping[["product_id", "unit_price", "unit_cost"]], on="product_id", how="inner", validate="many_to_one")
    detail["unit_price"] = detail["price"].where(detail["price"].gt(0)).fillna(detail["unit_price"])

    variants = {
        "direct": "forecast_qty",
        "old_loss_plus1": "old_loss_plus1_target",
        "avg2_strict": "avg2_target",
        "avg2_cold_start": "avg2_cold_start_target",
    }
    simulations = []
    simulations.append(build_actual_state(detail))
    for variant, column in variants.items():
        simulated = simulate_variant(detail, variant, column)
        simulated["target"] = simulated[column]
        simulations.append(simulated)
    rows = pd.concat(simulations, ignore_index=True)
    summary = summarize(rows)
    daily = rows.groupby(["date", "variant"], as_index=False).agg(
        target=("target", "sum"), production=("production", "sum"), demand=("demand", "sum"),
        served=("served", "sum"), lost=("lost", "sum"), lost_cases=("lost_case", "sum"),
        ending_stock=("ending_stock", "sum"), gross_profit=("gross_profit", "sum"),
    )
    coverage = pd.DataFrame(
        {
            "metric": [
                "rows", "both_history_points", "one_history_point", "zero_history_points",
                "shadow_signal_rows", "direct_plan_exact_match_rows", "direct_plan_abs_error_sum",
            ],
            "value": [
                len(detail),
                int(detail["avg2_history_points"].eq(2).sum()),
                int(detail["avg2_history_points"].eq(1).sum()),
                int(detail["avg2_history_points"].eq(0).sum()),
                int(detail["predictive_uplift"].gt(0).sum()),
                int(
                    rows.loc[rows["variant"].eq("direct"), "production"]
                    .reset_index(drop=True)
                    .eq(detail["issued_plan_qty"].reset_index(drop=True))
                    .sum()
                ),
                float(
                    (
                        rows.loc[rows["variant"].eq("direct"), "production"].reset_index(drop=True)
                        - detail["issued_plan_qty"].reset_index(drop=True)
                    ).abs().sum()
                ),
            ],
        }
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT / "summary.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(OUTPUT / "daily.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(OUTPUT / "coverage.csv", index=False, encoding="utf-8-sig")
    rows.to_parquet(OUTPUT / "rows.parquet", index=False)
    print(summary.to_string(index=False))
    print("\nCoverage")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()

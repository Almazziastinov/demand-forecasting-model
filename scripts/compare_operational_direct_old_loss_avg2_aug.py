"""Operational Aug 24-31 comparison on rows with an observed published stock contract."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_operational_direct_old_loss_avg2_sep import (  # noqa: E402
    KEYS,
    build_actual_state,
    load_avg2,
    load_multiples,
    simulate_variant,
    summarize,
)
from scripts.evaluate_corrected_direct_ablation import CURRENT, build_plan  # noqa: E402
from scripts.evaluate_weighted_loss_direct import SOURCE  # noqa: E402


DETAIL = ROOT / ".codex_tmp/pilot_recalc_report/detail.csv"
OUTPUT = ROOT / "reports/operational_direct_old_loss_avg2_aug24_31_20260914"


def main() -> None:
    source = pd.read_parquet(SOURCE / "evaluation_input.parquet")
    source["date"] = pd.to_datetime(source["date"]).dt.normalize()
    source = source.rename(
        columns={
            "plan_1": "forecast_qty",
            "observed_sales_qty": "sold_qty",
            "release_qty": "produced_qty",
            "incoming_move_qty": "received_qty",
            "outgoing_move_qty": "sent_qty",
            "written_off_qty": "written_off_qty_source",
            "observed_sales_amount": "revenue",
        }
    )
    published = pd.read_csv(DETAIL, encoding="utf-8-sig", low_memory=False)
    published["date"] = pd.to_datetime(published["business_date"]).dt.normalize()
    published = published[
        published["date"].between(pd.Timestamp("2026-08-24"), pd.Timestamp("2026-08-31"))
    ].drop_duplicates(KEYS)
    published = published.dropna(subset=["issued_yesterday_stock", "issued_plan_qty"])
    published = published[
        KEYS + ["issued_yesterday_stock", "issued_plan_qty", "available_to_sell_qty", "closing_stock_qty"]
    ]
    detail = source.merge(published, on=KEYS, how="inner", validate="one_to_one")
    detail["opening_stock"] = pd.to_numeric(detail["issued_yesterday_stock"], errors="coerce").fillna(0.0).clip(lower=0.0)
    detail["written_off_qty"] = detail["written_off_qty_source"].fillna(0.0).clip(lower=0.0)

    avg2 = load_avg2(detail)
    detail = detail.merge(avg2, on=KEYS, how="left", validate="one_to_one")
    detail["avg2_cold_start_target"] = detail["avg2_target"].where(
        detail["avg2_history_points"].gt(0), detail["forecast_qty"]
    )

    no_loss = build_plan(CURRENT, CURRENT, use_uplift=False, use_floor=False).rename(
        columns={"selected_sku_forecast": "no_loss_target"}
    )
    detail = detail.merge(no_loss, on=KEYS, how="left", validate="one_to_one")
    detail["no_loss_target"] = detail["no_loss_target"].fillna(detail["forecast_qty"])
    groups = [detail["date"], detail["bakery_id"]]
    direct_total = detail["forecast_qty"].groupby(groups).transform("sum")
    no_loss_total = detail["no_loss_target"].groupby(groups).transform("sum")
    no_loss_neutral = detail["no_loss_target"] * (
        direct_total / no_loss_total.replace(0.0, np.nan)
    ).fillna(1.0)
    signal = (detail["forecast_qty"] - no_loss_neutral).clip(lower=0.0)
    signal_total = signal.groupby(groups).transform("sum")
    detail["old_loss_plus1_target"] = detail["forecast_qty"] + signal * (
        0.01 * direct_total / signal_total.replace(0.0, np.nan)
    ).fillna(0.0)

    base_multiple, bakery_multiple = load_multiples(sorted(detail["product_id"].astype(int).unique()))
    detail["kratnost"] = [
        bakery_multiple.get((int(bakery_id), int(product_id)), base_multiple.get(int(product_id), 1))
        for bakery_id, product_id in zip(detail["bakery_id"], detail["product_id"])
    ]
    detail["price"] = detail["unit_price"]

    simulations = [build_actual_state(detail)]
    variants = {
        "direct": "forecast_qty",
        "old_loss_plus1": "old_loss_plus1_target",
        "avg2_strict": "avg2_target",
        "avg2_cold_start": "avg2_cold_start_target",
    }
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
            "metric": ["rows", "bakeries", "both_history_points", "one_history_point", "zero_history_points"],
            "value": [
                len(detail), detail["bakery_id"].nunique(), int(detail["avg2_history_points"].eq(2).sum()),
                int(detail["avg2_history_points"].eq(1).sum()), int(detail["avg2_history_points"].eq(0).sum()),
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

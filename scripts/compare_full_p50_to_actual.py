"""Compare full-period loss-0.75 P50 with the observed factual state."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = (
    ROOT
    / "reports/historical_daily_retraining_gate_loss075_full_20260910/predictions.parquet"
)
PANEL = ROOT / ".codex_tmp/historical_clickhouse_panel/causal_panel.parquet"
OUTPUT = ROOT / "reports/full_p50_loss075_vs_actual_20260910"
KEYS = ["date", "bakery_id", "product_id"]


def summarize(
    rows: pd.DataFrame, variant: str, plan: pd.Series, served: pd.Series
) -> dict[str, float | str]:
    demand = rows["demand"].sum()
    over = np.maximum(plan - rows["demand"], 0).sum()
    under = (rows["demand"] - served).clip(lower=0).sum()
    return {
        "variant": variant,
        "plan_or_available": float(plan.sum()),
        "served": float(served.sum()),
        "over": float(over),
        "under": float(under),
        "imbalance": float(over + under),
        "service_pct": float(100 * served.sum() / demand),
        "demand": float(demand),
    }


def main() -> None:
    rows = pd.read_parquet(PREDICTIONS)
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    facts = pd.read_parquet(
        PANEL,
        columns=KEYS
        + ["release_qty", "incoming_move_qty", "outgoing_move_qty", "written_off_qty"],
    )
    facts["date"] = pd.to_datetime(facts["date"]).dt.normalize()
    rows = rows.merge(facts, on=KEYS, how="left", validate="one_to_one")
    for column in [
        "release_qty",
        "incoming_move_qty",
        "outgoing_move_qty",
        "written_off_qty",
    ]:
        rows[column] = (
            pd.to_numeric(rows[column], errors="coerce").fillna(0).clip(lower=0)
        )
    rows["factual_closing_observable"] = (
        rows["release_qty"]
        + rows["incoming_move_qty"]
        - rows["outgoing_move_qty"]
        - rows["written_off_qty"]
        - rows["observed_sales_qty"]
    ).clip(lower=0)
    # Sales plus observable closing stock is the most defensible factual
    # availability because ClickHouse has no authoritative opening-stock ledger.
    rows["factual_available"] = (
        rows["observed_sales_qty"] + rows["factual_closing_observable"]
    )
    rows["factual_served"] = rows["observed_sales_qty"].clip(upper=rows["demand"])
    rows["p50_served"] = np.minimum(rows["p50_plan"], rows["demand"])

    totals = pd.DataFrame(
        [
            summarize(
                rows, "actual", rows["factual_available"], rows["factual_served"]
            ),
            summarize(rows, "p50_loss075", rows["p50_plan"], rows["p50_served"]),
        ]
    )
    actual = totals.iloc[0]
    for metric in ["plan_or_available", "over", "under", "imbalance", "service_pct"]:
        totals[f"{metric}_delta_vs_actual"] = totals[metric] - actual[metric]

    monthly_parts = []
    for period, part in rows.groupby(rows["date"].dt.to_period("M")):
        actual_row = summarize(
            part, "actual", part["factual_available"], part["factual_served"]
        )
        p50_row = summarize(part, "p50_loss075", part["p50_plan"], part["p50_served"])
        for record in [actual_row, p50_row]:
            record["period"] = str(period)
            monthly_parts.append(record)
    monthly = pd.DataFrame(monthly_parts)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    totals.to_csv(OUTPUT / "total_summary.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(OUTPUT / "monthly_summary.csv", index=False, encoding="utf-8-sig")
    print(totals.to_string(index=False))
    print("\nMONTHLY")
    print(monthly.to_string(index=False))


if __name__ == "__main__":
    main()

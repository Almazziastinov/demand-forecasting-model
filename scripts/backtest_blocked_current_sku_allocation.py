"""Blocked July validation of the frozen August causal SKU allocator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.backtest_current_sku_allocation import (  # noqa: E402
    GROUP,
    allocate_day,
    concentration,
    metric,
)


SNAPSHOT = ROOT / ".codex_tmp/blocked_sku_allocation_20260825/snapshot.parquet"
HISTORY = ROOT / ".codex_tmp/current_sku_allocation_20260825/sales_history.parquet"
OUTPUT = ROOT / "reports/blocked_sku_allocation_backtest_20260825"


def main() -> None:
    snapshot = pd.read_parquet(SNAPSHOT)
    history = pd.read_parquet(HISTORY)
    for frame in (snapshot, history):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["product_id"] = pd.to_numeric(frame["product_id"]).astype("int64")

    outputs = [
        allocate_day(day.reset_index(drop=True), history)
        for _, day in snapshot.groupby("date", sort=True)
    ]
    predicted = pd.concat(outputs, ignore_index=True)
    facts = history.rename(columns={"sold": "actual_sold"})[
        ["date", "bakery_id", "product_id", "actual_sold"]
    ]
    predicted = predicted.merge(
        facts, on=["date", "bakery_id", "product_id"], how="left"
    )
    predicted["actual_sold"] = predicted["actual_sold"].fillna(0)
    bakery_actual = predicted.groupby(["date", "bakery_id"])["actual_sold"].transform(
        "sum"
    )
    active = predicted[bakery_actual.gt(0)].copy()
    excluded = predicted[bakery_actual.le(0)].copy()
    methods = [
        "incumbent_sku_forecast",
        "same_weekday_forecast",
        "causal_trend_forecast",
        "blend_25",
        "blend_50",
        "blend_75",
    ]

    baseline_error = (
        (active[methods[0]] - active["actual_sold"])
        .abs()
        .groupby([active["date"], active["bakery_id"]])
        .sum()
    )
    stability = {}
    for method in methods[1:]:
        error = (
            (active[method] - active["actual_sold"])
            .abs()
            .groupby([active["date"], active["bakery_id"]])
            .sum()
        )
        stability[method] = {
            "better_bakery_days": int(error.lt(baseline_error).sum()),
            "bakery_days": int(len(error)),
        }

    date_wins = {}
    for method in methods[1:]:
        wins = 0
        for _, rows in active.groupby("date"):
            incumbent = metric(rows, methods[0], "actual_sold")["wape_pct"]
            challenger = metric(rows, method, "actual_sold")["wape_pct"]
            wins += challenger < incumbent
        date_wins[method] = int(wins)

    conservation = {}
    incumbent_totals = active.groupby(GROUP)[methods[0]].sum()
    for method in methods[1:]:
        conservation[method] = float(
            (active.groupby(GROUP)[method].sum() - incumbent_totals).abs().max()
        )

    summary = {
        "design": (
            "frozen August formula evaluated on earlier "
            "2026-07-17..2026-08-02 dates"
        ),
        "scope": {
            "dates": int(active["date"].nunique()),
            "bakeries": int(active["bakery_id"].nunique()),
            "bakery_days": int(active.groupby(["date", "bakery_id"]).ngroups),
            "sku_rows": int(len(active)),
        },
        "dq_excluded": {
            "bakery_days": int(excluded.groupby(["date", "bakery_id"]).ngroups),
            "forecast": float(excluded[methods[0]].sum()),
        },
        "metrics": {
            method: metric(active, method, "actual_sold") for method in methods
        },
        "concentration": {method: concentration(active, method) for method in methods},
        "stability": stability,
        "date_wins": date_wins,
        "conservation": conservation,
        "production_write": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    predicted.to_parquet(OUTPUT / "predictions.parquet", index=False)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

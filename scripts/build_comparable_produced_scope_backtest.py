"""Build a like-for-like Direct/P50 backtest on the produced forecast assortment.

For dates with an archived lead-1 forecast snapshot, that snapshot is the SKU
scope.  Earlier dates use a causal fallback: the bakery/SKU must have positive
production in the preceding 56 calendar days.  Plans are re-normalized inside
the selected scope so bakery totals are conserved.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/historical_daily_retraining_gate_loss075_full_20260910/predictions.parquet"
P50 = ROOT / "reports/exact_bakery_p50_loss075_full_20260910/bakery_totals.parquet"
PANEL = ROOT / ".codex_tmp/historical_clickhouse_panel/causal_panel.parquet"
SNAPSHOTS = ROOT / ".codex_tmp/exact_lead1_snapshots/selected_latest.parquet"
OUTPUT = ROOT / "reports/comparable_produced_scope_p50_loss075_20260911"
KEYS = ["date", "bakery_id", "product_id"]


def main() -> None:
    rows = pd.read_parquet(SOURCE)
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    totals = pd.read_parquet(P50).rename(columns={"p50_total_exact": "p50_total_new"})
    totals["date"] = pd.to_datetime(totals["date"]).dt.normalize()
    rows = rows.drop(columns=["p50_total", "p50_plan"], errors="ignore").merge(
        totals, on=["date", "bakery_id"], how="inner", validate="many_to_one"
    )

    flows = pd.read_parquet(PANEL, columns=KEYS + ["release_qty"])
    flows["date"] = pd.to_datetime(flows["date"]).dt.normalize()
    flows = flows.sort_values(["bakery_id", "product_id", "date"])
    flows["prior_release_56"] = flows.groupby(["bakery_id", "product_id"], sort=False)[
        "release_qty"
    ].transform(lambda value: value.shift(1).rolling(56, min_periods=1).sum())
    causal_keys = flows.loc[flows["prior_release_56"].gt(0), KEYS].assign(causal_scope=True)
    rows = rows.merge(causal_keys, on=KEYS, how="left", validate="one_to_one")

    snapshots = pd.read_parquet(SNAPSHOTS, columns=["forecast_date", "bakery_id", "product_id"])
    snapshots = snapshots.rename(columns={"forecast_date": "date"}).drop_duplicates(KEYS)
    snapshots["date"] = pd.to_datetime(snapshots["date"]).dt.normalize()
    snapshot_dates = set(snapshots["date"].unique())
    rows = rows.merge(snapshots.assign(snapshot_scope=True), on=KEYS, how="left")
    rows["scope_source"] = np.where(rows["date"].isin(snapshot_dates), "lead1_snapshot", "causal_production_56d")
    # A snapshot can also contain purchased/non-produced assortment.  Requiring
    # causal production prevents its checkout revenue from entering without a
    # corresponding production-cost basis.
    rows["in_scope"] = rows["causal_scope"].fillna(False) & np.where(
        rows["date"].isin(snapshot_dates), rows["snapshot_scope"].fillna(False), True
    )
    rows = rows[rows["in_scope"]].copy()

    denominator = rows.groupby(["date", "bakery_id"])["share"].transform("sum")
    rows = rows[denominator.gt(0)].copy()
    denominator = rows.groupby(["date", "bakery_id"])["share"].transform("sum")
    rows["share"] = rows["share"] / denominator
    rows["direct_plan"] = rows["direct_total"] * rows["share"]
    rows["p50_total"] = rows["p50_total_new"]
    rows["p50_plan"] = rows["p50_total"] * rows["share"]
    rows["gate_plan"] = rows["direct_plan"]
    rows["use_p50"] = False
    rows["gate_prior_advantage"] = np.nan

    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(OUTPUT / "predictions.parquet", index=False)
    audit = rows.groupby([rows["date"].dt.to_period("M").astype(str), "scope_source"], as_index=False).agg(
        dates=("date", "nunique"), bakeries=("bakery_id", "nunique"), skus=("product_id", "nunique"),
        rows=("product_id", "size"), sales=("observed_sales_qty", "sum")
    ).rename(columns={"date": "period"})
    audit.to_csv(OUTPUT / "scope_audit.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "production_write": False,
        "scope": "lead-1 snapshot when available; otherwise positive production in D-56..D-1",
        "snapshot_dates": len(snapshot_dates),
        "rows": len(rows),
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(audit.to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

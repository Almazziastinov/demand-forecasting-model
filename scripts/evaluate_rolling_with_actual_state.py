"""Add actual available-to-sell state to the 20-day rolling comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_relaxed_stockout_demand import client_from_env, query_components  # noqa: E402

ROWS = ROOT / "reports/rolling_floor_vs_no_floor_20260826/rows.parquet"
OUTPUT = ROOT / "reports/rolling_actual_state_comparison_20260826"
KEYS = ["date", "bakery_id", "product_id"]


def score(frame: pd.DataFrame, column: str, variant: str, fold: str) -> dict:
    error = frame[column] - frame["demand"]
    return {
        "fold": fold,
        "variant": variant,
        "dates": int(frame["date"].nunique()),
        "rows": int(len(frame)),
        "volume": float(frame[column].sum()),
        "surplus": float(error.clip(lower=0).sum()),
        "underbake": float((-error).clip(lower=0).sum()),
        "imbalance": float(error.abs().sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument(
        "--components-cache",
        type=Path,
        default=ROOT / ".codex_tmp/rolling_actual_components_20260721_20260823.parquet",
    )
    args = parser.parse_args()

    rows = pd.read_parquet(ROWS)
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    date_from = rows["date"].min() - pd.Timedelta(days=1)
    date_to = rows["date"].max()
    bakery_ids = tuple(sorted(rows["bakery_id"].astype(int).unique()))
    if args.components_cache.exists():
        components = pd.read_parquet(args.components_cache)
    else:
        components = query_components(
            client_from_env(args.env_file),
            str(date_from.date()),
            str(date_to.date()),
            bakery_ids,
        )
        args.components_cache.parent.mkdir(parents=True, exist_ok=True)
        components.to_parquet(args.components_cache, index=False)
    components["date"] = pd.to_datetime(components["date"]).dt.normalize()
    for column in ["produced", "sold", "received", "sent"]:
        components[column] = pd.to_numeric(components[column], errors="coerce").fillna(0.0)
    components["closing"] = (
        components["produced"]
        + components["received"]
        - components["sent"]
        - components["sold"]
    ).clip(lower=0.0)
    opening = components[KEYS + ["closing"]].copy()
    opening["date"] += pd.Timedelta(days=1)
    opening = opening.rename(columns={"closing": "opening_stock"})
    components = components.merge(opening, on=KEYS, how="left")
    components["opening_stock"] = components["opening_stock"].fillna(0.0)
    components["available_to_sell"] = (
        components["produced"]
        + components["opening_stock"]
        + components["received"]
        - components["sent"]
    ).clip(lower=0.0)

    controlled_pairs = components.loc[
        components[["produced", "received", "sent"]].sum(axis=1).gt(0),
        ["bakery_id", "product_id"],
    ].drop_duplicates()
    scope = rows.merge(controlled_pairs.assign(is_controlled=True), on=["bakery_id", "product_id"], how="inner")
    scope = scope.merge(
        components[KEYS + ["available_to_sell", "produced", "received", "sent", "opening_stock"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    for column in ["available_to_sell", "produced", "received", "sent", "opening_stock"]:
        scope[column] = scope[column].fillna(0.0)

    variants = [
        ("available_to_sell", "actual_state"),
        ("incumbent_sku_forecast", "current"),
        ("predictive_forecast", "predictive_same_volume"),
        ("p50_predictive", "p50_predictive"),
        ("p50_simple_floor", "p50_predictive_simple_floor"),
    ]
    metric_rows = []
    for fold, part in scope.groupby("rolling_fold"):
        for column, variant in variants:
            metric_rows.append(score(part, column, variant, str(fold)))
    metrics = pd.DataFrame(metric_rows)
    summary = metrics.groupby("variant", as_index=False).agg(
        folds=("fold", "nunique"),
        dates=("dates", "sum"),
        rows=("rows", "sum"),
        volume=("volume", "sum"),
        surplus=("surplus", "sum"),
        underbake=("underbake", "sum"),
        imbalance=("imbalance", "sum"),
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scope.to_parquet(OUTPUT / "rows.parquet", index=False)
    metrics.to_csv(OUTPUT / "fold_metrics.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    print(
        f"scope rows={len(scope)} dates={scope['date'].nunique()} "
        f"bakeries={scope['bakery_id'].nunique()} products={scope['product_id'].nunique()}"
    )
    print(summary.to_string(index=False))
    print("\nFold metrics")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()

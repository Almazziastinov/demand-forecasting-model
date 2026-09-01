"""Point-in-time comparison of incumbent, daily-profile and predictive SKU mix.

The input snapshot must already be exported read-only and contain an explicit
``source_run_id``.  The script never selects snapshots with ``argMax`` and
never writes to ClickHouse or production services.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments_v2.daily_sku_allocation import (  # noqa: E402
    allocate_category_totals,
    build_daily_sku_shares,
)


SNAPSHOT_KEYS = ["date", "bakery_id", "category", "product_id"]
GROUP_KEYS = ["date", "bakery_id", "category"]


def read_frame(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix == ".parquet":
        return pd.read_parquet(source)
    return pd.read_csv(source, low_memory=False)


def validate_snapshot(snapshot: pd.DataFrame) -> None:
    required = {*SNAPSHOT_KEYS, "city", "source_run_id", "incumbent_sku_forecast"}
    missing = sorted(required.difference(snapshot.columns))
    if missing:
        raise ValueError(f"snapshot is missing required columns: {missing}")
    if snapshot["source_run_id"].isna().any() or snapshot["source_run_id"].astype(str).str.strip().eq("").any():
        raise ValueError("every snapshot row must have source_run_id")
    run_count = snapshot.groupby("date")["source_run_id"].nunique()
    mixed = run_count[run_count.ne(1)]
    if not mixed.empty:
        raise ValueError(f"mixed source_run_id for forecast dates: {mixed.index.astype(str).tolist()}")
    duplicates = snapshot.duplicated(SNAPSHOT_KEYS, keep=False)
    if duplicates.any():
        raise ValueError(f"snapshot contains {int(duplicates.sum())} duplicate SKU keys")


def normalize_predictive_shares(rows: pd.DataFrame, daily_share_col: str) -> pd.Series:
    raw = pd.to_numeric(rows["predictive_share"], errors="coerce")
    raw = raw.where(raw.ge(0.0), np.nan).fillna(rows[daily_share_col])
    denominator = raw.groupby([rows[key] for key in GROUP_KEYS]).transform("sum")
    fallback = rows[daily_share_col]
    return (raw / denominator.replace(0.0, np.nan)).fillna(fallback)


def metrics(rows: pd.DataFrame, forecast_col: str, target_col: str) -> dict[str, float | int]:
    actual = rows[target_col].sum()
    error = rows[forecast_col] - rows[target_col]
    return {
        "rows": int(len(rows)),
        "actual": float(actual),
        "forecast": float(rows[forecast_col].sum()),
        "wape_pct": float(100.0 * error.abs().sum() / actual) if actual > 0 else 0.0,
        "bias_pct": float(100.0 * error.sum() / actual) if actual > 0 else 0.0,
        "mae": float(error.abs().mean()) if len(error) else 0.0,
    }


def concentration(rows: pd.DataFrame, forecast_col: str) -> dict[str, float | int]:
    group_total = rows.groupby(GROUP_KEYS)[forecast_col].transform("sum")
    shares = rows[forecast_col] / group_total.replace(0.0, np.nan)
    top = shares.groupby([rows[key] for key in GROUP_KEYS]).max().dropna()
    return {
        "groups": int(len(top)),
        "top_share_p95": float(top.quantile(0.95)) if len(top) else 0.0,
        "top_share_max": float(top.max()) if len(top) else 0.0,
        "groups_top_share_ge_20pct": int(top.ge(0.20).sum()),
        "groups_top_share_ge_30pct": int(top.ge(0.30).sum()),
        "groups_top_share_ge_40pct": int(top.ge(0.40).sum()),
    }


def allocation_metrics(rows: pd.DataFrame, forecast_col: str, target_col: str) -> dict[str, float | int]:
    """Measure mix quality at the same actual category total (diagnostic only)."""
    forecast_total = rows.groupby(GROUP_KEYS)[forecast_col].transform("sum")
    actual_total = rows.groupby(GROUP_KEYS)[target_col].transform("sum")
    share = rows[forecast_col] / forecast_total.replace(0.0, np.nan)
    oracle_forecast = (share * actual_total).fillna(0.0)
    error = oracle_forecast - rows[target_col]
    actual = rows[target_col].sum()
    return {
        "rows": int(len(rows)),
        "wape_pct": float(100.0 * error.abs().sum() / actual) if actual > 0 else 0.0,
        "bias_pct": float(100.0 * error.sum() / actual) if actual > 0 else 0.0,
        "note": "diagnostic SKU-mix metric using actual category total; not end-to-end",
    }


def date_stability(rows: pd.DataFrame, methods: list[str], target_col: str) -> dict[str, dict[str, int]]:
    by_date: dict[str, list[float]] = {method: [] for method in methods}
    for _, part in rows.groupby("date", sort=True):
        actual = part[target_col].sum()
        for method in methods:
            by_date[method].append(float((part[method] - part[target_col]).abs().sum() / actual))
    baseline = np.asarray(by_date[methods[0]])
    return {
        method: {
            "better_vs_incumbent_dates": int((np.asarray(by_date[method]) < baseline).sum()),
            "dates": int(len(baseline)),
        }
        for method in methods[1:]
    }


def run_backtest(
    snapshot: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    target_col: str,
    predictive: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    snapshot = snapshot.copy()
    snapshot["date"] = pd.to_datetime(snapshot["date"]).dt.normalize()
    snapshot["product_id"] = pd.to_numeric(snapshot["product_id"], errors="raise").astype("int64")
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["product_id"] = pd.to_numeric(panel["product_id"], errors="raise").astype("int64")
    validate_snapshot(snapshot)
    if target_col not in panel.columns:
        raise ValueError(f"panel is missing target column: {target_col}")

    predictive_lookup = None
    if predictive is not None:
        predictive_lookup = predictive.copy()
        predictive_lookup["date"] = pd.to_datetime(predictive_lookup["date"]).dt.normalize()
        predictive_lookup["product_id"] = pd.to_numeric(
            predictive_lookup["product_id"], errors="raise"
        ).astype("int64")
        required = {*SNAPSHOT_KEYS, "predictive_share"}
        missing = sorted(required.difference(predictive_lookup.columns))
        if missing:
            raise ValueError(f"predictive frame is missing required columns: {missing}")
        predictive_lookup = predictive_lookup[list(required)].drop_duplicates(SNAPSHOT_KEYS)

    outputs = []
    for forecast_date, day in snapshot.groupby("date", sort=True):
        history = panel[panel["date"].lt(forecast_date)]
        universe = day[["bakery_id", "city", "category", "product_id"]]
        daily = build_daily_sku_shares(history, universe, forecast_date, target_col=target_col)
        category_totals = day.groupby(["bakery_id", "category"], as_index=False).agg(
            category_forecast=("incumbent_sku_forecast", "sum")
        )
        daily_allocated = allocate_category_totals(category_totals, daily).rename(
            columns={"sku_day_forecast": "daily_profile_forecast", "sku_share": "daily_profile_share"}
        )
        result = day.merge(
            daily_allocated[
                ["bakery_id", "category", "product_id", "daily_profile_share", "daily_profile_forecast"]
            ],
            on=["bakery_id", "category", "product_id"],
            how="left",
            validate="one_to_one",
        )
        if predictive_lookup is not None:
            predicted_day = predictive_lookup[predictive_lookup["date"].eq(forecast_date)]
            result = result.merge(predicted_day, on=SNAPSHOT_KEYS, how="left", validate="one_to_one")
            result["predictive_share_full"] = normalize_predictive_shares(result, "daily_profile_share")
            result["predictive_forecast"] = result["predictive_share_full"] * result.groupby(GROUP_KEYS)[
                "incumbent_sku_forecast"
            ].transform("sum")
        outputs.append(result)

    rows = pd.concat(outputs, ignore_index=True)
    actual = panel[["date", "bakery_id", "product_id", target_col]].drop_duplicates(
        ["date", "bakery_id", "product_id"]
    )
    rows = rows.merge(actual, on=["date", "bakery_id", "product_id"], how="left")
    rows[target_col] = rows[target_col].fillna(0.0)
    methods = ["incumbent_sku_forecast", "daily_profile_forecast"]
    if "predictive_forecast" in rows.columns:
        methods.append("predictive_forecast")

    conservation = rows.groupby(GROUP_KEYS, as_index=False).agg(
        incumbent_total=("incumbent_sku_forecast", "sum"),
        daily_total=("daily_profile_forecast", "sum"),
        **({"predictive_total": ("predictive_forecast", "sum")} if "predictive_forecast" in rows else {}),
    )
    deltas = {"daily_max_abs_delta": float((conservation["daily_total"] - conservation["incumbent_total"]).abs().max())}
    if "predictive_total" in conservation:
        deltas["predictive_max_abs_delta"] = float(
            (conservation["predictive_total"] - conservation["incumbent_total"]).abs().max()
        )
    summary = {
        "coverage": {
            "date_from": str(rows["date"].min().date()),
            "date_to": str(rows["date"].max().date()),
            "rows": int(len(rows)),
            "bakeries": int(rows["bakery_id"].nunique()),
            "skus": int(rows["product_id"].nunique()),
            "source_runs": sorted(rows["source_run_id"].astype(str).unique().tolist()),
        },
        "target": target_col,
        "metrics": {method: metrics(rows, method, target_col) for method in methods},
        "allocation_metrics": {method: allocation_metrics(rows, method, target_col) for method in methods},
        "date_stability": date_stability(rows, methods, target_col),
        "concentration": {method: concentration(rows, method) for method in methods},
        "conservation": deltas,
        "causality": "history date < forecast date; universe and totals come from one explicit source_run_id per date",
        "production_write": False,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--predictive")
    parser.add_argument("--target-col", default="demand_mid")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows, summary = run_backtest(
        read_frame(args.snapshot),
        read_frame(args.panel),
        target_col=args.target_col,
        predictive=read_frame(args.predictive) if args.predictive else None,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output / "sku_day_predictions.parquet", index=False)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

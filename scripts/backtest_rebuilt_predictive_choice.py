"""Causal forecast-conditioned SKU-share model with frozen folds."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.backtest_current_sku_allocation import (  # noqa: E402
    GROUP,
    allocate_day,
    concentration,
    metric,
)


INPUT = (
    ROOT / ".codex_tmp/predictive_choice_rebuild_20260825/historical_snapshots.parquet"
)
HISTORY = ROOT / ".codex_tmp/current_sku_allocation_20260825/sales_history.parquet"
OUTPUT = ROOT / "reports/rebuilt_predictive_choice_20260825"
FEATURES = [
    "bakery_code",
    "product_code",
    "category_code",
    "dow",
    "incumbent_share",
    "same_weekday_share",
    "causal_trend_share",
    "log_category_total",
    "log_incumbent_qty",
]
CATEGORICAL = ["bakery_code", "product_code", "category_code", "dow"]


def build_features(snapshot: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    for _, day in snapshot.groupby("date", sort=True):
        rows = allocate_day(day.reset_index(drop=True), history)
        total = rows.groupby(GROUP)["incumbent_sku_forecast"].transform("sum")
        rows["incumbent_share"] = rows["incumbent_sku_forecast"] / total.replace(
            0, np.nan
        )
        rows["same_weekday_share"] = rows["same_weekday_forecast"] / total.replace(
            0, np.nan
        )
        rows["causal_trend_share"] = rows["causal_trend_forecast"] / total.replace(
            0, np.nan
        )
        rows["log_category_total"] = np.log1p(total)
        rows["log_incumbent_qty"] = np.log1p(rows["incumbent_sku_forecast"])
        rows["dow"] = rows["date"].dt.dayofweek
        outputs.append(rows)
    result = pd.concat(outputs, ignore_index=True)
    actual = history.rename(columns={"sold": "actual_sold"})[
        ["date", "bakery_id", "product_id", "actual_sold"]
    ]
    result = result.merge(actual, on=["date", "bakery_id", "product_id"], how="left")
    result["actual_sold"] = result["actual_sold"].fillna(0)
    actual_total = result.groupby(GROUP)["actual_sold"].transform("sum")
    result["target_share"] = result["actual_sold"] / actual_total.replace(0, np.nan)
    for source, target in [
        ("bakery_id", "bakery_code"),
        ("product_id", "product_code"),
        ("category", "category_code"),
    ]:
        result[target] = pd.Categorical(result[source]).codes
    return result


def fit_predict(
    rows: pd.DataFrame, train_end: str, test_dates: list[str]
) -> pd.DataFrame:
    train_end_ts = pd.Timestamp(train_end)
    test_ts = pd.to_datetime(test_dates)
    train = rows[rows["date"].le(train_end_ts) & rows["target_share"].notna()].copy()
    test = rows[rows["date"].isin(test_ts)].copy()
    model = lgb.LGBMRegressor(
        objective="l1",
        n_estimators=180,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=42,
        verbosity=-1,
    )
    model.fit(train[FEATURES], train["target_share"], categorical_feature=CATEGORICAL)
    test["predictive_raw"] = np.maximum(model.predict(test[FEATURES]), 0)
    denominator = test.groupby(GROUP)["predictive_raw"].transform("sum")
    fallback = test["causal_trend_share"]
    test["predictive_share"] = (
        test["predictive_raw"] / denominator.replace(0, np.nan)
    ).fillna(fallback)
    category_total = test.groupby(GROUP)["incumbent_sku_forecast"].transform("sum")
    test["predictive_forecast"] = test["predictive_share"] * category_total
    test["predictive_blend_25"] = (
        0.75 * test["incumbent_sku_forecast"] + 0.25 * test["predictive_forecast"]
    )
    return test


def summarize(rows: pd.DataFrame) -> dict:
    bakery_actual = rows.groupby(["date", "bakery_id"])["actual_sold"].transform("sum")
    active = rows[bakery_actual.gt(0)].copy()
    methods = [
        "incumbent_sku_forecast",
        "blend_25",
        "predictive_forecast",
        "predictive_blend_25",
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
        stability[method] = int(error.lt(baseline_error).sum())
    return {
        "scope": {
            "dates": int(active["date"].nunique()),
            "bakeries": int(active["bakery_id"].nunique()),
            "bakery_days": int(active.groupby(["date", "bakery_id"]).ngroups),
        },
        "metrics": {
            method: metric(active, method, "actual_sold") for method in methods
        },
        "concentration": {method: concentration(active, method) for method in methods},
        "better_bakery_days": stability,
    }


def main() -> None:
    snapshot = pd.read_parquet(INPUT)
    history = pd.read_parquet(HISTORY)
    snapshot = snapshot.rename(columns={"forecast_qty": "incumbent_sku_forecast"})
    for frame in (snapshot, history):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["product_id"] = pd.to_numeric(frame["product_id"]).astype("int64")
    rows = build_features(snapshot, history)
    blocked_dates = [
        str(date.date())
        for date in sorted(
            rows.loc[rows["date"].between("2026-07-22", "2026-08-02"), "date"].unique()
        )
    ]
    current_dates = [
        "2026-08-11",
        "2026-08-12",
        "2026-08-13",
        "2026-08-17",
        "2026-08-18",
        "2026-08-21",
        "2026-08-22",
        "2026-08-23",
    ]
    blocked = fit_predict(rows, "2026-07-21", blocked_dates)
    current = fit_predict(rows, "2026-08-10", current_dates)
    summary = {
        "design": "explicit runs; frozen folds; history strictly before each test fold",
        "blocked": summarize(blocked),
        "current": summarize(current),
        "production_write": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.concat(
        [blocked.assign(fold="blocked"), current.assign(fold="current")]
    ).to_parquet(OUTPUT / "predictions.parquet", index=False)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

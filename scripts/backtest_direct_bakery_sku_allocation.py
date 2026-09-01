"""Frozen-fold backtest for direct bakery-day to SKU allocation.

The incumbent SKU forecast is used only to recover the already approved bakery-day
volume and the forecast assortment. Incumbent SKU and category shares are never
used as model features or constraints.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    ROOT / ".codex_tmp/predictive_choice_rebuild_20260825/historical_snapshots.parquet"
)
HISTORY = ROOT / ".codex_tmp/current_sku_allocation_20260825/sales_history.parquet"
OLD_PREDICTIONS = (
    ROOT / "reports/rebuilt_predictive_choice_20260825/predictions.parquet"
)
OUTPUT = ROOT / "reports/direct_bakery_sku_allocation_20260827"
DAY = ["date", "bakery_id"]
KEYS = ["date", "bakery_id", "product_id"]

FEATURES = [
    "bakery_code",
    "product_code",
    "category_code",
    "dow",
    "log_bakery_total",
    "recent_7_mean",
    "prior_7_mean",
    "broad_56_mean",
    "same_weekday_4_mean",
    "presence_28",
    "recent_7_share",
    "broad_56_share",
    "same_weekday_4_share",
    "historical_category_share",
    "recent_trend",
]
CATEGORICAL = ["bakery_code", "product_code", "category_code", "dow"]


def reindex_sum(
    history: pd.DataFrame, day: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series:
    values = (
        history[history["date"].between(start, end)]
        .groupby(["bakery_id", "product_id"])["sold"]
        .sum()
    )
    index = pd.MultiIndex.from_frame(day[["bakery_id", "product_id"]])
    return pd.Series(
        values.reindex(index, fill_value=0.0).to_numpy(),
        index=day.index,
        dtype="float64",
    )


def normalized(values: pd.Series, groups: list[pd.Series]) -> pd.Series:
    total = values.groupby(groups).transform("sum")
    return (values / total.replace(0, np.nan)).fillna(0.0)


def build_day_features(day: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    result = day.copy().reset_index(drop=True)
    date = result["date"].iloc[0]
    recent = reindex_sum(
        history, result, date - pd.Timedelta(days=7), date - pd.Timedelta(days=1)
    )
    prior = reindex_sum(
        history, result, date - pd.Timedelta(days=14), date - pd.Timedelta(days=8)
    )
    broad = reindex_sum(
        history, result, date - pd.Timedelta(days=56), date - pd.Timedelta(days=1)
    )
    weekday_dates = [date - pd.Timedelta(days=7 * step) for step in range(1, 5)]
    weekday_values = (
        history[history["date"].isin(weekday_dates)]
        .groupby(["bakery_id", "product_id"])["sold"]
        .sum()
    )
    index = pd.MultiIndex.from_frame(result[["bakery_id", "product_id"]])
    weekday = pd.Series(
        weekday_values.reindex(index, fill_value=0.0).to_numpy(), index=result.index
    )

    presence_values = (
        history[
            history["date"].between(
                date - pd.Timedelta(days=28), date - pd.Timedelta(days=1)
            )
        ]
        .loc[lambda frame: frame["sold"].gt(0)]
        .groupby(["bakery_id", "product_id"])["date"]
        .nunique()
    )
    presence = pd.Series(
        presence_values.reindex(index, fill_value=0).to_numpy(), index=result.index
    )

    bakery_groups = [result["bakery_id"]]
    result["recent_7_mean"] = recent / 7.0
    result["prior_7_mean"] = prior / 7.0
    result["broad_56_mean"] = broad / 56.0
    result["same_weekday_4_mean"] = weekday / 4.0
    result["presence_28"] = presence / 28.0
    result["recent_7_share"] = normalized(recent, bakery_groups)
    result["broad_56_share"] = normalized(broad, bakery_groups)
    result["same_weekday_4_share"] = normalized(weekday, bakery_groups)
    category_broad = broad.groupby([result["bakery_id"], result["category"]]).transform(
        "sum"
    )
    bakery_broad = broad.groupby(result["bakery_id"]).transform("sum")
    result["historical_category_share"] = (
        category_broad / bakery_broad.replace(0, np.nan)
    ).fillna(0.0)
    result["recent_trend"] = ((recent + 1.0) / (prior + 1.0)).clip(0.25, 4.0)
    bakery_total = result.groupby(DAY)["incumbent_sku_forecast"].transform("sum")
    result["log_bakery_total"] = np.log1p(bakery_total)
    result["dow"] = date.dayofweek
    return result


def build_features(snapshot: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    outputs = [
        build_day_features(day, history)
        for _, day in snapshot.groupby("date", sort=True)
    ]
    rows = pd.concat(outputs, ignore_index=True)
    actual = history.rename(columns={"sold": "actual_sold"})[KEYS + ["actual_sold"]]
    rows = rows.merge(actual, on=KEYS, how="left")
    rows["actual_sold"] = rows["actual_sold"].fillna(0.0)
    for source, target in [
        ("bakery_id", "bakery_code"),
        ("product_id", "product_code"),
        ("category", "category_code"),
    ]:
        rows[target] = pd.Categorical(rows[source]).codes
    return rows


def fit_predict(
    rows: pd.DataFrame, train_end: str, test_dates: list[str]
) -> pd.DataFrame:
    train = rows[rows["date"].le(pd.Timestamp(train_end))].copy()
    test = rows[rows["date"].isin(pd.to_datetime(test_dates))].copy()
    model = lgb.LGBMRegressor(
        objective="poisson",
        n_estimators=240,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=120,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        random_state=42,
        verbosity=-1,
    )
    model.fit(train[FEATURES], train["actual_sold"], categorical_feature=CATEGORICAL)
    test["direct_raw_demand"] = np.maximum(model.predict(test[FEATURES]), 1e-9)
    raw_total = test.groupby(DAY)["direct_raw_demand"].transform("sum")
    test["direct_share"] = test["direct_raw_demand"] / raw_total.replace(0, np.nan)
    bakery_total = test.groupby(DAY)["incumbent_sku_forecast"].transform("sum")
    test["direct_forecast"] = test["direct_share"] * bakery_total
    return test


def score(rows: pd.DataFrame, column: str) -> dict[str, float]:
    error = rows[column] - rows["actual_sold"]
    actual = rows["actual_sold"].sum()
    return {
        "wape_pct": float(100 * error.abs().sum() / actual),
        "mae": float(error.abs().mean()),
        "bias_pct": float(100 * error.sum() / actual),
        "forecast": float(rows[column].sum()),
        "actual": float(actual),
    }


def shape(rows: pd.DataFrame, column: str) -> dict[str, float | int]:
    share = rows[column] / rows.groupby(DAY)[column].transform("sum").replace(0, np.nan)
    top = share.groupby([rows["date"], rows["bakery_id"]]).max()
    leaders = rows.loc[
        share.groupby([rows["date"], rows["bakery_id"]]).idxmax(), "product_id"
    ]
    return {
        "top_share_p95": float(top.quantile(0.95)),
        "top_share_max": float(top.max()),
        "top_ge20": int(top.ge(0.20).sum()),
        "top_ge30": int(top.ge(0.30).sum()),
        "top_ge40": int(top.ge(0.40).sum()),
        "sku_1071_is_top": int(leaders.eq(1071).sum()),
        "near_zero_rows": int(rows[column].le(1e-6).sum()),
    }


def category_wape(rows: pd.DataFrame, column: str) -> float:
    grouped = rows.groupby(["date", "bakery_id", "category"])[
        [column, "actual_sold"]
    ].sum()
    return float(
        100
        * (grouped[column] - grouped["actual_sold"]).abs().sum()
        / grouped["actual_sold"].sum()
    )


def summarize(rows: pd.DataFrame) -> dict:
    active_total = rows.groupby(DAY)["actual_sold"].transform("sum")
    active = rows[active_total.gt(0)].copy()
    columns = ["incumbent_sku_forecast", "predictive_forecast", "direct_forecast"]
    day_errors = {
        column: (active[column] - active["actual_sold"])
        .abs()
        .groupby([active["date"], active["bakery_id"]])
        .sum()
        for column in columns
    }
    summary = {
        "scope": {
            "dates": int(active["date"].nunique()),
            "bakeries": int(active["bakery_id"].nunique()),
            "bakery_days": int(active.groupby(DAY).ngroups),
            "sku_rows": int(len(active)),
        },
        "metrics": {column: score(active, column) for column in columns},
        "category_wape_pct": {
            column: category_wape(active, column) for column in columns
        },
        "shape": {column: shape(active, column) for column in columns},
        "better_bakery_days": {
            "direct_vs_incumbent": int(
                day_errors["direct_forecast"]
                .lt(day_errors["incumbent_sku_forecast"])
                .sum()
            ),
            "direct_vs_predictive": int(
                day_errors["direct_forecast"]
                .lt(day_errors["predictive_forecast"])
                .sum()
            ),
        },
        "positive_actual_rows_near_zero": {
            column: int(active["actual_sold"].gt(0).mul(active[column].le(1e-6)).sum())
            for column in columns
        },
        "conservation_max_abs": float(
            (
                active.groupby(DAY)["direct_forecast"].sum()
                - active.groupby(DAY)["incumbent_sku_forecast"].sum()
            )
            .abs()
            .max()
        ),
    }
    sku_1071 = active[active["product_id"].eq(1071)]
    summary["sku_1071_metrics"] = {
        column: score(sku_1071, column) for column in columns
    }
    case = active[
        active["date"].eq(pd.Timestamp("2026-08-23"))
        & active["bakery_id"].eq(29)
        & active["product_id"].eq(1071)
    ]
    if not case.empty:
        summary["bakery_29_sku_1071_2026_08_23"] = (
            case[
                [
                    "actual_sold",
                    "incumbent_sku_forecast",
                    "predictive_forecast",
                    "direct_forecast",
                    "direct_raw_demand",
                ]
            ]
            .iloc[0]
            .to_dict()
        )
        bakery = active[
            active["date"].eq(pd.Timestamp("2026-08-23")) & active["bakery_id"].eq(29)
        ]
        summary["bakery_29_2026_08_23_category_totals"] = (
            bakery.groupby("category")[
                [
                    "actual_sold",
                    "incumbent_sku_forecast",
                    "predictive_forecast",
                    "direct_forecast",
                ]
            ]
            .sum()
            .sort_values("actual_sold", ascending=False)
            .head(10)
            .reset_index()
            .to_dict(orient="records")
        )
    return summary


def main() -> None:
    snapshot = pd.read_parquet(INPUT).rename(
        columns={"forecast_qty": "incumbent_sku_forecast"}
    )
    history = pd.read_parquet(HISTORY)
    old = pd.read_parquet(OLD_PREDICTIONS)[KEYS + ["predictive_forecast"]]
    for frame in (snapshot, history, old):
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
    predictions = pd.concat(
        [blocked.assign(fold="blocked"), current.assign(fold="current")],
        ignore_index=True,
    )
    predictions = predictions.merge(old, on=KEYS, how="left")
    summary = {
        "design": "direct bakery-day to SKU; incumbent used only for bakery-day total and assortment universe",
        "forbidden_inputs": [
            "incumbent SKU share",
            "incumbent category total/share",
            "hourly profile",
            "old uplift",
        ],
        "target": "observed daily SKU sales (Poisson)",
        "blocked": summarize(predictions[predictions["fold"].eq("blocked")]),
        "current": summarize(predictions[predictions["fold"].eq("current")]),
        "production_write": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(OUTPUT / "predictions.parquet", index=False)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

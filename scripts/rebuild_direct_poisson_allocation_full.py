"""Rebuild the confirmed Direct Poisson SKU allocator in causal monthly folds."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / ".codex_tmp/historical_clickhouse_panel/causal_panel.parquet"
TOTALS = (
    ROOT
    / "reports/historical_daily_retraining_gate_loss075_full_20260910/predictions.parquet"
)
OUTPUT = ROOT / "reports/direct_poisson_allocation_full_20260910"
KEYS = ["date", "bakery_id", "product_id"]
DAY = ["date", "bakery_id"]
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
START = pd.Timestamp("2026-01-02")
END = pd.Timestamp("2026-08-31")
TRAIN_DAYS = 180


def period_sum(
    history: pd.DataFrame, day: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series:
    values = (
        history[history["date"].between(start, end)]
        .groupby(["bakery_id", "product_id"])["sold"]
        .sum()
    )
    index = pd.MultiIndex.from_frame(day[["bakery_id", "product_id"]])
    return pd.Series(values.reindex(index, fill_value=0.0).to_numpy(), index=day.index)


def normalized(values: pd.Series, group: pd.Series) -> pd.Series:
    total = values.groupby(group).transform("sum")
    return (values / total.replace(0, np.nan)).fillna(0.0)


def features_for_day(
    date: pd.Timestamp, panel: pd.DataFrame, totals: pd.DataFrame
) -> pd.DataFrame:
    history = panel[
        panel["date"].between(date - pd.Timedelta(days=56), date - pd.Timedelta(days=1))
    ]
    active = (
        history[history["activity"]]
        .sort_values("date")
        .drop_duplicates(["bakery_id", "product_id"], keep="last")
    )
    day = (
        active[["bakery_id", "product_id", "category_name"]]
        .rename(columns={"category_name": "category"})
        .reset_index(drop=True)
    )
    if day.empty:
        return day
    recent = period_sum(
        history, day, date - pd.Timedelta(days=7), date - pd.Timedelta(days=1)
    )
    prior = period_sum(
        history, day, date - pd.Timedelta(days=14), date - pd.Timedelta(days=8)
    )
    broad = period_sum(
        history, day, date - pd.Timedelta(days=56), date - pd.Timedelta(days=1)
    )
    weekday_dates = [date - pd.Timedelta(days=7 * step) for step in range(1, 5)]
    weekday_values = (
        history[history["date"].isin(weekday_dates)]
        .groupby(["bakery_id", "product_id"])["sold"]
        .sum()
    )
    index = pd.MultiIndex.from_frame(day[["bakery_id", "product_id"]])
    weekday = pd.Series(
        weekday_values.reindex(index, fill_value=0.0).to_numpy(), index=day.index
    )
    presence_values = (
        history[history["sold"].gt(0)]
        .groupby(["bakery_id", "product_id"])["date"]
        .nunique()
    )
    presence = pd.Series(
        presence_values.reindex(index, fill_value=0).to_numpy(), index=day.index
    )
    day["recent_7_mean"] = recent / 7
    day["prior_7_mean"] = prior / 7
    day["broad_56_mean"] = broad / 56
    day["same_weekday_4_mean"] = weekday / 4
    day["presence_28"] = presence / 28
    day["recent_7_share"] = normalized(recent, day["bakery_id"])
    day["broad_56_share"] = normalized(broad, day["bakery_id"])
    day["same_weekday_4_share"] = normalized(weekday, day["bakery_id"])
    category_broad = broad.groupby([day["bakery_id"], day["category"]]).transform("sum")
    bakery_broad = broad.groupby(day["bakery_id"]).transform("sum")
    day["historical_category_share"] = (
        category_broad / bakery_broad.replace(0, np.nan)
    ).fillna(0)
    day["recent_trend"] = ((recent + 1) / (prior + 1)).clip(0.25, 4)
    day["date"] = date
    day["dow"] = date.dayofweek
    day = day.merge(totals, on=DAY, how="inner", validate="many_to_one")
    day["log_bakery_total"] = np.log1p(day["direct_total"])
    actual = panel[panel["date"].eq(date)][KEYS + ["sold"]]
    day = day.merge(actual, on=KEYS, how="left", validate="one_to_one")
    day["sold"] = day["sold"].fillna(0.0)
    return day


def main() -> None:
    panel = pd.read_parquet(PANEL)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel = panel.rename(columns={"observed_sales_qty": "sold"})
    source = pd.read_parquet(
        TOTALS, columns=["date", "bakery_id", "direct_total", "p50_total"]
    )
    source["date"] = pd.to_datetime(source["date"]).dt.normalize()
    totals = source.drop_duplicates(DAY)
    dates = pd.date_range(START - pd.Timedelta(days=TRAIN_DAYS), END)
    frames = []
    for date in dates:
        # Historical training rows use the realized bakery sales total only as
        # the volume feature; evaluation rows receive the separately forecast total.
        if date < START and date not in set(totals["date"]):
            actual_total = (
                panel[panel["date"].eq(date)]
                .groupby("bakery_id", as_index=False)["sold"]
                .sum()
            )
            day_totals = actual_total.assign(
                date=date,
                direct_total=lambda x: x["sold"],
                p50_total=lambda x: x["sold"],
            )[["date", "bakery_id", "direct_total", "p50_total"]]
        else:
            day_totals = totals[totals["date"].eq(date)]
        frame = features_for_day(date, panel, day_totals)
        if not frame.empty:
            frames.append(frame)
        if date.day == 1:
            print(f"features {date.date()}", flush=True)
    rows = pd.concat(frames, ignore_index=True)
    all_bakeries = sorted(rows["bakery_id"].unique())
    all_products = sorted(rows["product_id"].unique())
    all_categories = sorted(rows["category"].fillna("unknown").unique())
    rows["bakery_code"] = pd.Categorical(
        rows["bakery_id"], categories=all_bakeries
    ).codes
    rows["product_code"] = pd.Categorical(
        rows["product_id"], categories=all_products
    ).codes
    rows["category_code"] = pd.Categorical(
        rows["category"].fillna("unknown"), categories=all_categories
    ).codes
    predictions = []
    for month in pd.date_range(START.normalize().replace(day=1), END, freq="MS"):
        month_end = min(month + pd.offsets.MonthEnd(0), END)
        train = rows[
            rows["date"].between(
                month - pd.Timedelta(days=TRAIN_DAYS), month - pd.Timedelta(days=1)
            )
        ]
        test = rows[rows["date"].between(max(month, START), month_end)].copy()
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
            n_jobs=-1,
        )
        model.fit(train[FEATURES], train["sold"], categorical_feature=CATEGORICAL)
        test["direct_raw_demand"] = np.maximum(model.predict(test[FEATURES]), 1e-9)
        raw_total = test.groupby(DAY)["direct_raw_demand"].transform("sum")
        test["direct_share"] = test["direct_raw_demand"] / raw_total
        test["direct_plan_poisson"] = test["direct_share"] * test["direct_total"]
        test["p50_plan_poisson"] = test["direct_share"] * test["p50_total"]
        predictions.append(test)
        print(f"predicted {month.date()} rows={len(test)}", flush=True)
    result = pd.concat(predictions, ignore_index=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT / "predictions.parquet", index=False)
    metadata = {
        "production_write": False,
        "allocator": "confirmed Direct Poisson",
        "fold": "monthly",
        "train_days": TRAIN_DAYS,
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        result.groupby(result["date"].dt.to_period("M"))
        .agg(rows=("product_id", "size"), forecast=("p50_plan_poisson", "sum"))
        .to_string()
    )


if __name__ == "__main__":
    main()

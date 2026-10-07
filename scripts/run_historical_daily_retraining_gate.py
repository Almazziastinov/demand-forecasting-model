"""Strict causal daily retraining of Direct, loss-0.75 P50, and bakery gate.

This is a research backtest only.  Every forecast for D is fitted on rows through
D-1; the SKU universe and SKU shares are also reconstructed only from D-1 data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = Path(
    os.environ.get(
        "HISTORICAL_GATE_INPUT",
        ROOT / ".codex_tmp/historical_gate_backtest/causal_mature_panel.parquet",
    )
)
OUTPUT = Path(
    os.environ.get(
        "HISTORICAL_GATE_OUTPUT",
        ROOT / "reports/historical_daily_retraining_gate_loss075_20260910",
    )
)
START = pd.Timestamp(os.environ.get("HISTORICAL_GATE_START", "2026-01-01"))
END = pd.Timestamp(os.environ.get("HISTORICAL_GATE_END", "2026-05-12"))
LOSS_WEIGHT = 0.75
FEATURES = [
    "bakery_id",
    "dow",
    "month",
    "day",
    "lag1",
    "lag2",
    "lag3",
    "lag7",
    "lag14",
    "lag28",
    "mean3",
    "mean7",
    "mean14",
    "mean28",
    "dow_mean4",
    "trend7_28",
]


def add_reconstructed_lost(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    available = (
        rows["release_qty"].fillna(0)
        + rows["incoming_move_qty"].fillna(0)
        - rows["outgoing_move_qty"].fillna(0)
    ).clip(lower=0)
    bakery_end = rows.groupby(["date", "bakery_id"])["last_sale_hour"].transform("max")
    elapsed = rows["last_sale_hour"] - 7.5
    remaining = (bakery_end - rows["last_sale_hour"]).clip(lower=0)
    eligible = (
        rows["is_mature_causal"]
        & rows["observed_sales_qty"].gt(0)
        & available.gt(0)
        & available.le(rows["observed_sales_qty"] + 1.0)
        & elapsed.ge(2.0)
        & remaining.gt(0)
    )
    raw = (rows["observed_sales_qty"] / elapsed * remaining).where(eligible, 0.0)
    cap = np.maximum(15.0, rows["observed_sales_qty"] * 1.5)
    rows["lost_reconstructed"] = raw.clip(lower=0, upper=cap).fillna(0)
    return rows


def daily_targets(rows: pd.DataFrame) -> pd.DataFrame:
    mature = rows[rows["is_mature_causal"]].copy()
    daily = mature.groupby(["date", "bakery_id"], as_index=False).agg(
        sales=("observed_sales_qty", "sum"),
        lost=("lost_reconstructed", "sum"),
    )
    bakeries = sorted(rows["bakery_id"].unique())
    dates = pd.date_range(rows["date"].min(), rows["date"].max())
    grid = pd.MultiIndex.from_product(
        [dates, bakeries], names=["date", "bakery_id"]
    ).to_frame(index=False)
    daily = grid.merge(daily, on=["date", "bakery_id"], how="left").fillna(
        {"sales": 0.0, "lost": 0.0}
    )
    daily["direct_target"] = daily["sales"]
    daily["p50_target"] = daily["sales"] + LOSS_WEIGHT * daily["lost"]
    return daily


def feature_frame(daily: pd.DataFrame, target: str) -> pd.DataFrame:
    frame = (
        daily[["date", "bakery_id", target]].copy().rename(columns={target: "target"})
    )
    frame = frame.sort_values(["bakery_id", "date"])
    grouped = frame.groupby("bakery_id", sort=False)["target"]
    for lag in [1, 2, 3, 7, 14, 28]:
        frame[f"lag{lag}"] = grouped.shift(lag)
    for window in [3, 7, 14, 28]:
        frame[f"mean{window}"] = grouped.transform(
            lambda x: x.shift(1).rolling(window, min_periods=2).mean()
        )
    frame["dow"] = frame["date"].dt.dayofweek
    frame["month"] = frame["date"].dt.month
    frame["day"] = frame["date"].dt.day
    frame["dow_mean4"] = grouped.transform(
        lambda x: x.shift(7).rolling(22, min_periods=1).mean()
    )
    frame["trend7_28"] = frame["mean7"] / frame["mean28"].replace(0, np.nan)
    return frame.replace([np.inf, -np.inf], np.nan)


def predict_one_day(frame: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    train = frame[
        frame["date"].between(
            date - pd.Timedelta(days=365), date - pd.Timedelta(days=1)
        )
    ].dropna(subset=["lag28"])
    test = frame[frame["date"].eq(date)].copy()
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=0.5,
        n_estimators=140,
        learning_rate=0.04,
        num_leaves=25,
        min_child_samples=80,
        reg_lambda=2.0,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )
    model.fit(
        train[FEATURES],
        train["target"],
        categorical_feature=["bakery_id", "dow", "month"],
    )
    test["prediction"] = np.maximum(model.predict(test[FEATURES]), 0)
    return test[["date", "bakery_id", "prediction"]]


def causal_sku_shares(rows: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    history = rows[
        rows["date"].between(date - pd.Timedelta(days=56), date - pd.Timedelta(days=1))
    ]
    active = (
        history[history["activity"]]
        .groupby(["bakery_id", "product_id"], as_index=False)
        .agg(
            product_name=("product_name", "last"),
            category_name=("category_name", "last"),
        )
    )

    def sums(start: int, end: int, name: str) -> pd.DataFrame:
        part = history[
            history["date"].between(
                date - pd.Timedelta(days=start), date - pd.Timedelta(days=end)
            )
        ]
        return (
            part.groupby(["bakery_id", "product_id"], as_index=False)[
                "observed_sales_qty"
            ]
            .sum()
            .rename(columns={"observed_sales_qty": name})
        )

    active = active.merge(sums(7, 1, "s7"), how="left").merge(
        sums(56, 1, "s56"), how="left"
    )
    weekdays = history[
        history["date"].isin([date - pd.Timedelta(days=7 * i) for i in range(1, 5)])
    ]
    sw = (
        weekdays.groupby(["bakery_id", "product_id"], as_index=False)[
            "observed_sales_qty"
        ]
        .sum()
        .rename(columns={"observed_sales_qty": "sw"})
    )
    active = active.merge(sw, how="left").fillna({"s7": 0.0, "s56": 0.0, "sw": 0.0})
    active["raw_share"] = (
        0.50 * active["s7"] / 7
        + 0.30 * active["sw"] / 4
        + 0.20 * active["s56"] / 56
        + 1e-6
    )
    active["share"] = active["raw_share"] / active.groupby("bakery_id")[
        "raw_share"
    ].transform("sum")
    active["date"] = date
    return active


def actual_for_day(rows: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    return rows[rows["date"].eq(date)][
        [
            "bakery_id",
            "product_id",
            "observed_sales_qty",
            "observed_sales_amount",
            "avg_sales_price",
            "lost_reconstructed",
        ]
    ]


def main() -> None:
    rows = pd.read_parquet(INPUT)
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    rows = add_reconstructed_lost(rows)
    daily = daily_targets(rows)
    frames = {
        name: feature_frame(daily, name) for name in ["direct_target", "p50_target"]
    }
    outputs = []
    for date in pd.date_range(START, END):
        forecasts = {}
        for name, frame in frames.items():
            forecasts[name] = predict_one_day(frame, date).rename(
                columns={"prediction": name.replace("_target", "_total")}
            )
        shares = causal_sku_shares(rows, date)
        day = shares.merge(forecasts["direct_target"], on=["date", "bakery_id"]).merge(
            forecasts["p50_target"], on=["date", "bakery_id"]
        )
        day["direct_plan"] = day["share"] * day["direct_total"]
        day["p50_plan"] = day["share"] * day["p50_total"]
        day = day.merge(
            actual_for_day(rows, date), on=["bakery_id", "product_id"], how="left"
        ).fillna(
            {
                "observed_sales_qty": 0.0,
                "observed_sales_amount": 0.0,
                "lost_reconstructed": 0.0,
            }
        )
        day["demand"] = day["observed_sales_qty"] + day["lost_reconstructed"]
        outputs.append(day)
        print(
            f"{date.date()} bakeries={day.bakery_id.nunique()} skus={len(day)}",
            flush=True,
        )
        if date.is_month_end or date == END:
            OUTPUT.mkdir(parents=True, exist_ok=True)
            pd.concat(outputs, ignore_index=True).to_parquet(
                OUTPUT / "predictions.partial.parquet", index=False
            )
    result = pd.concat(outputs, ignore_index=True)
    # Causal bakery gate: choose P50 only if its mean absolute demand error beat Direct
    # on available prior 14 days; the decision itself is shifted by one day.
    bd = result.groupby(["date", "bakery_id"], as_index=False).agg(
        demand=("demand", "sum"), direct=("direct_plan", "sum"), p50=("p50_plan", "sum")
    )
    bd["advantage"] = (bd["direct"] - bd["demand"]).abs() - (
        bd["p50"] - bd["demand"]
    ).abs()
    bd = bd.sort_values(["bakery_id", "date"])
    bd["gate_prior_advantage"] = bd.groupby("bakery_id")["advantage"].transform(
        lambda x: x.shift(1).rolling(14, min_periods=1).mean()
    )
    bd["use_p50"] = bd["gate_prior_advantage"].gt(0)
    result = result.merge(
        bd[["date", "bakery_id", "use_p50", "gate_prior_advantage"]],
        on=["date", "bakery_id"],
    )
    result["gate_plan"] = np.where(
        result["use_p50"], result["p50_plan"], result["direct_plan"]
    )
    summaries = []
    for period, part in result.groupby(result["date"].dt.to_period("M")):
        demand = part["demand"].sum()
        for variant in ["direct_plan", "p50_plan", "gate_plan"]:
            plan = part[variant]
            served = np.minimum(plan, part["demand"]).sum()
            summaries.append(
                {
                    "period": str(period),
                    "variant": variant,
                    "plan": float(plan.sum()),
                    "demand": float(demand),
                    "served": float(served),
                    "lost": float(demand - served),
                    "over": float(np.maximum(plan - part["demand"], 0).sum()),
                    "service_pct": float(100 * served / demand),
                }
            )
    summary = pd.DataFrame(summaries)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT / "predictions.parquet", index=False)
    summary.to_csv(OUTPUT / "monthly_summary.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "production_write": False,
        "start": str(START.date()),
        "end": str(END.date()),
        "loss_weight": LOSS_WEIGHT,
        "causal_cutoff": "D-1",
        "gate_window_days": 14,
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

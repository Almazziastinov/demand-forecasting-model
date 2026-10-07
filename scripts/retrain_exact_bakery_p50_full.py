"""Daily walk-forward enriched bakery P50 with corrected loss weight 0.75."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_demand_adjusted_bakery_target import rebuild_target_features  # noqa: E402
from src.experiments_v2.bakery_day_forecast import (  # noqa: E402
    BASE_FEATURES,
    TARGET_COL,
    build_model_frame,
    cast_category_columns,
)
from src.experiments_v2.common import predict_clipped, train_quantile  # noqa: E402


BASE = ROOT / ".codex_tmp/network_extension_20260826_bakery_daily_extended.csv.gz"
PANEL = ROOT / ".codex_tmp/historical_clickhouse_panel/causal_panel.parquet"
OUTPUT = ROOT / "reports/exact_bakery_p50_loss075_full_20260910"
START = pd.Timestamp("2026-01-02")
END = pd.Timestamp("2026-08-31")
LOSS_WEIGHT = 0.75


def main() -> None:
    base = pd.read_csv(BASE, encoding="utf-8-sig", low_memory=False)
    base["date"] = pd.to_datetime(base["date"]).dt.normalize()
    panel = pd.read_parquet(PANEL)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    pilot_ids = set(panel.loc[panel["date"].ge(START), "bakery_id"].unique())
    base = base[base["bakery_id"].isin(pilot_ids)].copy()
    daily = panel.groupby(["date", "bakery_id"], as_index=False).agg(
        sales=("observed_sales_qty", "sum"),
        lost=("lost_reconstructed", "sum")
        if "lost_reconstructed" in panel
        else ("observed_sales_qty", lambda x: 0.0),
    )
    if "lost_reconstructed" not in panel:
        available = (
            panel["release_qty"].fillna(0)
            + panel["incoming_move_qty"].fillna(0)
            - panel["outgoing_move_qty"].fillna(0)
        ).clip(lower=0)
        bakery_end = panel.groupby(["date", "bakery_id"])["last_sale_hour"].transform(
            "max"
        )
        elapsed = panel["last_sale_hour"] - 7.5
        remaining = (bakery_end - panel["last_sale_hour"]).clip(lower=0)
        eligible = (
            panel["is_mature_causal"]
            & panel["observed_sales_qty"].gt(0)
            & available.gt(0)
            & available.le(panel["observed_sales_qty"] + 1)
            & elapsed.ge(2)
            & remaining.gt(0)
        )
        raw = (
            (panel["observed_sales_qty"] / elapsed * remaining)
            .where(eligible, 0)
            .clip(lower=0)
        )
        cap = pd.concat(
            [panel["observed_sales_qty"] * 1.5, pd.Series(15.0, index=panel.index)],
            axis=1,
        ).max(axis=1)
        panel["lost"] = raw.clip(upper=cap).fillna(0)
        daily = panel.groupby(["date", "bakery_id"], as_index=False).agg(
            sales=("observed_sales_qty", "sum"), lost=("lost", "sum")
        )
    attributes = (
        base.sort_values("date")
        .groupby("bakery_id", as_index=False)
        .last()[["bakery_id", "bakery_name", "city"]]
    )
    future = (
        daily[daily["date"].gt(base["date"].max())]
        .rename(columns={"sales": TARGET_COL})
        .merge(attributes, on="bakery_id", how="left")
    )
    for column in base.columns:
        if column not in future:
            future[column] = pd.NA
    base = pd.concat([base, future[base.columns]], ignore_index=True).drop_duplicates(
        ["date", "bakery_id"], keep="last"
    )
    adjusted = base.merge(
        daily[["date", "bakery_id", "lost"]], on=["date", "bakery_id"], how="left"
    )
    adjusted["lost"] = adjusted["lost"].fillna(0)
    predictions = []
    for date in pd.date_range(START, END):
        work = adjusted.copy()
        history = work["date"].lt(date)
        work.loc[history, TARGET_COL] += LOSS_WEIGHT * work.loc[history, "lost"]
        frame = build_model_frame(rebuild_target_features(work.drop(columns="lost")))
        train = frame[frame["date"].lt(date)]
        test = frame[frame["date"].eq(date) & frame["bakery_id"].isin(pilot_ids)]
        features = [column for column in BASE_FEATURES if column in frame]
        train_x, test_x = cast_category_columns(
            train[features].copy(), test[features].copy(), features
        )
        model = train_quantile(
            train_x,
            train[TARGET_COL],
            alpha=0.5,
            params={
                "n_estimators": 180,
                "learning_rate": 0.04,
                "num_leaves": 31,
                "min_child_samples": 100,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": 2.0,
                "random_state": 42,
                "verbosity": -1,
            },
        )
        result = test[["date", "bakery_id"]].copy()
        result["p50_total_exact"] = predict_clipped(model, test_x)
        predictions.append(result)
        print(date.date(), len(result), flush=True)
    output = pd.concat(predictions, ignore_index=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT / "bakery_totals.parquet", index=False)


if __name__ == "__main__":
    main()

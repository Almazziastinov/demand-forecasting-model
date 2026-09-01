"""Generate a local predictive-choice shadow for one explicit production run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.backtest_current_sku_allocation import concentration  # noqa: E402
from scripts.backtest_rebuilt_predictive_choice import (  # noqa: E402
    build_features,
    fit_predict,
)


HISTORICAL_SNAPSHOTS = (
    ROOT / ".codex_tmp/predictive_choice_rebuild_20260825/historical_snapshots.parquet"
)
SHADOW_INPUT = ROOT / ".codex_tmp/predictive_choice_shadow_20260825"
OUTPUT = ROOT / "reports/predictive_choice_shadow_20260825"
RUN_ID = "prod_base_bakery_norm_recent_20260825_h14"
FORECAST_DATE = pd.Timestamp("2026-08-25")


def main() -> None:
    historical = pd.read_parquet(HISTORICAL_SNAPSHOTS).rename(
        columns={"forecast_qty": "incumbent_sku_forecast"}
    )
    shadow = pd.read_parquet(SHADOW_INPUT / "shadow_snapshot.parquet").rename(
        columns={"forecast_qty": "incumbent_sku_forecast"}
    )
    history = pd.read_parquet(SHADOW_INPUT / "sales_history_through_20260824.parquet")
    for frame in (historical, shadow, history):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["product_id"] = pd.to_numeric(frame["product_id"]).astype("int64")

    combined = pd.concat([historical, shadow], ignore_index=True)
    features = build_features(combined, history)
    prediction = fit_predict(features, "2026-08-23", ["2026-08-25"])

    prior = (
        history[history["date"].lt(FORECAST_DATE)]
        .groupby(["bakery_id", "product_id"])
        .agg(
            first_sale=("date", "min"),
            last_sale=("date", "max"),
            sales_days=("date", "nunique"),
        )
    )
    prediction = prediction.merge(prior, on=["bakery_id", "product_id"], how="left")
    prediction["cold_start"] = prediction["sales_days"].fillna(0).lt(7)
    prediction["no_prior_sales"] = prediction["sales_days"].isna()

    bakery_recent = (
        history[history["date"].between("2026-08-11", "2026-08-24")]
        .groupby("bakery_id")["sold"]
        .sum()
    )
    prediction["bakery_sales_14d"] = (
        prediction["bakery_id"].map(bakery_recent).fillna(0)
    )
    prediction["unobservable_bakery"] = prediction["bakery_sales_14d"].le(0)

    incumbent_total = prediction["incumbent_sku_forecast"].sum()
    predictive_total = prediction["predictive_forecast"].sum()
    category_delta = (
        prediction.groupby(["date", "bakery_id", "category"])[
            "predictive_forecast"
        ].sum()
        - prediction.groupby(["date", "bakery_id", "category"])[
            "incumbent_sku_forecast"
        ].sum()
    )
    summary = {
        "source_run_id": RUN_ID,
        "forecast_date": str(FORECAST_DATE.date()),
        "scope": {
            "rows": int(len(prediction)),
            "bakeries": int(prediction["bakery_id"].nunique()),
            "products": int(prediction["product_id"].nunique()),
        },
        "totals": {
            "incumbent": float(incumbent_total),
            "predictive": float(predictive_total),
            "delta": float(predictive_total - incumbent_total),
            "max_abs_category_delta": float(category_delta.abs().max()),
        },
        "concentration": {
            "incumbent": concentration(prediction, "incumbent_sku_forecast"),
            "predictive": concentration(prediction, "predictive_forecast"),
        },
        "coverage": {
            "cold_start_rows": int(prediction["cold_start"].sum()),
            "cold_start_incumbent_mass": float(
                prediction.loc[prediction["cold_start"], "incumbent_sku_forecast"].sum()
            ),
            "no_prior_sales_rows": int(prediction["no_prior_sales"].sum()),
            "unobservable_bakeries": int(
                prediction.loc[prediction["unobservable_bakery"], "bakery_id"].nunique()
            ),
            "unobservable_incumbent_mass": float(
                prediction.loc[
                    prediction["unobservable_bakery"], "incumbent_sku_forecast"
                ].sum()
            ),
        },
        "production_write": False,
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    prediction.to_parquet(OUTPUT / "shadow_predictions.parquet", index=False)
    prediction[["date", "bakery_id", "product_id", "predictive_forecast"]].rename(
        columns={"predictive_forecast": "forecast_qty"}
    ).to_csv(OUTPUT / "forecast_override_20260825.csv", index=False)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

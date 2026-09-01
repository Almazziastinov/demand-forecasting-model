"""Train network bakery quantiles on calibrated post-last-sale demand."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_demand_adjusted_bakery_target import rebuild_target_features  # noqa: E402
from src.experiments_v2.bakery_day_forecast import (  # noqa: E402
    BASE_FEATURES,
    BAKERY_ID_COL,
    BAKERY_NAME_COL,
    DATE_COL,
    TARGET_COL,
    build_model_frame,
    cast_category_columns,
)
from src.experiments_v2.common import predict_clipped, train_quantile  # noqa: E402

DATASET = ROOT / ".codex_tmp/network_extension_20260826_bakery_daily_extended.csv.gz"
LABELS = ROOT / "reports/calibrated_stockout_network_20260826/sku_day_demand.csv"
ROWS = ROOT / "reports/network_quantile_operational_balance_20260826/rows.parquet"
OUTPUT = ROOT / "reports/calibrated_network_quantiles_20260826"
CUTOFF = pd.Timestamp("2026-08-10")
QUANTILES = (0.50, 0.55, 0.60, 0.67, 0.75, 0.80, 0.85, 0.90, 0.95)


def main() -> None:
    base = pd.read_csv(DATASET, encoding="utf-8-sig", low_memory=False)
    base[DATE_COL] = pd.to_datetime(base[DATE_COL]).dt.normalize()
    labels = pd.read_csv(
        LABELS,
        usecols=["date", "bakery_id", "imputed_demand"],
        encoding="utf-8-sig",
    )
    labels[DATE_COL] = pd.to_datetime(labels[DATE_COL]).dt.normalize()
    daily_lost = labels.groupby([DATE_COL, BAKERY_ID_COL], as_index=False)["imputed_demand"].sum()

    adjusted = base.merge(daily_lost, on=[DATE_COL, BAKERY_ID_COL], how="left")
    adjusted["imputed_demand"] = adjusted["imputed_demand"].fillna(0.0)
    adjusted.loc[adjusted[DATE_COL] <= CUTOFF, TARGET_COL] += adjusted.loc[
        adjusted[DATE_COL] <= CUTOFF, "imputed_demand"
    ]
    model_frame = build_model_frame(rebuild_target_features(adjusted.drop(columns="imputed_demand")))

    operational = pd.read_parquet(ROWS)
    test_dates = sorted(pd.to_datetime(operational[DATE_COL]).dt.normalize().unique())
    bakery_ids = set(operational[BAKERY_ID_COL].astype(int).unique())
    train = model_frame[model_frame[DATE_COL] <= CUTOFF].copy()
    test = model_frame[
        model_frame[DATE_COL].isin(test_dates) & model_frame[BAKERY_ID_COL].isin(bakery_ids)
    ].copy()
    feature_cols = [column for column in BASE_FEATURES if column in model_frame.columns]
    train_x, test_x = cast_category_columns(
        train[feature_cols].copy(), test[feature_cols].copy(), feature_cols
    )

    parts = []
    for quantile in QUANTILES:
        model = train_quantile(
            train_x,
            train[TARGET_COL],
            alpha=quantile,
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
        part = test[[DATE_COL, BAKERY_ID_COL, BAKERY_NAME_COL]].copy()
        part["prediction"] = predict_clipped(model, test_x)
        part["variant"] = f"p{round(quantile * 100):02d}"
        parts.append(part)

    predictions = pd.concat(parts, ignore_index=True).merge(
        base[[DATE_COL, BAKERY_ID_COL, TARGET_COL]], on=[DATE_COL, BAKERY_ID_COL], how="left"
    ).merge(daily_lost, on=[DATE_COL, BAKERY_ID_COL], how="left")
    predictions = predictions.rename(columns={TARGET_COL: "bakery_sales", "imputed_demand": "calibrated_lost"})
    predictions["calibrated_lost"] = predictions["calibrated_lost"].fillna(0.0)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(OUTPUT / "predictions.parquet", index=False)
    print(predictions.groupby("variant").agg(prediction=("prediction", "sum"), sales=("bakery_sales", "sum"), lost=("calibrated_lost", "sum")).to_string())


if __name__ == "__main__":
    main()

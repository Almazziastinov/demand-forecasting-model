"""Causal rolling comparison of current, predictive, P50, and simple SKU floor."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_demand_adjusted_bakery_target import rebuild_target_features  # noqa: E402
from src.experiments_v2.bakery_day_forecast import (  # noqa: E402
    BASE_FEATURES,
    BAKERY_ID_COL,
    DATE_COL,
    TARGET_COL,
    build_model_frame,
    cast_category_columns,
)
from src.experiments_v2.common import predict_clipped, train_quantile  # noqa: E402

DATASET = ROOT / ".codex_tmp/network_extension_20260826_bakery_daily_extended.csv.gz"
LABELS = ROOT / "reports/relaxed_stockout_network_20260826/sku_day_demand.csv"
PREDICTIONS = ROOT / "reports/rebuilt_predictive_choice_20260825/predictions.parquet"
COEFFICIENTS = ROOT / "reports/rolling_post_last_sale_calibration_20260826/fold_coefficients.csv"
OUTPUT = ROOT / "reports/rolling_floor_vs_no_floor_20260826"
KEYS = ["date", "bakery_id", "product_id"]
FOLDS = {
    "2026-07-20": ("2026-07-19", "2026-07-22", "2026-07-26"),
    "2026-07-27": ("2026-07-26", "2026-07-27", "2026-08-02"),
    "2026-08-10": ("2026-08-09", "2026-08-11", "2026-08-13"),
    "2026-08-17": ("2026-08-16", "2026-08-17", "2026-08-23"),
}


def score(frame: pd.DataFrame, column: str, variant: str, fold: str) -> dict:
    error = frame[column] - frame["demand"]
    return {
        "fold": fold,
        "variant": variant,
        "dates": frame["date"].nunique(),
        "rows": len(frame),
        "volume": frame[column].sum(),
        "surplus": error.clip(lower=0).sum(),
        "underbake": (-error).clip(lower=0).sum(),
        "imbalance": error.abs().sum(),
    }


def calibrated_labels(source: pd.DataFrame, coefficients: pd.DataFrame) -> pd.DataFrame:
    work = source.copy()
    multiplier = np.interp(
        work["last_sale_hour"].fillna(coefficients["cutoff"].min()),
        coefficients["cutoff"],
        coefficients["multiplier"],
    )
    selected = work["is_clear_stockout"].fillna(False).astype(bool)
    work["lost"] = np.where(selected, work["raw_imputed_demand"] * multiplier, 0.0)
    work["demand"] = work["demand_lower_bound"] + work["lost"]
    return work


def main() -> None:
    base = pd.read_csv(DATASET, encoding="utf-8-sig", low_memory=False)
    base[DATE_COL] = pd.to_datetime(base[DATE_COL]).dt.normalize()
    labels = pd.read_csv(
        LABELS,
        usecols=[*KEYS, "demand_lower_bound", "is_clear_stockout", "last_sale_hour", "raw_imputed_demand"],
        encoding="utf-8-sig",
        low_memory=False,
    )
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    predictions = pd.read_parquet(PREDICTIONS)
    coefficients = pd.read_csv(COEFFICIENTS)
    coefficients["fold"] = coefficients["fold"].astype(str)
    results = []
    scored_parts = []

    for fold, (cutoff_value, start_value, end_value) in FOLDS.items():
        cutoff = pd.Timestamp(cutoff_value)
        start = pd.Timestamp(start_value)
        end = pd.Timestamp(end_value)
        fold_coefficients = coefficients[coefficients["fold"].eq(fold)].sort_values("cutoff")
        fold_labels = calibrated_labels(labels[labels["date"] <= end], fold_coefficients)
        daily_lost = fold_labels.groupby(["date", "bakery_id"], as_index=False)["lost"].sum()
        adjusted = base.merge(daily_lost, on=["date", "bakery_id"], how="left")
        adjusted["lost"] = adjusted["lost"].fillna(0.0)
        adjusted.loc[adjusted["date"] <= cutoff, TARGET_COL] += adjusted.loc[
            adjusted["date"] <= cutoff, "lost"
        ]
        model_frame = build_model_frame(rebuild_target_features(adjusted.drop(columns="lost")))
        train = model_frame[model_frame["date"] <= cutoff].copy()
        fold_predictions = predictions[predictions["date"].between(start, end)].copy()
        test_dates = fold_predictions["date"].unique()
        bakery_ids = fold_predictions["bakery_id"].unique()
        test = model_frame[
            model_frame["date"].isin(test_dates) & model_frame["bakery_id"].isin(bakery_ids)
        ].copy()
        feature_cols = [column for column in BASE_FEATURES if column in model_frame.columns]
        train_x, test_x = cast_category_columns(
            train[feature_cols].copy(), test[feature_cols].copy(), feature_cols
        )
        model = train_quantile(
            train_x,
            train[TARGET_COL],
            alpha=0.50,
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
        bakery_prediction = test[["date", "bakery_id"]].copy()
        bakery_prediction["p50_bakery"] = predict_clipped(model, test_x)
        bakery_totals = fold_predictions.groupby(["date", "bakery_id"], as_index=False).agg(
            predictive_total=("predictive_forecast", "sum")
        )
        factors = bakery_totals.merge(bakery_prediction, on=["date", "bakery_id"], validate="one_to_one")
        factors["p50_factor"] = factors["p50_bakery"] / factors["predictive_total"].replace(0.0, np.nan)
        scored = fold_predictions.merge(
            factors[["date", "bakery_id", "p50_factor"]], on=["date", "bakery_id"], validate="many_to_one"
        ).merge(
            fold_labels[KEYS + ["demand"]], on=KEYS, how="left", validate="one_to_one"
        )
        scored["demand"] = scored["demand"].fillna(0.0)
        scored["p50_predictive"] = scored["predictive_forecast"] * scored["p50_factor"]

        references = []
        for date in sorted(scored["date"].unique()):
            history = fold_labels[
                fold_labels["date"].between(date - pd.Timedelta(days=56), date - pd.Timedelta(days=1))
                & fold_labels["date"].dt.dayofweek.eq(date.dayofweek)
                & fold_labels["demand"].gt(0)
            ]
            reference = history.groupby(["bakery_id", "product_id"], as_index=False).agg(
                history_n=("demand", "size"), history_p67=("demand", lambda values: values.quantile(0.67))
            )
            reference["date"] = date
            references.append(reference)
        reference_frame = pd.concat(references, ignore_index=True)
        scored = scored.merge(reference_frame, on=KEYS, how="left", validate="many_to_one")
        eligible = scored["history_n"].fillna(0).ge(8)
        floor = 0.95 * scored["history_p67"].fillna(0.0)
        scored["p50_simple_floor"] = np.where(
            eligible,
            np.maximum(scored["p50_predictive"], np.minimum(floor, scored["p50_predictive"] + 8.0)),
            scored["p50_predictive"],
        )
        for column, variant in [
            ("incumbent_sku_forecast", "current"),
            ("predictive_forecast", "predictive_same_volume"),
            ("p50_predictive", "p50_predictive"),
            ("p50_simple_floor", "p50_predictive_simple_floor"),
        ]:
            results.append(score(scored, column, variant, fold))
        scored["rolling_fold"] = fold
        scored_parts.append(scored)

    metric_frame = pd.DataFrame(results)
    summary = metric_frame.groupby("variant", as_index=False).agg(
        folds=("fold", "nunique"),
        dates=("dates", "sum"),
        volume=("volume", "sum"),
        surplus=("surplus", "sum"),
        underbake=("underbake", "sum"),
        imbalance=("imbalance", "sum"),
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metric_frame.to_csv(OUTPUT / "fold_metrics.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    pd.concat(scored_parts, ignore_index=True).to_parquet(OUTPUT / "rows.parquet", index=False)
    print(summary.to_string(index=False))
    print("\nFold metrics")
    print(metric_frame.to_string(index=False))


if __name__ == "__main__":
    main()

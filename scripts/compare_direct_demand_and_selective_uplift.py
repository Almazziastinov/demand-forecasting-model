"""Compare direct demand-target training with a two-stage selective uplift."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_stockout_adjusted_bakery_targets import (  # noqa: E402
    CONSERVATIVE,
    build_adjustment_variants,
    build_evaluation_adjustments,
)
from scripts.experiment_demand_adjusted_bakery_target import (  # noqa: E402
    DEFAULT_RAW,
    extend_with_pilot_actuals,
    rebuild_target_features,
)
from scripts.experiment_demand_adjusted_profiles import load_hourly_raw  # noqa: E402
from src.experiments_v2.bakery_day_forecast import (  # noqa: E402
    BASE_FEATURES,
    BAKERY_ID_COL,
    DATE_COL,
    TARGET_COL,
    build_model_frame,
    cast_category_columns,
)

DATASET = ROOT / "data/processed/stg_daily_v1/bakery_daily_sales.csv"
DEMAND = ROOT / "reports/stockout_adjusted_demand_dataset/sku_day_demand.csv"
DIRECT = ROOT / "reports/stockout_adjusted_bakery_target_experiment/predictions.csv"
OUTPUT = ROOT / "reports/direct_demand_vs_selective_uplift_20260825"
CUTOFFS = ["2026-06-21", "2026-06-28", "2026-07-05"]
HOLDOUT_DAYS = 14


def metrics(rows: pd.DataFrame, prediction_col: str) -> dict[str, float | int]:
    prediction = rows[prediction_col].clip(lower=0.0)
    sold = rows[TARGET_COL].clip(lower=0.0)
    demand = sold + rows["conservative_imputed"].clip(lower=0.0)
    lost = demand - sold
    recognized = np.minimum(lost, (prediction - sold).clip(lower=0.0))
    error = prediction - demand
    return {
        "rows": int(len(rows)),
        "forecast_qty": float(prediction.sum()),
        "demand_qty": float(demand.sum()),
        "lost_qty": float(lost.sum()),
        "recognized_lost_qty": float(recognized.sum()),
        "recognized_lost_pct": float(100 * recognized.sum() / lost.sum())
        if lost.sum() > 0
        else 0.0,
        "wape_pct": float(100 * error.abs().sum() / demand.sum()),
        "bias_pct": float(100 * error.sum() / demand.sum()),
        "underforecast_qty": float((-error).clip(lower=0.0).sum()),
        "true_overforecast_qty": float(error.clip(lower=0.0).sum()),
        "true_overforecast_rows": int(error.gt(0).sum()),
    }


def selective_predictions(
    frame: pd.DataFrame,
    additions: pd.DataFrame,
    baseline: pd.DataFrame,
    pilot_ids: set[int],
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    feature_cols = [column for column in BASE_FEATURES if column in frame.columns]
    label_start = additions[DATE_COL].min()
    labels = additions[[DATE_COL, BAKERY_ID_COL, "imputed_demand"]]
    train = frame[
        frame[DATE_COL].between(label_start, cutoff)
        & frame[BAKERY_ID_COL].isin(pilot_ids)
    ].merge(labels, on=[DATE_COL, BAKERY_ID_COL], how="left")
    train["imputed_demand"] = train["imputed_demand"].fillna(0.0)
    test_end = cutoff + pd.Timedelta(days=HOLDOUT_DAYS)
    test = frame[
        frame[DATE_COL].gt(cutoff)
        & frame[DATE_COL].le(test_end)
        & frame[BAKERY_ID_COL].isin(pilot_ids)
    ].copy()
    train_x, test_x = cast_category_columns(
        train[feature_cols].copy(), test[feature_cols].copy(), feature_cols
    )
    label = train["imputed_demand"].gt(0).astype(int)
    classifier = lgb.LGBMClassifier(
        n_estimators=120,
        learning_rate=0.04,
        num_leaves=15,
        min_child_samples=30,
        reg_lambda=2.0,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    )
    classifier.fit(train_x, label)
    positive = train["imputed_demand"].gt(0)
    regressor = lgb.LGBMRegressor(
        objective="l1",
        n_estimators=120,
        learning_rate=0.04,
        num_leaves=15,
        min_child_samples=20,
        reg_lambda=2.0,
        random_state=42,
        verbosity=-1,
    )
    regressor.fit(train_x.loc[positive], train.loc[positive, "imputed_demand"])
    result = test[[DATE_COL, BAKERY_ID_COL]].copy()
    result["uplift_probability"] = classifier.predict_proba(test_x)[:, 1]
    result["uplift_amount_if_needed"] = np.maximum(regressor.predict(test_x), 0.0)
    result = result.merge(
        baseline[[DATE_COL, BAKERY_ID_COL, "baseline_prediction"]],
        on=[DATE_COL, BAKERY_ID_COL],
        how="inner",
        validate="one_to_one",
    )
    result["raw_selective_uplift"] = (
        result["uplift_probability"] * result["uplift_amount_if_needed"]
    )
    result["selective_uplift"] = np.minimum(
        result["raw_selective_uplift"], 0.08 * result["baseline_prediction"]
    )
    result["selective_prediction"] = (
        result["baseline_prediction"] + result["selective_uplift"]
    )
    return result


def main() -> None:
    dataset = pd.read_csv(DATASET, encoding="utf-8-sig", low_memory=False)
    dataset[DATE_COL] = pd.to_datetime(dataset[DATE_COL]).dt.normalize()
    demand = pd.read_csv(DEMAND, encoding="utf-8-sig", low_memory=False)
    demand[DATE_COL] = pd.to_datetime(demand[DATE_COL]).dt.normalize()
    demand[BAKERY_ID_COL] = demand[BAKERY_ID_COL].astype(int)
    pilot_ids = set(demand[BAKERY_ID_COL].unique())
    hourly = load_hourly_raw(
        DEFAULT_RAW,
        bakery_ids=pilot_ids,
        date_from=dataset[DATE_COL].max() + pd.Timedelta(days=1),
        date_to=max(pd.Timestamp(value) for value in CUTOFFS)
        + pd.Timedelta(days=HOLDOUT_DAYS),
    )
    dataset = extend_with_pilot_actuals(dataset, hourly)
    frame = build_model_frame(rebuild_target_features(dataset))
    additions = build_adjustment_variants(demand)
    additions = additions[additions["variant"].eq(CONSERVATIVE)].drop(columns="variant")
    evaluation = build_evaluation_adjustments(demand)[
        [DATE_COL, BAKERY_ID_COL, "conservative_imputed"]
    ]
    direct = pd.read_csv(DIRECT, encoding="utf-8-sig")
    direct[DATE_COL] = pd.to_datetime(direct[DATE_COL]).dt.normalize()

    prediction_parts = []
    metric_rows = []
    for cutoff_value in CUTOFFS:
        cutoff = pd.Timestamp(cutoff_value)
        fold = direct[direct["cutoff"].eq(cutoff_value)]
        baseline = fold[fold["variant"].eq("observed_sales_target")].rename(
            columns={"prediction": "baseline_prediction"}
        )
        direct_demand = fold[fold["variant"].eq(CONSERVATIVE)].rename(
            columns={"prediction": "direct_demand_prediction"}
        )
        selective = selective_predictions(frame, additions, baseline, pilot_ids, cutoff)
        result = baseline[
            [DATE_COL, BAKERY_ID_COL, "bakery_name", TARGET_COL, "baseline_prediction"]
        ].merge(
            direct_demand[[DATE_COL, BAKERY_ID_COL, "direct_demand_prediction"]],
            on=[DATE_COL, BAKERY_ID_COL],
            validate="one_to_one",
        )
        result = result.merge(
            selective.drop(columns="baseline_prediction"),
            on=[DATE_COL, BAKERY_ID_COL],
            validate="one_to_one",
        )
        result = result.merge(
            evaluation, on=[DATE_COL, BAKERY_ID_COL], how="left", validate="one_to_one"
        )
        result["conservative_imputed"] = result["conservative_imputed"].fillna(0.0)
        result["cutoff"] = cutoff_value
        prediction_parts.append(result)
        for variant, column in [
            ("sales_target", "baseline_prediction"),
            ("direct_demand_target", "direct_demand_prediction"),
            ("selective_uplift", "selective_prediction"),
        ]:
            metric_rows.append(
                {"cutoff": cutoff_value, "variant": variant, **metrics(result, column)}
            )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    metric_frame = pd.DataFrame(metric_rows)
    summary = (
        metric_frame.groupby("variant", as_index=False)
        .agg(
            folds=("cutoff", "nunique"),
            mean_forecast_qty=("forecast_qty", "mean"),
            mean_wape_pct=("wape_pct", "mean"),
            mean_bias_pct=("bias_pct", "mean"),
            mean_recognized_lost_pct=("recognized_lost_pct", "mean"),
            mean_true_overforecast_qty=("true_overforecast_qty", "mean"),
            mean_true_overforecast_rows=("true_overforecast_rows", "mean"),
        )
        .sort_values("mean_wape_pct")
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(OUTPUT / "predictions.parquet", index=False)
    metric_frame.to_csv(OUTPUT / "metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT / "summary.csv", index=False, encoding="utf-8-sig")
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary.to_dict("records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

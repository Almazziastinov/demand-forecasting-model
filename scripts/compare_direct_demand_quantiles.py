"""Frozen-fold comparison of direct-demand LightGBM quantiles."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_direct_demand_and_selective_uplift import (  # noqa: E402
    CUTOFFS,
    DEMAND,
    DIRECT,
    HOLDOUT_DAYS,
    metrics,
)
from scripts.experiment_demand_adjusted_bakery_target import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_RAW,
    apply_training_adjustments,
    extend_with_pilot_actuals,
    rebuild_target_features,
)
from scripts.experiment_demand_adjusted_profiles import load_hourly_raw  # noqa: E402
from scripts.experiment_stockout_adjusted_bakery_targets import (  # noqa: E402
    CONSERVATIVE,
    build_adjustment_variants,
    build_evaluation_adjustments,
)
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

OUTPUT = ROOT / "reports/direct_demand_quantiles_20260825"
QUANTILES = (0.50, 0.55, 0.60, 2 / 3, 0.75)


def predict_quantile(
    frame: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    holdout_end: pd.Timestamp,
    pilot_ids: set[int],
    quantile: float,
) -> pd.DataFrame:
    model_frame = build_model_frame(rebuild_target_features(frame))
    train = model_frame[model_frame[DATE_COL].le(cutoff)].copy()
    test = model_frame[
        model_frame[DATE_COL].gt(cutoff)
        & model_frame[DATE_COL].le(holdout_end)
        & model_frame[BAKERY_ID_COL].isin(pilot_ids)
    ].copy()
    feature_cols = [column for column in BASE_FEATURES if column in model_frame.columns]
    train_x, test_x = cast_category_columns(
        train[feature_cols].copy(), test[feature_cols].copy(), feature_cols
    )
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
    result = test[[DATE_COL, BAKERY_ID_COL, BAKERY_NAME_COL]].copy()
    result["prediction"] = predict_clipped(model, test_x)
    return result


def main() -> None:
    demand = pd.read_csv(DEMAND, encoding="utf-8-sig", low_memory=False)
    demand[DATE_COL] = pd.to_datetime(demand[DATE_COL]).dt.normalize()
    demand[BAKERY_ID_COL] = demand[BAKERY_ID_COL].astype(int)
    pilot_ids = set(demand[BAKERY_ID_COL].unique())
    additions = build_adjustment_variants(demand)
    additions = additions[additions["variant"].eq(CONSERVATIVE)].drop(columns="variant")
    evaluation = build_evaluation_adjustments(demand)[
        [DATE_COL, BAKERY_ID_COL, "conservative_imputed"]
    ]

    base = pd.read_csv(DEFAULT_DATASET, encoding="utf-8-sig", low_memory=False)
    base[DATE_COL] = pd.to_datetime(base[DATE_COL]).dt.normalize()
    max_holdout = max(pd.Timestamp(value) for value in CUTOFFS) + pd.Timedelta(
        days=HOLDOUT_DAYS
    )
    hourly = load_hourly_raw(
        DEFAULT_RAW,
        bakery_ids=pilot_ids,
        date_from=base[DATE_COL].max() + pd.Timedelta(days=1),
        date_to=max_holdout,
    )
    base = extend_with_pilot_actuals(base, hourly)
    observed = base[[DATE_COL, BAKERY_ID_COL, TARGET_COL]].copy()
    direct = pd.read_csv(DIRECT, encoding="utf-8-sig")
    direct[DATE_COL] = pd.to_datetime(direct[DATE_COL]).dt.normalize()

    predictions = []
    metric_rows = []
    for cutoff_value in CUTOFFS:
        cutoff = pd.Timestamp(cutoff_value)
        holdout_end = cutoff + pd.Timedelta(days=HOLDOUT_DAYS)
        adjusted = apply_training_adjustments(base, additions, train_end=cutoff)
        baseline = direct[
            direct["cutoff"].eq(cutoff_value)
            & direct["variant"].eq("observed_sales_target")
        ].rename(columns={"prediction": "sales_target_prediction"})
        for quantile in QUANTILES:
            variant = f"p{round(quantile * 100):02d}"
            result = predict_quantile(
                adjusted,
                cutoff=cutoff,
                holdout_end=holdout_end,
                pilot_ids=pilot_ids,
                quantile=quantile,
            )
            result = result.merge(
                observed,
                on=[DATE_COL, BAKERY_ID_COL],
                how="left",
                validate="one_to_one",
            )
            result = result.merge(
                evaluation,
                on=[DATE_COL, BAKERY_ID_COL],
                how="left",
                validate="one_to_one",
            )
            result = result.merge(
                baseline[[DATE_COL, BAKERY_ID_COL, "sales_target_prediction"]],
                on=[DATE_COL, BAKERY_ID_COL],
                validate="one_to_one",
            )
            result["conservative_imputed"] = result["conservative_imputed"].fillna(0.0)
            result["cutoff"] = cutoff_value
            result["variant"] = variant
            result["quantile"] = quantile
            predictions.append(result)
            metric_rows.append(
                {
                    "cutoff": cutoff_value,
                    "variant": variant,
                    "quantile": quantile,
                    **metrics(result, "prediction"),
                }
            )

    prediction_frame = pd.concat(predictions, ignore_index=True)
    metric_frame = pd.DataFrame(metric_rows)
    baseline_rows = []
    for cutoff_value, part in prediction_frame.groupby("cutoff"):
        sample = part[part["variant"].eq("p50")]
        baseline_rows.append(
            {
                "cutoff": cutoff_value,
                "variant": "sales_target",
                "quantile": None,
                **metrics(sample, "sales_target_prediction"),
            }
        )
    metric_frame = pd.concat(
        [pd.DataFrame(baseline_rows), metric_frame], ignore_index=True
    )
    baseline_mean = metric_frame[metric_frame["variant"].eq("sales_target")][
        ["underforecast_qty", "true_overforecast_qty"]
    ].mean()
    summary = metric_frame.groupby("variant", as_index=False).agg(
        quantile=("quantile", "first"),
        folds=("cutoff", "nunique"),
        mean_forecast_qty=("forecast_qty", "mean"),
        mean_wape_pct=("wape_pct", "mean"),
        mean_bias_pct=("bias_pct", "mean"),
        mean_recognized_lost_pct=("recognized_lost_pct", "mean"),
        mean_underforecast_qty=("underforecast_qty", "mean"),
        mean_overforecast_qty=("true_overforecast_qty", "mean"),
        mean_overforecast_rows=("true_overforecast_rows", "mean"),
    )
    summary["underforecast_reduction"] = (
        baseline_mean["underforecast_qty"] - summary["mean_underforecast_qty"]
    )
    summary["additional_overforecast"] = (
        summary["mean_overforecast_qty"] - baseline_mean["true_overforecast_qty"]
    )
    summary["break_even_underbake_weight"] = summary[
        "additional_overforecast"
    ] / summary["underforecast_reduction"].replace(0.0, pd.NA)
    summary = summary.sort_values("quantile", na_position="first")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_parquet(OUTPUT / "predictions.parquet", index=False)
    metric_frame.to_csv(OUTPUT / "metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT / "summary.csv", index=False, encoding="utf-8-sig")
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary.to_dict("records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

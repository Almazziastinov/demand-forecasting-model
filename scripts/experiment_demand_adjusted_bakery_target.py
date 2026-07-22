"""Offline A/B of observed versus stockout-adjusted bakery-day targets."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments_v2.bakery_day_forecast import (  # noqa: E402
    BASE_FEATURES,
    BAKERY_ID_COL,
    BAKERY_NAME_COL,
    CITY_COL,
    DATE_COL,
    TARGET_COL,
    build_model_frame,
    cast_category_columns,
)
from src.experiments_v2.build_bakery_daily_dataset import (  # noqa: E402
    add_calendar_features,
    add_lag_features,
)
from src.experiments_v2.common import predict_clipped, train_lgbm  # noqa: E402
from scripts.experiment_demand_adjusted_profiles import load_hourly_raw  # noqa: E402

DEFAULT_DATASET = ROOT / "data/processed/stg_daily_v1/bakery_daily_sales.csv"
DEFAULT_RAW = ROOT / "data/raw/pilot_stg_check_lines_2026-04-30_2026-07-19.csv"
DEFAULT_ADJUSTMENTS = ROOT / "reports/demand_adjusted_stockout_history_all_non_allocation/hourly_adjustments.csv"
DEFAULT_OUTPUT = ROOT / "reports/demand_adjusted_bakery_target_experiment"
TARGET_HISTORY_COLS = [
    "bakery_sales_lag1", "bakery_sales_lag2", "bakery_sales_lag3", "bakery_sales_lag7",
    "bakery_sales_lag14", "bakery_sales_lag30", "bakery_sales_lag365",
    "bakery_sales_roll_mean3", "bakery_sales_roll_mean7", "bakery_sales_roll_mean14",
    "bakery_sales_roll_mean30", "bakery_sales_roll_std7",
]


def extend_with_pilot_actuals(base: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    """Append pilot bakery-days beyond the materialized global dataset."""
    daily = (
        hourly.groupby([DATE_COL, BAKERY_ID_COL, BAKERY_NAME_COL, CITY_COL], as_index=False)
        ["sku_hour_sales"].sum()
        .rename(columns={"sku_hour_sales": TARGET_COL})
    )
    max_base_date = base[DATE_COL].max()
    future = daily[daily[DATE_COL].gt(max_base_date)].copy()
    if future.empty:
        return base
    for col in base.columns:
        if col not in future.columns:
            future[col] = pd.NA
    return pd.concat([base, future[base.columns]], ignore_index=True)


def apply_training_adjustments(
    frame: pd.DataFrame,
    adjustments: pd.DataFrame,
    *,
    train_end: pd.Timestamp,
) -> pd.DataFrame:
    """Add reconstructed demand only to history available by the cutoff."""
    work = frame.copy()
    addition = adjustments[adjustments[DATE_COL].le(train_end)].groupby(
        [DATE_COL, BAKERY_ID_COL], as_index=False
    )["imputed_demand"].sum()
    work = work.merge(addition, on=[DATE_COL, BAKERY_ID_COL], how="left")
    work["imputed_demand"] = pd.to_numeric(work["imputed_demand"], errors="coerce").fillna(0.0)
    work[TARGET_COL] = pd.to_numeric(work[TARGET_COL], errors="coerce").fillna(0.0) + work["imputed_demand"]
    return work.drop(columns="imputed_demand")


def rebuild_target_features(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.drop(columns=[col for col in TARGET_HISTORY_COLS if col in frame.columns]).copy()
    work[DATE_COL] = pd.to_datetime(work[DATE_COL]).dt.normalize()
    work = add_calendar_features(work)
    return add_lag_features(work)


def score_variant(
    frame: pd.DataFrame,
    *,
    train_end: pd.Timestamp,
    holdout_end: pd.Timestamp,
    pilot_ids: set[int],
) -> pd.DataFrame:
    model_frame = build_model_frame(rebuild_target_features(frame))
    train = model_frame[model_frame[DATE_COL].le(train_end)].copy()
    test = model_frame[
        model_frame[DATE_COL].gt(train_end)
        & model_frame[DATE_COL].le(holdout_end)
        & model_frame[BAKERY_ID_COL].isin(pilot_ids)
    ].copy()
    feature_cols = [col for col in BASE_FEATURES if col in model_frame.columns]
    train_x = train[feature_cols].copy()
    test_x = test[feature_cols].copy()
    train_x, test_x = cast_category_columns(train_x, test_x, feature_cols)
    model = train_lgbm(train_x, train[TARGET_COL])
    result = test[[DATE_COL, BAKERY_ID_COL, BAKERY_NAME_COL, TARGET_COL]].copy()
    result["prediction"] = predict_clipped(model, test_x)
    return result


def summarize_predictions(predictions: pd.DataFrame, demand_loss_days: pd.DataFrame, *, variant: str) -> pd.DataFrame:
    demand_loss = demand_loss_days.groupby(
        [DATE_COL, BAKERY_ID_COL], as_index=False
    )["imputed_demand"].sum()
    work = predictions.merge(
        demand_loss,
        on=[DATE_COL, BAKERY_ID_COL],
        how="left",
    )
    work["imputed_demand"] = work["imputed_demand"].fillna(0.0)
    work["is_demand_loss_bakery_day"] = work["imputed_demand"].gt(0)
    work["reconstructed_demand"] = work[TARGET_COL] + work["imputed_demand"]
    scopes = {
        "all_pilot_bakery_days_observed_sales": (
            pd.Series(True, index=work.index), TARGET_COL
        ),
        "no_demand_loss_pilot_bakery_days": (
            ~work["is_demand_loss_bakery_day"], TARGET_COL
        ),
        "demand_loss_days_observed_sales": (
            work["is_demand_loss_bakery_day"], TARGET_COL
        ),
        "demand_loss_days_reconstructed_demand": (
            work["is_demand_loss_bakery_day"], "reconstructed_demand"
        ),
    }
    rows = []
    for scope, (mask, evaluation_col) in scopes.items():
        part = work[mask]
        actual = float(part[evaluation_col].sum())
        error = part["prediction"] - part[evaluation_col]
        rows.append({
            "variant": variant,
            "scope": scope,
            "rows": int(len(part)),
            "actual_qty": actual,
            "predicted_qty": float(part["prediction"].sum()),
            "bias_qty": float(error.sum()),
            "mae": float(error.abs().mean()) if len(part) else None,
            "wape": float(error.abs().sum() / actual) if actual > 0 else None,
            "underforecast_qty": float((-error).clip(lower=0).sum()),
            "overforecast_qty": float(error.clip(lower=0).sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--adjustments", type=Path, default=DEFAULT_ADJUSTMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cutoffs", nargs="+", default=["2026-06-21", "2026-06-28", "2026-07-05"])
    parser.add_argument("--holdout-days", type=int, default=14)
    parser.add_argument("--predictions-input", type=Path)
    args = parser.parse_args()

    adjustments = pd.read_csv(args.adjustments)
    adjustments[DATE_COL] = pd.to_datetime(adjustments[DATE_COL]).dt.normalize()
    adjustments[BAKERY_ID_COL] = adjustments[BAKERY_ID_COL].astype(int)
    pilot_ids = set(adjustments[BAKERY_ID_COL].unique())
    if args.predictions_input:
        predictions = pd.read_csv(args.predictions_input, encoding="utf-8-sig")
        predictions[DATE_COL] = pd.to_datetime(predictions[DATE_COL]).dt.normalize()
    else:
        base = pd.read_csv(args.dataset, encoding="utf-8-sig")
        base[DATE_COL] = pd.to_datetime(base[DATE_COL]).dt.normalize()
        max_holdout = max(pd.Timestamp(value) for value in args.cutoffs) + pd.Timedelta(days=args.holdout_days)
        hourly = load_hourly_raw(
            args.raw,
            bakery_ids=pilot_ids,
            date_from=base[DATE_COL].max() + pd.Timedelta(days=1),
            date_to=max_holdout,
        )
        base = extend_with_pilot_actuals(base, hourly)
        observed_target = base[[DATE_COL, BAKERY_ID_COL, TARGET_COL]].rename(columns={TARGET_COL: "observed_target"})
        prediction_parts = []
        for cutoff_value in args.cutoffs:
            cutoff = pd.Timestamp(cutoff_value)
            holdout_end = cutoff + pd.Timedelta(days=args.holdout_days)
            for variant, frame in [
                ("observed_sales_target", base),
                ("demand_adjusted_target", apply_training_adjustments(base, adjustments, train_end=cutoff)),
            ]:
                part = score_variant(frame, train_end=cutoff, holdout_end=holdout_end, pilot_ids=pilot_ids)
                part = part.drop(columns=TARGET_COL).merge(
                    observed_target, on=[DATE_COL, BAKERY_ID_COL], how="left", validate="one_to_one"
                ).rename(columns={"observed_target": TARGET_COL})
                part["variant"] = variant
                part["cutoff"] = cutoff.date().isoformat()
                prediction_parts.append(part)
        predictions = pd.concat(prediction_parts, ignore_index=True)

    metric_parts = []
    for cutoff_value in args.cutoffs:
        cutoff = pd.Timestamp(cutoff_value)
        holdout_end = cutoff + pd.Timedelta(days=args.holdout_days)
        for variant in ["observed_sales_target", "demand_adjusted_target"]:
            part = predictions[
                predictions["cutoff"].eq(cutoff.date().isoformat())
                & predictions["variant"].eq(variant)
            ].copy()
            metrics = summarize_predictions(
                part,
                adjustments[
                    adjustments[DATE_COL].between(
                        cutoff + pd.Timedelta(days=1), holdout_end
                    )
                    & adjustments["imputed_demand"].gt(0)
                ],
                variant=variant,
            )
            metrics["cutoff"] = cutoff.date().isoformat()
            metric_parts.append(metrics)

    metrics = pd.concat(metric_parts, ignore_index=True)
    lookup = metrics.set_index(["cutoff", "variant", "scope"])
    deltas = []
    for cutoff in metrics["cutoff"].unique():
        for scope in metrics["scope"].unique():
            baseline = lookup.loc[(cutoff, "observed_sales_target", scope)]
            adjusted = lookup.loc[(cutoff, "demand_adjusted_target", scope)]
            deltas.append({
                "cutoff": cutoff,
                "scope": scope,
                "wape_delta": float(adjusted["wape"] - baseline["wape"]),
                "bias_qty_delta": float(adjusted["bias_qty"] - baseline["bias_qty"]),
                "underforecast_qty_delta": float(adjusted["underforecast_qty"] - baseline["underforecast_qty"]),
                "overforecast_qty_delta": float(adjusted["overforecast_qty"] - baseline["overforecast_qty"]),
            })
    deltas = pd.DataFrame(deltas)
    summary = deltas.groupby("scope", as_index=False).agg(
        cutoffs=("cutoff", "nunique"),
        wape_wins=("wape_delta", lambda value: int(value.lt(0).sum())),
        mean_wape_delta=("wape_delta", "mean"),
        mean_bias_qty_delta=("bias_qty_delta", "mean"),
        mean_underforecast_qty_delta=("underforecast_qty_delta", "mean"),
        mean_overforecast_qty_delta=("overforecast_qty_delta", "mean"),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(args.output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    deltas.to_csv(args.output_dir / "deltas.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary.to_dict("records"), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

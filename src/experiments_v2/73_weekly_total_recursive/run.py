"""
Experiment 73: strict recursive bakery forecasting.

Compare:
1) recursive daily baseline model
2) weekly total -> weekday share recursive model
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.experiments_v2.bakery_day_forecast import BASE_FEATURES
from src.experiments_v2.bakery_day_forecast import BAKERY_ID_COL
from src.experiments_v2.bakery_day_forecast import BAKERY_NAME_COL
from src.experiments_v2.bakery_day_forecast import CITY_COL
from src.experiments_v2.bakery_day_forecast import DATE_COL
from src.experiments_v2.bakery_day_forecast import TARGET_COL
from src.experiments_v2.bakery_day_forecast import build_model_frame
from src.experiments_v2.bakery_day_forecast import build_future_feature_rows
from src.experiments_v2.bakery_regime_shift_common import WEEKLY_FEATURES
from src.experiments_v2.bakery_regime_shift_common import build_bakery_weekly_frame
from src.experiments_v2.bakery_regime_shift_common import cast_category_columns
from src.experiments_v2.bakery_regime_shift_common import compute_adaptive_weekday_share_lookup
from src.experiments_v2.bakery_regime_shift_common import load_bakery_frame
from src.experiments_v2.bakery_regime_shift_common import make_train_test_split
from src.experiments_v2.bakery_regime_shift_common import regression_metrics
from src.experiments_v2.common import predict_clipped
from src.experiments_v2.common import train_lgbm


EXP_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "bakery_daily_sales.csv"

MODEL_NAMES = [
    "seasonal_naive_lag7_recursive",
    "repeat_last_week_recursive",
    "recursive_daily_baseline",
    "heuristic_blend_recursive",
    "weekly_total_daily_share_recursive",
]

OUTPUT_FILES = {
    name: {
        "metrics": EXP_DIR / f"metrics_{name}.csv",
        "predictions": EXP_DIR / f"predictions_{name}.csv",
    }
    for name in MODEL_NAMES
}

SUMMARY_FILES = {
    "summary_by_model": EXP_DIR / "summary_by_model.csv",
    "summary_best_by_bakery": EXP_DIR / "summary_best_by_bakery.csv",
    "training_log": EXP_DIR / "training_log.csv",
    "overview": EXP_DIR / "metrics.json",
}

DEFAULT_TEST_DAYS = 28
MIN_TRAIN_ROWS = 90


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def build_complete_weekly_history(history_daily: pd.DataFrame) -> pd.DataFrame:
    work = history_daily.sort_values([BAKERY_ID_COL, DATE_COL]).copy()
    work["week_start"] = work[DATE_COL] - pd.to_timedelta(work[DATE_COL].dt.dayofweek, unit="D")
    coverage = (
        work.groupby([BAKERY_ID_COL, "week_start"], as_index=False)
        .agg(n_days=(DATE_COL, "nunique"), n_dow=("dow", "nunique"))
    )
    complete_keys = coverage[(coverage["n_days"] == 7) & (coverage["n_dow"] == 7)][[BAKERY_ID_COL, "week_start"]]
    weekly = build_bakery_weekly_frame(work)
    return weekly.merge(complete_keys, on=[BAKERY_ID_COL, "week_start"], how="inner")


def select_feature_columns(train_df: pd.DataFrame, base_features: list[str]) -> list[str]:
    selected: list[str] = []
    for col in base_features:
        if col not in train_df.columns:
            continue
        series = train_df[col]
        if series.isna().all():
            continue
        if series.nunique(dropna=True) <= 1:
            continue
        selected.append(col)
    return selected


def build_prediction_frame(actual_df: pd.DataFrame, preds: pd.DataFrame, model_name: str) -> pd.DataFrame:
    frame = actual_df[[DATE_COL, BAKERY_ID_COL, BAKERY_NAME_COL, CITY_COL, TARGET_COL]].copy()
    merged = frame.merge(preds[[DATE_COL, BAKERY_ID_COL, "prediction"]], on=[DATE_COL, BAKERY_ID_COL], how="left")
    merged["model"] = model_name
    merged["prediction"] = pd.to_numeric(merged["prediction"], errors="coerce").fillna(0.0)
    merged["error"] = merged[TARGET_COL] - merged["prediction"]
    merged["abs_error"] = merged["error"].abs()
    return merged.sort_values([BAKERY_ID_COL, DATE_COL]).reset_index(drop=True)


def build_metrics_frame(pred_frame: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows: list[dict] = []
    for bakery_id, group in pred_frame.groupby(BAKERY_ID_COL, sort=False):
        m = regression_metrics(group[TARGET_COL], group["prediction"])
        rows.append(
            {
                BAKERY_ID_COL: bakery_id,
                BAKERY_NAME_COL: group[BAKERY_NAME_COL].iloc[0],
                CITY_COL: group[CITY_COL].iloc[0],
                "model": model_name,
                "n_test_days": int(group[DATE_COL].nunique()),
                "mae": round(m["mae"], 6),
                "mse": round(m["mse"], 6),
                "wmape": round(m["wmape"], 6),
                "bias": round(m["bias"], 6),
            }
        )
    return pd.DataFrame(rows)


def build_model_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in metrics_df.groupby("model", sort=False):
        rows.append(
            {
                "model": model_name,
                "n_bakeries": int(len(group)),
                "avg_mae": round(float(group["mae"].mean()), 6),
                "median_mae": round(float(group["mae"].median()), 6),
                "avg_mse": round(float(group["mse"].mean()), 6),
                "avg_wmape": round(float(group["wmape"].mean()), 6),
                "avg_bias": round(float(group["bias"].mean()), 6),
                "median_abs_bias": round(float(group["bias"].abs().median()), 6),
                "win_count": 0,
            }
        )
    return pd.DataFrame(rows)


def build_best_by_bakery(metrics_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    wide: pd.DataFrame | None = None
    for model_name, frame in metrics_frames.items():
        sub = frame[[BAKERY_ID_COL, BAKERY_NAME_COL, CITY_COL, "mae", "mse", "wmape", "bias"]].copy()
        sub = sub.rename(
            columns={
                "mae": f"{model_name}_mae",
                "mse": f"{model_name}_mse",
                "wmape": f"{model_name}_wmape",
                "bias": f"{model_name}_bias",
            }
        )
        merge_keys = [BAKERY_ID_COL, BAKERY_NAME_COL, CITY_COL]
        wide = sub if wide is None else wide.merge(sub, on=merge_keys, how="outer")

    if wide is None or wide.empty:
        return pd.DataFrame(columns=[BAKERY_ID_COL, "best_model", "best_mae"])

    mae_cols = [f"{name}_mae" for name in MODEL_NAMES]
    mae_values = wide[mae_cols].fillna(np.inf)
    wide["best_model"] = mae_values.idxmin(axis=1).str.replace("_mae", "", regex=False)
    wide["best_mae"] = mae_values.min(axis=1).replace(np.inf, np.nan)
    return wide


def train_daily_baseline_model(train_df: pd.DataFrame, *, min_train_rows: int) -> tuple[object | None, list[str], dict]:
    feature_cols = select_feature_columns(train_df, BASE_FEATURES)
    if len(feature_cols) == 0 or len(train_df) < min_train_rows:
        return None, feature_cols, {"status": "fallback_mean", "n_features": len(feature_cols)}

    train_x = train_df[feature_cols].copy()
    train_x, _ = cast_category_columns(train_x, train_x.copy(), feature_cols)
    model = train_lgbm(train_x, train_df[TARGET_COL])
    return model, feature_cols, {"status": "trained", "n_features": len(feature_cols)}


def seasonal_naive_lag7_recursive_backtest(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    history = train_df.sort_values([BAKERY_ID_COL, DATE_COL]).copy()
    pred_parts: list[pd.DataFrame] = []
    fallback_mean = float(train_df[TARGET_COL].mean()) if len(train_df) else 0.0

    for forecast_date in sorted(test_df[DATE_COL].unique()):
        feature_rows = build_future_feature_rows(history, pd.Timestamp(forecast_date))
        if "bakery_sales_lag7" in feature_rows.columns:
            preds = pd.to_numeric(feature_rows["bakery_sales_lag7"], errors="coerce").fillna(fallback_mean).clip(lower=0.0)
        else:
            preds = np.full(len(feature_rows), fallback_mean, dtype=float)
        feature_rows["prediction"] = preds
        pred_parts.append(feature_rows[[DATE_COL, BAKERY_ID_COL, "prediction"]].copy())
        feature_rows[TARGET_COL] = preds
        history = pd.concat([history, feature_rows], ignore_index=True, sort=False)

    return pd.concat(pred_parts, ignore_index=True), {"status": "naive_lag7", "n_features": 1}


def repeat_last_week_recursive_backtest(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    history_cols = [DATE_COL, BAKERY_ID_COL, BAKERY_NAME_COL, CITY_COL, TARGET_COL, "avg_price", "dow"]
    history = train_df[history_cols].sort_values([BAKERY_ID_COL, DATE_COL]).copy()
    pred_parts: list[pd.DataFrame] = []
    fallback_mean = float(train_df[TARGET_COL].mean()) if len(train_df) else 0.0

    for week_start in sorted((test_df[DATE_COL] - pd.to_timedelta(test_df[DATE_COL].dt.dayofweek, unit="D")).unique()):
        week_start = pd.Timestamp(week_start)
        future_days = test_df[(test_df[DATE_COL] >= week_start) & (test_df[DATE_COL] < week_start + pd.Timedelta(days=7))][
            [DATE_COL, BAKERY_ID_COL, BAKERY_NAME_COL, CITY_COL, "dow"]
        ].copy()
        last_week_start = week_start - pd.Timedelta(days=7)
        last_week = history[
            (history[DATE_COL] >= last_week_start) & (history[DATE_COL] < last_week_start + pd.Timedelta(days=7))
        ][[BAKERY_ID_COL, "dow", TARGET_COL]].copy()
        last_week = last_week.rename(columns={TARGET_COL: "prediction"})
        future_days = future_days.merge(last_week, on=[BAKERY_ID_COL, "dow"], how="left")
        future_days["prediction"] = pd.to_numeric(future_days["prediction"], errors="coerce").fillna(fallback_mean).clip(lower=0.0)
        pred_parts.append(future_days[[DATE_COL, BAKERY_ID_COL, "prediction"]].copy())

        appended = future_days[[DATE_COL, BAKERY_ID_COL, BAKERY_NAME_COL, CITY_COL, "dow", "prediction"]].copy()
        appended["avg_price"] = np.nan
        appended = appended.rename(columns={"prediction": TARGET_COL})
        history = pd.concat([history, appended], ignore_index=True, sort=False)

    return pd.concat(pred_parts, ignore_index=True), {"status": "repeat_last_week", "n_features": 1}


def recursive_daily_baseline_backtest(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    min_train_rows: int,
) -> tuple[pd.DataFrame, dict]:
    model, feature_cols, info = train_daily_baseline_model(train_df, min_train_rows=min_train_rows)
    history = train_df.sort_values([BAKERY_ID_COL, DATE_COL]).copy()
    pred_parts: list[pd.DataFrame] = []

    fallback_mean = float(train_df[TARGET_COL].mean()) if len(train_df) else 0.0
    for forecast_date in sorted(test_df[DATE_COL].unique()):
        feature_rows = build_future_feature_rows(history, pd.Timestamp(forecast_date))
        if model is None or len(feature_cols) == 0:
            preds = np.full(len(feature_rows), fallback_mean, dtype=float)
        else:
            predict_x = feature_rows[feature_cols].copy()
            _, predict_x = cast_category_columns(train_df[feature_cols].copy(), predict_x, feature_cols)
            preds = predict_clipped(model, predict_x)
        feature_rows["prediction"] = preds
        pred_parts.append(feature_rows[[DATE_COL, BAKERY_ID_COL, "prediction"]].copy())
        feature_rows[TARGET_COL] = preds
        history = pd.concat([history, feature_rows], ignore_index=True, sort=False)

    return pd.concat(pred_parts, ignore_index=True), info


def compute_heuristic_blend_prediction(
    feature_rows: pd.DataFrame,
    base_pred: np.ndarray,
    *,
    fallback_mean: float,
) -> pd.Series:
    lag7 = pd.to_numeric(feature_rows.get("bakery_sales_lag7", 0.0), errors="coerce").fillna(fallback_mean).clip(lower=0.0)
    lag14 = (
        pd.to_numeric(feature_rows.get("bakery_sales_lag14", 0.0), errors="coerce").fillna(lag7).clip(lower=0.0)
    )
    dow_mean = (
        pd.to_numeric(feature_rows.get("bakery_sales_dow_mean", 0.0), errors="coerce").fillna(lag7).clip(lower=0.0)
    )
    roll7 = (
        pd.to_numeric(feature_rows.get("bakery_sales_roll_mean7", 0.0), errors="coerce").fillna(lag7).clip(lower=0.0)
    )
    roll30 = (
        pd.to_numeric(feature_rows.get("bakery_sales_roll_mean30", 0.0), errors="coerce").fillna(roll7).clip(lower=0.0)
    )
    cv7 = pd.to_numeric(feature_rows.get("bakery_sales_cv_7d", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)

    trend_ratio = (roll7 / (roll30 + 1e-8)).clip(lower=0.75, upper=1.30)
    trend_scale = trend_ratio.clip(lower=0.90, upper=1.18)
    lag7_trend = lag7 * trend_scale
    seasonal_anchor = 0.70 * lag7 + 0.30 * dow_mean
    heuristic_anchor = 0.65 * lag7_trend + 0.35 * seasonal_anchor

    base_pred_series = pd.Series(base_pred, index=feature_rows.index, dtype=float)
    stability = (1.0 - cv7.clip(lower=0.0, upper=1.0)).clip(lower=0.0, upper=1.0)
    anchor_weight = (0.30 + 0.30 * stability).clip(lower=0.30, upper=0.60)

    ml_gap = ((lag7 - base_pred_series) / (lag7.abs() + 1.0)).clip(lower=-1.0, upper=1.0)
    uptrend_signal = ((trend_ratio - 1.02) / 0.16).clip(lower=0.0, upper=1.0)
    stable_signal = ((0.35 - cv7) / 0.20).clip(lower=0.0, upper=1.0)
    underpredict_signal = (ml_gap / 0.22).clip(lower=0.0, upper=1.0)
    lag_boost = 0.18 * uptrend_signal * stable_signal * underpredict_signal

    anchor_weight = (anchor_weight + lag_boost).clip(lower=0.30, upper=0.72)
    blend_pred = (1.0 - anchor_weight) * base_pred_series + anchor_weight * heuristic_anchor

    weekly_stability = (1.0 - ((lag7 - lag14).abs() / (lag14.abs() + 1.0)).clip(lower=0.0, upper=1.0)).clip(
        lower=0.0,
        upper=1.0,
    )
    gap_ratio = ((lag7 - blend_pred) / (lag7.abs() + 1.0)).clip(lower=-1.0, upper=1.0)
    trust_mask = (weekly_stability >= 0.90) & (gap_ratio >= 0.12) & (trend_ratio >= 1.00) & (cv7 <= 0.30)
    extra_weight = 0.24 * weekly_stability * gap_ratio.clip(lower=0.0, upper=1.0)
    final_pred = blend_pred.where(~trust_mask, (1.0 - extra_weight) * blend_pred + extra_weight * lag7)

    return pd.to_numeric(final_pred, errors="coerce").fillna(fallback_mean).clip(lower=0.0)


def heuristic_blend_recursive_backtest(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    min_train_rows: int,
) -> tuple[pd.DataFrame, dict]:
    model, feature_cols, info = train_daily_baseline_model(train_df, min_train_rows=min_train_rows)
    history = train_df.sort_values([BAKERY_ID_COL, DATE_COL]).copy()
    pred_parts: list[pd.DataFrame] = []
    fallback_mean = float(train_df[TARGET_COL].mean()) if len(train_df) else 0.0

    for forecast_date in sorted(test_df[DATE_COL].unique()):
        feature_rows = build_future_feature_rows(history, pd.Timestamp(forecast_date))
        if model is None or len(feature_cols) == 0:
            base_pred = np.full(len(feature_rows), fallback_mean, dtype=float)
        else:
            predict_x = feature_rows[feature_cols].copy()
            _, predict_x = cast_category_columns(train_df[feature_cols].copy(), predict_x, feature_cols)
            base_pred = predict_clipped(model, predict_x)

        feature_rows["prediction"] = compute_heuristic_blend_prediction(
            feature_rows,
            base_pred,
            fallback_mean=fallback_mean,
        )
        pred_parts.append(feature_rows[[DATE_COL, BAKERY_ID_COL, "prediction"]].copy())
        feature_rows[TARGET_COL] = feature_rows["prediction"]
        history = pd.concat([history, feature_rows], ignore_index=True, sort=False)

    out_info = dict(info)
    out_info["status"] = f"{info.get('status', 'unknown')}_plus_heuristic_blend"
    return pd.concat(pred_parts, ignore_index=True), out_info


def train_weekly_total_model(train_history_daily: pd.DataFrame, *, min_train_rows: int) -> tuple[object | None, list[str], dict]:
    weekly = build_complete_weekly_history(train_history_daily)
    feature_cols = select_feature_columns(weekly, WEEKLY_FEATURES)
    if len(feature_cols) == 0 or len(weekly) < min_train_rows:
        return None, feature_cols, {"status": "fallback_mean", "n_features": len(feature_cols)}

    train_x = weekly[feature_cols].copy()
    train_x, _ = cast_category_columns(train_x, train_x.copy(), feature_cols)
    model = train_lgbm(train_x, weekly["week_sales"])
    return model, feature_cols, {"status": "trained", "n_features": len(feature_cols)}


def build_future_week_rows(history_daily: pd.DataFrame, future_week_start: pd.Timestamp) -> pd.DataFrame:
    weekly_hist = build_complete_weekly_history(history_daily)
    last_daily = history_daily.sort_values([BAKERY_ID_COL, DATE_COL]).groupby(BAKERY_ID_COL, as_index=False).tail(1)
    rows: list[dict] = []

    for bakery_id, group in weekly_hist.groupby(BAKERY_ID_COL, sort=False):
        group = group.sort_values("week_start").copy()
        week_sales = pd.to_numeric(group["week_sales"], errors="coerce").fillna(0.0)
        last_meta = last_daily[last_daily[BAKERY_ID_COL] == bakery_id].iloc[-1]
        row = {
            BAKERY_ID_COL: bakery_id,
            BAKERY_NAME_COL: last_meta[BAKERY_NAME_COL],
            CITY_COL: last_meta[CITY_COL],
            "week_start": future_week_start,
            "month": int(future_week_start.month),
            "iso_week": int(future_week_start.isocalendar().week),
            "week_of_month": int(((future_week_start.day - 1) // 7) + 1),
            "avg_price": float(pd.to_numeric(group["avg_price"], errors="coerce").dropna().iloc[-1]) if group["avg_price"].notna().any() else 0.0,
            "week_sales_lag1": float(week_sales.iloc[-1]) if len(week_sales) >= 1 else 0.0,
            "week_sales_lag2": float(week_sales.iloc[-2]) if len(week_sales) >= 2 else 0.0,
            "week_sales_lag4": float(week_sales.iloc[-4]) if len(week_sales) >= 4 else 0.0,
            "week_sales_roll_mean2": float(week_sales.tail(2).mean()) if len(week_sales) else 0.0,
            "week_sales_roll_mean4": float(week_sales.tail(4).mean()) if len(week_sales) else 0.0,
            "week_sales_roll_mean8": float(week_sales.tail(8).mean()) if len(week_sales) else 0.0,
        }
        row["week_sales_trend"] = row["week_sales_roll_mean2"] / (row["week_sales_roll_mean8"] + 1e-8)
        rows.append(row)

    future = pd.DataFrame(rows)
    for col in [c for c in future.columns if c.startswith("week_sales_") or c == "avg_price"]:
        future[col] = pd.to_numeric(future[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return future


def build_elapsed_week_sales_lookup(history_daily: pd.DataFrame, week_start: pd.Timestamp) -> pd.DataFrame:
    current_week = history_daily[
        (history_daily[DATE_COL] >= week_start) & (history_daily[DATE_COL] < week_start + pd.Timedelta(days=7))
    ].copy()
    if current_week.empty:
        return pd.DataFrame(columns=[BAKERY_ID_COL, "elapsed_week_sales"])
    return (
        current_week.groupby(BAKERY_ID_COL, as_index=False)[TARGET_COL]
        .sum()
        .rename(columns={TARGET_COL: "elapsed_week_sales"})
    )


def weekly_total_daily_share_recursive_backtest(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    min_train_rows: int,
    recent_weeks: int,
) -> tuple[pd.DataFrame, dict]:
    history_cols = [DATE_COL, BAKERY_ID_COL, BAKERY_NAME_COL, CITY_COL, TARGET_COL, "avg_price", "dow"]
    history = train_df[history_cols].sort_values([BAKERY_ID_COL, DATE_COL]).copy()
    pred_parts: list[pd.DataFrame] = []
    info: dict = {"status": "trained", "n_features": 0}

    week_starts = sorted((test_df[DATE_COL] - pd.to_timedelta(test_df[DATE_COL].dt.dayofweek, unit="D")).unique())
    fallback_week_hist = build_complete_weekly_history(train_df)
    fallback_week_mean = float(fallback_week_hist["week_sales"].mean()) if len(fallback_week_hist) else 0.0

    for week_start in week_starts:
        week_start = pd.Timestamp(week_start)
        week_model, feature_cols, week_info = train_weekly_total_model(history, min_train_rows=min_train_rows)
        info = {"status": week_info["status"], "n_features": len(feature_cols)}

        future_week = build_future_week_rows(history, week_start)
        if week_model is None or len(feature_cols) == 0:
            future_week["pred_week_sales"] = fallback_week_mean
        else:
            predict_x = future_week[feature_cols].copy()
            for col in [c for c in ["bakery_id", "city", "month"] if c in feature_cols]:
                predict_x[col] = predict_x[col].astype("category")
            future_week["pred_week_sales"] = predict_clipped(week_model, predict_x)

        share_lookup = compute_adaptive_weekday_share_lookup(history, recent_weeks=recent_weeks)
        future_days = test_df[(test_df[DATE_COL] >= week_start) & (test_df[DATE_COL] < week_start + pd.Timedelta(days=7))][
            [DATE_COL, BAKERY_ID_COL, BAKERY_NAME_COL, CITY_COL, "dow"]
        ].copy()
        future_days = future_days.merge(future_week[[BAKERY_ID_COL, "pred_week_sales", "avg_price"]], on=BAKERY_ID_COL, how="left")
        future_days = future_days.merge(share_lookup, on=[BAKERY_ID_COL, "dow"], how="left")
        future_days["weekday_share"] = future_days["weekday_share"].fillna(1.0 / 7.0)
        future_day_counts = (
            future_days.groupby(BAKERY_ID_COL, as_index=False)
            .size()
            .rename(columns={"size": "n_future_days"})
        )
        share_sums = (
            future_days.groupby(BAKERY_ID_COL, as_index=False)["weekday_share"]
            .sum()
            .rename(columns={"weekday_share": "forecast_day_share_sum"})
        )
        elapsed = build_elapsed_week_sales_lookup(history, week_start)
        future_days = future_days.merge(future_day_counts, on=BAKERY_ID_COL, how="left")
        future_days = future_days.merge(share_sums, on=BAKERY_ID_COL, how="left")
        future_days = future_days.merge(elapsed, on=BAKERY_ID_COL, how="left")
        future_days["elapsed_week_sales"] = pd.to_numeric(future_days["elapsed_week_sales"], errors="coerce").fillna(0.0)
        future_days["remaining_week_sales"] = (
            pd.to_numeric(future_days["pred_week_sales"], errors="coerce").fillna(0.0) - future_days["elapsed_week_sales"]
        ).clip(lower=0.0)
        future_days["normalized_share"] = np.where(
            future_days["forecast_day_share_sum"] > 0,
            future_days["weekday_share"] / future_days["forecast_day_share_sum"],
            1.0 / future_days["n_future_days"].clip(lower=1),
        )
        future_days["prediction"] = np.where(
            future_days["elapsed_week_sales"] > 0,
            future_days["remaining_week_sales"] * future_days["normalized_share"],
            pd.to_numeric(future_days["pred_week_sales"], errors="coerce").fillna(0.0) * future_days["weekday_share"],
        )
        pred_parts.append(future_days[[DATE_COL, BAKERY_ID_COL, "prediction"]].copy())

        # Feed predicted days back into history for strict recursive weekly forecasting.
        appended = future_days[[DATE_COL, BAKERY_ID_COL, BAKERY_NAME_COL, CITY_COL, "avg_price", "dow", "prediction"]].copy()
        appended = appended.rename(columns={"prediction": TARGET_COL})
        history = pd.concat([history, appended], ignore_index=True, sort=False)

    return pd.concat(pred_parts, ignore_index=True), info


def evaluate_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    min_train_rows: int,
    recent_weeks: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], list[dict]]:
    prediction_frames: dict[str, pd.DataFrame] = {}
    metrics_frames: dict[str, pd.DataFrame] = {}
    training_log: list[dict] = []

    pred_naive_lag7, info_naive_lag7 = seasonal_naive_lag7_recursive_backtest(train_df, test_df)
    frame_naive_lag7 = build_prediction_frame(test_df, pred_naive_lag7, "seasonal_naive_lag7_recursive")
    prediction_frames["seasonal_naive_lag7_recursive"] = frame_naive_lag7
    metrics_frames["seasonal_naive_lag7_recursive"] = build_metrics_frame(
        frame_naive_lag7,
        "seasonal_naive_lag7_recursive",
    )
    gm_naive_lag7 = regression_metrics(frame_naive_lag7[TARGET_COL], frame_naive_lag7["prediction"])
    training_log.append(
        {
            "model": "seasonal_naive_lag7_recursive",
            "status": info_naive_lag7.get("status", "unknown"),
            "n_features": info_naive_lag7.get("n_features", 0),
            "rows_train": len(train_df),
            "rows_test": len(test_df),
            "mae": round(gm_naive_lag7["mae"], 6),
            "mse": round(gm_naive_lag7["mse"], 6),
            "wmape": round(gm_naive_lag7["wmape"], 6),
            "bias": round(gm_naive_lag7["bias"], 6),
        }
    )

    pred_repeat_last_week, info_repeat_last_week = repeat_last_week_recursive_backtest(train_df, test_df)
    frame_repeat_last_week = build_prediction_frame(test_df, pred_repeat_last_week, "repeat_last_week_recursive")
    prediction_frames["repeat_last_week_recursive"] = frame_repeat_last_week
    metrics_frames["repeat_last_week_recursive"] = build_metrics_frame(
        frame_repeat_last_week,
        "repeat_last_week_recursive",
    )
    gm_repeat_last_week = regression_metrics(frame_repeat_last_week[TARGET_COL], frame_repeat_last_week["prediction"])
    training_log.append(
        {
            "model": "repeat_last_week_recursive",
            "status": info_repeat_last_week.get("status", "unknown"),
            "n_features": info_repeat_last_week.get("n_features", 0),
            "rows_train": len(train_df),
            "rows_test": len(test_df),
            "mae": round(gm_repeat_last_week["mae"], 6),
            "mse": round(gm_repeat_last_week["mse"], 6),
            "wmape": round(gm_repeat_last_week["wmape"], 6),
            "bias": round(gm_repeat_last_week["bias"], 6),
        }
    )

    pred_daily, info_daily = recursive_daily_baseline_backtest(train_df, test_df, min_train_rows=min_train_rows)
    frame_daily = build_prediction_frame(test_df, pred_daily, "recursive_daily_baseline")
    prediction_frames["recursive_daily_baseline"] = frame_daily
    metrics_frames["recursive_daily_baseline"] = build_metrics_frame(frame_daily, "recursive_daily_baseline")
    gm_daily = regression_metrics(frame_daily[TARGET_COL], frame_daily["prediction"])
    training_log.append(
        {
            "model": "recursive_daily_baseline",
            "status": info_daily.get("status", "unknown"),
            "n_features": info_daily.get("n_features", 0),
            "rows_train": len(train_df),
            "rows_test": len(test_df),
            "mae": round(gm_daily["mae"], 6),
            "mse": round(gm_daily["mse"], 6),
            "wmape": round(gm_daily["wmape"], 6),
            "bias": round(gm_daily["bias"], 6),
        }
    )

    pred_blend, info_blend = heuristic_blend_recursive_backtest(train_df, test_df, min_train_rows=min_train_rows)
    frame_blend = build_prediction_frame(test_df, pred_blend, "heuristic_blend_recursive")
    prediction_frames["heuristic_blend_recursive"] = frame_blend
    metrics_frames["heuristic_blend_recursive"] = build_metrics_frame(frame_blend, "heuristic_blend_recursive")
    gm_blend = regression_metrics(frame_blend[TARGET_COL], frame_blend["prediction"])
    training_log.append(
        {
            "model": "heuristic_blend_recursive",
            "status": info_blend.get("status", "unknown"),
            "n_features": info_blend.get("n_features", 0),
            "rows_train": len(train_df),
            "rows_test": len(test_df),
            "mae": round(gm_blend["mae"], 6),
            "mse": round(gm_blend["mse"], 6),
            "wmape": round(gm_blend["wmape"], 6),
            "bias": round(gm_blend["bias"], 6),
        }
    )

    pred_weekly, info_weekly = weekly_total_daily_share_recursive_backtest(
        train_df,
        test_df,
        min_train_rows=min_train_rows,
        recent_weeks=recent_weeks,
    )
    frame_weekly = build_prediction_frame(test_df, pred_weekly, "weekly_total_daily_share_recursive")
    prediction_frames["weekly_total_daily_share_recursive"] = frame_weekly
    metrics_frames["weekly_total_daily_share_recursive"] = build_metrics_frame(
        frame_weekly,
        "weekly_total_daily_share_recursive",
    )
    gm_weekly = regression_metrics(frame_weekly[TARGET_COL], frame_weekly["prediction"])
    training_log.append(
        {
            "model": "weekly_total_daily_share_recursive",
            "status": info_weekly.get("status", "unknown"),
            "n_features": info_weekly.get("n_features", 0),
            "rows_train": len(train_df),
            "rows_test": len(test_df),
            "mae": round(gm_weekly["mae"], 6),
            "mse": round(gm_weekly["mse"], 6),
            "wmape": round(gm_weekly["wmape"], 6),
            "bias": round(gm_weekly["bias"], 6),
        }
    )

    return prediction_frames, metrics_frames, training_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 73: weekly total recursive")
    parser.add_argument("--dataset-path", default=str(DATA_PATH))
    parser.add_argument("--test-days", type=int, default=DEFAULT_TEST_DAYS)
    parser.add_argument("--min-train-rows", type=int, default=MIN_TRAIN_ROWS)
    parser.add_argument("--recent-weeks", type=int, default=4)
    args = parser.parse_args()

    start = time.time()
    print("=" * 72)
    print("EXPERIMENT 73: Weekly total recursive")
    print("=" * 72)

    print("\n[1/4] Loading bakery-day frame...")
    df = load_bakery_frame(args.dataset_path)
    print(
        f"  rows={len(df):,} | dates={df[DATE_COL].nunique()} | bakeries={df[BAKERY_ID_COL].nunique()} | "
        f"range={df[DATE_COL].min().date()}..{df[DATE_COL].max().date()}"
    )

    print("\n[2/4] Building holdout split...")
    train_df, test_df, test_start = make_train_test_split(df, args.test_days)
    print(
        f"  test_start={test_start.date()} | rows_train={len(train_df):,} | rows_test={len(test_df):,} | "
        f"train_days={train_df[DATE_COL].nunique()} | test_days={test_df[DATE_COL].nunique()}"
    )

    print("\n[3/4] Running strict recursive backtests...")
    prediction_frames, metrics_frames, training_log = evaluate_models(
        train_df,
        test_df,
        min_train_rows=args.min_train_rows,
        recent_weeks=args.recent_weeks,
    )

    print("\n[4/4] Saving artifacts...")
    for model_name in MODEL_NAMES:
        save_csv(metrics_frames[model_name], OUTPUT_FILES[model_name]["metrics"])
        save_csv(prediction_frames[model_name], OUTPUT_FILES[model_name]["predictions"])
        print(f"  saved {model_name}")

    metrics_all = pd.concat(metrics_frames.values(), ignore_index=True)
    summary_by_model = build_model_summary(metrics_all)
    best_by_bakery = build_best_by_bakery(metrics_frames)
    best_counts = best_by_bakery["best_model"].value_counts().reindex(MODEL_NAMES, fill_value=0)
    summary_by_model["win_count"] = summary_by_model["model"].map(best_counts).fillna(0).astype(int)

    save_csv(summary_by_model, SUMMARY_FILES["summary_by_model"])
    save_csv(best_by_bakery, SUMMARY_FILES["summary_best_by_bakery"])
    save_csv(pd.DataFrame(training_log), SUMMARY_FILES["training_log"])

    overview = {
        "experiment": "73_weekly_total_recursive",
        "dataset_path": str(args.dataset_path),
        "test_days": args.test_days,
        "min_train_rows": args.min_train_rows,
        "recent_weeks": args.recent_weeks,
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "bakeries_total": int(df[BAKERY_ID_COL].nunique()),
        "test_start": str(test_start.date()),
        "summary_by_model": summary_by_model.to_dict("records"),
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    SUMMARY_FILES["overview"].write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nModel summary:")
    print(summary_by_model.to_string(index=False))
    print(f"\nDone in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()

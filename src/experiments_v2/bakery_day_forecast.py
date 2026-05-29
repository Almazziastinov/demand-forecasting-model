"""
Train and run bakery-level daily sales forecasting on the English bakery dataset.

Modes:
  - train: fit one global LightGBM model on bakery-day rows and save artifacts
  - forecast: recursively predict future bakery-day sales for N days ahead
"""

from __future__ import annotations

# ruff: noqa: E402,E501

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.experiments_v2.common import predict_clipped
from src.experiments_v2.common import train_lgbm

DATE_COL = "date"
BAKERY_ID_COL = "bakery_id"
BAKERY_NAME_COL = "bakery_name"
CITY_COL = "city"
TARGET_COL = "bakery_sales"

FORECAST_COL = "bakery_day_forecast"
FORECAST_BIAS_ADJ_COL = "bakery_day_forecast_bias_adj"

DEFAULT_DATASET_PATH = ROOT / "data" / "processed" / "bakery_daily_sales.csv"
DEFAULT_MODEL_PATH = ROOT / "models" / "bakery_day_model.joblib"
DEFAULT_META_PATH = ROOT / "models" / "bakery_day_meta.joblib"
DEFAULT_TRAIN_SUMMARY_PATH = ROOT / "reports" / "bakery_day_model_summary.json"
DEFAULT_PREDICTIONS_PATH = ROOT / "reports" / "bakery_day_model_holdout_predictions.csv"
DEFAULT_BIAS_PATH = ROOT / "reports" / "bakery_day_model_bias_by_bakery.csv"
DEFAULT_FORECAST_PATH = ROOT / "data" / "processed" / "bakery_day_forecast.csv"

DEFAULT_TEST_DAYS = 30
DEFAULT_HORIZON_DAYS = 14

BASE_FEATURES = [
    BAKERY_ID_COL,
    CITY_COL,
    "dow",
    "day",
    "month",
    "iso_week",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_payday_week",
    "is_holiday",
    "is_pre_holiday",
    "current_event_cluster",
    "prev_event_cluster",
    "next_event_cluster",
    "days_since_prev_event",
    "days_to_next_event",
    "is_near_event_window",
    "avg_price",
    "bakery_sales_lag1",
    "bakery_sales_lag2",
    "bakery_sales_lag3",
    "bakery_sales_lag7",
    "bakery_sales_lag14",
    "bakery_sales_lag30",
    "bakery_sales_roll_mean3",
    "bakery_sales_roll_mean7",
    "bakery_sales_roll_mean14",
    "bakery_sales_roll_mean30",
    "bakery_sales_roll_std7",
    "bakery_sales_dow_mean",
    "bakery_sales_trend",
    "bakery_sales_cv_7d",
]

ENRICHED_EVENT_FEATURES = [
    "holiday_name_feature",
    "event_window_type",
    "event_distance_bin",
    "current_event_city",
    "nearest_event_city",
    "event_dow_interaction",
    "is_pre_event_1d",
    "is_pre_event_2d",
    "is_pre_event_3d",
    "is_post_event_1d",
    "is_post_event_2d",
    "is_post_event_3d",
    "is_event_window_7d",
    "days_to_payday",
    "days_since_payday",
    "payday_distance",
    "is_payday",
    "is_pre_payday_1d",
    "is_pre_payday_2d",
    "is_post_payday_1d",
    "is_post_payday_2d",
    "payday_window_type",
    "payday_dow_interaction",
]

BASE_FEATURES = [*BASE_FEATURES, *ENRICHED_EVENT_FEATURES]

CATEGORICAL_COLS = [
    BAKERY_ID_COL,
    CITY_COL,
    "month",
    "current_event_cluster",
    "prev_event_cluster",
    "next_event_cluster",
    "holiday_name_feature",
    "event_window_type",
    "event_distance_bin",
    "current_event_city",
    "nearest_event_city",
    "event_dow_interaction",
    "payday_window_type",
    "payday_dow_interaction",
]

PAYDAY_DAYS = (5, 20)
EVENT_WINDOW_DAYS = 7

FIXED_HOLIDAY_LABELS = {
    (1, 1): "new_year_day",
    (1, 2): "new_year_holiday_2",
    (1, 3): "new_year_holiday_3",
    (1, 4): "new_year_holiday_4",
    (1, 5): "new_year_holiday_5",
    (1, 6): "new_year_holiday_6",
    (1, 7): "christmas",
    (1, 8): "new_year_holiday_8",
    (2, 23): "defender_day",
    (3, 8): "womens_day",
    (5, 1): "spring_labor_day",
    (5, 9): "victory_day",
    (6, 12): "russia_day",
    (8, 30): "tatarstan_day",
    (11, 4): "unity_day",
    (11, 6): "constitution_day_rt",
}

SPECIAL_HOLIDAY_DATES = {
    pd.Timestamp("2025-03-30"): "uraza_bayram",
    pd.Timestamp("2025-04-20"): "orthodox_easter",
    pd.Timestamp("2025-06-06"): "kurban_bayram",
    pd.Timestamp("2026-03-20"): "uraza_bayram",
    pd.Timestamp("2026-04-12"): "orthodox_easter",
    pd.Timestamp("2026-05-27"): "kurban_bayram",
}

DISABLED_PRE_HOLIDAY_TARGETS = {
    (5, 1),
    (5, 9),
}

DISABLED_POST_HOLIDAY_DATES = {
    (5, 2),
    (5, 10),
}

EVENT_CLUSTER_MAP = {
    "new_year_day": "cluster_0",
    "new_year_holiday_2": "cluster_0",
    "new_year_holiday_3": "cluster_0",
    "new_year_holiday_4": "cluster_0",
    "new_year_holiday_5": "cluster_0",
    "new_year_holiday_6": "cluster_2",
    "new_year_holiday_8": "cluster_0",
    "christmas": "cluster_3",
    "defender_day": "cluster_1",
    "womens_day": "cluster_1",
    "spring_labor_day": "cluster_1",
    "victory_day": "cluster_3",
    "russia_day": "cluster_2",
    "tatarstan_day": "cluster_1",
    "unity_day": "cluster_3",
    "constitution_day_rt": "cluster_2",
    "orthodox_easter": "cluster_1",
    "uraza_bayram": "cluster_1",
    "kurban_bayram": "cluster_2",
}


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true_arr - y_pred_arr))) if len(y_true_arr) else np.nan
    mse = float(np.mean((y_true_arr - y_pred_arr) ** 2)) if len(y_true_arr) else np.nan
    wmape = float(np.sum(np.abs(y_true_arr - y_pred_arr)) / (np.sum(y_true_arr) + 1e-8) * 100) if len(y_true_arr) else np.nan
    bias = float(np.mean(y_true_arr - y_pred_arr)) if len(y_true_arr) else np.nan
    return {"mae": mae, "mse": mse, "wmape": wmape, "bias": bias}


def build_bias_by_bakery(holdout: pd.DataFrame) -> pd.DataFrame:
    work = holdout.copy()
    work["error"] = work[TARGET_COL] - work[FORECAST_COL]
    summary = (
        work.groupby([BAKERY_ID_COL, BAKERY_NAME_COL, CITY_COL], as_index=False)
        .agg(
            n_days=(DATE_COL, "nunique"),
            actual_mean=(TARGET_COL, "mean"),
            forecast_mean=(FORECAST_COL, "mean"),
            bias=("error", "mean"),
            mae=("error", lambda s: float(np.mean(np.abs(s)))),
            actual_sum=(TARGET_COL, "sum"),
            forecast_sum=(FORECAST_COL, "sum"),
        )
        .sort_values("bias", ascending=False)
        .reset_index(drop=True)
    )
    summary["abs_bias"] = summary["bias"].abs()
    summary["bias_pct_of_actual_mean"] = np.where(
        summary["actual_mean"].abs() > 1e-8,
        summary["bias"] / summary["actual_mean"] * 100.0,
        0.0,
    )
    summary["wmape"] = np.where(
        summary["actual_sum"].abs() > 1e-8,
        (summary["mae"] * summary["n_days"]) / summary["actual_sum"] * 100.0,
        0.0,
    )
    return summary.sort_values("abs_bias", ascending=False).reset_index(drop=True)


def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = [DATE_COL, BAKERY_ID_COL, TARGET_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required dataset columns: {missing}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, BAKERY_ID_COL, TARGET_COL]).copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0.0).clip(lower=0.0)
    if CITY_COL not in df.columns:
        df[CITY_COL] = "unknown"
    if BAKERY_NAME_COL not in df.columns:
        df[BAKERY_NAME_COL] = df[BAKERY_ID_COL].astype(str)
    return df.sort_values([BAKERY_ID_COL, DATE_COL]).reset_index(drop=True)


def build_event_calendar(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    rows: list[dict] = []
    years = range(start_date.year - 1, end_date.year + 2)
    for year in years:
        for (month, day), holiday_name in FIXED_HOLIDAY_LABELS.items():
            rows.append(
                {
                    DATE_COL: pd.Timestamp(year=year, month=month, day=day),
                    "holiday_name": holiday_name,
                }
            )
    for date_value, holiday_name in SPECIAL_HOLIDAY_DATES.items():
        rows.append({DATE_COL: date_value.normalize(), "holiday_name": holiday_name})

    events = pd.DataFrame(rows).drop_duplicates(subset=[DATE_COL], keep="last")
    events = events[(events[DATE_COL] >= start_date) & (events[DATE_COL] <= end_date)].copy()
    events["event_cluster"] = events["holiday_name"].map(EVENT_CLUSTER_MAP).fillna("cluster_other")
    return events.sort_values(DATE_COL).reset_index(drop=True)


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    calendar = build_event_calendar(
        work[DATE_COL].min() - pd.Timedelta(days=1),
        work[DATE_COL].max() + pd.Timedelta(days=1),
    )
    holiday_lookup = calendar.set_index(DATE_COL)["holiday_name"].to_dict()

    def is_holiday_date(d: pd.Timestamp) -> bool:
        return d.normalize() in holiday_lookup

    def holiday_name(d: pd.Timestamp) -> str:
        return str(holiday_lookup.get(d.normalize(), ""))

    def is_pre_holiday_date(d: pd.Timestamp) -> int:
        next_day = d + pd.Timedelta(days=1)
        if (next_day.month, next_day.day) in DISABLED_PRE_HOLIDAY_TARGETS:
            return 0
        return int(is_holiday_date(next_day))

    def is_post_holiday_date(d: pd.Timestamp) -> int:
        if (d.month, d.day) in DISABLED_POST_HOLIDAY_DATES:
            return 0
        prev_day = d - pd.Timedelta(days=1)
        return int(is_holiday_date(prev_day))

    work["holiday_name"] = work[DATE_COL].apply(holiday_name)
    work["is_holiday"] = work[DATE_COL].apply(lambda d: int(is_holiday_date(d)))
    work["is_pre_holiday"] = work[DATE_COL].apply(is_pre_holiday_date)
    work["is_post_holiday"] = work[DATE_COL].apply(is_post_holiday_date)
    return work


def add_event_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.sort_values(DATE_COL).copy().reset_index(drop=True)
    events = build_event_calendar(
        work[DATE_COL].min() - pd.Timedelta(days=370),
        work[DATE_COL].max() + pd.Timedelta(days=370),
    )[[DATE_COL, "holiday_name", "event_cluster"]].copy()
    events = events.sort_values(DATE_COL).reset_index(drop=True)

    work["current_event_cluster"] = np.where(
        work["is_holiday"] == 1,
        work["holiday_name"].map(EVENT_CLUSTER_MAP).fillna("cluster_other"),
        "cluster_none",
    )

    prev_events = pd.merge_asof(
        work[[DATE_COL]].sort_values(DATE_COL),
        events.rename(columns={DATE_COL: "prev_event_date", "event_cluster": "prev_event_cluster"}),
        left_on=DATE_COL,
        right_on="prev_event_date",
        direction="backward",
    )
    next_events = pd.merge_asof(
        work[[DATE_COL]].sort_values(DATE_COL),
        events.rename(columns={DATE_COL: "next_event_date", "event_cluster": "next_event_cluster"}),
        left_on=DATE_COL,
        right_on="next_event_date",
        direction="forward",
    )

    work["prev_event_cluster"] = prev_events["prev_event_cluster"].fillna("cluster_none")
    work["next_event_cluster"] = next_events["next_event_cluster"].fillna("cluster_none")
    work["days_since_prev_event"] = (
        (work[DATE_COL] - pd.to_datetime(prev_events["prev_event_date"], errors="coerce")).dt.days.fillna(999).clip(lower=0)
    )
    work["days_to_next_event"] = (
        (pd.to_datetime(next_events["next_event_date"], errors="coerce") - work[DATE_COL]).dt.days.fillna(999).clip(lower=0)
    )
    work["is_near_event_window"] = (
        np.minimum(work["days_since_prev_event"], work["days_to_next_event"]) <= 3
    ).astype(int)
    return work


def _string_feature(df: pd.DataFrame, col: str, default: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="object")
    return df[col].fillna(default).astype(str)


def add_enriched_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add explicit event/payday features for bakery-level top-down LGBM."""
    work = df.copy()
    days_to_next = pd.to_numeric(
        work.get("days_to_next_event", 999),
        errors="coerce",
    ).fillna(999)
    days_since_prev = pd.to_numeric(
        work.get("days_since_prev_event", 999),
        errors="coerce",
    ).fillna(999)
    is_holiday = pd.to_numeric(
        work.get("is_holiday", 0),
        errors="coerce",
    ).fillna(0).astype(int)

    work["holiday_name_feature"] = _string_feature(work, "holiday_name", "")
    work.loc[work["holiday_name_feature"].eq(""), "holiday_name_feature"] = (
        "no_holiday"
    )

    work["is_pre_event_1d"] = (days_to_next == 1).astype(int)
    work["is_pre_event_2d"] = (days_to_next == 2).astype(int)
    work["is_pre_event_3d"] = (days_to_next == 3).astype(int)
    work["is_post_event_1d"] = (days_since_prev == 1).astype(int)
    work["is_post_event_2d"] = (days_since_prev == 2).astype(int)
    work["is_post_event_3d"] = (days_since_prev == 3).astype(int)
    nearest_event_distance = np.minimum(days_to_next, days_since_prev)
    work["is_event_window_7d"] = (
        (nearest_event_distance <= EVENT_WINDOW_DAYS) | (is_holiday == 1)
    ).astype(int)

    work["event_window_type"] = "regular"
    work.loc[is_holiday == 1, "event_window_type"] = "event_day"
    work.loc[(is_holiday == 0) & (days_to_next <= 3), "event_window_type"] = (
        "pre_event_1_3"
    )
    work.loc[
        (is_holiday == 0) & (days_to_next > 3) & (days_to_next <= 7),
        "event_window_type",
    ] = "pre_event_4_7"
    work.loc[(is_holiday == 0) & (days_since_prev <= 3), "event_window_type"] = (
        "post_event_1_3"
    )
    work.loc[
        (is_holiday == 0) & (days_since_prev > 3) & (days_since_prev <= 7),
        "event_window_type",
    ] = "post_event_4_7"

    work["event_distance_bin"] = pd.cut(
        nearest_event_distance.clip(upper=999),
        bins=[-1, 0, 1, 3, 7, 14, 999],
        labels=["0", "1", "2_3", "4_7", "8_14", "far"],
    ).astype(str)

    current_cluster = _string_feature(work, "current_event_cluster", "cluster_none")
    next_cluster = _string_feature(work, "next_event_cluster", "cluster_none")
    prev_cluster = _string_feature(work, "prev_event_cluster", "cluster_none")
    nearest_cluster = np.where(
        days_to_next <= days_since_prev,
        next_cluster,
        prev_cluster,
    )
    nearest_cluster = pd.Series(nearest_cluster, index=work.index).fillna(
        "cluster_none"
    )

    work["current_event_city"] = current_cluster + "|" + work[CITY_COL].astype(str)
    work["nearest_event_city"] = nearest_cluster + "|" + work[CITY_COL].astype(str)
    work["event_dow_interaction"] = (
        nearest_cluster + "|dow_" + work["dow"].astype(str)
    )
    return add_payday_distance_features(work)


def add_payday_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    dates = pd.to_datetime(work[DATE_COL], errors="coerce")
    days_to = []
    days_since = []
    for date_value in dates:
        if pd.isna(date_value):
            days_to.append(999)
            days_since.append(999)
            continue
        candidates = []
        for month_offset in [-1, 0, 1]:
            shifted = date_value + pd.DateOffset(months=month_offset)
            for day in PAYDAY_DAYS:
                try:
                    candidates.append(pd.Timestamp(shifted.year, shifted.month, day))
                except ValueError:
                    continue
        future = [(c - date_value).days for c in candidates if c >= date_value]
        past = [(date_value - c).days for c in candidates if c <= date_value]
        days_to.append(min(future) if future else 999)
        days_since.append(min(past) if past else 999)

    work["days_to_payday"] = pd.Series(days_to, index=work.index).astype(int)
    work["days_since_payday"] = pd.Series(days_since, index=work.index).astype(int)
    work["payday_distance"] = np.minimum(
        work["days_to_payday"],
        work["days_since_payday"],
    )
    work["is_payday"] = (work["payday_distance"] == 0).astype(int)
    work["is_pre_payday_1d"] = (work["days_to_payday"] == 1).astype(int)
    work["is_pre_payday_2d"] = (work["days_to_payday"] == 2).astype(int)
    work["is_post_payday_1d"] = (work["days_since_payday"] == 1).astype(int)
    work["is_post_payday_2d"] = (work["days_since_payday"] == 2).astype(int)

    work["payday_window_type"] = "regular"
    work.loc[work["is_payday"] == 1, "payday_window_type"] = "payday"
    work.loc[work["is_pre_payday_1d"] == 1, "payday_window_type"] = "pre_payday_1"
    work.loc[work["is_pre_payday_2d"] == 1, "payday_window_type"] = "pre_payday_2"
    work.loc[work["is_post_payday_1d"] == 1, "payday_window_type"] = (
        "post_payday_1"
    )
    work.loc[work["is_post_payday_2d"] == 1, "payday_window_type"] = (
        "post_payday_2"
    )
    work["payday_dow_interaction"] = (
        work["payday_window_type"].astype(str) + "|dow_" + work["dow"].astype(str)
    )
    return work


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.sort_values([BAKERY_ID_COL, DATE_COL]).copy()
    grp = work.groupby(BAKERY_ID_COL, sort=False)
    dow_grp = work.groupby([BAKERY_ID_COL, "dow"], sort=False)

    work["bakery_sales_dow_mean"] = dow_grp[TARGET_COL].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    work["bakery_sales_trend"] = work["bakery_sales_roll_mean7"] / (work["bakery_sales_roll_mean30"] + 1e-8)
    work["bakery_sales_cv_7d"] = work["bakery_sales_roll_std7"] / (work["bakery_sales_roll_mean7"] + 1.0)

    if "avg_price" not in work.columns:
        work["avg_price"] = np.nan
    work["avg_price"] = grp["avg_price"].transform(lambda x: x.ffill().bfill())

    numeric_fill_cols = [
        "avg_price",
        "bakery_sales_dow_mean",
        "bakery_sales_trend",
        "bakery_sales_cv_7d",
        "bakery_sales_roll_std7",
        "bakery_sales_roll_mean3",
        "bakery_sales_roll_mean7",
        "bakery_sales_roll_mean14",
        "bakery_sales_roll_mean30",
        "bakery_sales_lag1",
        "bakery_sales_lag2",
        "bakery_sales_lag3",
        "bakery_sales_lag7",
        "bakery_sales_lag14",
        "bakery_sales_lag30",
    ]
    for col in numeric_fill_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
    return work


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work = add_holiday_features(work)
    work = add_event_cluster_features(work)
    work = add_enriched_event_features(work)
    work = add_derived_features(work)
    return work.sort_values([BAKERY_ID_COL, DATE_COL]).reset_index(drop=True)


def cast_category_columns(train_x: pd.DataFrame, test_x: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    for col in [c for c in CATEGORICAL_COLS if c in feature_cols]:
        train_x[col] = train_x[col].astype("category")
        test_x[col] = pd.Categorical(test_x[col], categories=train_x[col].cat.categories)
    return train_x, test_x


def make_train_test_split(df: pd.DataFrame, test_days: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    max_date = df[DATE_COL].max()
    test_start = max_date - pd.Timedelta(days=test_days - 1)
    train_df = df[df[DATE_COL] < test_start].copy()
    test_df = df[df[DATE_COL] >= test_start].copy()
    return train_df, test_df, test_start


def train_bakery_model(df: pd.DataFrame, *, test_days: int = DEFAULT_TEST_DAYS) -> tuple[object, dict, pd.DataFrame, pd.DataFrame]:
    frame = build_model_frame(df)
    train_df, test_df, test_start = make_train_test_split(frame, test_days)
    feature_cols = [col for col in BASE_FEATURES if col in frame.columns]

    train_x = train_df[feature_cols].copy()
    test_x = test_df[feature_cols].copy()
    train_x, test_x = cast_category_columns(train_x, test_x, feature_cols)

    model = train_lgbm(train_x, train_df[TARGET_COL])
    preds = predict_clipped(model, test_x)

    metrics = regression_metrics(test_df[TARGET_COL], preds)
    summary = {
        "rows_total": int(len(frame)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "bakeries": int(frame[BAKERY_ID_COL].nunique()),
        "date_min": str(frame[DATE_COL].min().date()),
        "date_max": str(frame[DATE_COL].max().date()),
        "test_start": str(test_start.date()),
        "feature_count": len(feature_cols),
        "feature_cols": feature_cols,
        "categorical_cols": [c for c in CATEGORICAL_COLS if c in feature_cols],
        "metrics": {k: round(v, 6) for k, v in metrics.items()},
    }

    holdout = test_df[[DATE_COL, BAKERY_ID_COL, BAKERY_NAME_COL, CITY_COL, TARGET_COL]].copy()
    holdout[FORECAST_COL] = preds
    bias_by_bakery = build_bias_by_bakery(holdout)
    summary["bias_top_overforecast"] = bias_by_bakery.sort_values("bias", ascending=True).head(10)[
        [BAKERY_ID_COL, BAKERY_NAME_COL, "bias", "bias_pct_of_actual_mean", "mae"]
    ].to_dict("records")
    summary["bias_top_underforecast"] = bias_by_bakery.sort_values("bias", ascending=False).head(10)[
        [BAKERY_ID_COL, BAKERY_NAME_COL, "bias", "bias_pct_of_actual_mean", "mae"]
    ].to_dict("records")
    return model, summary, holdout, bias_by_bakery


def save_training_artifacts(
    model: object,
    summary: dict,
    holdout: pd.DataFrame,
    bias_by_bakery: pd.DataFrame,
    model_path: str | Path,
    meta_path: str | Path,
    summary_path: str | Path,
    predictions_path: str | Path,
    bias_path: str | Path,
) -> dict[str, Path]:
    model_path = Path(model_path)
    meta_path = Path(meta_path)
    summary_path = Path(summary_path)
    predictions_path = Path(predictions_path)
    bias_path = Path(bias_path)
    for path in [model_path, meta_path, summary_path, predictions_path, bias_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(summary, meta_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    holdout.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    bias_by_bakery.to_csv(bias_path, index=False, encoding="utf-8-sig")
    return {
        "model": model_path,
        "meta": meta_path,
        "summary": summary_path,
        "holdout_predictions": predictions_path,
        "bias_by_bakery": bias_path,
    }


def _safe_tail_value(series: pd.Series, lag: int) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) < lag:
        return 0.0
    return float(numeric.iloc[-lag])


def _safe_rolling_mean(series: pd.Series, window: int, min_periods: int = 1) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    count = min(len(numeric), window)
    if count < min_periods:
        return float(numeric.mean())
    return float(numeric.iloc[-count:].mean())


def _safe_rolling_std(series: pd.Series, window: int, min_periods: int = 2) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    count = min(len(numeric), window)
    if count < min_periods:
        return 0.0
    return float(numeric.iloc[-count:].std(ddof=1) or 0.0)


def build_future_feature_rows(history_df: pd.DataFrame, forecast_date: pd.Timestamp) -> pd.DataFrame:
    rows: list[dict] = []
    dow = int(forecast_date.dayofweek)
    day = int(forecast_date.day)
    month = int(forecast_date.month)
    iso_week = int(forecast_date.isocalendar().week)

    for bakery_id, group in history_df.groupby(BAKERY_ID_COL, sort=False):
        group = group.sort_values(DATE_COL).copy()
        sales = pd.to_numeric(group[TARGET_COL], errors="coerce").fillna(0.0)
        same_dow = group.loc[group["dow"] == dow, TARGET_COL] if "dow" in group.columns else pd.Series(dtype=float)
        avg_price = pd.to_numeric(group.get("avg_price", pd.Series(dtype=float)), errors="coerce").dropna()

        rows.append(
            {
                DATE_COL: forecast_date,
                BAKERY_ID_COL: bakery_id,
                BAKERY_NAME_COL: group[BAKERY_NAME_COL].iloc[-1],
                CITY_COL: group[CITY_COL].iloc[-1] if CITY_COL in group.columns else "unknown",
                "dow": dow,
                "day": day,
                "month": month,
                "iso_week": iso_week,
                "is_weekend": int(dow >= 5),
                "is_month_start": int(day <= 5),
                "is_month_end": int(day >= 25),
                "is_payday_week": int(day in [4, 5, 6, 19, 20, 21]),
                "avg_price": float(avg_price.iloc[-1]) if len(avg_price) else 0.0,
                "bakery_sales_lag1": _safe_tail_value(sales, 1),
                "bakery_sales_lag2": _safe_tail_value(sales, 2),
                "bakery_sales_lag3": _safe_tail_value(sales, 3),
                "bakery_sales_lag7": _safe_tail_value(sales, 7),
                "bakery_sales_lag14": _safe_tail_value(sales, 14),
                "bakery_sales_lag30": _safe_tail_value(sales, 30),
                "bakery_sales_roll_mean3": _safe_rolling_mean(sales, 3, min_periods=1),
                "bakery_sales_roll_mean7": _safe_rolling_mean(sales, 7, min_periods=1),
                "bakery_sales_roll_mean14": _safe_rolling_mean(sales, 14, min_periods=7),
                "bakery_sales_roll_mean30": _safe_rolling_mean(sales, 30, min_periods=14),
                "bakery_sales_roll_std7": _safe_rolling_std(sales, 7, min_periods=2),
                "bakery_sales_dow_mean": float(pd.to_numeric(same_dow, errors="coerce").dropna().mean()) if len(same_dow) else 0.0,
            }
        )

    future = pd.DataFrame(rows)
    future = add_holiday_features(future)
    future = add_event_cluster_features(future)
    future = add_enriched_event_features(future)
    future["bakery_sales_trend"] = future["bakery_sales_roll_mean7"] / (future["bakery_sales_roll_mean30"] + 1e-8)
    future["bakery_sales_cv_7d"] = future["bakery_sales_roll_std7"] / (future["bakery_sales_roll_mean7"] + 1.0)
    return future


def recursive_forecast(
    history_df: pd.DataFrame,
    model: object,
    feature_cols: list[str],
    *,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    start_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    history = build_model_frame(history_df).copy()
    history = history.sort_values([BAKERY_ID_COL, DATE_COL]).reset_index(drop=True)

    next_date = pd.Timestamp(start_date) if start_date is not None else history[DATE_COL].max() + pd.Timedelta(days=1)
    future_frames: list[pd.DataFrame] = []

    for step in range(horizon_days):
        forecast_date = next_date + pd.Timedelta(days=step)
        future_rows = build_future_feature_rows(history, forecast_date)
        predict_x = future_rows[feature_cols].copy()
        for col in [c for c in CATEGORICAL_COLS if c in feature_cols]:
            predict_x[col] = predict_x[col].astype("category")
        preds = predict_clipped(model, predict_x)
        future_rows[FORECAST_COL] = preds
        future_rows[TARGET_COL] = preds
        future_frames.append(
            future_rows[[DATE_COL, BAKERY_ID_COL, BAKERY_NAME_COL, CITY_COL, FORECAST_COL]].copy()
        )
        history = pd.concat([history, future_rows], ignore_index=True, sort=False)

    forecast = pd.concat(future_frames, ignore_index=True)
    return forecast.sort_values([BAKERY_ID_COL, DATE_COL]).reset_index(drop=True)


def load_bias_table(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = [BAKERY_ID_COL, "bias"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required bias columns: {missing}")

    work = df[[BAKERY_ID_COL, "bias"]].copy()
    work["bias"] = pd.to_numeric(work["bias"], errors="coerce").fillna(0.0)
    return work.groupby(BAKERY_ID_COL, as_index=False).agg(bias=("bias", "mean"))


def apply_bias_correction(
    forecast: pd.DataFrame,
    bias_table: pd.DataFrame,
    *,
    clip_pct: float | None = None,
) -> pd.DataFrame:
    work = forecast.merge(bias_table, on=BAKERY_ID_COL, how="left", validate="many_to_one")
    work["bias"] = pd.to_numeric(work["bias"], errors="coerce").fillna(0.0)

    adjustment = work["bias"].copy()
    if clip_pct is not None:
        max_adjustment = pd.to_numeric(work[FORECAST_COL], errors="coerce").fillna(0.0) * float(clip_pct)
        adjustment = adjustment.clip(lower=-max_adjustment, upper=max_adjustment)

    work[FORECAST_BIAS_ADJ_COL] = (
        pd.to_numeric(work[FORECAST_COL], errors="coerce").fillna(0.0) + adjustment
    ).clip(lower=0.0)
    return work.drop(columns=["bias"])


def run_train_mode(args: argparse.Namespace) -> dict[str, Path]:
    df = load_dataset(args.dataset_path)
    model, summary, holdout, bias_by_bakery = train_bakery_model(df, test_days=args.test_days)
    return save_training_artifacts(
        model,
        summary,
        holdout,
        bias_by_bakery,
        args.model_path,
        args.meta_path,
        args.summary_path,
        args.predictions_path,
        args.bias_path,
    )


def run_forecast_mode(args: argparse.Namespace) -> Path:
    df = load_dataset(args.dataset_path)
    model = joblib.load(args.model_path)
    meta = joblib.load(args.meta_path)
    feature_cols = meta["feature_cols"]
    forecast = recursive_forecast(
        df,
        model,
        feature_cols,
        horizon_days=args.horizon_days,
        start_date=args.start_date,
    )
    if args.apply_bias_correction:
        bias_table = load_bias_table(args.bias_path)
        clip_pct = None if args.bias_clip_pct is None else float(args.bias_clip_pct)
        forecast = apply_bias_correction(forecast, bias_table, clip_pct=clip_pct)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bakery-level daily forecast pipeline")
    parser.add_argument("--mode", choices=["train", "forecast"], default="train")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--meta-path", default=str(DEFAULT_META_PATH))
    parser.add_argument("--summary-path", default=str(DEFAULT_TRAIN_SUMMARY_PATH))
    parser.add_argument("--predictions-path", default=str(DEFAULT_PREDICTIONS_PATH))
    parser.add_argument("--bias-path", default=str(DEFAULT_BIAS_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_FORECAST_PATH))
    parser.add_argument("--test-days", type=int, default=DEFAULT_TEST_DAYS)
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS)
    parser.add_argument("--start-date", default=None, help="Optional forecast start date YYYY-MM-DD")
    parser.add_argument("--apply-bias-correction", action="store_true")
    parser.add_argument("--bias-clip-pct", type=float, default=0.15, help="Cap bakery bias correction by share of forecast")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "train":
        paths = run_train_mode(args)
        print("=" * 72)
        print("BAKERY DAY MODEL TRAIN")
        print("=" * 72)
        for name, path in paths.items():
            print(f"{name}: {path}")
    else:
        path = run_forecast_mode(args)
        print("=" * 72)
        print("BAKERY DAY FORECAST")
        print("=" * 72)
        print(f"forecast: {path}")


if __name__ == "__main__":
    main()

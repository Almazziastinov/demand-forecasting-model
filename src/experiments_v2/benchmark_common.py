"""
Shared helpers for benchmark-style experiment runners in experiments_v2.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Allow imports from project root
ROOT = Path(__file__).resolve().parent.parent.parent
import sys

sys.path.insert(0, str(ROOT))

from src.experiments_v2.common import (  # noqa: E402
    CATEGORICAL_COLS_V2,
    DEMAND_8M_PATH,
    DEMAND_TARGET,
    TEST_DAYS,
    predict_clipped,
    train_quantile,
    wmape,
)


DATE_COL = "Дата"
BAKERY_COL = "Пекарня"
PRODUCT_COL = "Номенклатура"
DOW_COL = "ДеньНедели"
SOLD_COL = "Продано"


def load_daily_demand(selected_bakeries: list[str] | None = None) -> pd.DataFrame:
    """Load the processed daily demand dataset and optionally filter bakeries."""
    df = pd.read_csv(DEMAND_8M_PATH, encoding="utf-8-sig")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, BAKERY_COL, PRODUCT_COL]).copy()
    if selected_bakeries:
        df = df[df[BAKERY_COL].isin(selected_bakeries)].copy()
    df = df.sort_values([BAKERY_COL, PRODUCT_COL, DATE_COL]).reset_index(drop=True)
    return df


def load_global_model(model_path: str | Path) -> dict:
    """Load the best global model artifact."""
    return joblib.load(Path(model_path))


def load_best_by_sku(summary_path: str | Path) -> pd.DataFrame:
    """Load the benchmark-level best-by-SKU comparison table."""
    path = Path(summary_path)
    if not path.exists():
        raise FileNotFoundError(f"Best-by-SKU summary not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def build_r2_filtered_subset(
    best_by_sku: pd.DataFrame,
    r2_threshold: float,
    base_model_col: str = "global_best_r2",
) -> pd.DataFrame:
    """Return SKU pairs where the chosen model's R2 is below the threshold.

    The default base model is `global_best_r2` so the subset highlights rows where
    the current production-like model is weak enough to justify extra benchmarking.
    """
    if base_model_col not in best_by_sku.columns:
        raise KeyError(f"Missing required column: {base_model_col}")

    work = best_by_sku.copy()
    work[base_model_col] = pd.to_numeric(work[base_model_col], errors="coerce")
    work = work.dropna(subset=[base_model_col]).copy()
    return work.loc[work[base_model_col] < r2_threshold].reset_index(drop=True)


def add_censoring_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values([BAKERY_COL, PRODUCT_COL, DATE_COL]).copy()
    grp = df.groupby([BAKERY_COL, PRODUCT_COL])
    df["is_censored_lag1"] = grp["is_censored"].shift(1).fillna(0).astype(int)
    df["lost_qty_lag1"] = grp["lost_qty"].shift(1).fillna(0)
    df["pct_censored_7d"] = (
        grp["is_censored"].transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean() * 100)
    ).fillna(0)
    return df


def add_dow_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values([BAKERY_COL, PRODUCT_COL, DATE_COL]).copy()
    grp = df.groupby([BAKERY_COL, PRODUCT_COL, DOW_COL])
    df["sales_dow_mean"] = grp[SOLD_COL].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    ).fillna(0)
    df["demand_dow_mean"] = grp[DEMAND_TARGET].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    ).fillna(0)
    return df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df["demand_trend"] = df["demand_roll_mean7"] / (df["demand_roll_mean30"] + 1e-8)
    df["cv_7d"] = df["sales_roll_std7"] / (df["sales_roll_mean7"] + 1)
    return df


def add_assortment_features(df: pd.DataFrame) -> pd.DataFrame:
    df["items_in_bakery_today"] = df.groupby([BAKERY_COL, DATE_COL])[PRODUCT_COL].transform("nunique")
    bakery_day = (
        df.groupby([BAKERY_COL, DATE_COL])[PRODUCT_COL]
        .nunique()
        .reset_index(name="__bakery_items")
        .sort_values([BAKERY_COL, DATE_COL])
    )
    bakery_day["items_in_bakery_lag1"] = bakery_day.groupby(BAKERY_COL)["__bakery_items"].shift(1)
    bakery_day["items_change"] = bakery_day["__bakery_items"] - bakery_day["items_in_bakery_lag1"]
    bakery_day.drop(columns=["__bakery_items"], inplace=True)
    df = df.merge(bakery_day, on=[BAKERY_COL, DATE_COL], how="left")
    for col in ["items_in_bakery_today", "items_in_bakery_lag1", "items_change"]:
        df[col] = df[col].fillna(0)
    return df


def assign_ts_clusters(df: pd.DataFrame, ts_pipeline: dict) -> pd.DataFrame:
    """Assign cluster_ts using the saved KMeans pipeline from the global model."""
    scaler = ts_pipeline["scaler"]
    kmeans = ts_pipeline["kmeans"]
    feat_cols = ts_pipeline["feat_cols"]

    cluster_df = (
        df.groupby([BAKERY_COL, PRODUCT_COL])
        .agg(
            demand_mean=(DEMAND_TARGET, "mean"),
            demand_std=(DEMAND_TARGET, "std"),
            sales_std=(SOLD_COL, "std"),
            n_days=(DATE_COL, "count"),
            lost_mean=("lost_qty", "mean"),
            censored_pct=("is_censored", "mean"),
            dow_std=(SOLD_COL, lambda x: x.groupby(df.loc[x.index, DOW_COL]).sum().std()),
        )
        .reset_index()
    )
    for col in ["demand_std", "sales_std", "dow_std"]:
        cluster_df[col] = cluster_df[col].fillna(0)
    cluster_df["cv"] = cluster_df["demand_std"] / (cluster_df["demand_mean"] + 1e-8)

    last_week = df[df[DATE_COL] >= df[DATE_COL].max() - pd.Timedelta(days=7)]
    last_week_agg = (
        last_week.groupby([BAKERY_COL, PRODUCT_COL])[DEMAND_TARGET].mean().reset_index(name="last_week_mean")
    )
    cluster_df = cluster_df.merge(last_week_agg, on=[BAKERY_COL, PRODUCT_COL], how="left")
    cluster_df["last_week_mean"] = cluster_df["last_week_mean"].fillna(cluster_df["demand_mean"])
    cluster_df["trend"] = (cluster_df["last_week_mean"] / (cluster_df["demand_mean"] + 1e-8)).clip(0.5, 2.0)

    X = cluster_df[feat_cols].copy()
    X["demand_mean"] = np.log1p(X["demand_mean"])
    X["n_days"] = np.log1p(X["n_days"])
    X_scaled = scaler.transform(X.values)
    cluster_df["cluster_ts"] = kmeans.predict(X_scaled).astype(int)

    return cluster_df[[BAKERY_COL, PRODUCT_COL, "cluster_ts"]]


def build_benchmark_features(df: pd.DataFrame, model_data: dict) -> pd.DataFrame:
    """Build all global-model features used by the benchmark."""
    df = df.copy()
    df = add_censoring_features(df)
    df = add_dow_features(df)
    df = add_trend_features(df)
    if "stale_ratio_lag1" not in df.columns:
        df["stale_ratio_lag1"] = 0.0
    df = add_assortment_features(df)

    loc_clusters = model_data["loc_clusters"]
    df = df.merge(loc_clusters, on=BAKERY_COL, how="left")
    df["cluster_loc"] = df["cluster_loc"].fillna(-1).astype(int)

    ts_clusters = assign_ts_clusters(df, model_data["ts_pipeline"])
    df = df.merge(ts_clusters, on=[BAKERY_COL, PRODUCT_COL], how="left")
    df["cluster_ts"] = df["cluster_ts"].fillna(-1).astype(int)

    for col in CATEGORICAL_COLS_V2:
        if col in df.columns:
            df[col] = df[col].astype("category")
    if "cluster_loc" in df.columns:
        df["cluster_loc"] = df["cluster_loc"].astype("category")
    if "cluster_ts" in df.columns:
        df["cluster_ts"] = df["cluster_ts"].astype("category")

    return df


def predict_global_from_model_data(df: pd.DataFrame, model_data: dict) -> np.ndarray:
    """Predict using the saved global model artifact."""
    variant = model_data["variant"]
    model = model_data["model"]
    features = model_data["features"]
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features for global model: {missing[:10]}")

    X = df[features].copy()
    if variant == "E_routed_cluster_ts":
        cluster_models = model["cluster_models"]
        fallback = model["fallback"]
        preds = np.full(len(df), np.nan, dtype=float)
        for cluster_id, cluster_model in cluster_models.items():
            mask = df["cluster_ts"].astype(int) == cluster_id
            if mask.any():
                preds[mask] = predict_clipped(cluster_model, X.loc[mask])
        missing_mask = np.isnan(preds)
        if missing_mask.any():
            fb = fallback if fallback is not None else next(iter(cluster_models.values()))
            preds[missing_mask] = predict_clipped(fb, X.loc[missing_mask])
        return preds

    return predict_clipped(model, X)


def make_train_test_split(df: pd.DataFrame, test_days: int = TEST_DAYS) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split by the last `test_days` days, inclusive of the max date."""
    max_date = df[DATE_COL].max()
    test_start = max_date - pd.Timedelta(days=test_days - 1)
    train = df[df[DATE_COL] < test_start].copy()
    test = df[df[DATE_COL] >= test_start].copy()
    return train, test, test_start


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict:
    """Compute regression metrics with safe handling for tiny series."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if len(y_true_arr) == 0:
        return {"mae": np.nan, "mse": np.nan, "r2": np.nan, "wmape": np.nan}
    r2 = np.nan
    if len(y_true_arr) >= 2 and not np.allclose(y_true_arr, y_true_arr[0]):
        r2 = float(r2_score(y_true_arr, y_pred_arr))
    return {
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "mse": float(mean_squared_error(y_true_arr, y_pred_arr)),
        "r2": r2,
        "wmape": float(wmape(y_true_arr, y_pred_arr)),
    }


def cast_category_columns(train_x: pd.DataFrame, test_x: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Align categorical dtypes between train and test."""
    cat_cols = [c for c in CATEGORICAL_COLS_V2 + ["cluster_loc", "cluster_ts"] if c in feature_cols]
    for col in cat_cols:
        if col not in train_x.columns or col not in test_x.columns:
            continue
        train_x[col] = train_x[col].astype("category")
        test_x[col] = pd.Categorical(test_x[col], categories=train_x[col].cat.categories)
    return train_x, test_x, cat_cols


def select_feature_columns(train_df: pd.DataFrame, base_features: list[str], drop_constants: bool = True) -> list[str]:
    """Select usable features for a specific training subset."""
    available = [f for f in base_features if f in train_df.columns]
    if not drop_constants:
        return available
    selected: list[str] = []
    for col in available:
        series = train_df[col]
        if series.isna().all():
            continue
        if series.nunique(dropna=True) <= 1:
            continue
        selected.append(col)
    return selected


def train_and_predict_quantile(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = DEMAND_TARGET,
    min_train_rows: int = 30,
) -> tuple[np.ndarray, dict]:
    """Train a LightGBM quantile model and predict on test; fallback to train-mean when needed."""
    train_df = train_df.dropna(subset=[target_col]).copy()
    test_df = test_df.copy()
    feature_cols = [f for f in feature_cols if f in train_df.columns and f in test_df.columns]
    feature_cols = [f for f in feature_cols if train_df[f].notna().any()]

    if len(feature_cols) == 0 or len(train_df) < min_train_rows:
        fallback = float(train_df[target_col].mean()) if len(train_df) else 0.0
        return np.full(len(test_df), fallback, dtype=float), {
            "status": "fallback_mean",
            "n_train": len(train_df),
            "n_features": len(feature_cols),
        }

    train_x = train_df[feature_cols].copy()
    test_x = test_df[feature_cols].copy()
    train_x, test_x, cat_cols = cast_category_columns(train_x, test_x, feature_cols)

    train_mask = train_x.notna().all(axis=1)
    if not train_mask.any():
        fallback = float(train_df[target_col].mean()) if len(train_df) else 0.0
        return np.full(len(test_df), fallback, dtype=float), {
            "status": "fallback_mean_no_train_rows",
            "n_train": len(train_df),
            "n_features": len(feature_cols),
        }

    train_x = train_x.loc[train_mask]
    y_train = train_df.loc[train_x.index, target_col]

    model = train_quantile(train_x, y_train, alpha=0.5)
    preds = predict_clipped(model, test_x)
    return preds, {
        "status": "trained",
        "n_train": len(train_x),
        "n_features": len(feature_cols),
        "categorical_cols": cat_cols,
    }


def two_week_average_predictions(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str = DEMAND_TARGET,
    window: int = 14,
) -> np.ndarray:
    """Predict each row as the mean of the previous `window` observations within group."""
    work = df.sort_values(group_cols + [DATE_COL]).copy()
    grp = work.groupby(group_cols, sort=False)
    pred = grp[target_col].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    pred = pred.fillna(work[target_col].mean() if len(work) else 0.0)
    return pred.to_numpy(dtype=float)

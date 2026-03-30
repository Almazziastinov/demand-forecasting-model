"""
Shared utilities for architecture experiments.
Data loading, metrics, model params -- extracted from tune_v6.py.
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import (
    FEATURES, TARGET, CATEGORICAL_COLS, MODEL_PARAMS, TEST_DAYS,
    MERGED_DATA_PATH,
)

warnings.filterwarnings("ignore")

DATA_PATH = MERGED_DATA_PATH
BASELINE_MAE = None  # will be computed dynamically


def load_data():
    """Load data, split into train/test (last TEST_DAYS days = test).
    Returns: df, X_train, y_train, X_test, y_test, features list.
    """
    print(f"Zagruzka {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df["Дата"] = pd.to_datetime(df["Дата"])

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    available = [f for f in FEATURES if f in df.columns]
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"  [!] Otsutstvuyut priznaki: {missing}")

    test_start = df["Дата"].max() - pd.Timedelta(days=TEST_DAYS - 1)
    train = df[df["Дата"] < test_start].copy()
    test = df[df["Дата"] >= test_start].copy()

    print(f"  Train: {len(train):,} strok ({train['Дата'].nunique()} dnej)")
    print(f"  Test:  {len(test):,} strok ({test['Дата'].nunique()} dnej: "
          f"{test['Дата'].min().date()} -- {test['Дата'].max().date()})")
    print(f"  Priznakov: {len(available)}")

    X_train = train[available]
    y_train = train[TARGET]
    X_test = test[available]
    y_test = test[TARGET]

    return df, X_train, y_train, X_test, y_test, available


def wmape(y_true, y_pred):
    """Weighted Mean Absolute Percentage Error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(y_true) + 1e-8) * 100


def print_metrics(name, y_true, y_pred, baseline_mae=None):
    """Print MAE, WMAPE, Bias, delta vs baseline."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    wm = wmape(y_true, y_pred)
    bias = np.mean(y_true - y_pred)

    delta_str = ""
    if baseline_mae is not None:
        delta = mae - baseline_mae
        delta_str = f"  (delta vs baseline: {delta:+.4f})"

    print(f"  {name}:")
    print(f"    MAE   = {mae:.4f}{delta_str}")
    print(f"    WMAPE = {wm:.2f}%")
    print(f"    Bias  = {bias:+.4f}")
    return mae, wm, bias


def print_category_metrics(y_true, y_pred, categories):
    """Per-category MAE breakdown."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    categories = np.asarray(categories)

    print(f"    {'Kategoriya':<25} {'N':>6} {'MAE':>8} {'Bias':>8}")
    print(f"    {'-' * 49}")

    for cat in sorted(set(categories)):
        mask = categories == cat
        n = mask.sum()
        mae = mean_absolute_error(y_true[mask], y_pred[mask])
        bias = np.mean(y_true[mask] - y_pred[mask])
        print(f"    {str(cat):<25} {n:>6} {mae:>8.4f} {bias:>+8.4f}")


def train_lgbm(X_train, y_train, params=None):
    """Train LightGBM with given or default params. Returns fitted model."""
    p = (params or MODEL_PARAMS).copy()
    model = LGBMRegressor(**p)
    model.fit(X_train, y_train)
    return model


def predict_clipped(model, X):
    """Predict and clip to >= 0."""
    return np.maximum(model.predict(X), 0)


def save_predictions(X_test, y_test, y_pred, filepath, extra_cols=None):
    """Save predictions to CSV."""
    result = pd.DataFrame({"fact": np.asarray(y_test), "pred": np.asarray(y_pred)})
    result["abs_error"] = np.abs(result["pred"] - result["fact"])
    if "Категория" in X_test.columns:
        result["Категория"] = X_test["Категория"].values
    if extra_cols:
        for col_name, col_data in extra_cols.items():
            result[col_name] = col_data
    result.to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"  Predskazaniya sokhraneny: {filepath}")

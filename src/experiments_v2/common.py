"""
Shared utilities for experiments_v2 (hourly check-level data, 8 months).
Based on src/experiments/common.py but adapted for the new dataset.
"""

import sys
import os
import warnings
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import TARGET, CATEGORICAL_COLS, MODEL_PARAMS, TEST_DAYS

warnings.filterwarnings("ignore")

# New data paths
SALES_HRS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "sales_hrs_all.csv"
DAILY_8M_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "daily_sales_8m.csv"

# Experiment results go into each experiment's own folder
RESULTS_DIR = Path(__file__).resolve().parent

# --- Features v2: 45 features (no stock features, no data on Vypusk/Ostatok in checks) ---
FEATURES_V2 = [
    # Categorical
    "Пекарня", "Номенклатура", "Категория", "Город",
    # Calendar
    "ДеньНедели", "День", "IsWeekend", "Месяц", "НомерНедели",
    # Sales lags
    "sales_lag1", "sales_lag2", "sales_lag3", "sales_lag7",
    "sales_lag14", "sales_lag30",
    # Rolling stats
    "sales_roll_mean3", "sales_roll_mean7", "sales_roll_std7",
    "sales_roll_mean14", "sales_roll_mean30",
    # Holiday / calendar
    "is_holiday", "is_pre_holiday", "is_post_holiday", "is_payday_week",
    "is_month_start", "is_month_end",
    # Weather (15 features)
    "temp_max", "temp_min", "temp_mean", "temp_range",
    "precipitation", "rain", "snowfall", "windspeed_max",
    "is_rainy", "is_snowy", "is_cold", "is_warm",
    "is_windy", "is_bad_weather", "weather_cat_code",
]

CATEGORICAL_COLS_V2 = ["Пекарня", "Номенклатура", "Категория", "Город", "Месяц"]

# City coordinates for weather (original 7 + Kursk, Moskva)
CITY_COORDS = {
    "Казань":             {"lat": 55.7879, "lon": 49.1233},
    "Чебоксары":          {"lat": 56.1439, "lon": 47.2489},
    "Набережные Челны":   {"lat": 55.7439, "lon": 52.4042},
    "Нижнекамск":         {"lat": 55.6386, "lon": 51.8221},
    "Зеленодольск":       {"lat": 55.8430, "lon": 48.5206},
    "Бугульма":           {"lat": 54.5393, "lon": 52.7959},
    "Заинск":             {"lat": 55.2978, "lon": 52.0046},
    "Курск":              {"lat": 51.7373, "lon": 36.1874},
    "Москва":             {"lat": 55.7558, "lon": 37.6173},
}

# City extraction: last word(s) of bakery name -> city
# Special cases: villages near Kazan, Novocheboksarsk -> Cheboksary
CITY_SUFFIXES = {
    "Казань": "Казань",
    "Челны": "Набережные Челны",
    "Чебоксары": "Чебоксары",
    "Нижнекамск": "Нижнекамск",
    "Зеленодольск": "Зеленодольск",
    "Бугульма": "Бугульма",
    "Заинск": "Заинск",
    "Курск": "Курск",
    "Москва": "Москва",
    "Новочебоксарск": "Чебоксары",
    # Villages near Kazan
    "Куюки": "Казань",
    "Сокуры": "Казань",
    "Дербышки": "Казань",
    "Васильево": "Казань",
    "Габишево": "Казань",
}


def extract_city(bakery_name):
    """Extract city from bakery name using suffix matching."""
    if pd.isna(bakery_name):
        return "Казань"
    name = str(bakery_name).strip()
    parts = name.split()
    if not parts:
        return "Казань"
    # Check last word
    last = parts[-1]
    if last in CITY_SUFFIXES:
        return CITY_SUFFIXES[last]
    # Check last two words (e.g. "Наб Челны")
    if len(parts) >= 2:
        last2 = parts[-2] + " " + parts[-1]
        if "Челны" in last2:
            return "Набережные Челны"
    # Default to Kazan
    return "Казань"


def load_checks(path=None):
    """Load the merged check-level data (sales_hrs_all.csv).
    Returns raw DataFrame with all 30M rows.
    """
    p = path or SALES_HRS_PATH
    print(f"Zagruzka {p}...", flush=True)
    df = pd.read_csv(p, encoding='utf-8-sig')
    print(f"  {len(df):,} strok, {df.columns.tolist()}", flush=True)
    return df


def aggregate_daily(df, date_col="Дата продажи", shop_col="Касса.Торговая точка",
                    cat_col="Категория", product_col="Номенклатура", qty_col="Кол-во"):
    """Aggregate check-level data to daily: shop x product x date -> total sold."""
    daily = (df.groupby([date_col, shop_col, cat_col, product_col])[qty_col]
             .sum()
             .reset_index()
             .rename(columns={qty_col: TARGET}))
    print(f"  Agregirovano: {len(daily):,} strok "
          f"({daily[date_col].nunique()} dnej)", flush=True)
    return daily


def wmape(y_true, y_pred):
    """Weighted Mean Absolute Percentage Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(y_true) + 1e-8) * 100


def print_metrics(name, y_true, y_pred, baseline_mae=None):
    """Print MAE, WMAPE, Bias, delta vs baseline."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
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
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
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


def save_results(exp_dir, metrics, predictions=None, extra=None):
    """Save experiment results into its folder.
    - metrics.json: MAE, WMAPE, Bias, etc.
    - predictions.csv: optional fact vs pred
    """
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    metrics["timestamp"] = datetime.now().isoformat()
    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  Metriki sokhraneny: {exp_dir / 'metrics.json'}", flush=True)

    if predictions is not None:
        pred_path = exp_dir / "predictions.csv"
        predictions.to_csv(pred_path, index=False, encoding="utf-8-sig")
        print(f"  Predskazaniya: {pred_path}", flush=True)

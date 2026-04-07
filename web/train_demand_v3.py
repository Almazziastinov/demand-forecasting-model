"""
Обучение модели V3: Demand Target + Price Features + Quantile P50.
Лучший результат: MAE 2.8816 (exp 60).

Usage:
    .venv/Scripts/python.exe web/train_demand_v3.py
"""

import os
import sys
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

from src.config import MODEL_PARAMS, TEST_DAYS
from src.experiments_v2.common import (
    DEMAND_8M_PATH, CITY_COORDS, extract_city,
)
from src.logger import get_logger

logger = get_logger("train_demand_v3", log_file="train_demand_v3.log")

MODEL_OUTPUT = ROOT / "models" / "demand_model_v3.pkl"
META_OUTPUT = ROOT / "models" / "model_meta_v3.pkl"
ARCHIVE_DIR = ROOT / "models" / "archive"


def add_price_features(df):
    """Добавить ценовые фичи."""
    print("  Добавление ценовых фичей...")
    
    df = df.sort_values(["Пекарня", "Номенклатура", "Дата"]).reset_index(drop=True)
    grouped = df.groupby(["Пекарня", "Номенклатура"])
    
    for lag in [1, 7]:
        df[f"price_lag{lag}"] = grouped["Цена"].shift(lag)
    
    for window in [7]:
        df[f"price_roll_mean{window}"] = grouped["Цена"].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f"price_roll_std{window}"] = grouped["Цена"].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=2).std()
        )
    
    median_price = df.groupby(["Пекарня", "Номенклатура"])["Цена"].transform("median")
    df["price_vs_median"] = df["Цена"] / (median_price + 1e-8)
    
    df["price_change_7d"] = (df["Цена"] - df["price_lag7"]) / (df["price_lag7"] + 1e-8)
    
    return df


def main():
    print("=" * 60)
    print("  TRAIN V3: Demand + Price + Quantile P50")
    print("=" * 60)
    
    # Загрузка данных
    print("\n[1] Загрузка данных...")
    df = pd.read_csv(str(DEMAND_8M_PATH), encoding="utf-8-sig")
    df["Дата"] = pd.to_datetime(df["Дата"])
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['Дата'].min().date()} -- {df['Дата'].max().date()}")
    
    # Добавляем ценовые фичи
    if "Цена" not in df.columns:
        print("  [!] Нет колонки Цена, добавляем...")
        # Цена не в данных - создадим заглушку (средняя по продукту)
        df["Цена"] = 100  # placeholder
    else:
        df = add_price_features(df)
    
    # Фичи
    features = [
        "Пекарня", "Номенклатура", "Категория", "Город",
        "ДеньНедели", "День", "IsWeekend", "Месяц", "НомерНедели",
        "sales_lag1", "sales_lag2", "sales_lag3", "sales_lag7", 
        "sales_lag14", "sales_lag30",
        "sales_roll_mean3", "sales_roll_mean7", "sales_roll_std7",
        "sales_roll_mean14", "sales_roll_mean30",
        "is_holiday", "is_pre_holiday", "is_post_holiday",
        "is_payday_week", "is_month_start", "is_month_end",
        "temp_max", "temp_min", "temp_mean", "temp_range",
        "precipitation", "rain", "snowfall", "windspeed_max",
        "is_rainy", "is_snowy", "is_cold", "is_warm",
        "is_windy", "is_bad_weather", "weather_cat_code",
        "avg_price", "price_vs_median", "price_lag7",
        "price_change_7d", "price_roll_mean7", "price_roll_std7",
    ]
    
    # Demand lags (если есть в данных)
    demand_features = [
        "demand_lag1", "demand_lag2", "demand_lag3", "demand_lag7",
        "demand_lag14", "demand_lag30",
        "demand_roll_mean3", "demand_roll_mean7", "demand_roll_mean14",
        "demand_roll_mean30", "demand_roll_std7",
    ]
    
    available = [f for f in features if f in df.columns]
    available_demand = [f for f in demand_features if f in df.columns]
    all_features = available + available_demand
    
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"  [!] Missing features: {missing}")
    
    print(f"  Features: {len(all_features)} ({len(available)} base + {len(available_demand)} demand)")
    
    # Категориальные колонки
    categorical = ["Пекарня", "Номенклатура", "Категория", "Город", "Месяц"]
    for col in categorical:
        if col in df.columns:
            df[col] = df[col].astype("category")
    
    # Split
    print("\n[2] Train/test split...")
    test_start = df["Дата"].max() - pd.Timedelta(days=TEST_DAYS - 1)
    train = df[df["Дата"] < test_start].copy()
    test = df[df["Дата"] >= test_start].copy()
    
    print(f"  Train: {len(train):,} rows ({train['Дата'].min().date()} -- {train['Дата'].max().date()})")
    print(f"  Test:  {len(test):,} rows ({test['Дата'].min().date()} -- {test['Дата'].max().date()})")
    
    X_train = train[all_features]
    y_train = train["Спрос"]
    X_test = test[all_features]
    y_test = test["Спрос"]
    
    # Обучение с Quantile P50
    print("\n[3] Training LightGBM (Quantile P50)...")
    params = MODEL_PARAMS.copy()
    params["objective"] = "quantile"
    params["alpha"] = 0.5
    params.pop("metric", None)
    params["n_estimators"] = min(params["n_estimators"], 1000)  # Limit for speed
    
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    
    # Предсказания
    y_pred = np.maximum(model.predict(X_test), 0)
    
    # Метрики
    mae = mean_absolute_error(y_test, y_pred)
    wmape = np.sum(np.abs(y_test - y_pred)) / (np.sum(y_test) + 1e-8) * 100
    bias = np.mean(y_test - y_pred)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    
    print(f"\n  === RESULTS ===")
    print(f"  MAE:   {mae:.4f}")
    print(f"  WMAPE: {wmape:.2f}%")
    print(f"  Bias:  {bias:+.4f}")
    print(f"  RMSE:  {rmse:.4f}")
    
    # Feature importance
    print(f"\n[4] Feature importance (top 10)...")
    imp = pd.DataFrame({
        "feature": all_features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    
    for _, row in imp.head(10).iterrows():
        print(f"  {row['feature']:<25} {row['importance']:>8.0f}")
    
    # Сохранение
    print("\n[5] Saving model...")
    
    # Архив
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = ARCHIVE_DIR / f"v3_{version}"
    os.makedirs(version_dir, exist_ok=True)
    
    joblib.dump(model, version_dir / "demand_model.pkl")
    
    meta = {
        "version": version,
        "experiment": "V3_demand_price_quantile",
        "features": all_features,
        "categorical_cols": [c for c in categorical if c in all_features],
        "target": "Спрос",
        "objective": "quantile_p50",
        "metrics": {
            "mae": round(mae, 4),
            "wmape": round(wmape, 2),
            "bias": round(bias, 4),
            "rmse": round(rmse, 4),
        },
        "train_rows": len(train),
        "test_rows": len(test),
        "model_params": {k: v for k, v in params.items() if k != "verbose"},
    }
    
    joblib.dump(meta, version_dir / "model_meta.pkl")
    
    # Копируем как latest
    shutil.copy2(version_dir / "demand_model.pkl", MODEL_OUTPUT)
    shutil.copy2(version_dir / "model_meta.pkl", META_OUTPUT)
    
    # Метрики JSON
    with open(version_dir / "metrics.json", "w") as f:
        json.dump(meta["metrics"], f, indent=2)
    
    print(f"  Model: {MODEL_OUTPUT}")
    print(f"  Meta:  {META_OUTPUT}")
    print(f"  Archive: {version_dir}")
    
    logger.info(f"V3 trained: MAE={mae:.4f}, WMAPE={wmape:.2f}%")
    print("\n  Done!")


if __name__ == "__main__":
    main()

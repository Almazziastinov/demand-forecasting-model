"""
Обучение LightGBM модели (v6 параметры) и сохранение в .pkl
Данные: preprocessed_data_3month_enriched.csv
"""

import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# Пути
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "preprocessed_data_3month_enriched.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(OUTPUT_DIR, "demand_model.pkl")
META_PATH = os.path.join(OUTPUT_DIR, "model_meta.pkl")

TARGET = "Продано"

FEATURES = [
    "Пекарня", "Номенклатура", "Категория", "Город",
    "ДеньНедели", "День", "IsWeekend", "Месяц", "НомерНедели",
    "sales_lag1", "sales_lag2", "sales_lag3", "sales_lag7",
    "sales_lag14", "sales_lag30",
    "sales_roll_mean3", "sales_roll_mean7", "sales_roll_std7",
    "sales_roll_mean14", "sales_roll_mean30",
    "stock_lag1", "stock_sales_ratio", "stock_deficit",
    "is_holiday", "is_pre_holiday", "is_post_holiday",
    "is_payday_week", "is_month_start", "is_month_end",
    "temp_mean", "temp_range", "precipitation",
    "is_cold", "is_bad_weather", "weather_cat_code",
]

CATEGORICAL_COLS = ["Пекарня", "Номенклатура", "Категория", "Город", "Месяц"]

# v6 лучшие параметры
MODEL_PARAMS = {
    "n_estimators": 2304,
    "learning_rate": 0.016085231060110994,
    "num_leaves": 151,
    "max_depth": 7,
    "min_child_samples": 5,
    "subsample": 0.8012777641245349,
    "colsample_bytree": 0.6654847541174173,
    "reg_alpha": 6.633500153052146,
    "reg_lambda": 2.204771489304501e-06,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def main():
    print("=" * 60)
    print("  ОБУЧЕНИЕ И СОХРАНЕНИЕ МОДЕЛИ v6")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"Ошибка: файл не найден: {DATA_PATH}")
        sys.exit(1)

    # --- Загрузка ---
    print(f"\nЗагрузка данных...")
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["Дата"] = pd.to_datetime(df["Дата"])

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    available = [f for f in FEATURES if f in df.columns]
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"  [!] Отсутствуют признаки: {missing}")
    print(f"  Признаков: {len(available)}")
    print(f"  Строк: {len(df):,}")

    # --- Сплит ---
    test_start = df["Дата"].max() - pd.Timedelta(days=2)
    train = df[df["Дата"] < test_start].copy()
    test = df[df["Дата"] >= test_start].copy()

    print(f"  Train: {len(train):,} строк ({train['Дата'].min().date()} — {train['Дата'].max().date()})")
    print(f"  Test:  {len(test):,} строк  ({test['Дата'].min().date()} — {test['Дата'].max().date()})")

    X_train, y_train = train[available], train[TARGET]
    X_test, y_test = test[available], test[TARGET]

    # --- Обучение ---
    print("\nОбучение LightGBM (v6 params)...")
    model = LGBMRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    print("  Обучение завершено.")

    # --- Метрики ---
    y_pred = np.maximum(model.predict(X_test), 0)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    wmape = np.sum(np.abs(y_test.values - y_pred)) / (np.sum(y_test.values) + 1e-8) * 100
    bias = np.mean(y_test.values - y_pred)

    print(f"\n{'=' * 60}")
    print(f"  МЕТРИКИ НА ТЕСТЕ")
    print(f"{'=' * 60}")
    print(f"  MAE:   {mae:.4f} шт.")
    print(f"  RMSE:  {rmse:.4f} шт.")
    print(f"  R2:    {r2:.4f}")
    print(f"  WMAPE: {wmape:.2f}%")
    print(f"  Bias:  {bias:+.4f}")

    # Важность признаков
    imp = pd.DataFrame({"feature": available, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    print(f"\nТоп-10 признаков:")
    for _, row in imp.head(10).iterrows():
        print(f"  {row['feature']:<25} {row['importance']:>8.0f}")

    # --- Сохранение ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    print(f"\nМодель сохранена: {MODEL_PATH}")

    # Метаданные для веб-приложения
    meta = {
        "features": available,
        "categorical_cols": [c for c in CATEGORICAL_COLS if c in available],
        "target": TARGET,
        "bakeries": sorted(df["Пекарня"].unique().tolist()),
        "products": sorted(df["Номенклатура"].unique().tolist()),
        "categories": sorted(df["Категория"].unique().tolist()),
        "cities": sorted(df["Город"].unique().tolist()),
        "metrics": {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4), "wmape": round(wmape, 2)},
        "feature_defaults": {
            col: float(df[col].median()) for col in available if col not in CATEGORICAL_COLS
        },
        "product_category": df.groupby("Номенклатура")["Категория"].first().to_dict(),
        "bakery_city": df.groupby("Пекарня")["Город"].first().to_dict(),
    }

    # Дефолты по паре (пекарня, продукт) — для умного автозаполнения
    num_cols = [c for c in available if c not in CATEGORICAL_COLS]
    pair_defaults = (
        df.groupby(["Пекарня", "Номенклатура"])[num_cols]
        .median()
        .to_dict(orient="index")
    )
    # Ключ: "пекарня|||продукт" -> {фича: значение}
    meta["pair_defaults"] = {
        f"{bak}|||{prod}": {k: round(float(v), 2) for k, v in vals.items()}
        for (bak, prod), vals in pair_defaults.items()
    }

    joblib.dump(meta, META_PATH)
    print(f"Метаданные сохранены: {META_PATH}")
    print(f"\nГотово!")


if __name__ == "__main__":
    main()

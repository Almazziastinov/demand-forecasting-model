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

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import (
    FEATURES, TARGET, CATEGORICAL_COLS, MODEL_PARAMS,
    PROCESSED_DATA_PATH, MODELS_DIR, MODEL_PATH, META_PATH, ARCHIVE_DIR,
)
from src.logger import get_logger
from src.tracking import log_experiment

warnings.filterwarnings("ignore")

logger = get_logger("train", log_file="train.log")

DATA_PATH = PROCESSED_DATA_PATH
OUTPUT_DIR = MODELS_DIR


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

    # --- Сохранение с версионированием ---
    import json
    import shutil
    from datetime import datetime

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Архивная копия: models/archive/YYYYMMDD_HHMMSS/
    version_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = os.path.join(ARCHIVE_DIR, version_tag)
    os.makedirs(version_dir, exist_ok=True)

    archive_model = os.path.join(version_dir, "demand_model.pkl")
    archive_meta = os.path.join(version_dir, "model_meta.pkl")
    archive_metrics = os.path.join(version_dir, "metrics.json")

    joblib.dump(model, archive_model)

    # Копируем как latest
    shutil.copy2(archive_model, MODEL_PATH)
    logger.info(f"Модель сохранена: {MODEL_PATH}")
    logger.info(f"Архив: {version_dir}")

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

    joblib.dump(meta, archive_meta)
    shutil.copy2(archive_meta, META_PATH)
    logger.info(f"Метаданные сохранены: {META_PATH}")

    # Сохраняем метрики в JSON (для CI/мониторинга)
    metrics_dict = {
        "version": version_tag,
        "mae": round(mae, 4), "rmse": round(rmse, 4),
        "r2": round(r2, 4), "wmape": round(wmape, 2),
        "bias": round(bias, 4),
        "train_rows": len(train), "test_rows": len(test),
        "features": len(available), "data_path": DATA_PATH,
    }
    with open(archive_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=2)

    # --- Трекинг ---
    log_experiment(
        name="v6_train",
        metrics=metrics_dict,
        params=MODEL_PARAMS,
        model_path=archive_model,
        data_path=DATA_PATH,
    )
    logger.info("Готово!")


if __name__ == "__main__":
    main()

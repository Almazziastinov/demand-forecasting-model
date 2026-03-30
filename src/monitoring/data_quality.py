"""
Проверки качества данных перед обучением.
Запуск: python -m src.monitoring.data_quality [path_to_csv]
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import FEATURES, TARGET, CATEGORICAL_COLS, PROCESSED_DATA_PATH
from src.logger import get_logger

logger = get_logger("data_quality")


def check_data_quality(df=None, data_path=None):
    """
    Проверяет качество данных. Возвращает список проблем.
    Если проблем нет, возвращает пустой список.
    """
    if df is None:
        path = data_path or PROCESSED_DATA_PATH
        if not os.path.exists(path):
            return [f"Файл не найден: {path}"]
        df = pd.read_csv(path)

    issues = []

    # 1. Проверка наличия target
    if TARGET not in df.columns:
        issues.append(f"Отсутствует целевая переменная: {TARGET}")
        return issues

    # 2. Проверка наличия ключевых признаков
    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        issues.append(f"Отсутствуют признаки ({len(missing_features)}): {missing_features[:5]}")

    # 3. Проверка NaN
    null_counts = df[FEATURES + [TARGET]].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls) > 0:
        issues.append(f"NaN в {len(cols_with_nulls)} колонках: "
                      f"{dict(cols_with_nulls.head(5))}")

    # 4. Отрицательные продажи
    if (df[TARGET] < 0).any():
        n_neg = (df[TARGET] < 0).sum()
        issues.append(f"Отрицательные значения {TARGET}: {n_neg} строк")

    # 5. Проверка минимального размера
    if len(df) < 1000:
        issues.append(f"Мало данных: {len(df)} строк (ожидается >= 1000)")

    # 6. Проверка дат
    if "Дата" in df.columns:
        df["Дата"] = pd.to_datetime(df["Дата"])
        n_days = df["Дата"].nunique()
        if n_days < 14:
            issues.append(f"Мало дней в данных: {n_days} (ожидается >= 14)")

    # 7. Проверка дубликатов
    if {"Дата", "Пекарня", "Номенклатура"}.issubset(df.columns):
        dups = df.duplicated(subset=["Дата", "Пекарня", "Номенклатура"]).sum()
        if dups > 0:
            issues.append(f"Дубликаты (Дата+Пекарня+Номенклатура): {dups}")

    # Логируем результат
    if issues:
        logger.warning(f"Найдено {len(issues)} проблем:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info(f"Данные OK: {len(df):,} строк, {len(df.columns)} колонок")

    return issues


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    issues = check_data_quality(data_path=path)
    if not issues:
        print("\nВсе проверки пройдены!")
    else:
        print(f"\nНайдено проблем: {len(issues)}")
        sys.exit(1)

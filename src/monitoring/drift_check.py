"""
Проверка дрифта предсказаний.
Сравнивает распределение предсказаний из logs/predictions.csv
с историческими данными.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import LOGS_DIR, PROCESSED_DATA_PATH, TARGET
from src.logger import get_logger

logger = get_logger("drift_check")

PREDICTION_LOG = os.path.join(LOGS_DIR, "predictions.csv")


def load_prediction_log():
    """Load prediction log CSV."""
    if not os.path.exists(PREDICTION_LOG):
        logger.warning(f"Лог предсказаний не найден: {PREDICTION_LOG}")
        return pd.DataFrame()
    return pd.read_csv(PREDICTION_LOG)


def check_prediction_drift(recent_days=7):
    """
    Сравнивает средние предсказания за последние N дней
    с историческими средними из обучающих данных.
    """
    log_df = load_prediction_log()
    if log_df.empty:
        logger.info("Нет данных для проверки дрифта")
        return None

    log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])
    cutoff = log_df["timestamp"].max() - pd.Timedelta(days=recent_days)
    recent = log_df[log_df["timestamp"] >= cutoff]

    if len(recent) < 10:
        logger.info(f"Мало данных за последние {recent_days} дней: {len(recent)} строк")
        return None

    # Статистика предсказаний
    pred_mean = recent["prediction"].mean()
    pred_std = recent["prediction"].std()

    # Историческая статистика
    if os.path.exists(PROCESSED_DATA_PATH):
        hist = pd.read_csv(PROCESSED_DATA_PATH)
        hist_mean = hist[TARGET].mean()
        hist_std = hist[TARGET].std()
    else:
        logger.warning("Нет исторических данных для сравнения")
        return None

    drift_ratio = abs(pred_mean - hist_mean) / (hist_std + 1e-8)

    result = {
        "recent_pred_mean": round(pred_mean, 2),
        "recent_pred_std": round(pred_std, 2),
        "historical_mean": round(hist_mean, 2),
        "historical_std": round(hist_std, 2),
        "drift_ratio": round(drift_ratio, 3),
        "drift_detected": drift_ratio > 2.0,
        "n_predictions": len(recent),
    }

    if result["drift_detected"]:
        logger.warning(f"ДРИФТ ОБНАРУЖЕН! ratio={drift_ratio:.3f} "
                       f"(pred_mean={pred_mean:.2f} vs hist_mean={hist_mean:.2f})")
    else:
        logger.info(f"Дрифт не обнаружен (ratio={drift_ratio:.3f})")

    return result


if __name__ == "__main__":
    result = check_prediction_drift()
    if result:
        for k, v in result.items():
            print(f"  {k}: {v}")

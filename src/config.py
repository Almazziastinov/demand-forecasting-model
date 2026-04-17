"""
Единый источник конфигурации проекта.
Все пути, признаки, параметры модели -- здесь.
"""

import os

# --- Пути ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Файлы данных
RAW_DATA_PATH = os.path.join(RAW_DIR, "beigl_data.xlsx")
RAW_DATA_3MONTH = os.path.join(RAW_DIR, "3_month_data.xlsx")
PROCESSED_DATA_PATH = os.path.join(
    PROCESSED_DIR, "preprocessed_data_3month_enriched.csv"
)
MERGED_DATA_PATH = os.path.join(PROCESSED_DIR, "preprocessed_data_merged_enriched.csv")

# Файлы моделей
MODEL_PATH = os.path.join(MODELS_DIR, "demand_model.pkl")
META_PATH = os.path.join(MODELS_DIR, "model_meta.pkl")
ARCHIVE_DIR = os.path.join(MODELS_DIR, "archive")

# --- Целевая переменная ---
TARGET = "Продано"

# --- Признаки ---
FEATURES = [
    # Categorical
    "Пекарня",
    "Номенклатура",
    "Категория",
    "Город",
    # Calendar
    "ДеньНедели",
    "День",
    "IsWeekend",
    "Месяц",
    "НомерНедели",
    # Sales lags
    "sales_lag1",
    "sales_lag2",
    "sales_lag3",
    "sales_lag7",
    "sales_lag14",
    "sales_lag30",
    # Rolling stats
    "sales_roll_mean3",
    "sales_roll_mean7",
    "sales_roll_std7",
    "sales_roll_mean14",
    "sales_roll_mean30",
    # Stock features
    "stock_lag1",
    "stock_sales_ratio",
    "stock_deficit",
    # Holiday / calendar
    "is_holiday",
    "is_pre_holiday",
    "is_post_holiday",
    "is_payday_week",
    "is_month_start",
    "is_month_end",
    # Weather
    "temp_mean",
    "temp_range",
    "precipitation",
    "is_cold",
    "is_bad_weather",
    "weather_cat_code",
]

CATEGORICAL_COLS = ["Пекарня", "Номенклатура", "Категория", "Город", "Месяц"]

# --- Параметры модели (v6-best) ---
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
    "verbose": 5000,
}

# --- Параметры обучения ---
TEST_DAYS = 7  # последние 7 дней = тест (1 полная неделя)

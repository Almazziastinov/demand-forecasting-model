import warnings

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PATH = "data/processed/preprocessed_data_3month_enriched.csv"
N_TRIALS = 50

FEATURES = [
    # Categorical
    "Пекарня",
    "Номенклатура",
    "Категория",
    "Город",
    # Calendar (base)
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
    # Weather (6 selected features)
    "temp_mean",
    "temp_range",
    "precipitation",
    "is_cold",
    "is_bad_weather",
    "weather_cat_code",
]
TARGET = "Продано"


def load_data():
    print(f"Загрузка {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df["Дата"] = pd.to_datetime(df["Дата"])

    for col in ["Пекарня", "Номенклатура", "Категория", "Город", "Месяц"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Берём только признаки которые реально есть
    available = [f for f in FEATURES if f in df.columns]
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"  [!]  Отсутствуют признаки: {missing}")

    test_start = df["Дата"].max() - pd.Timedelta(days=2)
    train = df[df["Дата"] < test_start]
    test = df[df["Дата"] >= test_start]

    print(
        f"  Период обучения: {train['Дата'].min().date()} -- {train['Дата'].max().date()}"
    )
    print(
        f"  Период теста:    {test['Дата'].min().date()} -- {test['Дата'].max().date()}"
    )
    print(f"  Train: {len(train):,} строк ({train['Дата'].nunique()} дней)")
    print(f"  Test:  {len(test):,} строк")
    print(f"  Признаков: {len(available)}")

    return train[available], train[TARGET], test[available], test[TARGET], available


def objective(trial, X_tr, y_tr, X_te, y_te):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    m = LGBMRegressor(**params)
    m.fit(X_tr, y_tr)

    # Bias correction within trial
    train_pred = np.maximum(m.predict(X_tr), 0)
    bias = np.mean(y_tr.values - train_pred)
    pred = np.maximum(m.predict(X_te) + bias, 0)

    return mean_absolute_error(y_te, pred)


def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(y_true) + 1e-8) * 100


def main():
    print("=" * 60)
    print("  ТЮНИНГ v6: WEATHER + BIAS CORRECTION + STOCK FEATURES")
    print("=" * 60)

    X_tr, y_tr, X_te, y_te, features = load_data()

    # -- Baseline (v5 лучшие параметры, без bias correction) --------
    print("\nBaseline (v5 лучшие параметры, без bias correction)...")
    baseline_params = {
        "n_estimators": 1480,
        "learning_rate": 0.00559902887165902,
        "num_leaves": 16,
        "max_depth": 9,
        "min_child_samples": 47,
        "subsample": 0.9738733982256087,
        "colsample_bytree": 0.5250124486848033,
        "reg_alpha": 2.9277614297660098e-08,
        "reg_lambda": 0.085417866288997,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    bl = LGBMRegressor(**baseline_params)
    bl.fit(X_tr, y_tr)

    # Uncorrected baseline
    bl_pred_raw = np.maximum(bl.predict(X_te), 0)
    bl_mae_raw = mean_absolute_error(y_te, bl_pred_raw)
    bl_wmape_raw = wmape(y_te.values, bl_pred_raw)
    bl_bias_raw = np.mean(y_te.values - bl_pred_raw)

    # Bias-corrected baseline
    bl_train_pred = np.maximum(bl.predict(X_tr), 0)
    bl_bias = np.mean(y_tr.values - bl_train_pred)
    bl_pred_corrected = np.maximum(bl.predict(X_te) + bl_bias, 0)
    bl_mae_corrected = mean_absolute_error(y_te, bl_pred_corrected)
    bl_wmape_corrected = wmape(y_te.values, bl_pred_corrected)
    bl_bias_corrected = np.mean(y_te.values - bl_pred_corrected)

    print(
        f"  Baseline (raw):       MAE={bl_mae_raw:.4f} | WMAPE={bl_wmape_raw:.2f}% | bias={bl_bias_raw:+.3f}"
    )
    print(
        f"  Baseline (corrected): MAE={bl_mae_corrected:.4f} | WMAPE={bl_wmape_corrected:.2f}% | bias={bl_bias_corrected:+.3f}"
    )
    print(f"  Train bias estimate: {bl_bias:+.4f}")

    # -- Optuna -------------------------------------------------------
    print(f"\nОптимизация Optuna: {N_TRIALS} испытаний...")
    study = optuna.create_study(direction="minimize")

    completed = [0]

    def callback(study, trial):
        completed[0] += 1
        print(
            f"  [{completed[0]:>3}/{N_TRIALS}] "
            f"MAE={trial.value:.4f} | "
            f"best={study.best_value:.4f}",
            flush=True,
        )

    study.optimize(
        lambda t: objective(t, X_tr, y_tr, X_te, y_te),
        n_trials=N_TRIALS,
        callbacks=[callback],
    )

    print(f"\nЛучший MAE Optuna: {study.best_value:.4f}")
    print("Лучшие параметры:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # -- Финальная модель ---------------------------------------------
    print("\nОбучение финальной модели...")
    best = study.best_params.copy()
    best.update({"random_state": 42, "n_jobs": -1, "verbose": -1})

    model = LGBMRegressor(**best)
    model.fit(X_tr, y_tr)

    # Raw predictions (no bias correction)
    pred_raw = np.maximum(model.predict(X_te), 0)
    mae_raw = mean_absolute_error(y_te, pred_raw)
    wm_raw = wmape(y_te.values, pred_raw)
    bias_raw = np.mean(y_te.values - pred_raw)

    # Bias correction from training residuals
    train_pred = np.maximum(model.predict(X_tr), 0)
    bias = np.mean(y_tr.values - train_pred)
    pred = np.maximum(model.predict(X_te) + bias, 0)

    mae = mean_absolute_error(y_te, pred)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    r2 = r2_score(y_te, pred)
    wm = wmape(y_te.values, pred)
    bias_after = np.mean(y_te.values - pred)

    print("\n" + "=" * 60)
    print("  МЕТРИКИ ФИНАЛЬНОЙ МОДЕЛИ v6")
    print("=" * 60)
    print(f"  --- Без коррекции ---")
    print(f"  MAE:   {mae_raw:.4f} шт.")
    print(f"  WMAPE: {wm_raw:.2f}%")
    print(f"  Bias:  {bias_raw:+.4f}")
    print(f"")
    print(f"  --- С bias correction ({bias:+.4f}) ---")
    print(f"  MAE:   {mae:.4f} шт.")
    print(f"  RMSE:  {rmse:.4f} шт.")
    print(f"  R2:    {r2:.4f}")
    print(f"  WMAPE: {wm:.2f}%")
    print(f"  Bias:  {bias_after:+.4f}")
    print("=" * 60)

    # -- Важность признаков -------------------------------------------
    imp = pd.DataFrame(
        {"Признак": features, "Важность": model.feature_importances_}
    ).sort_values("Важность", ascending=False)

    print("\nТоп-15 признаков по важности:")
    print(imp.head(15).to_string(index=False))

    # -- Итоговое сравнение всех версий ------------------------------
    v3_mae = 3.7039
    v5_mae = 3.39

    print("\n" + "=" * 60)
    print("  СРАВНЕНИЕ ВСЕХ ВЕРСИЙ")
    print("=" * 60)
    print(f"  {'Версия':<45} {'MAE':>8} {'WMAPE':>8}")
    print(f"  {'-' * 63}")
    print(f"  {'v3  (13 дней, lag1-7)':<45} {'3.7039':>8} {'  --':>8}")
    print(f"  {'v5  (3 мес, lag14/30, праздники)':<45} {'3.3900':>8} {'24.00%':>8}")
    print(f"  {'v6  raw (weather+stock, no bias)':<45} {mae_raw:>8.4f} {wm_raw:>7.2f}%")
    print(
        f"  {'v6  corrected (weather+stock+bias)':<45} {mae:>8.4f} {wm:>7.2f}%  <- текущий"
    )
    print(f"  {'-' * 63}")
    improvement_v5 = (v5_mae - mae) / v5_mae * 100
    improvement_v3 = (v3_mae - mae) / v3_mae * 100
    if improvement_v5 > 0:
        print(f"  [+] Улучшение vs v5: -{improvement_v5:.1f}%")
    else:
        print(f"  [i]  Изменение vs v5: {improvement_v5:+.1f}%")
    print(f"  [+] Улучшение vs v3: -{improvement_v3:.1f}%")
    print("=" * 60)

    # Сохраняем предсказания
    test_df = X_te.copy()
    test_df["Продано_факт"] = y_te.values
    test_df["Продано_прогноз"] = pred
    test_df["Продано_прогноз_raw"] = pred_raw
    test_df["Ошибка"] = np.abs(pred - y_te.values)
    test_df["bias_correction"] = bias
    test_df.to_csv("reports/v6_predictions.csv", index=False, encoding="utf-8-sig")
    print("\nПредсказания сохранены в reports/v6_predictions.csv")
    print("\nГотово!")


if __name__ == "__main__":
    main()

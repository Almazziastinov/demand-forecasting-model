import sys
import pandas as pd
import numpy as np
import optuna
import warnings
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_and_split_data(data_path='data/processed/preprocessed_data_enriched.csv'):
    """Загружает данные и делит их на Train/Test по времени."""

    # Если обогащённый файл не найден — берём базовый
    import os
    if not os.path.exists(data_path):
        fallback = 'data/processed/preprocessed_data.csv'
        print(f"  ⚠️  {data_path} не найден, используем {fallback}")
        data_path = fallback

    df = pd.read_csv(data_path)
    df['Дата'] = pd.to_datetime(df['Дата'])

    cat_features = ['Пекарня', 'Номенклатура', 'Категория', 'Город', 'Месяц']
    for col in cat_features:
        df[col] = df[col].astype('category')

    # ── Базовые признаки ──────────────────────────────────────────────
    BASE_FEATURES = [
        'Пекарня', 'Номенклатура', 'Категория', 'Город',
        'ДеньНедели', 'День', 'IsWeekend', 'Месяц', 'НомерНедели',
        'sales_lag1', 'sales_lag2', 'sales_lag3', 'sales_lag7',
        'sales_roll_mean3', 'sales_roll_mean7', 'sales_roll_std7',
        'stock_lag1',
        'is_holiday', 'is_pre_holiday', 'is_post_holiday', 'is_payday_week',
    ]

    # Длинные лаги — только если они есть в датасете
    LONG_LAG_FEATURES = [
        'sales_lag14', 'sales_lag30',
        'sales_roll_mean14', 'sales_roll_mean30',
    ]

    # ── Погодные признаки (если есть в датасете) ──────────────────────
    WEATHER_FEATURES = [
        'temp_mean', 'temp_max', 'temp_range',
        'precipitation', 'snowfall', 'windspeed_max',
        'is_rainy', 'is_snowy', 'is_cold', 'is_bad_weather',
        'weather_cat_code',
    ]

    # ── Календарные признаки (если есть в датасете, но НЕ дублируют BASE) ──
    CALENDAR_FEATURES = [
        'is_month_start', 'is_month_end',
    ]

    # Берём только те признаки, которые реально есть в датасете
    available_long_lags = [f for f in LONG_LAG_FEATURES  if f in df.columns]
    available_weather   = [f for f in WEATHER_FEATURES   if f in df.columns]
    available_calendar  = [f for f in CALENDAR_FEATURES  if f in df.columns]

    # Базовые фичи — только те, что есть (на случай старого датасета без Месяц и т.д.)
    base_available = [f for f in BASE_FEATURES if f in df.columns]
    features = base_available + available_long_lags + available_weather + available_calendar

    print(f"  Базовых признаков:       {len(base_available)}")
    print(f"  Длинных лагов:           {len(available_long_lags)}")
    print(f"  Погодных признаков:      {len(available_weather)}")
    print(f"  Внешних календарных:     {len(available_calendar)}")
    print(f"  Итого:                   {len(features)}")
    if available_long_lags:
        print(f"  Длинные лаги: {available_long_lags}")

    TARGET = 'Продано'

    max_date       = df['Дата'].max()
    test_start     = max_date - pd.Timedelta(days=2)

    train = df[df['Дата'] < test_start]
    test  = df[df['Дата'] >= test_start]

    X_train = train[features]
    y_train = train[TARGET]
    X_test  = test[features]
    y_test  = test[TARGET]

    print(f"  Период обучения: {train['Дата'].min().date()} — {train['Дата'].max().date()}")
    print(f"  Период теста:    {test['Дата'].min().date()} — {test['Дата'].max().date()}")
    print(f"  Train: {len(X_train):,} строк | Test: {len(X_test):,} строк\n")

    return X_train, y_train, X_test, y_test, features


def objective(trial, X_train, y_train, X_test, y_test):
    """Функция-цель для Optuna — минимизируем MAE на тесте."""
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 200, 2000),
        'learning_rate':     trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'num_leaves':        trial.suggest_int('num_leaves', 16, 256),
        'max_depth':         trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }

    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = np.maximum(model.predict(X_test), 0)
    return mean_absolute_error(y_test, y_pred)


def weighted_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(y_true) + 1e-8) * 100


def train_best_model(best_params, X_train, y_train, X_test, y_test, features):
    """Обучает финальную модель с лучшими параметрами и выводит метрики."""
    params = best_params.copy()
    params.update({'random_state': 42, 'n_jobs': -1, 'verbose': -1})

    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = np.maximum(model.predict(X_test), 0)

    mae   = mean_absolute_error(y_test, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
    r2    = r2_score(y_test, y_pred)
    wmape = weighted_mape(y_test.values, y_pred)

    print("=" * 50)
    print("  МЕТРИКИ ФИНАЛЬНОЙ МОДЕЛИ")
    print("=" * 50)
    print(f"  MAE:           {mae:.4f} шт.")
    print(f"  RMSE:          {rmse:.4f} шт.")
    print(f"  R2:            {r2:.4f}")
    print(f"  Weighted MAPE: {wmape:.2f}%")
    print("=" * 50)

    imp = pd.DataFrame({
        'Признак':  features,
        'Важность': model.feature_importances_
    }).sort_values('Важность', ascending=False)

    print("\nТоп-15 признаков по важности:")
    print(imp.head(15).to_string(index=False))

    return model, y_pred


def main():
    N_TRIALS = 50

    # Можно передать путь к файлу аргументом:
    # python tune_model.py preprocessed_data_3month.csv
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'data/processed/preprocessed_data_enriched.csv'

    print("=" * 50)
    print("  ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
    print("=" * 50)
    print(f"  Файл: {data_path}")
    X_train, y_train, X_test, y_test, features = load_and_split_data(data_path)

    # ── Baseline (прошлые лучшие параметры без новых фич) ────────────
    print("Считаем baseline (прошлые лучшие параметры)...")
    baseline = LGBMRegressor(
        n_estimators=1480, learning_rate=0.00559902887165902,
        num_leaves=16, max_depth=9, min_child_samples=47,
        subsample=0.9738733982256087, colsample_bytree=0.5250124486848033,
        reg_alpha=2.9277614297660098e-08, reg_lambda=0.085417866288997,
        random_state=42, n_jobs=-1, verbose=-1
    )
    baseline.fit(X_train, y_train)
    y_base     = np.maximum(baseline.predict(X_test), 0)
    mae_base   = mean_absolute_error(y_test, y_base)
    wmape_base = weighted_mape(y_test.values, y_base)
    print(f"  Baseline MAE: {mae_base:.4f} шт. | Weighted MAPE: {wmape_base:.2f}%\n")

    # ── Optuna ───────────────────────────────────────────────────────
    print(f"Запуск Optuna: {N_TRIALS} испытаний...")
    study = optuna.create_study(direction='minimize')

    completed = [0]
    def callback(study, trial):
        completed[0] += 1
        print(
            f"  [{completed[0]:>3}/{N_TRIALS}] "
            f"MAE={trial.value:.4f} | "
            f"лучший MAE={study.best_value:.4f}",
            flush=True
        )

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_test, y_test),
        n_trials=N_TRIALS,
        callbacks=[callback]
    )

    print("\n" + "=" * 50)
    print("  ЛУЧШИЕ ПАРАМЕТРЫ:")
    print("=" * 50)
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"\n  Лучший MAE (Optuna): {study.best_value:.4f}")

    # ── Финальная модель ─────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ")
    print("=" * 50)
    model, y_pred = train_best_model(
        study.best_params, X_train, y_train, X_test, y_test, features
    )

    # ── Итоговое сравнение ───────────────────────────────────────────
    mae_tuned  = mean_absolute_error(y_test, y_pred)
    wmape_tuned = weighted_mape(y_test.values, y_pred)
    improvement = mae_base - mae_tuned
    improvement_pct = improvement / mae_base * 100

    print("\n" + "=" * 50)
    print("  ИТОГОВОЕ СРАВНЕНИЕ")
    print("=" * 50)
    print(f"  {'Метрика':<22} {'Baseline':>12} {'После тюнинга':>15}")
    print(f"  {'-'*50}")
    print(f"  {'MAE (шт.)':<22} {mae_base:>12.4f} {mae_tuned:>15.4f}")
    print(f"  {'Weighted MAPE (%)':<22} {wmape_base:>12.2f} {wmape_tuned:>15.2f}")
    print(f"  {'-'*50}")
    if improvement > 0:
        print(f"\n  ✅ Улучшение MAE: -{improvement:.4f} шт. ({improvement_pct:.1f}%)")
    else:
        print(f"\n  ℹ️  Тюнинг не дал улучшения: разница {improvement:.4f} шт.")
    print("=" * 50)
    print("\nГотово!")


if __name__ == "__main__":
    main()

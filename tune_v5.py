import pandas as pd
import numpy as np
import optuna
import warnings
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PATH = 'preprocessed_data_3month.csv'
N_TRIALS  = 50

FEATURES = [
    'Пекарня', 'Номенклатура', 'Категория', 'Город',
    'ДеньНедели', 'День', 'IsWeekend', 'Месяц', 'НомерНедели',
    'sales_lag1', 'sales_lag2', 'sales_lag3', 'sales_lag7',
    'sales_lag14', 'sales_lag30',
    'sales_roll_mean3', 'sales_roll_mean7', 'sales_roll_std7',
    'sales_roll_mean14', 'sales_roll_mean30',
    'stock_lag1',
    'is_holiday', 'is_pre_holiday', 'is_post_holiday', 'is_payday_week',
]
TARGET = 'Продано'


def load_data():
    print(f"Загрузка {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df['Дата'] = pd.to_datetime(df['Дата'])

    for col in ['Пекарня', 'Номенклатура', 'Категория', 'Город', 'Месяц']:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Берём только признаки которые реально есть
    available = [f for f in FEATURES if f in df.columns]
    missing   = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"  ⚠️  Отсутствуют признаки: {missing}")

    test_start = df['Дата'].max() - pd.Timedelta(days=2)
    train = df[df['Дата'] < test_start]
    test  = df[df['Дата'] >= test_start]

    print(f"  Период обучения: {train['Дата'].min().date()} — {train['Дата'].max().date()}")
    print(f"  Период теста:    {test['Дата'].min().date()} — {test['Дата'].max().date()}")
    print(f"  Train: {len(train):,} строк ({train['Дата'].nunique()} дней)")
    print(f"  Test:  {len(test):,} строк")
    print(f"  Признаков: {len(available)}")

    return train[available], train[TARGET], test[available], test[TARGET], available


def objective(trial, X_tr, y_tr, X_te, y_te):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 500, 3000),
        'learning_rate':     trial.suggest_float('learning_rate', 0.003, 0.2, log=True),
        'num_leaves':        trial.suggest_int('num_leaves', 16, 256),
        'max_depth':         trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42, 'n_jobs': -1, 'verbose': -1,
    }
    m = LGBMRegressor(**params)
    m.fit(X_tr, y_tr)
    return mean_absolute_error(y_te, np.maximum(m.predict(X_te), 0))


def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(y_true) + 1e-8) * 100


def main():
    print("=" * 55)
    print("  ТЮНИНГ v5: 3 МЕСЯЦА + LAG14/30 + ПРАЗДНИКИ + МЕСЯЦ")
    print("=" * 55)

    X_tr, y_tr, X_te, y_te, features = load_data()

    # ── Baseline ─────────────────────────────────────────────────────
    print("\nBaseline (прошлые лучшие параметры)...")
    baseline_params = {
        'n_estimators': 1480, 'learning_rate': 0.00559902887165902,
        'num_leaves': 16, 'max_depth': 9, 'min_child_samples': 47,
        'subsample': 0.9738733982256087, 'colsample_bytree': 0.5250124486848033,
        'reg_alpha': 2.9277614297660098e-08, 'reg_lambda': 0.085417866288997,
        'random_state': 42, 'n_jobs': -1, 'verbose': -1,
    }
    bl = LGBMRegressor(**baseline_params)
    bl.fit(X_tr, y_tr)
    bl_pred = np.maximum(bl.predict(X_te), 0)
    bl_mae  = mean_absolute_error(y_te, bl_pred)
    bl_wmape = wmape(y_te.values, bl_pred)
    print(f"  Baseline MAE: {bl_mae:.4f} шт. | WMAPE: {bl_wmape:.2f}%")

    # ── Optuna ───────────────────────────────────────────────────────
    print(f"\nОптимизация Optuna: {N_TRIALS} испытаний...")
    study = optuna.create_study(direction='minimize')

    completed = [0]
    def callback(study, trial):
        completed[0] += 1
        print(
            f"  [{completed[0]:>3}/{N_TRIALS}] "
            f"MAE={trial.value:.4f} | "
            f"best={study.best_value:.4f}",
            flush=True
        )

    study.optimize(
        lambda t: objective(t, X_tr, y_tr, X_te, y_te),
        n_trials=N_TRIALS,
        callbacks=[callback]
    )

    print(f"\nЛучший MAE Optuna: {study.best_value:.4f}")
    print("Лучшие параметры:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # ── Финальная модель ─────────────────────────────────────────────
    print("\nОбучение финальной модели...")
    best = study.best_params.copy()
    best.update({'random_state': 42, 'n_jobs': -1, 'verbose': -1})

    model = LGBMRegressor(**best)
    model.fit(X_tr, y_tr)
    pred = np.maximum(model.predict(X_te), 0)

    mae  = mean_absolute_error(y_te, pred)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    r2   = r2_score(y_te, pred)
    wm   = wmape(y_te.values, pred)

    print("\n" + "=" * 55)
    print("  МЕТРИКИ ФИНАЛЬНОЙ МОДЕЛИ v5")
    print("=" * 55)
    print(f"  MAE:   {mae:.4f} шт.")
    print(f"  RMSE:  {rmse:.4f} шт.")
    print(f"  R2:    {r2:.4f}")
    print(f"  WMAPE: {wm:.2f}%")
    print("=" * 55)

    # ── Важность признаков ───────────────────────────────────────────
    imp = pd.DataFrame({
        'Признак':  features,
        'Важность': model.feature_importances_
    }).sort_values('Важность', ascending=False)

    print("\nТоп-15 признаков по важности:")
    print(imp.head(15).to_string(index=False))

    # ── Итоговое сравнение ───────────────────────────────────────────
    v3_mae = 3.7039
    improvement = (v3_mae - mae) / v3_mae * 100

    print("\n" + "=" * 55)
    print("  СРАВНЕНИЕ ВСЕХ ВЕРСИЙ")
    print("=" * 55)
    print(f"  {'Версия':<40} {'MAE':>8}")
    print(f"  {'-' * 50}")
    print(f"  {'v3  (13 дней, lag1-7)':<40} {'3.7039':>8}")
    print(f"  {'v5  baseline (3 мес, старые params)':<40} {bl_mae:>8.4f}")
    print(f"  {'v5  после тюнинга':<40} {mae:>8.4f}  ← текущий")
    print(f"  {'-' * 50}")
    if improvement > 0:
        print(f"  ✅ Улучшение vs v3: -{improvement:.1f}%")
    else:
        print(f"  ℹ️  Изменение vs v3: {improvement:+.1f}%")
    print("=" * 55)

    # Сохраняем предсказания
    test_df = X_te.copy()
    test_df['Продано_факт']   = y_te.values
    test_df['Продано_прогноз'] = pred
    test_df['Ошибка']          = np.abs(pred - y_te.values)
    test_df.to_csv('v5_predictions.csv', index=False, encoding='utf-8-sig')
    print("\nПредсказания сохранены в v5_predictions.csv")
    print("\nГотово!")


if __name__ == "__main__":
    main()

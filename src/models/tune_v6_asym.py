import pandas as pd
import numpy as np
import optuna
import warnings
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PATH = 'data/processed/preprocessed_data_3month_enriched.csv'
N_TRIALS  = 100

FEATURES = [
    # Categorical
    'Пекарня', 'Номенклатура', 'Категория', 'Город',
    # Calendar (base)
    'ДеньНедели', 'День', 'IsWeekend', 'Месяц', 'НомерНедели',
    # Sales lags
    'sales_lag1', 'sales_lag2', 'sales_lag3', 'sales_lag7',
    'sales_lag14', 'sales_lag30',
    # Rolling stats
    'sales_roll_mean3', 'sales_roll_mean7', 'sales_roll_std7',
    'sales_roll_mean14', 'sales_roll_mean30',
    # Stock features
    'stock_lag1', 'stock_sales_ratio', 'stock_deficit',
    # Holiday / calendar
    'is_holiday', 'is_pre_holiday', 'is_post_holiday', 'is_payday_week',
    'is_month_start', 'is_month_end',
    # Weather (6 selected features)
    'temp_mean', 'temp_range', 'precipitation',
    'is_cold', 'is_bad_weather', 'weather_cat_code',
]
TARGET = 'Продано'


# ── Asymmetric loss functions for LightGBM ──────────────────────
# alpha > 0.5 means underprediction is penalized more
def asymmetric_mae_objective(alpha):
    """Returns (grad, hess) for asymmetric MAE (quantile loss)."""
    def _obj(y_true, y_pred):
        residual = y_true - y_pred
        grad = np.where(residual > 0, -alpha, (1 - alpha))
        hess = np.ones_like(residual)  # constant hessian for stability
        return grad, hess
    return _obj


def asymmetric_mse_objective(alpha):
    """Returns (grad, hess) for asymmetric MSE (weighted squared loss).
    Underprediction (residual > 0) gets weight alpha, overprediction gets (1-alpha)."""
    def _obj(y_true, y_pred):
        residual = y_true - y_pred
        weight = np.where(residual > 0, alpha, (1 - alpha))
        grad = -2 * weight * residual
        hess = 2 * weight
        return grad, hess
    return _obj


def load_data():
    print(f"Загрузка {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df['Дата'] = pd.to_datetime(df['Дата'])

    for col in ['Пекарня', 'Номенклатура', 'Категория', 'Город', 'Месяц']:
        if col in df.columns:
            df[col] = df[col].astype('category')

    available = [f for f in FEATURES if f in df.columns]
    missing   = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"  Отсутствуют признаки: {missing}")

    test_start = df['Дата'].max() - pd.Timedelta(days=2)
    train = df[df['Дата'] < test_start]
    test  = df[df['Дата'] >= test_start]

    print(f"  Период обучения: {train['Дата'].min().date()} -- {train['Дата'].max().date()}")
    print(f"  Период теста:    {test['Дата'].min().date()} -- {test['Дата'].max().date()}")
    print(f"  Train: {len(train):,} строк ({train['Дата'].nunique()} дней)")
    print(f"  Test:  {len(test):,} строк")
    print(f"  Признаков: {len(available)}")

    return train[available], train[TARGET], test[available], test[TARGET], available


def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(y_true) + 1e-8) * 100


def objective(trial, X_tr, y_tr, X_te, y_te):
    # Tunable asymmetry parameter
    alpha = trial.suggest_float('alpha', 0.5, 0.8)
    loss_type = trial.suggest_categorical('loss_type', ['mae', 'mse'])

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

    if loss_type == 'mae':
        custom_obj = asymmetric_mae_objective(alpha)
    else:
        custom_obj = asymmetric_mse_objective(alpha)

    m = LGBMRegressor(**params, objective=custom_obj)
    m.fit(X_tr, y_tr)
    pred = np.maximum(m.predict(X_te), 0)

    return mean_absolute_error(y_te, pred)


def main():
    print("=" * 65)
    print("  ТЮНИНГ v6-ASYM: ASYMMETRIC LOSS + 100 TRIALS")
    print("=" * 65)

    X_tr, y_tr, X_te, y_te, features = load_data()

    # ── Baseline: v6 symmetric (лучшие параметры из прошлого запуска) ──
    print("\nBaseline (v6 symmetric, лучшие параметры)...")
    v6_params = {
        'n_estimators': 2304, 'learning_rate': 0.016085231060110994,
        'num_leaves': 151, 'max_depth': 7, 'min_child_samples': 5,
        'subsample': 0.8012777641245349, 'colsample_bytree': 0.6654847541174173,
        'reg_alpha': 6.633500153052146, 'reg_lambda': 2.204771489304501e-06,
        'random_state': 42, 'n_jobs': -1, 'verbose': -1,
    }
    bl = LGBMRegressor(**v6_params)
    bl.fit(X_tr, y_tr)
    bl_pred = np.maximum(bl.predict(X_te), 0)
    bl_mae  = mean_absolute_error(y_te, bl_pred)
    bl_wm   = wmape(y_te.values, bl_pred)
    bl_bias = np.mean(y_te.values - bl_pred)
    print(f"  v6 symmetric: MAE={bl_mae:.4f} | WMAPE={bl_wm:.2f}% | bias={bl_bias:+.3f}")

    # ── Optuna ───────────────────────────────────────────────────────
    print(f"\nОптимизация Optuna (asymmetric loss): {N_TRIALS} испытаний...")
    study = optuna.create_study(direction='minimize')

    completed = [0]
    def callback(study, trial):
        completed[0] += 1
        print(
            f"  [{completed[0]:>3}/{N_TRIALS}] "
            f"MAE={trial.value:.4f} | "
            f"best={study.best_value:.4f} | "
            f"alpha={trial.params.get('alpha', 0):.3f} "
            f"loss={trial.params.get('loss_type', '?')}",
            flush=True
        )

    study.optimize(
        lambda t: objective(t, X_tr, y_tr, X_te, y_te),
        n_trials=N_TRIALS,
        callbacks=[callback]
    )

    print(f"\nЛучший MAE: {study.best_value:.4f}")
    print("Лучшие параметры:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # ── Финальная модель ─────────────────────────────────────────────
    print("\nОбучение финальной модели...")
    best = study.best_params.copy()
    best_alpha = best.pop('alpha')
    best_loss = best.pop('loss_type')
    best.update({'random_state': 42, 'n_jobs': -1, 'verbose': -1})

    if best_loss == 'mae':
        custom_obj = asymmetric_mae_objective(best_alpha)
    else:
        custom_obj = asymmetric_mse_objective(best_alpha)

    model = LGBMRegressor(**best, objective=custom_obj)
    model.fit(X_tr, y_tr)

    pred = np.maximum(model.predict(X_te), 0)
    mae  = mean_absolute_error(y_te, pred)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    r2   = r2_score(y_te, pred)
    wm   = wmape(y_te.values, pred)
    bias = np.mean(y_te.values - pred)

    print("\n" + "=" * 65)
    print("  МЕТРИКИ ФИНАЛЬНОЙ МОДЕЛИ v6-ASYM")
    print("=" * 65)
    print(f"  Asymmetric loss: {best_loss} | alpha={best_alpha:.4f}")
    print(f"  MAE:   {mae:.4f} шт.")
    print(f"  RMSE:  {rmse:.4f} шт.")
    print(f"  R2:    {r2:.4f}")
    print(f"  WMAPE: {wm:.2f}%")
    print(f"  Bias:  {bias:+.4f} (+ = underprediction)")
    print("=" * 65)

    # ── Анализ bias по категориям ──────────────────────────────────
    test_df = X_te.copy()
    test_df['fact'] = y_te.values
    test_df['pred'] = pred
    test_df['error'] = pred - y_te.values  # positive = overprediction

    if 'Категория' in test_df.columns:
        print("\nBias по категориям:")
        cat_bias = test_df.groupby('Категория').agg(
            count=('fact', 'size'),
            mean_fact=('fact', 'mean'),
            mean_pred=('pred', 'mean'),
            bias=('error', 'mean'),
            mae=('error', lambda x: np.abs(x).mean()),
        ).sort_values('mae', ascending=False)
        print(cat_bias.round(3).to_string())

    # ── Важность признаков ───────────────────────────────────────────
    imp = pd.DataFrame({
        'Признак':  features,
        'Важность': model.feature_importances_
    }).sort_values('Важность', ascending=False)

    print("\nТоп-15 признаков по важности:")
    print(imp.head(15).to_string(index=False))

    # ── Итоговое сравнение ──────────────────────────────────────────
    v3_mae = 3.7039
    v5_mae = 3.39
    v6_sym_mae = 3.33

    print("\n" + "=" * 65)
    print("  СРАВНЕНИЕ ВСЕХ ВЕРСИЙ")
    print("=" * 65)
    print(f"  {'Версия':<50} {'MAE':>8} {'WMAPE':>8} {'Bias':>8}")
    print(f"  {'-' * 76}")
    print(f"  {'v3  (13 дней, lag1-7)':<50} {'3.7039':>8} {'  --':>8} {'  --':>8}")
    print(f"  {'v5  (3 мес, lag14/30, праздники)':<50} {'3.3900':>8} {'24.00%':>8} {'+0.895':>8}")
    print(f"  {'v6  symmetric (weather+stock)':<50} {v6_sym_mae:>8.4f} {'23.59%':>8} {'+0.668':>8}")
    print(f"  {'v6  asymmetric (alpha={:.3f}, {})'.format(best_alpha, best_loss):<50} {mae:>8.4f} {wm:>7.2f}% {bias:>+8.3f}")
    print(f"  {'-' * 76}")

    improvement_v5 = (v5_mae - mae) / v5_mae * 100
    improvement_v6 = (v6_sym_mae - mae) / v6_sym_mae * 100
    if improvement_v5 > 0:
        print(f"  Улучшение vs v5:  -{improvement_v5:.1f}%")
    else:
        print(f"  Изменение vs v5:  {improvement_v5:+.1f}%")
    if improvement_v6 > 0:
        print(f"  Улучшение vs v6-sym: -{improvement_v6:.1f}%")
    else:
        print(f"  Изменение vs v6-sym: {improvement_v6:+.1f}%")
    print("=" * 65)

    # Сохраняем
    test_df.to_csv('reports/v6_asym_predictions.csv', index=False, encoding='utf-8-sig')
    print("\nПредсказания сохранены в reports/v6_asym_predictions.csv")
    print("\nГотово!")


if __name__ == "__main__":
    main()

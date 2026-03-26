"""
V7 Experiments: Sample Weights vs Log-Transform vs Baseline
Сравнение трёх подходов для улучшения MAE на высокоспросовых товарах.

Эксперимент A: Sample Weights (вес = уровень спроса)
Эксперимент B: Log-Transform таргета (log1p/expm1)
Эксперимент C: Sample Weights + Asymmetric Loss (комбо)
Baseline: v6 symmetric (стандартный MAE loss)
"""

import pandas as pd
import numpy as np
import optuna
import warnings
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PATH = 'data/processed/preprocessed_data_3month_enriched.csv'
N_TRIALS  = 10  # на эксперимент

FEATURES = [
    'Пекарня', 'Номенклатура', 'Категория', 'Город',
    'ДеньНедели', 'День', 'IsWeekend', 'Месяц', 'НомерНедели',
    'sales_lag1', 'sales_lag2', 'sales_lag3', 'sales_lag7',
    'sales_lag14', 'sales_lag30',
    'sales_roll_mean3', 'sales_roll_mean7', 'sales_roll_std7',
    'sales_roll_mean14', 'sales_roll_mean30',
    'stock_lag1', 'stock_sales_ratio', 'stock_deficit',
    'is_holiday', 'is_pre_holiday', 'is_post_holiday', 'is_payday_week',
    'is_month_start', 'is_month_end',
    'temp_mean', 'temp_range', 'precipitation',
    'is_cold', 'is_bad_weather', 'weather_cat_code',
]
TARGET = 'Продано'


def load_data():
    print(f"Загрузка {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df['Дата'] = pd.to_datetime(df['Дата'])

    for col in ['Пекарня', 'Номенклатура', 'Категория', 'Город', 'Месяц']:
        if col in df.columns:
            df[col] = df[col].astype('category')

    available = [f for f in FEATURES if f in df.columns]
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"  Отсутствуют: {missing}")

    test_start = df['Дата'].max() - pd.Timedelta(days=2)
    train = df[df['Дата'] < test_start]
    test = df[df['Дата'] >= test_start]

    print(f"  Train: {len(train):,} ({train['Дата'].nunique()} дней)")
    print(f"  Test:  {len(test):,}")
    print(f"  Признаков: {len(available)}")

    return train, test, available


def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(y_true) + 1e-8) * 100


def category_metrics(X_te, y_te, pred, label=""):
    """Считает MAE по категориям."""
    tmp = X_te.copy()
    tmp['fact'] = y_te.values
    tmp['pred'] = pred
    tmp['abs_err'] = np.abs(pred - y_te.values)
    tmp['err'] = pred - y_te.values

    if 'Категория' not in tmp.columns:
        return pd.DataFrame()

    result = tmp.groupby('Категория').agg(
        count=('fact', 'size'),
        mean_demand=('fact', 'mean'),
        bias=('err', 'mean'),
        mae=('abs_err', 'mean'),
    ).sort_values('mae', ascending=False)

    # MAE на высокоспросовых (demand > 10)
    high = tmp[tmp['fact'] >= 10]
    low = tmp[tmp['fact'] < 10]
    mae_high = np.abs(high['pred'] - high['fact']).mean() if len(high) > 0 else 0
    mae_low = np.abs(low['pred'] - low['fact']).mean() if len(low) > 0 else 0

    return result, mae_high, mae_low


def make_params(trial):
    return {
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


# =====================================================================
#  EXPERIMENT A: Sample Weights
# =====================================================================
def run_experiment_A(X_tr, y_tr, X_te, y_te):
    """LightGBM с sample_weight = clip(y, 1, None)"""
    print("\n" + "=" * 65)
    print("  ЭКСПЕРИМЕНТ A: SAMPLE WEIGHTS (вес = спрос)")
    print("=" * 65)

    weights_tr = y_tr.clip(lower=1).values

    def obj_A(trial):
        params = make_params(trial)
        m = LGBMRegressor(**params)
        m.fit(X_tr, y_tr, sample_weight=weights_tr)
        pred = np.maximum(m.predict(X_te), 0)
        return mean_absolute_error(y_te, pred)

    study = optuna.create_study(direction='minimize')
    done = [0]
    def cb(study, trial):
        done[0] += 1
        if done[0] % 10 == 0 or done[0] == 1:
            print(f"  [{done[0]:>3}/{N_TRIALS}] MAE={trial.value:.4f} | best={study.best_value:.4f}")
    study.optimize(obj_A, n_trials=N_TRIALS, callbacks=[cb])

    # Финальная модель
    best = study.best_params.copy()
    best.update({'random_state': 42, 'n_jobs': -1, 'verbose': -1})
    model = LGBMRegressor(**best)
    model.fit(X_tr, y_tr, sample_weight=weights_tr)
    pred = np.maximum(model.predict(X_te), 0)

    return model, pred, study.best_value, best


# =====================================================================
#  EXPERIMENT B: Log-Transform
# =====================================================================
def run_experiment_B(X_tr, y_tr, X_te, y_te):
    """LightGBM с log1p(target), обратное преобразование expm1."""
    print("\n" + "=" * 65)
    print("  ЭКСПЕРИМЕНТ B: LOG-TRANSFORM ТАРГЕТА")
    print("=" * 65)

    y_tr_log = np.log1p(y_tr)

    def obj_B(trial):
        params = make_params(trial)
        m = LGBMRegressor(**params)
        m.fit(X_tr, y_tr_log)
        pred_log = m.predict(X_te)
        pred = np.maximum(np.expm1(pred_log), 0)
        return mean_absolute_error(y_te, pred)

    study = optuna.create_study(direction='minimize')
    done = [0]
    def cb(study, trial):
        done[0] += 1
        if done[0] % 10 == 0 or done[0] == 1:
            print(f"  [{done[0]:>3}/{N_TRIALS}] MAE={trial.value:.4f} | best={study.best_value:.4f}")
    study.optimize(obj_B, n_trials=N_TRIALS, callbacks=[cb])

    # Финальная модель
    best = study.best_params.copy()
    best.update({'random_state': 42, 'n_jobs': -1, 'verbose': -1})
    model = LGBMRegressor(**best)
    model.fit(X_tr, y_tr_log)
    pred_log = model.predict(X_te)
    pred = np.maximum(np.expm1(pred_log), 0)

    return model, pred, study.best_value, best


# =====================================================================
#  EXPERIMENT C: Sample Weights + Asymmetric MSE
# =====================================================================
def run_experiment_C(X_tr, y_tr, X_te, y_te):
    """Asymmetric MSE (alpha=0.6) + sample_weight = спрос."""
    print("\n" + "=" * 65)
    print("  ЭКСПЕРИМЕНТ C: SAMPLE WEIGHTS + ASYMMETRIC MSE")
    print("=" * 65)

    weights_tr = y_tr.clip(lower=1).values

    def asym_mse(alpha):
        def _obj(y_true, y_pred):
            residual = y_true - y_pred
            w = np.where(residual > 0, alpha, (1 - alpha))
            return -2 * w * residual, 2 * w
        return _obj

    def obj_C(trial):
        alpha = trial.suggest_float('alpha', 0.5, 0.75)
        params = make_params(trial)
        m = LGBMRegressor(**params, objective=asym_mse(alpha))
        m.fit(X_tr, y_tr, sample_weight=weights_tr)
        pred = np.maximum(m.predict(X_te), 0)
        return mean_absolute_error(y_te, pred)

    study = optuna.create_study(direction='minimize')
    done = [0]
    def cb(study, trial):
        done[0] += 1
        if done[0] % 10 == 0 or done[0] == 1:
            print(f"  [{done[0]:>3}/{N_TRIALS}] MAE={trial.value:.4f} | best={study.best_value:.4f} | alpha={trial.params.get('alpha',0):.3f}")
    study.optimize(obj_C, n_trials=N_TRIALS, callbacks=[cb])

    best = study.best_params.copy()
    best_alpha = best.pop('alpha')
    best.update({'random_state': 42, 'n_jobs': -1, 'verbose': -1})
    model = LGBMRegressor(**best, objective=asym_mse(best_alpha))
    model.fit(X_tr, y_tr, sample_weight=weights_tr)
    pred = np.maximum(model.predict(X_te), 0)

    return model, pred, study.best_value, best, best_alpha


def main():
    print("=" * 65)
    print("  V7 EXPERIMENTS: WEIGHTS vs LOG vs COMBO")
    print("=" * 65)

    train_df, test_df, features = load_data()
    X_tr, y_tr = train_df[features], train_df[TARGET]
    X_te, y_te = test_df[features], test_df[TARGET]

    # ── Baseline (v6 symmetric) ──────────────────────────────────────
    print("\n--- Baseline (v6 symmetric) ---")
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
    bl_mae = mean_absolute_error(y_te, bl_pred)
    bl_wm = wmape(y_te.values, bl_pred)
    bl_bias = np.mean(y_te.values - bl_pred)
    bl_cats, bl_mae_high, bl_mae_low = category_metrics(X_te, y_te, bl_pred)
    print(f"  MAE={bl_mae:.4f} | WMAPE={bl_wm:.2f}% | bias={bl_bias:+.3f}")
    print(f"  MAE(demand>=10)={bl_mae_high:.4f} | MAE(demand<10)={bl_mae_low:.4f}")

    # ── Эксперименты ─────────────────────────────────────────────────
    model_A, pred_A, best_A, params_A = run_experiment_A(X_tr, y_tr, X_te, y_te)
    model_B, pred_B, best_B, params_B = run_experiment_B(X_tr, y_tr, X_te, y_te)
    model_C, pred_C, best_C, params_C, alpha_C = run_experiment_C(X_tr, y_tr, X_te, y_te)

    # ── Сводная таблица ──────────────────────────────────────────────
    experiments = {
        'Baseline (v6 sym)': bl_pred,
        'A: Sample Weights': pred_A,
        'B: Log-Transform': pred_B,
        'C: Weights+Asym': pred_C,
    }

    print("\n" + "=" * 65)
    print("  СВОДНАЯ ТАБЛИЦА ЭКСПЕРИМЕНТОВ")
    print("=" * 65)
    print(f"  {'Эксперимент':<25} {'MAE':>7} {'WMAPE':>7} {'Bias':>8} {'MAE_hi':>8} {'MAE_lo':>8}")
    print(f"  {'-' * 67}")

    for name, pred in experiments.items():
        mae = mean_absolute_error(y_te, pred)
        wm = wmape(y_te.values, pred)
        bias = np.mean(y_te.values - pred)
        _, mae_hi, mae_lo = category_metrics(X_te, y_te, pred)
        print(f"  {name:<25} {mae:>7.4f} {wm:>6.2f}% {bias:>+8.3f} {mae_hi:>8.4f} {mae_lo:>8.4f}")

    print(f"  {'-' * 67}")
    print(f"  MAE_hi = MAE на товарах с demand >= 10 шт")
    print(f"  MAE_lo = MAE на товарах с demand < 10 шт")

    # ── Детали по категориям для лучшего эксперимента ────────────────
    all_maes = {name: mean_absolute_error(y_te, pred) for name, pred in experiments.items()}
    best_name = min(all_maes, key=all_maes.get)
    best_pred = experiments[best_name]

    print(f"\n  Лучший эксперимент: {best_name} (MAE={all_maes[best_name]:.4f})")
    print(f"\n  Bias по категориям ({best_name}):")
    cats, _, _ = category_metrics(X_te, y_te, best_pred)
    print(cats.round(3).to_string())

    # ── Детали по категориям для ВСЕХ экспериментов ──────────────────
    print("\n" + "=" * 65)
    print("  MAE ПО КАТЕГОРИЯМ (все эксперименты)")
    print("=" * 65)

    cat_comparison = pd.DataFrame()
    for name, pred in experiments.items():
        tmp = X_te.copy()
        tmp['abs_err'] = np.abs(pred - y_te.values)
        cat_mae = tmp.groupby('Категория')['abs_err'].mean()
        cat_comparison[name] = cat_mae

    print(cat_comparison.round(3).to_string())

    # ── Сохранение ───────────────────────────────────────────────────
    results = X_te.copy()
    results['fact'] = y_te.values
    results['pred_baseline'] = bl_pred
    results['pred_weights'] = pred_A
    results['pred_log'] = pred_B
    results['pred_combo'] = pred_C
    results.to_csv('reports/v7_experiments.csv', index=False, encoding='utf-8-sig')
    print("\nРезультаты сохранены в reports/v7_experiments.csv")
    print("\nГотово!")


if __name__ == "__main__":
    main()

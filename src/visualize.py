import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Настройки графиков
# ─────────────────────────────────────────────
plt.rcParams['figure.dpi'] = 130
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

COLORS = {
    'actual':    '#2196F3',
    'predicted': '#FF5722',
    'error':     '#9C27B0',
    'bar':       '#4CAF50',
    'bar2':      '#FF9800',
    'bar3':      '#F44336',
    'baseline':  '#9E9E9E',
}

DAY_NAMES = {0: 'Пн', 1: 'Вт', 2: 'Ср', 3: 'Чт', 4: 'Пт', 5: 'Сб', 6: 'Вс'}

# ─────────────────────────────────────────────
# Загрузка данных
# ─────────────────────────────────────────────
print("Загрузка данных...")
df = pd.read_csv('data/processed/preprocessed_data.csv')
df['Дата'] = pd.to_datetime(df['Дата'])

cat_features = ['Пекарня', 'Номенклатура', 'Категория', 'Город']
for col in cat_features:
    df[col] = df[col].astype('category')

FEATURES = [
    'Пекарня', 'Номенклатура', 'Категория', 'Город',
    'ДеньНедели', 'День', 'IsWeekend',
    'sales_lag1', 'sales_lag2', 'sales_lag3', 'sales_lag7',
    'sales_roll_mean3', 'sales_roll_mean7', 'sales_roll_std7',
    'stock_lag1',
]
TARGET = 'Продано'

max_date = df['Дата'].max()
test_start_date = max_date - pd.Timedelta(days=2)

train = df[df['Дата'] < test_start_date].copy()
test  = df[df['Дата'] >= test_start_date].copy()

X_train, y_train = train[FEATURES], train[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

print(f"Обучение: {train['Дата'].min().date()} — {train['Дата'].max().date()}")
print(f"Тест:     {test['Дата'].min().date()} — {test['Дата'].max().date()}")

# ─────────────────────────────────────────────
# Обучение — три версии для сравнения
# ─────────────────────────────────────────────

# 1. Старый baseline (lag 1-3 only, стандартные параметры)
OLD_FEATURES = [
    'Пекарня', 'Номенклатура', 'Категория', 'Город',
    'ДеньНедели', 'День',
    'sales_lag1', 'sales_lag2', 'sales_lag3', 'sales_roll_mean3',
]
# Все старые признаки есть в новом датасете — просто берём подмножество
X_train_old = train[OLD_FEATURES]
X_test_old  = test[OLD_FEATURES]

print("\nОбучение старой модели (v1)...")
model_old = LGBMRegressor(
    n_estimators=1000, learning_rate=0.05, num_leaves=31,
    random_state=42, n_jobs=-1, verbose=-1
)
model_old.fit(X_train_old, y_train)
y_pred_old = np.maximum(model_old.predict(X_test_old), 0)

# 2. Новый baseline (все фичи, стандартные параметры)
print("Обучение baseline новых фич (v2)...")
model_base = LGBMRegressor(
    n_estimators=1000, learning_rate=0.05, num_leaves=31,
    random_state=42, n_jobs=-1, verbose=-1
)
model_base.fit(X_train, y_train)
y_pred_base = np.maximum(model_base.predict(X_test), 0)

# 3. Финальная модель (лучшие параметры из Optuna)
BEST_PARAMS = {
    'n_estimators':      1480,
    'learning_rate':     0.00559902887165902,
    'num_leaves':        16,
    'max_depth':         9,
    'min_child_samples': 47,
    'subsample':         0.9738733982256087,
    'colsample_bytree':  0.5250124486848033,
    'reg_alpha':         2.9277614297660098e-08,
    'reg_lambda':        0.085417866288997,
    'random_state':      42,
    'n_jobs':            -1,
    'verbose':           -1,
}
print("Обучение финальной модели с лучшими параметрами (v3)...")
model_best = LGBMRegressor(**BEST_PARAMS)
model_best.fit(X_train, y_train)
y_pred_best = np.maximum(model_best.predict(X_test), 0)

# ─────────────────────────────────────────────
# Метрики
# ─────────────────────────────────────────────
def metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    wmape = np.sum(np.abs(y_true - y_pred)) / (np.sum(y_true) + 1e-8) * 100
    return mae, rmse, r2, wmape

mae_old,  rmse_old,  r2_old,  wmape_old  = metrics(y_test, y_pred_old)
mae_base, rmse_base, r2_base, wmape_base = metrics(y_test, y_pred_base)
mae_best, rmse_best, r2_best, wmape_best = metrics(y_test, y_pred_best)

print(f"\nv1 (старые фичи):       MAE={mae_old:.4f} | R2={r2_old:.4f} | WMAPE={wmape_old:.2f}%")
print(f"v2 (новые фичи, base): MAE={mae_base:.4f} | R2={r2_base:.4f} | WMAPE={wmape_base:.2f}%")
print(f"v3 (новые фичи + tune): MAE={mae_best:.4f} | R2={r2_best:.4f} | WMAPE={wmape_best:.2f}%")

# Добавляем предсказания в test
test = test.copy()
test['Predicted'] = y_pred_best
test['Error']     = np.abs(test['Predicted'] - test[TARGET])

print("\nСтроим графики...")

# ═══════════════════════════════════════════════════════
# ГРАФИК 1: Факт vs Прогноз по дням (суммарно по сети)
# ═══════════════════════════════════════════════════════
daily_actual = test.groupby('Дата')[TARGET].sum()
daily_pred   = test.groupby('Дата')['Predicted'].sum()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(daily_actual.index, daily_actual.values, 'o-',
        color=COLORS['actual'], linewidth=2.5, markersize=9, label='Факт')
ax.plot(daily_pred.index, daily_pred.values, 's--',
        color=COLORS['predicted'], linewidth=2.5, markersize=9, label='Прогноз')
ax.fill_between(daily_actual.index, daily_actual.values, daily_pred.values,
                alpha=0.12, color=COLORS['error'], label='Расхождение')
for date in daily_actual.index:
    ax.annotate(f"{daily_actual[date]:,.0f}",
                (date, daily_actual[date]), textcoords="offset points",
                xytext=(0, 12), ha='center', fontsize=10, color=COLORS['actual'], fontweight='bold')
    ax.annotate(f"{daily_pred[date]:,.0f}",
                (date, daily_pred[date]), textcoords="offset points",
                xytext=(0, -20), ha='center', fontsize=10, color=COLORS['predicted'], fontweight='bold')
ax.set_title('Суммарные продажи по всей сети: Факт vs Прогноз (тестовый период)')
ax.set_xlabel('Дата')
ax.set_ylabel('Суммарные продажи (шт.)')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('reports/plots/plot1_timeseries.png')
plt.close()
print("  ✓ plot1_timeseries.png")

# ═══════════════════════════════════════════════════════
# ГРАФИК 2: Scatter plot — Факт vs Прогноз
# ═══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, y_pred_best, alpha=0.12, s=12, color=COLORS['actual'])
max_val = max(y_test.max(), y_pred_best.max())
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Идеальный прогноз')
ax.set_xlabel('Реальные продажи (шт.)')
ax.set_ylabel('Предсказанные продажи (шт.)')
ax.set_title(f'Scatter: Факт vs Прогноз\nMAE={mae_best:.2f} | R²={r2_best:.4f}')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.4)
ax.text(0.05, 0.91,
        f'MAE  = {mae_best:.2f} шт.\nRMSE = {rmse_best:.2f} шт.\nR²   = {r2_best:.4f}',
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray'))
plt.tight_layout()
plt.savefig('reports/plots/plot2_scatter.png')
plt.close()
print("  ✓ plot2_scatter.png")

# ═══════════════════════════════════════════════════════
# ГРАФИК 3: Распределение ошибок
# ═══════════════════════════════════════════════════════
errors = y_pred_best - y_test.values
pct_over  = (errors > 0).mean() * 100
pct_under = (errors < 0).mean() * 100

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(errors, bins=70, color=COLORS['error'], alpha=0.75, edgecolor='white')
ax.axvline(0, color='black', linewidth=2, linestyle='-', label='Нулевая ошибка')
ax.axvline(errors.mean(), color='red', linewidth=2, linestyle='--',
           label=f'Среднее смещение = {errors.mean():.2f}')
ax.set_title('Распределение ошибок прогноза\n(+ = перепрогноз / − = недопрогноз)')
ax.set_xlabel('Ошибка = Прогноз − Факт (шт.)')
ax.set_ylabel('Кол-во позиций')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.4)
ax.text(0.70, 0.88,
        f'Перепрогноз:  {pct_over:.1f}%\nНедопрогноз: {pct_under:.1f}%',
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray'))
plt.tight_layout()
plt.savefig('reports/plots/plot3_error_dist.png')
plt.close()
print("  ✓ plot3_error_dist.png")

# ═══════════════════════════════════════════════════════
# ГРАФИК 4: Важность признаков
# ═══════════════════════════════════════════════════════
importance_df = pd.DataFrame({
    'Признак':  FEATURES,
    'Важность': model_best.feature_importances_
}).sort_values('Важность', ascending=True)

fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.barh(importance_df['Признак'], importance_df['Важность'],
               color=COLORS['bar'], edgecolor='white', height=0.65)
for bar, val in zip(bars, importance_df['Важность']):
    ax.text(val + importance_df['Важность'].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{val:,.0f}', va='center', fontsize=9)
ax.set_title('Важность признаков модели LightGBM (финальная версия)')
ax.set_xlabel('Важность (gain)')
ax.grid(True, axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('reports/plots/plot4_feature_importance.png')
plt.close()
print("  ✓ plot4_feature_importance.png")

# ═══════════════════════════════════════════════════════
# ГРАФИК 5: MAE по категориям товаров
# ═══════════════════════════════════════════════════════
mae_by_cat = (
    test.groupby('Категория', observed=True)
    .apply(lambda g: mean_absolute_error(g[TARGET], g['Predicted']), include_groups=False)
    .sort_values(ascending=True)
    .reset_index()
)
mae_by_cat.columns = ['Категория', 'MAE']

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(mae_by_cat['Категория'].astype(str), mae_by_cat['MAE'],
               color=COLORS['bar2'], edgecolor='white', height=0.55)
for bar, val in zip(bars, mae_by_cat['MAE']):
    ax.text(val + 0.03, bar.get_y() + bar.get_height() / 2,
            f'{val:.2f} шт.', va='center', fontsize=10)
ax.axvline(mae_best, color='red', linewidth=1.8, linestyle='--',
           label=f'Общий MAE = {mae_best:.2f}')
ax.set_title('MAE по категориям товаров')
ax.set_xlabel('MAE (шт.)')
ax.legend()
ax.grid(True, axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('reports/plots/plot5_mae_by_category.png')
plt.close()
print("  ✓ plot5_mae_by_category.png")

# ═══════════════════════════════════════════════════════
# ГРАФИК 6: MAE по дням недели
# ═══════════════════════════════════════════════════════
mae_by_dow = (
    test.groupby('ДеньНедели', observed=True)
    .apply(lambda g: mean_absolute_error(g[TARGET], g['Predicted']), include_groups=False)
    .reset_index()
)
mae_by_dow.columns = ['ДеньНедели', 'MAE']
mae_by_dow['ДеньНедели_int'] = mae_by_dow['ДеньНедели'].astype(int)
mae_by_dow['День'] = mae_by_dow['ДеньНедели_int'].map(DAY_NAMES)
mae_by_dow = mae_by_dow.sort_values('ДеньНедели_int')

bar_colors = [COLORS['bar2'] if d >= 5 else COLORS['bar']
              for d in mae_by_dow['ДеньНедели_int']]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(mae_by_dow['День'], mae_by_dow['MAE'],
              color=bar_colors, edgecolor='white', width=0.6)
for bar, val in zip(bars, mae_by_dow['MAE']):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.05,
            f'{val:.2f}', ha='center', va='bottom', fontsize=10)
ax.axhline(mae_best, color='red', linewidth=1.8, linestyle='--',
           label=f'Общий MAE = {mae_best:.2f}')
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLORS['bar'],  label='Будни'),
    Patch(facecolor=COLORS['bar2'], label='Выходные'),
    plt.Line2D([0], [0], color='red', linestyle='--', label=f'Общий MAE = {mae_best:.2f}')
]
ax.legend(handles=legend_elements)
ax.set_title('MAE по дням недели')
ax.set_xlabel('День недели')
ax.set_ylabel('MAE (шт.)')
ax.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('reports/plots/plot6_mae_by_weekday.png')
plt.close()
print("  ✓ plot6_mae_by_weekday.png")

# ═══════════════════════════════════════════════════════
# ГРАФИК 7 (НОВЫЙ): Сравнение всех версий модели
# ═══════════════════════════════════════════════════════
versions   = ['v1\n(lag1-3)', 'v2\n(+lag7, std7,\nstock_lag1)', 'v3\n(+ тюнинг)']
mae_vals   = [mae_old,   mae_base,   mae_best]
rmse_vals  = [rmse_old,  rmse_base,  rmse_best]
r2_vals    = [r2_old,    r2_base,    r2_best]
wmape_vals = [wmape_old, wmape_base, wmape_best]
bar_colors_v = [COLORS['baseline'], COLORS['bar2'], COLORS['bar']]

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle('Сравнение всех версий модели', fontsize=15, fontweight='bold')

# MAE
ax = axes[0]
bars = ax.bar(versions, mae_vals, color=bar_colors_v, edgecolor='white', width=0.5)
for bar, val in zip(bars, mae_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
            f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_title('MAE (шт.) ↓ меньше = лучше')
ax.set_ylabel('MAE (шт.)')
ax.set_ylim(0, max(mae_vals) * 1.2)
ax.grid(True, axis='y', linestyle='--', alpha=0.4)

# R²
ax = axes[1]
bars = ax.bar(versions, r2_vals, color=bar_colors_v, edgecolor='white', width=0.5)
for bar, val in zip(bars, r2_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
            f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_title('R² ↑ больше = лучше')
ax.set_ylabel('R²')
ax.set_ylim(min(r2_vals) * 0.97, 1.0)
ax.grid(True, axis='y', linestyle='--', alpha=0.4)

# Weighted MAPE
ax = axes[2]
bars = ax.bar(versions, wmape_vals, color=bar_colors_v, edgecolor='white', width=0.5)
for bar, val in zip(bars, wmape_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_title('Weighted MAPE (%) ↓ меньше = лучше')
ax.set_ylabel('Weighted MAPE (%)')
ax.set_ylim(0, max(wmape_vals) * 1.2)
ax.grid(True, axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('reports/plots/plot7_model_comparison.png')
plt.close()
print("  ✓ plot7_model_comparison.png")

# ═══════════════════════════════════════════════════════
# ГРАФИК 8 (НОВЫЙ): Топ-15 пекарен по суммарной ошибке
# ═══════════════════════════════════════════════════════
store_mae = (
    test.groupby('Пекарня', observed=True)
    .apply(lambda g: pd.Series({
        'MAE':         mean_absolute_error(g[TARGET], g['Predicted']),
        'Продано_сум': g[TARGET].sum(),
        'Ошибка_сум':  g['Error'].sum(),
    }), include_groups=False)
    .reset_index()
)

# Топ-15 пекарен с наибольшей суммарной ошибкой
top15_worst = store_mae.nlargest(15, 'Ошибка_сум')
# Топ-15 пекарен с наименьшей MAE (лучшие)
top15_best  = store_mae.nsmallest(15, 'MAE')

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Анализ качества прогноза по пекарням', fontsize=14, fontweight='bold')

# Левый: худшие по суммарной ошибке
ax = axes[0]
labels_w = [str(s)[:30] for s in top15_worst['Пекарня']]
bars = ax.barh(labels_w, top15_worst['Ошибка_сум'],
               color=COLORS['bar3'], edgecolor='white', height=0.7)
for bar, val in zip(bars, top15_worst['Ошибка_сум']):
    ax.text(val + top15_worst['Ошибка_сум'].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.0f} шт.', va='center', fontsize=8)
ax.set_title('Топ-15 пекарен\nс наибольшей суммарной ошибкой')
ax.set_xlabel('Суммарная |ошибка| (шт.)')
ax.grid(True, axis='x', linestyle='--', alpha=0.4)

# Правый: лучшие по MAE
ax = axes[1]
labels_b = [str(s)[:30] for s in top15_best['Пекарня']]
bars = ax.barh(labels_b, top15_best['MAE'],
               color=COLORS['bar'], edgecolor='white', height=0.7)
for bar, val in zip(bars, top15_best['MAE']):
    ax.text(val + top15_best['MAE'].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.2f}', va='center', fontsize=8)
ax.axvline(mae_best, color='red', linewidth=1.5, linestyle='--',
           label=f'Общий MAE = {mae_best:.2f}')
ax.set_title('Топ-15 пекарен\nс наименьшей MAE (лучшие)')
ax.set_xlabel('MAE (шт.)')
ax.legend(fontsize=9)
ax.grid(True, axis='x', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('reports/plots/plot8_stores_analysis.png')
plt.close()
print("  ✓ plot8_stores_analysis.png")

print("\nВсе 8 графиков готовы!")
print("Файлы: plot1_timeseries.png, plot2_scatter.png, plot3_error_dist.png,")
print("       plot4_feature_importance.png, plot5_mae_by_category.png,")
print("       plot6_mae_by_weekday.png, plot7_model_comparison.png, plot8_stores_analysis.png")

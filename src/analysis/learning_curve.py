import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 130
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13

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

# ─────────────────────────────────────────────
# Фиксируем тестовый период (последние 3 дня)
# ─────────────────────────────────────────────
max_date   = df['Дата'].max()
test_start = max_date - pd.Timedelta(days=2)
test       = df[df['Дата'] >= test_start].copy()
X_test     = test[FEATURES]
y_test     = test[TARGET]

all_train   = df[df['Дата'] < test_start].copy()
train_dates = sorted(all_train['Дата'].unique())
n_days_max  = len(train_dates)

print(f"Тест:  {test['Дата'].min().date()} — {test['Дата'].max().date()} ({test['Дата'].nunique()} дня)")
print(f"Максимум доступных дней для обучения: {n_days_max}")

# ─────────────────────────────────────────────
# Симуляция кривой обучения
# Берём последние N дней (скользящее окно с конца → к тесту)
# ─────────────────────────────────────────────
# Минимум: 3 дня (меньше нет смысла), шаг: 1 день
results = []

print("\nЗапуск симуляции кривой обучения...")
print(f"{'Дней':>6} {'Train строк':>12} {'MAE':>8} {'R2':>8} {'WMAPE':>8}")
print("-" * 48)

for n_days in range(3, n_days_max + 1):
    # Берём последние n_days из тренировочного периода
    window_dates = train_dates[-n_days:]
    train_window = all_train[all_train['Дата'].isin(window_dates)]

    X_train = train_window[FEATURES]
    y_train = train_window[TARGET]

    model = LGBMRegressor(**BEST_PARAMS)
    model.fit(X_train, y_train)

    y_pred = np.maximum(model.predict(X_test), 0)

    mae   = mean_absolute_error(y_test, y_pred)
    r2    = r2_score(y_test, y_pred)
    wmape = np.sum(np.abs(y_test.values - y_pred)) / (np.sum(y_test.values) + 1e-8) * 100

    results.append({
        'n_days':      n_days,
        'train_rows':  len(train_window),
        'mae':         mae,
        'r2':          r2,
        'wmape':       wmape,
        'date_start':  window_dates[0],
    })

    print(f"{n_days:>6} {len(train_window):>12,} {mae:>8.4f} {r2:>8.4f} {wmape:>7.2f}%")

results_df = pd.DataFrame(results)

# ─────────────────────────────────────────────
# Аналитика
# ─────────────────────────────────────────────
best_row = results_df.loc[results_df['mae'].idxmin()]
last_row = results_df.iloc[-1]   # максимум доступных дней

print("\n" + "=" * 55)
print("  ИТОГИ СИМУЛЯЦИИ")
print("=" * 55)
print(f"\n  Лучший результат:")
print(f"    Дней обучения: {int(best_row['n_days'])}")
print(f"    MAE:           {best_row['mae']:.4f} шт.")
print(f"    R2:            {best_row['r2']:.4f}")
print(f"    WMAPE:         {best_row['wmape']:.2f}%")

print(f"\n  При максимуме доступных данных ({int(last_row['n_days'])} дней):")
print(f"    MAE:           {last_row['mae']:.4f} шт.")
print(f"    R2:            {last_row['r2']:.4f}")
print(f"    WMAPE:         {last_row['wmape']:.2f}%")

# Тренд: помогают ли дополнительные дни?
first_row = results_df.iloc[0]
improvement_total = first_row['mae'] - last_row['mae']
print(f"\n  Изменение MAE: {first_row['mae']:.4f} → {last_row['mae']:.4f}")
sign = '-' if improvement_total > 0 else '+'
print(f"  ({sign}{abs(improvement_total):.4f} шт., "
      f"{'улучшение' if improvement_total > 0 else 'ухудшение'} "
      f"{abs(improvement_total)/first_row['mae']*100:.1f}%)")

print(f"\n  Вывод о вероятности улучшения с новыми данными:")
# Оцениваем наклон кривой на последних 5 точках
if len(results_df) >= 5:
    tail = results_df.tail(5)
    slope = np.polyfit(tail['n_days'], tail['mae'], 1)[0]
    if slope < -0.01:
        print(f"  ✅ Кривая обучения всё ещё падает (slope={slope:.4f})")
        print(f"     Добавление данных ОЧЕНЬ ВЕРОЯТНО улучшит модель.")
        probability = "ВЫСОКАЯ (>85%)"
    elif slope < 0:
        print(f"  🟡 Кривая обучения замедляется (slope={slope:.4f})")
        print(f"     Добавление данных ВЕРОЯТНО улучшит модель.")
        probability = "СРЕДНЯЯ (60-85%)"
    else:
        print(f"  🟠 Кривая обучения выровнялась (slope={slope:.4f})")
        print(f"     Польза от новых данных неочевидна — нужны другие признаки.")
        probability = "НИЗКАЯ (<60%)"
    print(f"\n  Оценочная вероятность улучшения с 3 мес. данными: {probability}")

# ─────────────────────────────────────────────
# Экстраполяция: что ожидать от 30/60/90 дней
# ─────────────────────────────────────────────
print(f"\n  Экстраполяция (если добавить больше данных):")
print(f"  {'Горизонт':<20} {'Оценка MAE':>12} {'Оценка R2':>12}")
print(f"  {'─'*46}")

# Фитируем логарифмическую кривую: MAE = a + b * log(n_days)
# Это стандартная форма кривой обучения
from scipy.optimize import curve_fit

def log_curve(x, a, b):
    return a + b * np.log(x)

try:
    popt, _ = curve_fit(log_curve, results_df['n_days'], results_df['mae'], maxfev=5000)
    a, b = popt

    # Для R2 — тоже логарифмическая, но возрастающая
    def log_curve_r2(x, a, b):
        return a - b * np.log(x)

    popt_r2, _ = curve_fit(log_curve_r2, results_df['n_days'],
                           -results_df['r2'] + 1, maxfev=5000)

    horizons = {
        'Текущий (13 дн.)':  13,
        '1 месяц (30 дн.)':  30,
        '2 месяца (60 дн.)': 60,
        '3 месяца (90 дн.)': 90,
        '6 месяцев (180 дн.)': 180,
    }

    for label, n in horizons.items():
        est_mae = max(log_curve(n, a, b), 0.5)  # не ниже 0.5 шт.
        est_r2  = min(1 - log_curve_r2(n, *popt_r2), 0.99)
        marker = ' ← текущий' if n == 13 else ''
        print(f"  {label:<20} {est_mae:>12.2f} шт. {est_r2:>11.4f}{marker}")

except Exception as e:
    print(f"  Не удалось построить экстраполяцию: {e}")

# ─────────────────────────────────────────────
# Визуализация
# ─────────────────────────────────────────────
print("\nСтроим графики...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    'Кривая обучения: как MAE зависит от объёма обучающих данных',
    fontsize=14, fontweight='bold'
)

# ── MAE vs кол-во дней ───────────────────────────────────────────────
ax = axes[0]
ax.plot(results_df['n_days'], results_df['mae'],
        'o-', color='#2196F3', linewidth=2, markersize=5, label='Реальный MAE')

# Логарифмическая аппроксимация
try:
    x_smooth = np.linspace(results_df['n_days'].min(), 90, 200)
    y_smooth = log_curve(x_smooth, a, b)
    y_smooth = np.maximum(y_smooth, 0.5)
    ax.plot(x_smooth, y_smooth, '--', color='#FF9800', linewidth=2,
            label='Тренд (экстраполяция)')

    # Отметки для 30/60/90 дней
    for n, clr, lbl in [(30, '#4CAF50', '30 дн.'), (60, '#FF5722', '60 дн.'), (90, '#9C27B0', '90 дн.')]:
        est = max(log_curve(n, a, b), 0.5)
        ax.axvline(n, color=clr, linestyle=':', alpha=0.6)
        ax.plot(n, est, 's', color=clr, markersize=9, zorder=5)
        ax.annotate(f'{lbl}\n≈{est:.2f}', (n, est),
                    textcoords='offset points', xytext=(6, 4),
                    fontsize=8, color=clr, fontweight='bold')
except Exception:
    pass

ax.axvline(13, color='red', linewidth=1.5, linestyle='--', alpha=0.7, label='Текущий (13 дн.)')
ax.set_xlabel('Дней обучения')
ax.set_ylabel('MAE (шт.)')
ax.set_title('MAE vs кол-во дней обучения')
ax.legend(fontsize=9)
ax.grid(True, linestyle='--', alpha=0.4)

# ── R2 vs кол-во дней ───────────────────────────────────────────────
ax = axes[1]
ax.plot(results_df['n_days'], results_df['r2'],
        'o-', color='#4CAF50', linewidth=2, markersize=5, label='Реальный R²')
ax.axvline(13, color='red', linewidth=1.5, linestyle='--', alpha=0.7, label='Текущий (13 дн.)')
ax.axhline(results_df['r2'].max(), color='gray', linewidth=1, linestyle=':',
           label=f'Макс. R²={results_df["r2"].max():.4f}')

try:
    y_r2_smooth = np.array([min(1 - log_curve_r2(x, *popt_r2), 0.99) for x in x_smooth])
    ax.plot(x_smooth, y_r2_smooth, '--', color='#FF9800', linewidth=2, label='Тренд')
except Exception:
    pass

ax.set_xlabel('Дней обучения')
ax.set_ylabel('R²')
ax.set_title('R² vs кол-во дней обучения')
ax.legend(fontsize=9)
ax.grid(True, linestyle='--', alpha=0.4)

# ── WMAPE vs кол-во дней ─────────────────────────────────────────────
ax = axes[2]
ax.plot(results_df['n_days'], results_df['wmape'],
        'o-', color='#FF5722', linewidth=2, markersize=5, label='Weighted MAPE')
ax.axvline(13, color='red', linewidth=1.5, linestyle='--', alpha=0.7, label='Текущий (13 дн.)')

# Минимум
min_wmape_row = results_df.loc[results_df['wmape'].idxmin()]
ax.plot(min_wmape_row['n_days'], min_wmape_row['wmape'],
        '*', color='gold', markersize=14, zorder=5,
        label=f'Лучший: {min_wmape_row["wmape"]:.2f}% ({int(min_wmape_row["n_days"])} дн.)')

ax.set_xlabel('Дней обучения')
ax.set_ylabel('Weighted MAPE (%)')
ax.set_title('Weighted MAPE vs кол-во дней обучения')
ax.legend(fontsize=9)
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('reports/plots/plot13_learning_curve.png')
plt.close()
print("  ✓ plot13_learning_curve.png")

print("\nГотово!")

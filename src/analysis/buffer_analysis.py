import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 130
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13

# ─────────────────────────────────────────────
# 1. Загрузка данных и обучение модели
# ─────────────────────────────────────────────
print("Загрузка данных и обучение модели...")
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

max_date   = df['Дата'].max()
test_start = max_date - pd.Timedelta(days=2)

train = df[df['Дата'] < test_start].copy()
test  = df[df['Дата'] >= test_start].copy()

X_train, y_train = train[FEATURES], train[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

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

model = LGBMRegressor(**BEST_PARAMS)
model.fit(X_train, y_train)
y_pred_base = np.maximum(model.predict(X_test), 0)

print(f"Базовый MAE (без буфера): {mean_absolute_error(y_test, y_pred_base):.4f} шт.")

# ─────────────────────────────────────────────
# 2. Эталон — реальная система
# ─────────────────────────────────────────────
REAL_MEAN_STOCK = df['Остаток'].clip(lower=0).mean()   # ~2.0 шт. по всему датасету
test_real_stock = test['Остаток'].clip(lower=0).mean() # среднее в тестовом периоде

print(f"\nРеальное среднее Остаток (весь датасет): {REAL_MEAN_STOCK:.4f} шт.")
print(f"Реальное среднее Остаток (тест):         {test_real_stock:.4f} шт.")

# ─────────────────────────────────────────────
# 3. Перебор буферов
# ─────────────────────────────────────────────
# Тип буфера 1: мультипликативный — Производим = Predicted * (1 + pct)
# Тип буфера 2: аддитивный        — Производим = Predicted + N штук

n_pos = len(test)

# --- Мультипликативный буфер: 0% → 100% ---
pct_buffers = np.arange(0.0, 1.01, 0.01)   # 0% до 100%
results_pct = []

for buf in pct_buffers:
    y_plan = np.round(y_pred_base * (1 + buf)).astype(int)

    # Остаток с моделью = max(0, план - реальные продажи)
    model_stock   = np.maximum(y_plan - y_test.values, 0)
    # Дефицит с моделью = max(0, реальные продажи - план)
    model_deficit = np.maximum(y_test.values - y_plan, 0)

    n_under = (y_plan < y_test.values).sum()  # позиций с недобором
    n_over  = (y_plan > y_test.values).sum()  # позиций с остатком
    n_exact = (y_plan == y_test.values).sum()

    results_pct.append({
        'buffer_pct':       buf * 100,
        'mean_stock':       model_stock.mean(),
        'total_stock':      model_stock.sum(),
        'mean_deficit':     model_deficit.mean(),
        'total_deficit':    model_deficit.sum(),
        'pct_underpredict': n_under / n_pos * 100,
        'pct_overpredict':  n_over  / n_pos * 100,
        'pct_exact':        n_exact / n_pos * 100,
        'mae':              mean_absolute_error(y_test, y_plan),
    })

results_pct_df = pd.DataFrame(results_pct)

# --- Аддитивный буфер: 0 → 10 штук ---
add_buffers = np.arange(0, 11, 1)
results_add = []

for buf in add_buffers:
    y_plan = np.round(y_pred_base + buf).astype(int)

    model_stock   = np.maximum(y_plan - y_test.values, 0)
    model_deficit = np.maximum(y_test.values - y_plan, 0)

    n_under = (y_plan < y_test.values).sum()
    n_over  = (y_plan > y_test.values).sum()
    n_exact = (y_plan == y_test.values).sum()

    results_add.append({
        'buffer_add':       buf,
        'mean_stock':       model_stock.mean(),
        'total_stock':      model_stock.sum(),
        'mean_deficit':     model_deficit.mean(),
        'total_deficit':    model_deficit.sum(),
        'pct_underpredict': n_under / n_pos * 100,
        'pct_overpredict':  n_over  / n_pos * 100,
        'pct_exact':        n_exact / n_pos * 100,
        'mae':              mean_absolute_error(y_test, y_plan),
    })

results_add_df = pd.DataFrame(results_add)

# ─────────────────────────────────────────────
# 4. Находим ключевые точки
# ─────────────────────────────────────────────

# Точка где среднее Остаток модели = реальное среднее по всему датасету (~2.0)
target_stock = REAL_MEAN_STOCK

# Мультипликативный: найти ближайший буфер к target
pct_match = results_pct_df.iloc[
    (results_pct_df['mean_stock'] - target_stock).abs().idxmin()
]

# Аддитивный: найти ближайший буфер к target
add_match = results_add_df.iloc[
    (results_add_df['mean_stock'] - target_stock).abs().idxmin()
]

# ─────────────────────────────────────────────
# 5. Вывод таблицы
# ─────────────────────────────────────────────
print("\n" + "=" * 75)
print("  МУЛЬТИПЛИКАТИВНЫЙ БУФЕР: Производим = Predicted × (1 + буфер%)")
print("=" * 75)
print(f"  {'Буфер':>8} {'Ср. остаток':>12} {'Ср. дефицит':>13} "
      f"{'Недобор%':>10} {'Остаток%':>10}")
print(f"  {'─' * 58}")

key_pcts = [0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
for _, row in results_pct_df[results_pct_df['buffer_pct'].isin(key_pcts)].iterrows():
    marker = ' ← ~реальн.' if abs(row['mean_stock'] - target_stock) < 0.15 else ''
    print(f"  {row['buffer_pct']:>7.0f}% "
          f"{row['mean_stock']:>12.2f} "
          f"{row['mean_deficit']:>13.2f} "
          f"{row['pct_underpredict']:>9.1f}% "
          f"{row['pct_overpredict']:>9.1f}%"
          f"{marker}")

print(f"\n  ★  Буфер где ср. остаток ≈ реальному ({target_stock:.2f} шт.):")
print(f"     Мультипликативный: +{pct_match['buffer_pct']:.0f}%")
print(f"     Ср. остаток:       {pct_match['mean_stock']:.2f} шт.")
print(f"     Ср. дефицит:       {pct_match['mean_deficit']:.2f} шт.")
print(f"     Позиций с недобором: {pct_match['pct_underpredict']:.1f}%")

print("\n" + "=" * 75)
print("  АДДИТИВНЫЙ БУФЕР: Производим = Predicted + N штук")
print("=" * 75)
print(f"  {'Буфер':>8} {'Ср. остаток':>12} {'Ср. дефицит':>13} "
      f"{'Недобор%':>10} {'Остаток%':>10}")
print(f"  {'─' * 58}")

for _, row in results_add_df.iterrows():
    marker = ' ← ~реальн.' if abs(row['mean_stock'] - target_stock) < 0.15 else ''
    print(f"  {row['buffer_add']:>7.0f} шт. "
          f"{row['mean_stock']:>12.2f} "
          f"{row['mean_deficit']:>13.2f} "
          f"{row['pct_underpredict']:>9.1f}% "
          f"{row['pct_overpredict']:>9.1f}%"
          f"{marker}")

print(f"\n  ★  Буфер где ср. остаток ≈ реальному ({target_stock:.2f} шт.):")
print(f"     Аддитивный:        +{add_match['buffer_add']:.0f} шт.")
print(f"     Ср. остаток:       {add_match['mean_stock']:.2f} шт.")
print(f"     Ср. дефицит:       {add_match['mean_deficit']:.2f} шт.")
print(f"     Позиций с недобором: {add_match['pct_underpredict']:.1f}%")

# ─────────────────────────────────────────────
# 6. Визуализация
# ─────────────────────────────────────────────
print("\nСтроим графики...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    'Trade-off: буфер к прогнозу vs остатки и дефицит\n'
    '(чем больше буфер — тем меньше недобора, но больше списаний)',
    fontsize=13, fontweight='bold'
)

# ── Левый верхний: остаток vs дефицит (мультипликативный) ────────────
ax = axes[0, 0]
ax.plot(results_pct_df['buffer_pct'], results_pct_df['mean_stock'],
        'o-', color='#F44336', linewidth=2, markersize=3, label='Ср. остаток (списания)')
ax.plot(results_pct_df['buffer_pct'], results_pct_df['mean_deficit'],
        's-', color='#2196F3', linewidth=2, markersize=3, label='Ср. дефицит (недобор)')
ax.axhline(REAL_MEAN_STOCK, color='green', linewidth=2, linestyle='--',
           label=f'Реальный ср. остаток ({REAL_MEAN_STOCK:.2f} шт.)')
ax.axvline(pct_match['buffer_pct'], color='orange', linewidth=1.5, linestyle=':',
           label=f'Точка паритета: +{pct_match["buffer_pct"]:.0f}%')
ax.set_xlabel('Буфер (%)')
ax.set_ylabel('Штук на позицию')
ax.set_title('Мультипликативный буфер\nПроизводим = Predicted × (1 + буфер%)')
ax.legend(fontsize=9)
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_xlim(0, 60)

# ── Правый верхний: остаток vs дефицит (аддитивный) ──────────────────
ax = axes[0, 1]
ax.plot(results_add_df['buffer_add'], results_add_df['mean_stock'],
        'o-', color='#F44336', linewidth=2, markersize=6, label='Ср. остаток (списания)')
ax.plot(results_add_df['buffer_add'], results_add_df['mean_deficit'],
        's-', color='#2196F3', linewidth=2, markersize=6, label='Ср. дефицит (недобор)')
ax.axhline(REAL_MEAN_STOCK, color='green', linewidth=2, linestyle='--',
           label=f'Реальный ср. остаток ({REAL_MEAN_STOCK:.2f} шт.)')
ax.axvline(add_match['buffer_add'], color='orange', linewidth=1.5, linestyle=':',
           label=f'Точка паритета: +{add_match["buffer_add"]:.0f} шт.')
# Подписываем каждую точку
for _, row in results_add_df.iterrows():
    ax.annotate(f"{row['mean_stock']:.1f}",
                (row['buffer_add'], row['mean_stock']),
                textcoords='offset points', xytext=(0, 7),
                ha='center', fontsize=8, color='#F44336')
ax.set_xlabel('Буфер (шт.)')
ax.set_ylabel('Штук на позицию')
ax.set_title('Аддитивный буфер\nПроизводим = Predicted + N штук')
ax.legend(fontsize=9)
ax.grid(True, linestyle='--', alpha=0.4)

# ── Левый нижний: % позиций с недобором (мультипликативный) ──────────
ax = axes[1, 0]
ax.plot(results_pct_df['buffer_pct'], results_pct_df['pct_underpredict'],
        'o-', color='#2196F3', linewidth=2, markersize=3, label='% позиций с недобором')
ax.plot(results_pct_df['buffer_pct'], results_pct_df['pct_overpredict'],
        's-', color='#F44336', linewidth=2, markersize=3, label='% позиций с остатком')
ax.axvline(pct_match['buffer_pct'], color='orange', linewidth=1.5, linestyle=':',
           label=f'Точка паритета: +{pct_match["buffer_pct"]:.0f}%')
# Текущее значение (буфер = 0)
zero_row = results_pct_df[results_pct_df['buffer_pct'] == 0].iloc[0]
ax.annotate(f"Без буфера:\n{zero_row['pct_underpredict']:.1f}% недобор",
            (0, zero_row['pct_underpredict']),
            textcoords='offset points', xytext=(15, -20),
            fontsize=9, color='#2196F3',
            arrowprops=dict(arrowstyle='->', color='#2196F3', lw=1.5))
ax.set_xlabel('Буфер (%)')
ax.set_ylabel('% позиций')
ax.set_title('% позиций с недобором vs остатком\n(мультипликативный буфер)')
ax.legend(fontsize=9)
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_xlim(0, 60)

# ── Правый нижний: аддитивный — % позиций ────────────────────────────
ax = axes[1, 1]
x = results_add_df['buffer_add']
ax.bar(x - 0.2, results_add_df['pct_underpredict'], width=0.35,
       color='#2196F3', label='% недобора', edgecolor='white')
ax.bar(x + 0.2, results_add_df['pct_overpredict'],  width=0.35,
       color='#F44336', label='% остатка',  edgecolor='white')
# Подписи
for _, row in results_add_df.iterrows():
    ax.text(row['buffer_add'] - 0.2, row['pct_underpredict'] + 0.3,
            f"{row['pct_underpredict']:.0f}%", ha='center', fontsize=8, color='#2196F3')
    ax.text(row['buffer_add'] + 0.2, row['pct_overpredict'] + 0.3,
            f"{row['pct_overpredict']:.0f}%", ha='center', fontsize=8, color='#F44336')
ax.axvline(add_match['buffer_add'], color='orange', linewidth=1.5, linestyle=':',
           label=f'Точка паритета: +{add_match["buffer_add"]:.0f} шт.')
ax.set_xlabel('Буфер (шт.)')
ax.set_ylabel('% позиций')
ax.set_title('% позиций с недобором vs остатком\n(аддитивный буфер)')
ax.legend(fontsize=9)
ax.grid(True, axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('reports/plots/plot14_buffer_tradeoff.png')
plt.close()
print("  ✓ plot14_buffer_tradeoff.png")

# ─────────────────────────────────────────────
# 7. Итоговое резюме
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  ИТОГОВОЕ РЕЗЮМЕ")
print("=" * 65)
print(f"\n  Реальное среднее Остаток (бенчмарк): {REAL_MEAN_STOCK:.2f} шт.")
print(f"\n  Без буфера (чистый прогноз модели):")
zero = results_pct_df[results_pct_df['buffer_pct'] == 0].iloc[0]
print(f"    Ср. остаток:         {zero['mean_stock']:.2f} шт.")
print(f"    Ср. дефицит:         {zero['mean_deficit']:.2f} шт.")
print(f"    Позиций с недобором: {zero['pct_underpredict']:.1f}%")
print(f"\n  Мультипликативный буфер +{pct_match['buffer_pct']:.0f}%:")
print(f"    Ср. остаток:         {pct_match['mean_stock']:.2f} шт.  ≈ реальный")
print(f"    Ср. дефицит:         {pct_match['mean_deficit']:.2f} шт.")
print(f"    Позиций с недобором: {pct_match['pct_underpredict']:.1f}%")
print(f"\n  Аддитивный буфер +{add_match['buffer_add']:.0f} шт.:")
print(f"    Ср. остаток:         {add_match['mean_stock']:.2f} шт.  ≈ реальный")
print(f"    Ср. дефицит:         {add_match['mean_deficit']:.2f} шт.")
print(f"    Позиций с недобором: {add_match['pct_underpredict']:.1f}%")
print(f"\n  Вывод: чтобы выйти на текущий уровень остатков (~{REAL_MEAN_STOCK:.1f} шт.)  ")
print(f"  и при этом минимизировать дефицит, рекомендуется:")
print(f"  → Аддитивный буфер +{add_match['buffer_add']:.0f} шт. к прогнозу модели")
print(f"  → или Мультипликативный +{pct_match['buffer_pct']:.0f}% к прогнозу модели")

print("\nГотово!")

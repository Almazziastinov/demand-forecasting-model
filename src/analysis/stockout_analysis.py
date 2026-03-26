import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 130
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13

# ─────────────────────────────────────────────
# 1. Загрузка данных и обучение модели
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

print("Обучение модели...")
model = LGBMRegressor(**BEST_PARAMS)
model.fit(X_train, y_train)
y_pred = np.maximum(model.predict(X_test), 0)

test = test.copy()
test['Predicted']        = np.round(y_pred).astype(int)
test['Реальный_Остаток'] = test['Остаток'].clip(lower=0).round().astype(int)
test['Реальный_Выпуск']  = test['Выпуск'].round().astype(int)

# ─────────────────────────────────────────────
# 2. Делим на две группы
# ─────────────────────────────────────────────
mask_stockout    = test['Реальный_Остаток'] == 0   # продали всё
mask_no_stockout = test['Реальный_Остаток'] > 0    # остался товар

sub = test[mask_stockout].copy()

# Для stockout позиций:
#   Если Predicted > Продано → модель считает, что спрос был выше → недозакупка
#   Если Predicted ≤ Продано → сигнала нет
sub['Сигнал_Недозакупки'] = (sub['Predicted'] - sub[TARGET]).clip(lower=0)
sub['Есть_Сигнал']        = sub['Сигнал_Недозакупки'] > 0

n_total     = len(test)
n_stockout  = len(sub)
n_signal    = sub['Есть_Сигнал'].sum()
n_no_signal = n_stockout - n_signal

total_signal_units = sub['Сигнал_Недозакупки'].sum()
avg_signal_units   = sub.loc[sub['Есть_Сигнал'], 'Сигнал_Недозакупки'].mean()

# ─────────────────────────────────────────────
# 3. Вывод статистики
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  АНАЛИЗ ПОЗИЦИЙ С НУЛЕВЫМ ОСТАТКОМ (ВОЗМОЖНАЯ НЕДОЗАКУПКА)")
print("=" * 65)

print(f"\n  Тестовый период: {test['Дата'].min().date()} — {test['Дата'].max().date()}")
print(f"  Всего позиций:                        {n_total:,}")
print(f"  ├─ Остаток > 0 (спрос известен):      {mask_no_stockout.sum():,} ({mask_no_stockout.sum()/n_total*100:.1f}%)")
print(f"  └─ Остаток = 0 (продали всё):         {n_stockout:,} ({n_stockout/n_total*100:.1f}%)")

print(f"\n  Из {n_stockout:,} позиций с нулевым остатком:")
print(f"  ┌──────────────────────────────────────────────────────────┐")
print(f"  │  Модель даёт сигнал недозакупки                          │")
print(f"  │  (Predicted > Продано):  {n_signal:>6,} поз. ({n_signal/n_stockout*100:.1f}%)          │")
print(f"  │  → ср. потенц. недобор:  {avg_signal_units:>5.1f} шт. на позицию         │")
print(f"  │                                                          │")
print(f"  │  Сигнала нет                                             │")
print(f"  │  (Predicted ≤ Продано):  {n_no_signal:>6,} поз. ({n_no_signal/n_stockout*100:.1f}%)          │")
print(f"  └──────────────────────────────────────────────────────────┘")
print(f"\n  Суммарный потенциальный недобор (оценка модели):")
print(f"  {total_signal_units:,.0f} шт. за тестовый период")
print(f"  ({total_signal_units / test['Дата'].nunique():,.0f} шт./день по всей сети)")

print(f"\n  ⚠️  Важно: это ОЦЕНКА модели, а не точный факт.")
print(f"      Истинный спрос на stockout-позициях неизвестен.")
print(f"      Модель предсказывала до того, как узнала о нехватке,")
print(f"      поэтому её прогноз — лучшая доступная аппроксимация.")

# ─────────────────────────────────────────────
# 4. По категориям
# ─────────────────────────────────────────────
print(f"\n  По категориям (позиции с Остаток = 0):")
print(f"  {'Категория':<22} {'Всего поз':>10} {'Сигнал':>8} {'% сигн':>8} "
      f"{'Ср. недобор':>13} {'Сум. недобор':>14}")
print(f"  {'─' * 78}")

cat_stats = sub.groupby('Категория', observed=True).agg(
    Всего     = ('Есть_Сигнал', 'count'),
    Сигналов  = ('Есть_Сигнал', 'sum'),
    Сум_Сигн  = ('Сигнал_Недозакупки', 'sum'),
).reset_index()
cat_stats['Пct']       = cat_stats['Сигналов'] / cat_stats['Всего'] * 100
cat_stats['Ср_Сигнал'] = cat_stats['Сум_Сигн'] / (cat_stats['Сигналов'] + 1e-8)

for _, row in cat_stats.sort_values('Сум_Сигн', ascending=False).iterrows():
    print(f"  {str(row['Категория']):<22} {row['Всего']:>10,.0f} "
          f"{row['Сигналов']:>8,.0f} "
          f"{row['Пct']:>7.1f}% "
          f"{row['Ср_Сигнал']:>13.1f} "
          f"{row['Сум_Сигн']:>14,.0f}")

# ─────────────────────────────────────────────
# 5. По дням
# ─────────────────────────────────────────────
print(f"\n  По дням:")
print(f"  {'Дата':<14} {'Поз. с Ост=0':>13} {'Сигналов':>10} {'%':>6} "
      f"{'Суммарный недобор':>19}")
print(f"  {'─' * 68}")

day_stats = sub.groupby('Дата').agg(
    Всего    = ('Есть_Сигнал', 'count'),
    Сигналов = ('Есть_Сигнал', 'sum'),
    Сумма    = ('Сигнал_Недозакупки', 'sum'),
).reset_index()

for _, row in day_stats.iterrows():
    pct = row['Сигналов'] / row['Всего'] * 100
    print(f"  {str(row['Дата'].date()):<14} {row['Всего']:>13,.0f} "
          f"{row['Сигналов']:>10,.0f} "
          f"{pct:>5.1f}% "
          f"{row['Сумма']:>19,.0f} шт.")

print(f"  {'ИТОГО':<14} {day_stats['Всего'].sum():>13,.0f} "
      f"{day_stats['Сигналов'].sum():>10,.0f} "
      f"{day_stats['Сигналов'].sum()/day_stats['Всего'].sum()*100:>5.1f}% "
      f"{day_stats['Сумма'].sum():>19,.0f} шт.")

# ─────────────────────────────────────────────
# 6. Топ товаров с наибольшим сигналом недозакупки
# ─────────────────────────────────────────────
print(f"\n  Топ-10 товаров с наибольшим суммарным сигналом недозакупки:")
print(f"  {'Номенклатура':<35} {'Сигналов':>10} {'Сум. недобор':>14} {'Ср. недобор':>13}")
print(f"  {'─' * 75}")

prod_stats = (
    sub[sub['Есть_Сигнал']]
    .groupby('Номенклатура', observed=True)
    .agg(
        Сигналов = ('Сигнал_Недозакупки', 'count'),
        Сумма    = ('Сигнал_Недозакупки', 'sum'),
        Среднее  = ('Сигнал_Недозакупки', 'mean'),
    )
    .sort_values('Сумма', ascending=False)
    .head(10)
    .reset_index()
)

for _, row in prod_stats.iterrows():
    print(f"  {str(row['Номенклатура']):<35} {row['Сигналов']:>10,.0f} "
          f"{row['Сумма']:>14,.0f} "
          f"{row['Среднее']:>13.1f}")

# ─────────────────────────────────────────────
# 7. Распределение сигнала по бакетам
# ─────────────────────────────────────────────
print(f"\n  Распределение сигнала недозакупки по величине:")
bins   = [1, 3, 5, 10, 20, np.inf]
labels = ['1–2 шт.', '3–5 шт.', '6–10 шт.', '11–20 шт.', '20+ шт.']
signal_only = sub[sub['Есть_Сигнал']].copy()
signal_only['Бакет'] = pd.cut(
    signal_only['Сигнал_Недозакупки'], bins=bins, labels=labels, right=False
)
bucket_counts = signal_only['Бакет'].value_counts().sort_index()
print(f"\n  {'Сигнал':<12} {'Позиций':>10} {'Доля':>8}")
print(f"  {'─' * 33}")
for label, count in bucket_counts.items():
    pct = count / n_signal * 100
    bar = '█' * int(pct / 3)
    print(f"  {str(label):<12} {count:>10,}  {pct:>6.1f}%  {bar}")

# ─────────────────────────────────────────────
# 8. Визуализация
# ─────────────────────────────────────────────
print("\nСтроим графики...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    'Анализ позиций с нулевым остатком — оценка потенциальной недозакупки\n'
    '(Остаток = 0: продали всё; сигнал = Predicted > Продано)',
    fontsize=13, fontweight='bold'
)

# ── Левый верхний: структура stockout позиций ────────────────────────
ax = axes[0, 0]
sizes  = [n_signal, n_no_signal, mask_no_stockout.sum()]
clrs   = ['#FF9800', '#4CAF50', '#2196F3']
labels_pie = [
    f'Сигнал недозакупки\n{n_signal:,} поз. ({n_signal/n_stockout*100:.1f}%)',
    f'Сигнала нет\n{n_no_signal:,} поз. ({n_no_signal/n_stockout*100:.1f}%)',
    f'Остаток > 0\n{mask_no_stockout.sum():,} поз.',
]
wedges, texts, autotexts = ax.pie(
    sizes, labels=labels_pie, colors=clrs,
    autopct='%1.1f%%', startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=2),
    textprops=dict(fontsize=9)
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight('bold')
ax.set_title('Структура всех позиций тестового периода')

# ── Правый верхний: сигнал недозакупки по категориям ─────────────────
ax = axes[0, 1]
cat_plot = cat_stats.sort_values('Сум_Сигн', ascending=True)
bar_colors = ['#FF9800' if v > 0 else '#4CAF50' for v in cat_plot['Сум_Сигн']]
bars = ax.barh(
    cat_plot['Категория'].astype(str), cat_plot['Сум_Сигн'],
    color=bar_colors, edgecolor='white', height=0.55
)
for bar, val in zip(bars, cat_plot['Сум_Сигн']):
    ax.text(val + cat_plot['Сум_Сигн'].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{val:,.0f} шт.', va='center', fontsize=10)
ax.set_title('Суммарный сигнал недозакупки\nпо категориям товаров')
ax.set_xlabel('Потенциальный недобор (шт., оценка модели)')
ax.grid(True, axis='x', linestyle='--', alpha=0.4)

# ── Левый нижний: распределение сигнала по бакетам ───────────────────
ax = axes[1, 0]
bkt_vals   = bucket_counts.values
bkt_labels = [str(l) for l in bucket_counts.index]
bkt_colors = ['#4CAF50', '#8BC34A', '#FF9800', '#FF5722', '#F44336']
bars = ax.bar(bkt_labels, bkt_vals, color=bkt_colors, edgecolor='white', width=0.6)
for bar, val in zip(bars, bkt_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + bkt_vals.max() * 0.01,
            f'{val:,}\n({val/n_signal*100:.1f}%)',
            ha='center', va='bottom', fontsize=10)
ax.set_title('Распределение сигнала недозакупки\nпо величине (только позиции с сигналом)')
ax.set_xlabel('Потенциальный недобор на позицию (шт.)')
ax.set_ylabel('Кол-во позиций')
ax.grid(True, axis='y', linestyle='--', alpha=0.4)
ax.set_ylim(0, bkt_vals.max() * 1.25)

# ── Правый нижний: топ-10 товаров ────────────────────────────────────
ax = axes[1, 1]
top10 = prod_stats.sort_values('Сумма', ascending=True)
bars = ax.barh(
    top10['Номенклатура'].astype(str).str[:28],
    top10['Сумма'],
    color='#FF9800', edgecolor='white', height=0.65
)
for bar, val in zip(bars, top10['Сумма']):
    ax.text(val + top10['Сумма'].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{val:,.0f} шт.', va='center', fontsize=9)
ax.set_title('Топ-10 товаров по суммарному\nсигналу недозакупки')
ax.set_xlabel('Потенциальный недобор (шт., оценка модели)')
ax.grid(True, axis='x', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('reports/plots/plot11_stockout_analysis.png')
plt.close()
print("  ✓ plot11_stockout_analysis.png")

print("\nГотово!")

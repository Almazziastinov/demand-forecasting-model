import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# ─────────────────────────────────────────────
# 2. Определяем недопредсказание
# ─────────────────────────────────────────────
# Недопредсказание: Predicted < Продано
# Shortfall = Продано - Predicted — это минимальные единицы,
# которых бы не хватило если производить по прогнозу.
# Для позиций где Остаток > 0: спрос точно известен → shortfall точный.
# Для позиций где Остаток = 0: реальный спрос мог быть ещё выше,
# но мы берём только то, что точно знаем (нижняя граница).

test['Shortfall']     = (test[TARGET] - test['Predicted']).clip(lower=0)
test['Недопред']      = test['Shortfall'] > 0
test['С_остатком']    = test['Реальный_Остаток'] > 0   # спрос известен точно

n_total       = len(test)
n_underpred   = test['Недопред'].sum()
n_overpred    = (test['Predicted'] > test[TARGET]).sum()
n_exact       = (test['Predicted'] == test[TARGET]).sum()
total_shortfall = test['Shortfall'].sum()

# ─────────────────────────────────────────────
# 3. Сводная статистика
# ─────────────────────────────────────────────
PRICE = 100  # средняя цена, руб.

print("\n" + "=" * 65)
print("  АНАЛИЗ НЕДОПРЕДСКАЗАНИЯ: Predicted < Продано")
print("  (нижняя граница потенциальных потерь выручки)")
print("=" * 65)

print(f"\n  Тестовый период: {test['Дата'].min().date()} — {test['Дата'].max().date()}")
print(f"  Всего позиций:                         {n_total:,}")
print(f"  Всего продано (факт):                  {test[TARGET].sum():,} шт.")
print(f"  Всего предсказано:                     {test['Predicted'].sum():,} шт.")

print(f"\n  {'Predicted < Продано (недопред.):':<40} {n_underpred:,} поз. ({n_underpred/n_total*100:.1f}%)")
print(f"  {'Predicted > Продано (перепред.):':<40} {n_overpred:,} поз. ({n_overpred/n_total*100:.1f}%)")
print(f"  {'Predicted = Продано (точно):':<40} {n_exact:,} поз. ({n_exact/n_total*100:.1f}%)")

sub = test[test['Недопред']].copy()
avg_shortfall     = sub['Shortfall'].mean()
median_shortfall  = sub['Shortfall'].median()
n_days            = test['Дата'].nunique()

print(f"\n  ┌──────────────────────────────────────────────────────────┐")
print(f"  │  НИЖНЯЯ ГРАНИЦА ПОТЕРЬ (только Predicted < Продано)      │")
print(f"  ├──────────────────────────────────────────────────────────┤")
print(f"  │  Суммарный shortfall:       {total_shortfall:>8,.0f} шт.              │")
print(f"  │  Средний shortfall:         {avg_shortfall:>8.1f} шт. на позицию    │")
print(f"  │  Медианный shortfall:       {median_shortfall:>8.1f} шт. на позицию    │")
print(f"  │  Потенц. потери выручки:    {total_shortfall * PRICE:>8,.0f} руб.             │")
print(f"  │  В среднем за день:         {total_shortfall / n_days * PRICE:>8,.0f} руб./день           │")
print(f"  │  Экстраполяция на месяц:    {total_shortfall / n_days * 30 * PRICE:>8,.0f} руб./мес.           │")
print(f"  └──────────────────────────────────────────────────────────┘")

print(f"\n  ⚠️  Это НИЖНЯЯ ГРАНИЦА потерь:")
print(f"      — для позиций где Остаток > 0 (спрос известен точно): shortfall точный")
print(f"      — для позиций где Остаток = 0 (продали всё): реальный спрос мог")
print(f"        быть выше Продано, значит потери могут быть ещё больше")

# Разбивка по группам остатка
sub_known   = sub[sub['С_остатком']]    # Остаток > 0: спрос точно известен
sub_unknown = sub[~sub['С_остатком']]   # Остаток = 0: спрос мог быть выше

print(f"\n  Разбивка по надёжности оценки:")
print(f"  {'─' * 62}")
print(f"  {'Группа':<38} {'Поз.':>6} {'Shortfall':>12} {'Руб.':>12}")
print(f"  {'─' * 62}")
print(f"  {'Остаток > 0 (точная оценка)':<38} "
      f"{len(sub_known):>6,} "
      f"{sub_known['Shortfall'].sum():>12,.0f} "
      f"{sub_known['Shortfall'].sum() * PRICE:>12,.0f}")
print(f"  {'Остаток = 0 (нижняя граница)':<38} "
      f"{len(sub_unknown):>6,} "
      f"{sub_unknown['Shortfall'].sum():>12,.0f} "
      f"{sub_unknown['Shortfall'].sum() * PRICE:>12,.0f}")
print(f"  {'ИТОГО':<38} "
      f"{len(sub):>6,} "
      f"{sub['Shortfall'].sum():>12,.0f} "
      f"{sub['Shortfall'].sum() * PRICE:>12,.0f}")

# ─────────────────────────────────────────────
# 4. По категориям
# ─────────────────────────────────────────────
print(f"\n  По категориям:")
print(f"  {'Категория':<22} {'Поз. с недопр.':>15} {'%':>6} "
      f"{'Ср. shortfall':>14} {'Сум. shortfall':>15}")
print(f"  {'─' * 75}")

cat_stats = test.groupby('Категория', observed=True).agg(
    Всего       = (TARGET, 'count'),
    Недопред    = ('Недопред', 'sum'),
    Сум_Short   = ('Shortfall', 'sum'),
    Ср_Short    = ('Shortfall', 'mean'),
).reset_index()
cat_stats['Pct'] = cat_stats['Недопред'] / cat_stats['Всего'] * 100

for _, row in cat_stats.sort_values('Сум_Short', ascending=False).iterrows():
    print(f"  {str(row['Категория']):<22} {row['Недопред']:>15,.0f} "
          f"{row['Pct']:>5.1f}% "
          f"{row['Ср_Short']:>14.1f} "
          f"{row['Сум_Short']:>15,.0f}")

# ─────────────────────────────────────────────
# 5. По дням
# ─────────────────────────────────────────────
print(f"\n  По дням:")
print(f"  {'Дата':<14} {'Поз. с недопр.':>15} {'%':>6} "
      f"{'Shortfall':>11} {'Потери выручки':>16}")
print(f"  {'─' * 68}")

day_stats = test.groupby('Дата').agg(
    Всего     = (TARGET, 'count'),
    Недопред  = ('Недопред', 'sum'),
    Сумма     = ('Shortfall', 'sum'),
).reset_index()

for _, row in day_stats.iterrows():
    pct = row['Недопред'] / row['Всего'] * 100
    print(f"  {str(row['Дата'].date()):<14} {row['Недопред']:>15,.0f} "
          f"{pct:>5.1f}% "
          f"{row['Сумма']:>11,.0f} шт. "
          f"{row['Сумма'] * PRICE:>14,.0f} руб.")

print(f"  {'ИТОГО':<14} {day_stats['Недопред'].sum():>15,.0f} "
      f"{day_stats['Недопред'].sum()/day_stats['Всего'].sum()*100:>5.1f}% "
      f"{day_stats['Сумма'].sum():>11,.0f} шт. "
      f"{day_stats['Сумма'].sum() * PRICE:>14,.0f} руб.")

# ─────────────────────────────────────────────
# 6. Топ-10 товаров с наибольшим shortfall
# ─────────────────────────────────────────────
print(f"\n  Топ-10 товаров по суммарному shortfall:")
print(f"  {'Номенклатура':<35} {'Поз.':>6} {'Сум. shf.':>11} {'Ср. shf.':>10} {'Потери, руб.':>14}")
print(f"  {'─' * 80}")

prod_stats = (
    sub.groupby('Номенклатура', observed=True)
    .agg(
        Позиций = ('Shortfall', 'count'),
        Сумма   = ('Shortfall', 'sum'),
        Среднее = ('Shortfall', 'mean'),
    )
    .sort_values('Сумма', ascending=False)
    .head(10)
    .reset_index()
)

for _, row in prod_stats.iterrows():
    print(f"  {str(row['Номенклатура']):<35} {row['Позиций']:>6,.0f} "
          f"{row['Сумма']:>11,.0f} "
          f"{row['Среднее']:>10.1f} "
          f"{row['Сумма'] * PRICE:>14,.0f}")

# ─────────────────────────────────────────────
# 7. Распределение shortfall по бакетам
# ─────────────────────────────────────────────
print(f"\n  Распределение shortfall по величине:")
bins   = [1, 3, 5, 10, 20, np.inf]
labels = ['1–2 шт.', '3–5 шт.', '6–10 шт.', '11–20 шт.', '20+ шт.']
sub_buckets = sub.copy()
sub_buckets['Бакет'] = pd.cut(sub_buckets['Shortfall'], bins=bins, labels=labels, right=False)
bucket_counts = sub_buckets['Бакет'].value_counts().sort_index()

print(f"\n  {'Shortfall':<12} {'Позиций':>10} {'Доля':>8}   Доля от shortfall-позиций")
print(f"  {'─' * 55}")
for label, count in bucket_counts.items():
    pct = count / n_underpred * 100
    bar = '█' * int(pct / 2)
    print(f"  {str(label):<12} {count:>10,}  {pct:>6.1f}%  {bar}")

# ─────────────────────────────────────────────
# 8. Итоговое резюме
# ─────────────────────────────────────────────
print(f"\n" + "=" * 65)
print(f"  ИТОГОВОЕ РЕЗЮМЕ")
print(f"=" * 65)
waste_rub_saved = 16561 * PRICE * 0.6   # из предыдущего анализа
lost_revenue    = total_shortfall * PRICE

print(f"\n  Что даёт модель (оценка за {n_days} дня):")
print(f"  {'─' * 55}")
print(f"  {'✅ Экономия на списаниях:':<40} ~{waste_rub_saved:>10,.0f} руб.")
print(f"  {'❌ Нижняя граница потерь выручки:':<40} ~{lost_revenue:>10,.0f} руб.")
print(f"  {'─' * 55}")
net = waste_rub_saved - lost_revenue
sign = '✅' if net > 0 else '❌'
print(f"  {sign + ' Чистый эффект (приближённо):':<40} ~{net:>+10,.0f} руб.")
print(f"\n  ⚠️  Напоминание: потери выручки занижены,")
print(f"      т.к. не учитывают спрос сверх Продано")
print(f"      на stockout-позициях.")

# ─────────────────────────────────────────────
# 9. Визуализация
# ─────────────────────────────────────────────
print("\nСтроим графики...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    'Анализ недопредсказания: Predicted < Продано\n'
    '(нижняя граница потенциальных потерь выручки)',
    fontsize=13, fontweight='bold'
)

# ── Левый верхний: соотношение перепред / недопред / точно ───────────
ax = axes[0, 0]
sizes  = [n_underpred, n_overpred, n_exact]
clrs   = ['#2196F3', '#F44336', '#4CAF50']
labels_pie = [
    f'Недопредсказание\n(потенц. потери выручки)\n{n_underpred:,} поз. ({n_underpred/n_total*100:.1f}%)',
    f'Перепредсказание\n(потенц. списания)\n{n_overpred:,} поз. ({n_overpred/n_total*100:.1f}%)',
    f'Точное попадание\n{n_exact:,} поз. ({n_exact/n_total*100:.1f}%)',
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
ax.set_title('Структура ошибок прогноза\n(все позиции тестового периода)')

# ── Правый верхний: shortfall по категориям ──────────────────────────
ax = axes[0, 1]
cat_plot = cat_stats.sort_values('Сум_Short', ascending=True)
bars = ax.barh(
    cat_plot['Категория'].astype(str),
    cat_plot['Сум_Short'],
    color='#2196F3', edgecolor='white', height=0.55
)
for bar, val, rub in zip(bars, cat_plot['Сум_Short'], cat_plot['Сум_Short'] * PRICE):
    ax.text(val + cat_plot['Сум_Short'].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{val:,.0f} шт.  (~{rub/1000:.0f}к руб.)',
            va='center', fontsize=9)
ax.set_title('Суммарный shortfall по категориям\n(потенц. потери выручки)')
ax.set_xlabel('Shortfall (шт.)')
ax.grid(True, axis='x', linestyle='--', alpha=0.4)
ax.set_xlim(0, cat_plot['Сум_Short'].max() * 1.4)

# ── Левый нижний: распределение shortfall по бакетам ─────────────────
ax = axes[1, 0]
bkt_vals   = bucket_counts.values
bkt_labels = [str(l) for l in bucket_counts.index]
bkt_colors = ['#4CAF50', '#8BC34A', '#FF9800', '#FF5722', '#F44336']
bars = ax.bar(bkt_labels, bkt_vals, color=bkt_colors, edgecolor='white', width=0.6)
for bar, val in zip(bars, bkt_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + bkt_vals.max() * 0.01,
            f'{val:,}\n({val/n_underpred*100:.1f}%)',
            ha='center', va='bottom', fontsize=10)
ax.set_title('Распределение shortfall по величине\n(только позиции с недопредсказанием)')
ax.set_xlabel('Shortfall на позицию (шт.)')
ax.set_ylabel('Кол-во позиций')
ax.grid(True, axis='y', linestyle='--', alpha=0.4)
ax.set_ylim(0, bkt_vals.max() * 1.25)

# ── Правый нижний: топ-10 товаров ────────────────────────────────────
ax = axes[1, 1]
top10 = prod_stats.sort_values('Сумма', ascending=True)
bars = ax.barh(
    top10['Номенклатура'].astype(str).str[:30],
    top10['Сумма'],
    color='#2196F3', edgecolor='white', height=0.65
)
for bar, val in zip(bars, top10['Сумма']):
    ax.text(val + top10['Сумма'].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{val:,.0f} шт.  (~{val*PRICE/1000:.0f}к руб.)',
            va='center', fontsize=9)
ax.set_title('Топ-10 товаров по shortfall\n(потенц. упущенная выручка)')
ax.set_xlabel('Shortfall (шт.)')
ax.grid(True, axis='x', linestyle='--', alpha=0.4)
ax.set_xlim(0, top10['Сумма'].max() * 1.45)

plt.tight_layout()
plt.savefig('reports/plots/plot12_underprediction_analysis.png')
plt.close()
print("  ✓ plot12_underprediction_analysis.png")

print("\nГотово!")

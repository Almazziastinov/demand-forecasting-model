import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
test['Predicted']         = np.round(y_pred).astype(int)
test['Реальный_Остаток']  = test['Остаток'].clip(lower=0).round().astype(int)
test['Реальный_Выпуск']   = test['Выпуск'].round().astype(int)

# Гипотетический остаток если бы производили по прогнозу:
# max(0, Predicted - Продано)
# Это честно ТОЛЬКО на позициях где не было дефицита (Остаток > 0),
# потому что там Продано == реальный спрос.
test['Модель_Остаток'] = (test['Predicted'] - test[TARGET]).clip(lower=0)

# Маска: позиции без дефицита (остаток > 0 → спрос был полностью удовлетворён)
mask_no_stockout = test['Реальный_Остаток'] > 0
mask_stockout    = test['Реальный_Остаток'] == 0

n_total        = len(test)
n_no_stockout  = mask_no_stockout.sum()
n_stockout     = mask_stockout.sum()

# ─────────────────────────────────────────────
# 2. Сводная статистика
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  ОЦЕНКА МОДЕЛИ ПО ИСТОРИЧЕСКИМ ОСТАТКАМ")
print("  (только то, что можно честно измерить)")
print("=" * 65)

print(f"\n  Тестовый период: {test['Дата'].min().date()} — {test['Дата'].max().date()}")
print(f"  Всего позиций (товар × пекарня × день): {n_total:,}")
print()
print(f"  ┌──────────────────────────────────────────────────────┐")
print(f"  │  Позиции С остатком  (спрос известен точно): {n_no_stockout:>6,} │")
print(f"  │  Позиции БЕЗ остатка (спрос мог быть выше): {n_stockout:>6,} │")
print(f"  └──────────────────────────────────────────────────────┘")

print(f"\n  ⚠️  Важно: упущенную выручку из исторических данных")
print(f"      оценить невозможно — когда товар кончается, мы не")
print(f"      знаем, сколько покупателей ушли ни с чем.")
print(f"      Поэтому анализ остатков ведётся только там, где")
print(f"      реальный спрос известен точно (Остаток > 0).\n")

# ── Анализ только на позициях без дефицита ────────────────────────────
sub = test[mask_no_stockout].copy()

real_waste_known  = sub['Реальный_Остаток'].sum()
model_waste_known = sub['Модель_Остаток'].sum()
waste_saved       = real_waste_known - model_waste_known
waste_saved_pct   = waste_saved / (real_waste_known + 1e-8) * 100
avg_real          = sub['Реальный_Остаток'].mean()
avg_model         = sub['Модель_Остаток'].mean()

print(f"  {'АНАЛИЗ НА ПОЗИЦИЯХ БЕЗ ДЕФИЦИТА ({:,} шт.)'.format(n_no_stockout)}")
print(f"  {'─' * 60}")
print(f"  {'Показатель':<42} {'Реально':>9} {'Модель':>9}")
print(f"  {'─' * 60}")
print(f"  {'Суммарный остаток (шт.)':<42} {real_waste_known:>9,} {model_waste_known:>9,}")
print(f"  {'Средний остаток на позицию (шт.)':<42} {avg_real:>9.2f} {avg_model:>9.2f}")
print(f"  {'─' * 60}")
print(f"  {'Сокращение остатков с моделью':<42} "
      f"{waste_saved:>+9,} шт. ({waste_saved_pct:+.1f}%)")

PRICE      = 100
COST_RATIO = 0.6

real_waste_rub  = real_waste_known  * PRICE * COST_RATIO
model_waste_rub = model_waste_known * PRICE * COST_RATIO
saved_rub       = real_waste_rub - model_waste_rub
n_days          = test['Дата'].nunique()

print(f"\n  Финансовая оценка (цена ≈ {PRICE} руб./шт., себест. ≈ {int(COST_RATIO*100)}%)")
print(f"  {'─' * 60}")
print(f"  {'Потери от списаний, руб.':<42} {real_waste_rub:>9,.0f} {model_waste_rub:>9,.0f}")
print(f"  {'─' * 60}")
if saved_rub > 0:
    print(f"  💰 Экономия на списаниях: ~{saved_rub:,.0f} руб. за {n_days} дня")
    print(f"     (~{saved_rub / n_days:,.0f} руб./день | ~{saved_rub / n_days * 30:,.0f} руб./месяц)")
else:
    print(f"  Модель увеличивает списания на ~{-saved_rub:,.0f} руб. за {n_days} дня")

# ── Анализ по категориям (только no-stockout) ─────────────────────────
print(f"\n  По категориям товаров (только позиции с остатком):")
print(f"  {'Категория':<22} {'Реал. остаток':>14} {'Мод. остаток':>13} {'Экономия':>10}")
print(f"  {'─' * 62}")
cat_stats = sub.groupby('Категория', observed=True).agg(
    Реал=('Реальный_Остаток', 'sum'),
    Мод=('Модель_Остаток',    'sum'),
).reset_index()
cat_stats['Экономия']  = cat_stats['Реал'] - cat_stats['Мод']
cat_stats['Экон_pct']  = cat_stats['Экономия'] / (cat_stats['Реал'] + 1e-8) * 100
for _, row in cat_stats.sort_values('Реал', ascending=False).iterrows():
    sign = '+' if row['Экономия'] >= 0 else ''
    print(f"  {str(row['Категория']):<22} {row['Реал']:>14,.0f} "
          f"{row['Мод']:>13,.0f} "
          f"  {sign}{row['Экономия']:,.0f} шт. ({sign}{row['Экон_pct']:.1f}%)")

# ── Анализ по дням ────────────────────────────────────────────────────
print(f"\n  По дням тестового периода (только позиции с остатком):")
print(f"  {'Дата':<14} {'Позиций':>9} {'Реал. ост.':>12} {'Мод. ост.':>11} {'Экономия':>10}")
print(f"  {'─' * 60}")
day_stats = sub.groupby('Дата').agg(
    Позиций=('Реальный_Остаток', 'count'),
    Реал   =('Реальный_Остаток', 'sum'),
    Мод    =('Модель_Остаток',   'sum'),
).reset_index()
for _, row in day_stats.iterrows():
    econ = row['Реал'] - row['Мод']
    sign = '+' if econ >= 0 else ''
    print(f"  {str(row['Дата'].date()):<14} {row['Позиций']:>9,.0f} "
          f"{row['Реал']:>12,.0f} "
          f"{row['Мод']:>11,.0f} "
          f"  {sign}{econ:,.0f} шт.")
print(f"  {'ИТОГО':<14} {day_stats['Позиций'].sum():>9,.0f} "
      f"{day_stats['Реал'].sum():>12,.0f} "
      f"{day_stats['Мод'].sum():>11,.0f} "
      f"  {'+' if waste_saved >= 0 else ''}{waste_saved:,.0f} шт.")

# ─────────────────────────────────────────────
# 3. Визуализация
# ─────────────────────────────────────────────
print("\nСтроим графики...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    'Оценка модели по историческим остаткам\n'
    '(только позиции где реальный спрос известен точно — Остаток > 0)',
    fontsize=13, fontweight='bold'
)

# ── Левый верхний: остатки по дням ───────────────────────────────────
ax = axes[0, 0]
x = np.arange(len(day_stats))
w = 0.35
b1 = ax.bar(x - w/2, day_stats['Реал'], width=w,
            color='#F44336', label='Реальный остаток', edgecolor='white')
b2 = ax.bar(x + w/2, day_stats['Мод'],  width=w,
            color='#4CAF50', label='Остаток с моделью', edgecolor='white')
for bar in [*b1, *b2]:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + day_stats['Реал'].max() * 0.01,
            f'{h:,.0f}', ha='center', va='bottom', fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([str(d.date()) for d in day_stats['Дата']], rotation=10)
ax.set_title('Остатки по дням\n(позиции без дефицита)')
ax.set_ylabel('Штук')
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.4)

# ── Правый верхний: остатки по категориям ────────────────────────────
ax = axes[0, 1]
cats = cat_stats.sort_values('Реал', ascending=True)
x2 = np.arange(len(cats))
ax.barh(x2 - 0.2, cats['Реал'], height=0.38,
        color='#F44336', label='Реальный остаток', edgecolor='white')
ax.barh(x2 + 0.2, cats['Мод'],  height=0.38,
        color='#4CAF50', label='Остаток с моделью', edgecolor='white')
ax.set_yticks(x2)
ax.set_yticklabels(cats['Категория'].astype(str), fontsize=10)
ax.set_title('Остатки по категориям\n(позиции без дефицита)')
ax.set_xlabel('Суммарный остаток (шт.)')
ax.legend()
ax.grid(True, axis='x', linestyle='--', alpha=0.4)

# ── Левый нижний: структура всех позиций ─────────────────────────────
ax = axes[1, 0]
# Для модельного сценария: сколько позиций имело бы остаток/ноль
n_model_has_waste  = (test['Модель_Остаток'] > 0).sum()
n_model_zero_waste = (test['Модель_Остаток'] == 0).sum()
n_real_has_waste   = (test['Реальный_Остаток'] > 0).sum()
n_real_zero_waste  = (test['Реальный_Остаток'] == 0).sum()

labels_sc = ['Реальный\nсценарий', 'Сценарий\nс моделью']
vals_waste = [n_real_has_waste  / n_total * 100, n_model_has_waste  / n_total * 100]
vals_zero  = [n_real_zero_waste / n_total * 100, n_model_zero_waste / n_total * 100]
x3 = np.arange(2)

p1 = ax.bar(x3, vals_zero,  color='#4CAF50', label='Остаток = 0 (продано всё)', edgecolor='white')
p2 = ax.bar(x3, vals_waste, bottom=vals_zero, color='#F44336', label='Есть остаток', edgecolor='white')
for bar, val in zip(p1, vals_zero):
    ax.text(bar.get_x() + bar.get_width()/2, val/2,
            f'{val:.1f}%', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
for bar, val, bot in zip(p2, vals_waste, vals_zero):
    ax.text(bar.get_x() + bar.get_width()/2, bot + val/2,
            f'{val:.1f}%', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
ax.set_xticks(x3)
ax.set_xticklabels(labels_sc, fontsize=11)
ax.set_title('Доля позиций с остатком vs без\n(все позиции тестового периода)')
ax.set_ylabel('Доля позиций (%)')
ax.set_ylim(0, 115)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, axis='y', linestyle='--', alpha=0.3)

# ── Правый нижний: финансовая оценка (только списания) ───────────────
ax = axes[1, 1]
fin_labels = [f'Потери от списаний\n(реально)\n{real_waste_rub:,.0f} руб.',
              f'Потери от списаний\n(с моделью)\n{model_waste_rub:,.0f} руб.',
              f'Экономия\nна списаниях\n{saved_rub:,.0f} руб.']
fin_values = [real_waste_rub, model_waste_rub, saved_rub]
fin_colors = ['#F44336', '#FF9800', '#4CAF50' if saved_rub >= 0 else '#F44336']
bars = ax.bar(fin_labels, fin_values, color=fin_colors, edgecolor='white', width=0.5)
for bar, val in zip(bars, fin_values):
    sign = '+' if val >= 0 else ''
    ax.text(bar.get_x() + bar.get_width()/2, max(val, 0) + max(fin_values) * 0.01,
            f'{sign}{val:,.0f} руб.', ha='center', va='bottom',
            fontsize=10, fontweight='bold')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title(f'Финансовая оценка\n'
             f'(цена ≈ {PRICE} руб., себест. {int(COST_RATIO*100)}%, период: {n_days} дня)\n'
             f'⚠️ Упущенная выручка не включена — её нельзя измерить')
ax.set_ylabel('Рублей')
ax.grid(True, axis='y', linestyle='--', alpha=0.4)
ax.set_ylim(min(0, min(fin_values)) * 1.2, max(fin_values) * 1.35)

plt.tight_layout()
plt.savefig('reports/plots/plot10_historical_evaluation.png')
plt.close()
print("  ✓ plot10_historical_evaluation.png")
print("\nГотово!")

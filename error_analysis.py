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
# Загрузка данных и обучение финальной модели
# ─────────────────────────────────────────────
print("Загрузка данных...")
df = pd.read_csv('preprocessed_data.csv')
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
test['Predicted'] = y_pred
test['Ошибка']    = test['Predicted'] - test[TARGET]   # со знаком: + = перепрогноз, - = недопрогноз
test['|Ошибка|']  = test['Ошибка'].abs()

mae = mean_absolute_error(y_test, y_pred)
avg_sales = y_test.mean()

# ─────────────────────────────────────────────
# 1. ОБЩАЯ ИНТЕРПРЕТАЦИЯ
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ПРАКТИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ ОШИБКИ МОДЕЛИ")
print("=" * 60)
print(f"\n  MAE = {mae:.2f} шт.")
print(f"  Это значит: на каждую позицию (товар × пекарня × день)")
print(f"  модель в среднем ошибается на {mae:.2f} штуки.")
print(f"\n  Средние продажи одной позиции: {avg_sales:.2f} шт.")
print(f"  Относительная ошибка:          {mae / avg_sales * 100:.1f}%")

n_over  = (test['Ошибка'] > 0).sum()
n_under = (test['Ошибка'] < 0).sum()
n_exact = (test['Ошибка'] == 0).sum()
total   = len(test)

pct_over  = n_over  / total * 100
pct_under = n_under / total * 100
pct_exact = n_exact / total * 100

mean_over_val  = test.loc[test['Ошибка'] > 0, 'Ошибка'].mean()
mean_under_val = test.loc[test['Ошибка'] < 0, 'Ошибка'].mean()

print(f"\n  Из {total:,} позиций в тестовом периоде:")
print(f"  ┌─────────────────────────────────────────────────────┐")
print(f"  │  Перепрогноз (остаток): {n_over:>6,} поз. ({pct_over:>5.1f}%)     │")
print(f"  │  → ср. излишек:  +{mean_over_val:.2f} шт. на позицию             │")
print(f"  │                                                     │")
print(f"  │  Недопрогноз (недобор): {n_under:>6,} поз. ({pct_under:>5.1f}%)     │")
print(f"  │  → ср. недобор:  {mean_under_val:.2f} шт. на позицию              │")
print(f"  │                                                     │")
print(f"  │  Точное попадание:      {n_exact:>6,} поз. ({pct_exact:>5.1f}%)     │")
print(f"  └─────────────────────────────────────────────────────┘")

# ─────────────────────────────────────────────
# 2. РАСПРЕДЕЛЕНИЕ ОШИБОК ПО БАКЕТАМ
# ─────────────────────────────────────────────
print("\n  Распределение абсолютной ошибки по позициям:")
bins   = [0, 1, 3, 5, 10, 20, np.inf]
labels = ['0 шт.', '1–2 шт.', '3–5 шт.', '6–10 шт.', '11–20 шт.', '20+ шт.']
test['Бакет'] = pd.cut(test['|Ошибка|'], bins=bins, labels=labels, right=False)
bucket_counts = test['Бакет'].value_counts().sort_index()
print(f"\n  {'Ошибка':<15} {'Кол-во позиций':>15} {'Доля':>8}")
print(f"  {'-'*40}")
for label, count in bucket_counts.items():
    pct = count / total * 100
    bar = '█' * int(pct / 2)
    print(f"  {str(label):<15} {count:>15,}  {pct:>6.1f}%  {bar}")

# ─────────────────────────────────────────────
# 3. БИЗНЕС-ЭФФЕКТ: потери от перепрогноза и недопрогноза
# ─────────────────────────────────────────────
total_overstock  = test.loc[test['Ошибка'] > 0, 'Ошибка'].sum()
total_understock = test.loc[test['Ошибка'] < 0, 'Ошибка'].abs().sum()

# Оцениваем в деньгах (примерная стоимость одного изделия ~100 руб.)
PRICE_PER_UNIT = 100
WASTE_COST_PCT = 0.6   # списание = 60% от цены (себестоимость)
LOST_SALE_PCT  = 1.0   # недопродажа = 100% упущенная выручка

waste_rub      = total_overstock  * PRICE_PER_UNIT * WASTE_COST_PCT
lost_sale_rub  = total_understock * PRICE_PER_UNIT * LOST_SALE_PCT

print(f"\n" + "=" * 60)
print(f"  БИЗНЕС-ЭФФЕКТ (тестовый период: {test['Дата'].min().date()} — {test['Дата'].max().date()})")
print(f"=" * 60)
print(f"\n  Если производить ровно по прогнозу модели:")
print(f"\n  Суммарный перепрогноз (потенц. остатки): {total_overstock:>8,.0f} шт.")
print(f"  → Потери от списаний (~{WASTE_COST_PCT*100:.0f}% себест.):  ~{waste_rub:>10,.0f} руб.")
print(f"\n  Суммарный недопрогноз (потенц. недобор):  {total_understock:>8,.0f} шт.")
print(f"  → Упущенная выручка:                      ~{lost_sale_rub:>10,.0f} руб.")

per_day_over  = total_overstock  / test['Дата'].nunique()
per_day_under = total_understock / test['Дата'].nunique()
print(f"\n  В среднем за день по всей сети:")
print(f"  → Излишек:   {per_day_over:,.0f} шт.")
print(f"  → Недобор:   {per_day_under:,.0f} шт.")

# ─────────────────────────────────────────────
# 4. ВИЗУАЛИЗАЦИЯ
# ─────────────────────────────────────────────
print("\nСтроим графики...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Практическая интерпретация ошибки модели (MAE)', fontsize=15, fontweight='bold')

# ── Левый верхний: гистограмма ошибок со знаком ──────────────────────
ax = axes[0, 0]
errors_signed = test['Ошибка'].values
clip_val = np.percentile(np.abs(errors_signed), 95)
errors_clipped = np.clip(errors_signed, -clip_val, clip_val)
n, bins_h, patches = ax.hist(errors_clipped, bins=60, edgecolor='white', linewidth=0.4)
for patch, left in zip(patches, bins_h[:-1]):
    patch.set_facecolor('#F44336' if left >= 0 else '#2196F3')
ax.axvline(0, color='black', linewidth=2, linestyle='-', label='Нулевая ошибка')
ax.axvline(errors_signed.mean(), color='orange', linewidth=2, linestyle='--',
           label=f'Среднее смещение = {errors_signed.mean():.2f}')
over_patch  = mpatches.Patch(color='#F44336', label=f'Перепрогноз → остаток ({pct_over:.1f}%)')
under_patch = mpatches.Patch(color='#2196F3', label=f'Недопрогноз → недобор ({pct_under:.1f}%)')
ax.legend(handles=[over_patch, under_patch,
                   mpatches.Patch(color='black', label=f'Среднее смещение = {errors_signed.mean():.2f}')],
          fontsize=9)
ax.set_title('Распределение ошибок прогноза\n(красный = остаток, синий = недобор)')
ax.set_xlabel('Прогноз − Факт (шт.)')
ax.set_ylabel('Кол-во позиций')
ax.grid(True, linestyle='--', alpha=0.4)

# ── Правый верхний: распределение абс. ошибки по бакетам ─────────────
ax = axes[0, 1]
bucket_pcts = bucket_counts / total * 100
bar_colors  = ['#4CAF50', '#8BC34A', '#FF9800', '#FF5722', '#F44336', '#9C27B0']
bars = ax.bar(labels, bucket_pcts.values, color=bar_colors, edgecolor='white', width=0.65)
for bar, val, cnt in zip(bars, bucket_pcts.values, bucket_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
            f'{val:.1f}%\n({cnt:,})', ha='center', va='bottom', fontsize=9)
ax.set_title('Сколько позиций попадает\nв каждый диапазон ошибки?')
ax.set_xlabel('Абсолютная ошибка (шт.)')
ax.set_ylabel('Доля позиций (%)')
ax.grid(True, axis='y', linestyle='--', alpha=0.4)
ax.set_ylim(0, bucket_pcts.max() * 1.25)

# ── Левый нижний: перепрогноз vs недопрогноз по категориям ───────────
ax = axes[1, 0]
cat_err = test.groupby('Категория', observed=True).agg(
    Перепрогноз=('Ошибка', lambda x: x[x > 0].sum()),
    Недопрогноз=('Ошибка', lambda x: x[x < 0].abs().sum()),
).reset_index()
cat_err['Категория'] = cat_err['Категория'].astype(str)
x = np.arange(len(cat_err))
w = 0.38
ax.bar(x - w/2, cat_err['Перепрогноз'], width=w, color='#F44336', label='Остаток (перепрогноз)', edgecolor='white')
ax.bar(x + w/2, cat_err['Недопрогноз'], width=w, color='#2196F3', label='Недобор (недопрогноз)', edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(cat_err['Категория'], rotation=15, ha='right', fontsize=9)
ax.set_title('Суммарный остаток vs недобор\nпо категориям товаров (тест)')
ax.set_ylabel('Суммарная ошибка (шт.)')
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.4)

# ── Правый нижний: бизнес-эффект ─────────────────────────────────────
ax = axes[1, 1]
categories  = ['Потери от\nсписаний', 'Упущенная\nвыручка']
values      = [waste_rub, lost_sale_rub]
bar_c       = ['#FF5722', '#9C27B0']
bars = ax.bar(categories, values, color=bar_c, edgecolor='white', width=0.45)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + max(values) * 0.01,
            f'{val:,.0f} руб.', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_title(f'Оценка бизнес-потерь за тестовый период\n'
             f'(цена ≈ {PRICE_PER_UNIT} руб./шт., {test["Дата"].nunique()} дня)')
ax.set_ylabel('Сумма (руб.)')
ax.grid(True, axis='y', linestyle='--', alpha=0.4)
ax.set_ylim(0, max(values) * 1.2)

plt.tight_layout()
plt.savefig('plot9_error_business_analysis.png')
plt.close()
print("  ✓ plot9_error_business_analysis.png")

print("\nГотово! Открой plot9_error_business_analysis.png для просмотра.")

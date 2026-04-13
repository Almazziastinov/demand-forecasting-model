# Experiments V2 — Plan

Baseline: **01_baseline_8m** (MAE 2.29, WMAPE 25.9%, R2 0.918)
Data: 3.5M rows, 207 days, 199 bakeries, 610 products, 27 categories

---

## Tier 1: Tuning & Validation

### 02. Optuna 8m ✅
Перетюнить гиперпараметры на 8 месяцах. v6 params оптимальны для 3m/5cat.
- Потенциал: +3-7% MAE
- Сложность: низкая
- **Результат**: Best MAE 2.2812 за 16 trials (20 max). Улучшение минимальное (-0.009). v6 params уже близки к оптимуму.

### 03. Demand Target (fair comparison) ✅
Два варианта: A (train on Продано) и B (train on Спрос). Оба оценены vs Спрос.
- Потенциал: информативный + бизнес-ценность
- Сложность: низкая
- **Результат**: Model B (demand) wins: MAE 2.99 vs 3.13 (demand metric). При P&L-анализе Model B экономически выгоднее (+673К руб/день vs пекарь).

---

## Tier 2: Feature Engineering

### 04. Ramadan feature ⏭️ SKIPPED
is_ramadan, is_iftar_period, ramadan_day (1-30). Рамадан 2026: 18 фев — 19 мар.
- Потенциал: средний (30 дней из 207)
- Сложность: низкая
- **Пропущен**: только 1 Рамадан в данных — невозможно протестировать effect vs non-Ramadan. Нужен минимум 2 года данных.

### 05. Product & bakery profiles ✅
bakery_avg_sales, product_avg_sales, product_cv, bakery_x_product_avg.
- Потенциал: средний
- Сложность: низкая
- **Результат**: MAE 2.31 (+0.02). Нейтрально — LightGBM уже извлекает эту информацию через категориальные энкодинги. Явные профили избыточны.

### 06. Demand correction via cumulative profiles (censored demand)
**Ключевой эксперимент.** Таргет = Продано + Упущенно.

Источник: `sales_hrs_all.csv` (30M чеков с `Дата время чека`) — все 199 пекарен, 8 месяцев.
Раньше профили строились только по Тукая 62А (checks_analysis Section 16, 350К руб/3мес).
Теперь масштабируем на все точки.

Алгоритм:
1. Для каждого (товар × пекарня × день) находим час последней продажи
2. "Полный день" = last_sale >= 17:00 (товар ещё был в наличии к вечеру)
3. Из полных дней строим cumulative profile по часам для каждого товара×пекарня:
   "к 10:00 = 30%, к 12:00 = 55%, к 14:00 = 75%, к 17:00 = 95%"
4. Для дня с ранним стопом (last_sale=14:00, продано=20):
   estimated_demand = 20 / 0.75 = 26.7, lost = 6.7 шт
5. Защиты: min 5 полных дней, cum >= 5%, cap x5, estimated >= actual
6. Новый таргет: Спрос = Продано + Упущенно (или Продано / cum_profile)

Дополнительные фичи из профилей:
- `is_censored` — бинарный: ранний стоп (last_sale <= 14:00)
- `censoring_ratio` — какая доля спроса реализована (cum_profile at stop hour)
- `cum_profile_14h` — типичная доля продаж к 14:00 (характеристика товара)
- `pct_censored_days` — % дней с ранним стопом (как часто товар продаётся в ноль)

Два варианта эксперимента:
  a) Таргет = скорректированный спрос (обучаем предсказывать demand, не sales)
  b) Таргет = Продано, но добавляем censoring-фичи (модель знает о проблеме)

Ожидаемый эффект: сильнее всего улучшит Выпечку сытную (51% дней продано в ноль).
Прямо решает проблему bias на high demand (+14.1 на 100+).
- Потенциал: **высокий** (единственный способ приблизиться к real demand)
- Сложность: средняя-высокая (чанковый парсинг часов, агрегация профилей)

### 07. Temporal features v2
day_of_year, season, Fourier-компоненты недельной/месячной сезонности.
- Потенциал: низкий-средний
- Сложность: низкая

### 08. Location features (анализ локаций по БП) ✅ NEUTRAL
Источник: `data/raw/анализ локаций по БП 02.03 ( 1) (2) (4) (1) (1).xlsx`
228 пекарен, 23 parsed фичи. Покрытие: 138 из 194 (71%).
Target: Спрос. Оценка на 138 matched пекарнях.

Model A (baseline + demand lags, 52 фичи): MAE 3.0248
Model B (+ 23 location фичи, 75 фич): MAE 3.0222

- **Результат**: Delta **-0.003 MAE** — статистически незначимо. Location features избыточны при наличии `Пекарня` как категориальной фичи — LightGBM уже знает уровень продаж каждой точки. Все loc_* фичи на последних местах feature importance (ранги 34-71 из 75). High demand: без улучшений (>=50: +0.09 хуже).
- Потенциал: **нулевой** при наличии bakery ID (полезно только для cold start новых пекарен)
- Сложность: средняя

### 09. Price features (ценовые признаки)
Источник: `data/raw/sales_hrs_all.csv` (30M чеков, колонка `Цена`).
Target: Спрос. Агрегация средней цены на уровне день×пекарня×продукт.

Фичи:
- `avg_price` — средняя цена товара за день
- `price_change_vs_lag7` — изменение цены за неделю (%)
- `price_vs_median` — цена относительно медианы по продукту (ratio)
- `price_roll_mean7` / `price_roll_std7` — скользящие по цене

Гипотеза: цена — динамическая фича (меняется день ото дня, акции, скидки на вчерашний товар). В отличие от location, это **новая информация** которую модель не может извлечь из bakery/product ID.
- Потенциал: средний-высокий
- Сложность: средняя (агрегация 30M строк + merge)
- **Результат**: MAE 2.9364, delta **-0.056** — **лучшее улучшение среди всех экспериментов на demand target**. price_vs_median (ранг 8) и price_change_7d (ранг 10) в топ-10 фич. High demand >=100: delta -0.51. Подтверждение: динамические фичи > статические.

---

## Tier 2b: High Demand Problem

Факт: corr(std, MAE) = 0.936. Bias на 100+ = +14.1 (модель занижает).
Треугольник курица MAE 19-48, Кыстыбый MAE 24-48.
MSE loss учит условное среднее → "усредняет" при широком распределении.

### 40. Tweedie regression ✅
`objective='tweedie', tweedie_variance_power=1.5`.
Var(y) ∝ mean^p — автоматически учитывает рост дисперсии с mean.
Между Poisson (p=1) и Gamma (p=2). Одна строка изменений.
- Потенциал: средний (снижает bias на high demand)
- Сложность: минимальная
- **Результат**: MAE 2.2778, WMAPE 25.77%, R2 0.9185. Улучшение -0.013 MAE. Маргинальный прирост.

### 41. Log-transform target ✅
`y = log1p(Продано)`, обратно `expm1(pred)`. Сжимает шкалу, разница
между 150 и 200 = 0.28 вместо 50. Снижает доминирование high demand.
- Потенциал: средний
- Сложность: минимальная
- Риск: может ухудшить low demand (0 vs 1 раздувается)
- **Результат**: MAE 2.2889, WMAPE 25.89%, Bias 0.60, R2 0.914. Почти baseline, но bias вырос. Не оправдал.

### 42. Asymmetric loss (штраф за занижение) ❌ FAILED
Custom objective: alpha > 0.5 штрафует недопрогноз сильнее.
Бизнес-мотивация: упущенные продажи хуже списания.
alpha=0.6 сдвинет bias с +14 к ~0 на high demand.
- Потенциал: средний (bias), слабый (MAE)
- Сложность: низкая (~20 строк custom loss)
- **Результат**: MAE 8.84, WMAPE 100%, R2 -0.33. Сломано — gradient implementation некорректная. Нужна переделка.

### 43. Quantile regression (интервалы) ✅ BEST MAE
Три модели: P25, P50, P75. Для Треугольника курица (mean 198, std 63):
вместо "198" → "160–200–240". Пекарь сам выбирает стратегию.
P50 (медиана) устойчивее к правому хвосту, чем mean.
- Потенциал: высокий (бизнес-ценность), средний (MAE)
- Сложность: средняя (3 модели, UI)
- **Результат**: P50 MAE 2.2655, WMAPE 25.63%. Лучший MAE среди всех экспериментов (-0.025). Coverage 51.9%, interval width 3.67. Практически полезно: интервалы вместо точки.

### 44. Mixture of Experts (demand-level split) ✅
product_avg >= 15 → model_high (depth 9, 3000+ trees, Huber loss)
product_avg < 15 → model_low (стандартный)
Доп. фичи для high: product_std, recent_max, days_since_max.
- Потенциал: высокий
- Сложность: средняя
- **Результат**: MAE 2.54, WMAPE 28.7%, R2 0.803. Хуже baseline (+0.25 MAE). Split-стратегия не работает — модели на подвыборках теряют генерализацию.

### 45. Log-target + residual correction (двухэтапный) ✅
Этап 1: LightGBM на log1p(y) → base (хорош для high demand)
Этап 2: LightGBM на (y - expm1(base)) → correction (правит low demand)
Итог: expm1(base) + correction. Лучшее из двух миров.
- Потенциал: высокий
- Сложность: средняя (2 модели, 2 этапа)
- **Результат**: MAE 2.28, WMAPE 25.79%, R2 0.9185. Второй лучший результат (-0.010). Быстрее квантилей (482s vs 1207s).

### 46. Variance-weighted training ✅
sample_weight = f(product_std). Два варианта:
  a) weight ∝ std — фокус на high demand → MAE high demand улучшится
  b) weight ∝ 1/std — фокус на предсказуемых → общий MAE улучшится
Выбор зависит от бизнес-приоритета.
- Потенциал: средний
- Сложность: минимальная
- **Результат**: Best: B (1/std) MAE 2.2806, WMAPE 25.8%, R2 0.9191 (лучший R2). Маргинальное улучшение (-0.010).

---

## Results Summary (completed experiments)

| # | Experiment | MAE | WMAPE | Bias | R² | Delta MAE |
|---|---|---|---|---|---|---|
| 01 | **Baseline 8m** | 2.2904 | 25.91% | +0.08 | 0.918 | — |
| 02 | Optuna 8m | 2.2812 | — | — | — | -0.009 |
| 03 | Demand target | 2.9923* | 28.19%* | +0.11* | 0.896* | +0.70* |
| 05 | Profiles | 2.3109 | 26.14% | +0.04 | 0.918 | +0.02 |
| 40 | Tweedie | 2.2778 | 25.77% | +0.11 | 0.919 | -0.013 |
| 41 | Log target | 2.2889 | 25.89% | +0.60 | 0.914 | -0.002 |
| 42 | Asymmetric loss | ~~8.84~~ | ~~100%~~ | — | -0.33 | BROKEN |
| **43** | **Quantile P50** | **2.2655** | **25.63%** | +0.41 | 0.915 | **-0.025** |
| 44 | Mixture of Experts | 2.5371 | 28.70% | +0.54 | 0.803 | +0.247 |
| 45 | Log + residual | 2.2800 | 25.79% | +0.10 | 0.919 | -0.010 |
| 46 | Variance weighted | 2.2806 | 25.80% | +0.08 | 0.919 | -0.010 |
| 08 | Location features | 3.0222* | 27.74%* | +0.16* | — | -0.003* |
| **09** | **Price features** | **2.9364** | **27.66%** | +0.14 | — | **-0.056** |

\* Exp 03, 08 оценены vs Спрос (demand) на подмножестве пекарен — нечестно сравнивать с baseline.

**Вывод**: потолок MAE ~2.27 при текущих данных. Лучший: квантильная P50 (-0.025).
Ни один подход не решил high demand problem кардинально.

---

## Tier 2c: High Demand Problem (Demand Target)

Те же 7 подходов (40-46), но обученные на Спрос вместо Продано.
Data: `daily_sales_8m_demand.csv`, Target: `Спрос`, Features: FEATURES_V2 + 11 demand lags.
Baseline: exp 03 Model B (MAE 2.9923, WMAPE 28.19% vs Спрос).

### 50. Tweedie (demand) ✅
- **Результат**: MAE 2.9798, WMAPE 28.07%. Delta -0.013. Маргинально.

### 51. Log target (demand) ✅
- **Результат**: MAE 2.9642, WMAPE 27.93%. Delta -0.028. Bias +0.85. Аналогично exp 41.

### 52. Asymmetric loss (demand) ❌ BROKEN
- **Результат**: MAE 10.61, WMAPE 100%. Та же проблема что exp 42 — gradient broken.

### 53. Quantile P50 (demand) ✅ BEST DEMAND
- **Результат**: P50 MAE 2.9437, WMAPE 27.73%. Delta **-0.049**. Coverage 48.9%. Лучший demand result.

### 54. Mixture of Experts (demand) ✅
- **Результат**: MAE 3.2865, WMAPE 30.96%. Delta +0.294. Хуже baseline, как и exp 44.

### 55. Log + residual (demand) ✅
- **Результат**: MAE 2.9897, WMAPE 28.17%. Delta -0.003. Stage 2 correction ухудшила (+0.026 MAE).

### 56. Variance-weighted (demand) ✅
- **Результат**: Best B (1/std) MAE 2.9907, WMAPE 28.18%. Delta -0.002.

### Demand Target Results Summary

| # | Experiment | MAE | WMAPE | Bias | R² | Delta MAE |
|---|---|---|---|---|---|---|
| 03 | **Baseline (demand)** | 2.9923 | 28.19% | +0.11 | 0.896 | — |
| 50 | Tweedie | 2.9798 | 28.07% | +0.21 | 0.895 | -0.013 |
| 51 | Log target | 2.9642 | 27.93% | +0.85 | 0.890 | -0.028 |
| 52 | Asymmetric loss | ~~10.61~~ | ~~100%~~ | — | -0.43 | BROKEN |
| **53** | **Quantile P50** | **2.9437** | **27.73%** | +0.74 | 0.892 | **-0.049** |
| 54 | MoE | 3.2865 | 30.96% | +0.70 | 0.777 | +0.294 |
| 55 | Log + residual | 2.9897 | 28.17% | +0.14 | 0.895 | -0.003 |
| 56 | Variance weighted | 2.9907 | 28.18% | +0.10 | 0.896 | -0.002 |

**Вывод demand экспериментов:**
- Рейтинг подходов идентичен sales (quantile > log > tweedie > variance ≈ log+residual > MoE > asymmetric).
- Quantile P50 на demand даёт delta -0.049 (вдвое больше чем на sales -0.025) — квантильная регрессия лучше работает с некупированным распределением спроса.
- Потолок demand MAE ~2.94. Ни один подход не решает high demand problem.

---

## Baseline V3 (новый reference)

### 60. Baseline V3 ✅ NEW BASELINE
Комбинация лучших находок:
- Target: Спрос (demand) — из exp 03
- Features: FEATURES_V3 = V2 + demand lags + price (58 фич) — из exp 09
- Objective: Quantile P50 (медиана) — из exp 53

**Результат**: MAE **2.8816**, WMAPE 27.15%, R2 0.895, coverage 48.8%
- Delta vs exp 03 (demand/MSE): **-0.111** — синергия price + quantile (каждый ~-0.05, вместе -0.11)
- price_vs_median (ранг 3) и price_change_7d (ранг 4) в топ-5 feature importance
- Prediction intervals: mean width 4.48, coverage 48.8% (~50% expected)
- vs Продано (для бизнес-оценки): MAE 2.5587, WMAPE 27.67%

**Проблема:** Bias +0.77 general, +15.49 на demand>=100. P50 завышает для high demand.
MSE на тех же фичах: MAE 2.936 — на high demand точнее (MAE 6.91 vs 7.23).

| # | Experiment | MAE (vs Спрос) | WMAPE | Delta vs exp03 |
|---|---|---|---|---|
| 03 | Baseline demand/MSE | 2.9923 | 28.19% | — |
| 53 | Quantile P50 | 2.9437 | 27.73% | -0.049 |
| 09 | Price features/MSE | 2.9364 | 27.66% | -0.056 |
| **60** | **V3 (price+P50)** | **2.8816** | **27.15%** | **-0.111** |

---

## Tier 2d: Additional Features

### 61. Censoring & behavioral features ✅ POSITIVE (+8 фич, MAE -0.0215)
Baseline: exp 60 V3 (MAE 2.8816, 58 фич).

**Результаты:**
| Вариант | Фичи | MAE | Delta |
|---------|------|-----|-------|
| baseline | 58 | 2.8816 | — |
| +A (censoring) | 61 | 2.8809 | -0.0007 |
| +A+B (dow) | 63 | 2.8662 | -0.0154 |
| +A+B+C (trend) | 65 | 2.8648 | -0.0168 |
| +A+B+C+D (stale) | 66 | 2.8601 | -0.0215 |

**Лучшие новые фичи по importance:**
- #6: demand_trend (5766)
- #10: cv_7d (4908)
- #17: demand_dow_mean (4360)
- #19: sales_dow_mean (4025)
- #33: lost_qty_lag1 (2660)
- #34: stale_ratio_lag1 (2583)

**Вывод:** Группа B (DOW means) дала основной прирост. is_censored_lag1 бесполезен (#59), но lost_qty_lag1 работает. High demand: MAE 22.13 vs 22.45 (-0.32).

**Группа A — Censoring features (из daily_sales_8m_demand.csv):**
- `is_censored` — товар продан в ноль вчера (бинарный). Сигнал: вчера был стоп → сегодня может быть отложенный спрос
- `lost_qty` — оценка упущенного спроса вчера (числовой). Масштаб потерь
- `pct_censored_7d` — % дней со стопом за последние 7 дней (rolling). Характеризует дефицитность товара

**Группа B — Day-of-week features:**
- `sales_dow_mean` — средние продажи для (пекарня × продукт × день_недели) за 30 дней. Понедельник ≠ пятница, модель видит ДеньНедели, но не кросс с историей
- `demand_dow_mean` — то же для спроса

**Группа C — Trend & volatility:**
- `demand_trend` — demand_roll_mean7 / demand_roll_mean30. Растёт/падает спрос? Модель видит оба числа, но ratio неявный
- `cv_7d` — sales_roll_std7 / (sales_roll_mean7 + 1). Коэффициент вариации — стабильный vs волатильный товар

**Группа D — Stale product (из сырых чеков):**
- `stale_ratio` — доля вчерашнего товара в продажах (из Свежесть). Высокая доля = перепроизводство, низкая = дефицит

Эксперимент: последовательное добавление групп A → A+B → A+B+C → A+B+C+D.
Каждая группа оценивается отдельно для определения вклада.

- Потенциал: средний (is_censored и sales_dow_mean — наиболее перспективны)
- Сложность: низкая (A, B, C из существующих колонок), средняя (D — парсинг чеков)

### 62. Adaptive demand profiles (DOW + scaling coefficient)
Улучшение фундамента — пересчёт таргета "Спрос" с адаптивными профилями.
Baseline: текущий `build_demand_profiles.py` (один профиль на пекарню×продукт для всех дней).

**Проблема текущего подхода:**
Один профиль на все дни: понедельник и суббота используют одну кривую, хотя:
- Будни: пик 8-9 (завтраки), обед 12-13
- Выходные: пик позже 10-11, равномернее
- Пятничный вечер ≠ вторничный вечер

**Решение A — DOW-разбивка профилей:**
Fallback chain:
1. (пекарня, продукт, день_недели) — если >= 3 полных дня для этого DOW
2. (пекарня, продукт, будни/выходные) — если мало на конкретный DOW, >= 5 полных дней
3. (пекарня, продукт) — текущий вариант
4. Нет профиля → demand = sales

**Решение B — Скользящий коэффициент масштабирования:**
Абсолютный профиль даёт **форму**, а последняя неделя — **уровень**.
- Для каждого (пекарня, продукт, дата): берём продажи за последние 7 дней
- Сравниваем фактические продажи к часу стопа с ожидаемыми по профилю
- Коэффициент = actual_at_stop / expected_at_stop (усреднённый за 7 дней)
- Адаптированный спрос = базовый профиль × коэффициент

Выигрыш: устойчивость (среднее за неделю vs один день), ловит тренды ("этот товар в последнюю неделю +20% к историческому профилю"), работает даже без полных дней за неделю.

**Эксперимент:**
1. Пересчитать `demand_profiles.json` с DOW-разбивкой (A)
2. Пересчитать `daily_sales_8m_demand.csv` с новыми профилями (A)
3. Добавить скользящий коэффициент (B) → ещё одна версия `Спрос`
4. Обучить V3 модель на каждой версии, сравнить MAE

Варианты оценки: baseline (текущий Спрос) → A (DOW profiles) → B (DOW + scaling) → оба.

- Потенциал: **высокий** (улучшение таргета влияет на всю модель, а не только на одну фичу)
- Сложность: средняя (рефактор build_demand_profiles, пересчёт 30M чеков)

### 66. Cluster features (exp63 + location/TS clusters) ✅ POSITIVE
Baseline: exp 63 combined (MAE 2.8540, 69 фич). Идея: не только кластерные модели как в exp 23, а кластеры как дополнительные сигналы поверх сильного feature set exp 63.

Пять вариантов:
- A: exp63 baseline
- B: A + `cluster_loc` (кластер пекарни по локационным фичам)
- C: A + `cluster_ts` (кластер пары bakery×product по поведению ряда)
- D: A + оба кластера
- E: routed model по `cluster_ts` (отдельная модель на каждый TS cluster)

**Результаты:**
| Вариант | Фичи | MAE | Delta vs exp63 |
|---------|------|-----|----------------|
| A exp63 baseline | 69 | 2.8579 | +0.0039 |
| B + cluster_loc | 70 | 2.8563 | +0.0023 |
| C + cluster_ts | 70 | 2.8587 | +0.0047 |
| D + both clusters | 71 | 2.8561 | +0.0021 |
| **E routed cluster_ts** | **71** | **2.8469** | **-0.0071** |

Дополнительные наблюдения:
- `cluster_loc` как обычная фича почти нейтрален
- `cluster_ts` как обычная фича тоже почти нейтрален
- routing по `cluster_ts` даёт реальный прирост и стал лучшим вариантом
- по high demand улучшение ограниченное: demand>=100 MAE 21.81 vs 22.13 у exp63, но demand>=200 не улучшился

Структура TS-кластеров при K=3:
- cluster 0: 5,177 пар, mean demand 4.23, MAE 1.41
- cluster 1: 5,818 пар, mean demand 24.32, MAE 5.33
- cluster 2: 15,748 пар, mean demand 4.33, MAE 1.83

Метаданные кластеризации:
- location clusters: K=3, silhouette=0.447
- TS clusters: K=3, silhouette=0.226

**Вывод:** Кластеризация как routing-сигнал работает лучше, чем кластеризация как обычная categorical feature. Это подтверждает гипотезу из exp 23: segment-aware modeling полезен, но не в виде грубого "одна глобальная модель + cluster id". Следующее разумное направление — развивать routing/mixture strategy именно для demand-target, а не пытаться просто добавлять cluster labels в таблицу.

---

## Tier 3: Ensembles & Chains

### 10. Stacking (LightGBM + CatBoost + Ridge) ✅ NEGATIVE
Level 1: LightGBM P50 + CatBoost P50 + Ridge, 3-fold OOF. Level 2: Ridge meta-learner.
Target: Спрос, FEATURES_V3.
- **Результат**: Stacking MAE 2.9221 — хуже чем LightGBM один (2.8801). Meta-веса: LGBM=1.065, CatBoost=0.040, Ridge=-0.090 — мета-модель по сути игнорирует CatBoost и Ridge. На high demand стекинг лучше (>=100: -1.00 MAE), но портит массовые мелкие позиции. Модели слишком похожи — одни фичи, недостаточно разнообразия.
- Потенциал: **нулевой** при текущей конфигурации
- Сложность: высокая (50 минут на 3-fold × 3 модели)

### 11. Residual Chain
Model 1 (structure): calendar + bakery/product averages -> pred1
Model 2 (dynamics): lags + rolling на residual (y - pred1) -> pred2
Model 3 (context): weather + holidays на residual -> pred3
Final: pred1 + pred2 + pred3
- Потенциал: высокий
- Сложность: средняя

### 12. Mixture of Experts (demand-level routing)
High demand (>15): deep LightGBM. Medium (3-15): standard. Low (<3): zero-inflated.
- Потенциал: высокий (решает разброс MAE по категориям)
- Сложность: средняя

### 13. Temporal Ensemble
Short (30d) + Medium (90d) + Long (all) models, weighted average.
- Потенциал: средний
- Сложность: низкая

---

## Tier 4: Per-Segment Models

### 20. Per-bakery models
199 отдельных моделей (или топ-50 крупных + одна общая для остальных).
Гипотеза: у каждой пекарни свой паттерн, общая модель — компромисс.
- Потенциал: средний (мало дней на пекарню: 207 дней × N товаров)
- Сложность: средняя
- Риск: переобучение на малых пекарнях

### 21. Per-product (номенклатура) models
Отдельная модель на каждый из 610 товаров или топ-100 + общая.
Гипотеза: Самса и Капучино — разные паттерны спроса.
- Потенциал: низкий-средний (мало данных на товар: 207 дней × N пекарен)
- Сложность: средняя
- Риск: высокий — у редких товаров < 100 строк

### 22. Per-category models (re-test on 8m)
27 отдельных моделей. На 3m/5cat было хуже (+0.04 MAE).
На 8m данных больше — стоит проверить заново.
- Потенциал: средний
- Сложность: низкая

### 23. Cluster-based models ✅ POSITIVE (+MAE -0.0233)
Baseline: exp 60 V3 (MAE 2.8816, 58 фич).
K-Means кластеризация по профилю (avg demand, cv, seasonality).
- **Результаты:**
| Вариант | MAE | Delta |
|---------|-----|-------|
| baseline | 2.8816 | — |
| K=5 clusters | 2.8583 | **-0.0233** |

- Silhouette: 0.279
- Cluster 2 (high demand, avg=59.3): MAE 10.57 — хуже чем baseline (ухудшило)
- Cluster 3 (low demand, avg=3.9): MAE 1.38 — значительное улучшение
- Cluster 1 (low/medium): MAE 1.66
- Cluster 4 (medium): MAE 4.08
- Cluster 0 (low): MAE 2.59

**Вывод:** Cluster-based модели дают прирост на low/medium demand сегментах, но страдает high-demand кластер. Стратегия: оставить единую модель для high-demand, применить кластерные для остальных. Или K=3 (объединить 0+1+3 в один cluster, 4 в medium).

**Методология (Dodo approach):**
1. Агрегация: (пекарня × продукт) → временной ряд
2. Фичи кластеризации: avg_demand, std_demand, cv, dow_pattern (вектор 7 значений), trend (first/last ratio)
3. K-Means с scaler, silhouette для выбора K
4. Обучение: отдельная модель на кластер (или единая модель с cluster ID как фича)

- Потенциал: **высокий** (Dodo claim: +5-7% WAPE)
- Сложность: средняя

### 24. City-level models
9 городов. Казань (115 пекарен) — отдельная модель,
остальные — общая или по группам.
- Потенциал: низкий-средний
- Сложность: низкая

### 25. Hierarchical models + reconciliation
Bottom-up / Top-down / Middle-out / MinT optimal reconciliation.
Уровни: Город → Пекарня → Категория → Товар.
- Потенциал: средний (+ бизнес-ценность: согласованные прогнозы)
- Сложность: высокая

---

## Tier 5: Deep Learning

### 30. LSTM baseline
Seq2seq LSTM: окно 30 дней → прогноз 7 дней.
Input: lag-фичи + calendar + weather (per bakery×product series).
Библиотека: PyTorch.
- Потенциал: средний (обычно хуже бустинга на табличных данных)
- Сложность: высокая
- Нужно: GPU, нормализация, embedding для категорий

### 31. GRU + Attention
GRU легче LSTM, + attention mechanism на временное окно.
Может лучше ловить "важные" дни в истории (праздники, аномалии).
- Потенциал: средний
- Сложность: высокая

### 32. N-BEATS (Neural Basis Expansion)
Специализированная DL-архитектура для временных рядов (Oreshkin et al. 2019).
Стеки: trend + seasonality + generic. Не требует фичей — чистые ряды.
Библиотека: pytorch-forecasting или darts.
- Потенциал: средний-высокий
- Сложность: средняя (готовые реализации)

### 33. Temporal Fusion Transformer (TFT)
Google 2019. Лучшая DL-модель для multi-horizon forecasting.
Умеет: static covariates (пекарня, товар), known future (календарь, погода),
observed past (лаги). Interpretable attention.
Библиотека: pytorch-forecasting.
- Потенциал: высокий
- Сложность: высокая
- Фишка: встроенная интерпретируемость (variable selection, attention weights)

### 34. DeepAR (Amazon)
Probabilistic forecasting — выдаёт распределение, не точку.
Авторегрессивная RNN, обучается на всех сериях одновременно (global model).
Аналог того что делает Foodforecast.
Библиотека: pytorch-forecasting или GluonTS.
- Потенциал: высокий (+ квантили бесплатно)
- Сложность: средняя-высокая

### 35. PatchTST / iTransformer (2023-2024 SOTA)
Современные трансформеры для временных рядов.
PatchTST: патчи вместо отдельных точек, channel-independent.
iTransformer: inverted — attention поверх переменных, не времени.
- Потенциал: высокий (SOTA на бенчмарках)
- Сложность: высокая
- Нужно: GPU, много данных (у нас есть)

### 36. LightGBM + LSTM hybrid
LightGBM для основного прогноза, LSTM для моделирования residual
(ловит нелинейные временные паттерны которые бустинг пропускает).
- Потенциал: высокий
- Сложность: высокая

---

## Recommended Sequence

**Phase 1 — Quick wins (tuning + features):**
02 (Optuna) → 03 (5 cat filter) → 04 (Ramadan) → 05 (profiles) → 08 (locations)

**Phase 2 — High demand problem:**
40 (Tweedie) → 41 (log-target) → 42 (asymmetric) → 43 (quantiles) → 44 (MoE) → 45 (log+residual)

**Phase 3 — Ensembles:**
10 (Stacking) → 11 (Residual Chain) → 12 (MoE full) → 23 (clusters)

**Phase 4 — Per-segment:**
22 (per-category) → 23 (clusters) → 20 (per-bakery) → 25 (hierarchical)

**Phase 5 — Deep Learning:**
32 (N-BEATS) → 33 (TFT) → 34 (DeepAR) → 35 (PatchTST) → 36 (Hybrid)

**Phase 6 — Advanced:**
30 (LSTM) → 31 (GRU+Attention) → 25 (Hierarchical reconciliation)

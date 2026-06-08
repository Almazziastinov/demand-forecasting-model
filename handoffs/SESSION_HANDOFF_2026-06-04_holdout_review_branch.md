# Session Handoff - 2026-06-04 - Holdout Review (Bakery + SKU)

## Context

Параллельная ветка работы, не связанная с продовым redeploy'ем VibeCode (см.
`SESSION_HANDOFF_2026-06-04_vibecode_redeploy_smoke.md`). Пользователя смутили
агрегатные метрики из weather-rollout (MAE −6, WMAPE −0.64 п.п.), и он
запросил детальный визуальный разбор: «прогноз vs факт за последние 30 дней,
сначала по пекарням, затем по SKU внутри пекарен».

После обсуждения договорились идти от готовых артефактов, без ретрейнов:

1. **Bakery**: использовать готовые holdout-предикты weather-ретрейна
   (окно 2026-05-02..05-31, 30 дней, 187 пекарен).
2. **SKU**: построить детерминированную аллокацию `bakery_day_forecast`
   через `share-профили (raw + smoothed)`, сравнить с фактом из чеков
   (`actual_sku_daily_clickhouse_eval30d.csv`).

В прод ничего не катилось. Это **чисто аналитика на существующих файлах**.

## New Artifacts (untracked)

### Scripts

```text
scripts/build_holdout_30d_compare.py
scripts/build_holdout_sku_compare.py
```

### Reports

```text
reports/holdout_30d_bakery_compare.csv         5,474 rows  1.1 MB
reports/holdout_30d_bakery_charts/             187 PNG
reports/holdout_sku_compare.csv                526,446 rows  ~75 MB
reports/holdout_sku_charts/bakery_NNNN/        182 dirs × top-20 SKU = 3,640 PNG
```

### Notebooks

```text
notebooks/bakery_holdout_30d_review.ipynb
notebooks/sku_holdout_review.ipynb
```

### Reference

```text
data/processed/product_lookup.csv              product_id -> product_name, category_name
```

## Bakery Layer (30 days, 2026-05-02..05-31)

### Pipeline

`scripts/build_holdout_30d_compare.py` склеивает два готовых артефакта
weather-ретрейна:

```text
reports/bakery_day_model_holdout_predictions.csv          (base target = "Продано")
reports/bakery_day_model_uplifted_holdout_predictions.csv (uplifted target)
```

Каждый из них уже содержит `(date, bakery_id, bakery_name, city,
bakery_sales, bakery_day_forecast)`. Скрипт переименовывает колонки,
делает outer-join по `(date, bakery_id)`, добавляет `err_*` и `abs_err_*`.

### Numbers (sanity vs `bakery_day_model_summary.json` — бит в бит)

```text
rows:            5,474
bakeries:        187          (из 217 в обучении — 30 без наблюдений в окне)
date range:      2026-05-02 .. 2026-05-31

base track:
  MAE   = 97.1842
  WMAPE = 10.2906 %
  bias  = -21.27       (модель в среднем занижает)

uplifted track:
  MAE   = 137.3487
  WMAPE = 12.0341 %
  bias  = -26.79
```

### Notebook structure (`bakery_holdout_30d_review.ipynb`)

1. Загрузка + sanity
2. Aggregate-метрики
3. WMAPE по дням окна
4. По городам
5. Per-bakery рейтинг (top-15 worst, over-, under-forecast)
6. Графики: 4 линии (fact_base, forecast_base, fact_uplifted, forecast_uplifted)
   - по умолчанию top-20 худших по WMAPE base в ноутбуке
   - все 187 пекарен сохраняются в PNG
7. Конкретный bakery_id (по умолчанию 114 = Энергетиков 3 Казань)
8. Top-5 лучших для контраста

### Key per-bakery findings (top of by_bakery)

```text
under-forecast (модель занижает):
  250 Вокзальная 1 Курск             bias +351   WMAPE base 56.7% — резкий рост трафика
  273 Универсиады 10 Казань           bias +282   WMAPE 25.1%
  272 Четаева 4 Казань                bias +197   WMAPE 19.7%
  77  Гагарина 17Б Чебоксары          bias +101
  89  Парина 6 Казань                 bias +95

over-forecast (модель завышает):
  270 Восточная 1к1 Новочебоксарск    bias -147   WMAPE 31.9% — обороты падают
  18  Хади Такташ 105 Казань          bias -58
  266 Мусина 61В Казань               bias -57
  30  Баумана 29/11 Казань            bias -45
  142 Баки Урманче 6 Казань           bias -44
```

Интерпретация: новые/выросшие пекарни модель догоняет с лагом
(`bakery_sales_lag*` смотрит в прошлое); угасающие — тоже с лагом.
Это ожидаемое поведение для модели с lag-фичами, не баг.

## SKU Layer (11 days overlap, 2026-05-02..05-12)

### Why 11 days

bakery-holdout window = 05-02..05-31 (30 дней).
SKU-fact (`actual_sku_daily_clickhouse_eval30d.csv`) = 04-13..05-12 (30 дней).
Overlap = **2026-05-02..05-12 = 11 дней**.

Пользователь согласился делать MVP на 11 днях и решать, нужно ли
догружать SKU-факт за 05-13..05-31 из ClickHouse, после просмотра
графиков.

### Allocation math

Профили в `bakery_hour_profile.csv` и
`sku_hour_share_profile{,_smoothed}.csv` — hour-уровня. Чтобы получить
SKU-уровень дня без раздувания таблицы до часов:

```text
sku_day_share(bakery, sku, dow) =
    sum_hour( mean_hour_share_norm(bakery, dow, hour)
              * mean_sku_share_in_hour_norm(bakery, sku, dow, hour) )

sku_day_forecast(date, bakery, sku) =
    bakery_day_forecast(date, bakery) * sku_day_share(bakery, sku, dow(date))
```

Sanity: `sku_day_share` по продуктам в (bakery, dow) суммируется ровно
в 1.0000 для всех (bakery, dow) (min=mean=max=1.0).

### Fact rescaling — IMPORTANT

`bakery_fact_base` ("Продано") и SKU-fact (`quantity` из чеков) —
**разные метрики**:

```text
window 2026-05-02..05-12 (11d):
  sum bakery fact_base       1,797,364
  sum bakery forecast_base   1,869,856
  sum SKU fact (raw)         2,024,437   <- 12.6% больше fact_base!
  sum bakery fact_uplifted   2,145,099
```

Скрипт ремасштабирует SKU-fact per (date, bakery) так, чтобы
`sum(sku_fact) == bakery_fact_base` на этом (date, bakery). Это
сохраняет shape распределения по SKU из чеков, но приводит уровень в
ту же метрику, в которой работает прогноз. Без этого WMAPE
систематически плыл бы на 12% независимо от качества аллокации.

### Numbers

```text
rows:               526,446
bakeries:           182
products:           1,145
days:               11

sum fact:           1,790,927   (rescaled to fact_base level)
sum forecast raw:   1,868,890
sum forecast smooth:1,868,890   (same — sum is invariant to profile choice)
```

ALL rows:
```text
WMAPE raw:       100.7743 %
WMAPE smoothed:   99.5003 %
bias raw:        +0.148   (≈ 0)
bias smoothed:   +0.148
```

Only `fact > 0` (исключая мёртвые корзины):
```text
WMAPE raw:        59.37 %
WMAPE smoothed:   59.94 %
bias raw:         -2.93   (систематическая недооценка на живых SKU)
bias smoothed:    -2.78
```

### Cell structure (главное наблюдение)

```text
                          rows      % rows    % abs_err
both > 0 (real):         226,600    43.0%     58.9%
forecast > 0, fact = 0:  299,840    57.0%     41.1%    <- мёртвые корзины
forecast = 0, fact > 0:        0     0.0%      0.0%    <- profile никогда не пропускает SKU
both = 0:                      6     0.0%      0.0%
```

### "Dead bucket" leakage

```text
(bakery, sku) пар с фактом = 0 все 11 дней
  и заметным forecast (>5):                15,835
сумма forecast по ним:                     500,615 ед
% от всего forecast:                        26.8 %
```

То есть **четверть** bakery-прогноза «утекает» в SKU, которые в этой
пекарне в окне ни разу не продавались. Скорее всего это:

- SKU убраны из ассортимента, но остались в долгой истории профиля
- сезонные SKU вне сезона
- SKU мигрировали между пекарнями

### Top dead examples

```text
b=262 (Харьковская 3 Курск)   sku=11469   forecast ~120 ед/день, fact=0
b=232                         sku=11021   forecast ~97
b=110                         sku=10082   forecast ~78
```

### Top "both > 0" errors (модель недодаёт на flagman SKU)

```text
b=242  sku=11018   forecast    4 vs fact  293   delta +289
b=30   sku=1076    forecast   45 vs fact  315   delta +270
b=30   sku=1071    forecast   58 vs fact  313   delta +255
b=79   sku=1071    forecast   86 vs fact  330   delta +244
```

Bakery 30 (Баумана 29/11 Казань) недополучает на flagman-выпечке
(Вак-бэлиш и т.п.) ~270 ед/день на каждом из топ-SKU.

### Smoothed vs raw

Smoothed профиль почти не отличается от raw:

```text
ALL WMAPE:   100.77% raw → 99.50% smoothed     (delta -1.27 п.п.)
fact>0 MAE:    4.69   raw →  4.74    smoothed   (smoothed чуть хуже на ненулевых)
fact>0 bias:  -2.93   raw → -2.78    smoothed   (smoothed чуть меньше недодаёт)
```

Сглаживание профиля **не решает** главную проблему — раздачу прогноза
на мёртвые корзины. Структурно raw и smoothed эквивалентны.

### Notebook structure (`sku_holdout_review.ipynb`)

1. Загрузка + sanity
2. Cell-структура (fact × forecast buckets)
3. Aggregate-метрики (ALL / fact>0 / both>0)
4. По дням окна
5. По категориям
6. **Мёртвые SKU** (top-30 по фиктивному forecast)
7. Топ-20 SKU по факту цепочки целиком
8. **Графики**: per-bakery top-20 SKU по mean_fact, 3 линии
   (fact, forecast_raw, forecast_smoothed)
   - по умолчанию ноутбук рисует `SHOW_FOR_BAKERIES = [114, 30, 250]`
   - все 3640 PNG сохраняются в `reports/holdout_sku_charts/bakery_NNNN/`
9. Конкретная пара (BAKERY_ID=30, PRODUCT_ID=1076 — Вак-бэлиш как пример)
10. Per-bakery WMAPE SKU rating

## Why None of This Is Committed

Вся работа — debug / discovery branch. Артефакты крупные
(`holdout_sku_compare.csv` ~75 MB, 3640 PNG). До решения, что с этим
делать дальше (см. ниже), коммит не делается.

В `.gitignore` уже есть `reports/bakery_day_model*_holdout_predictions.csv`
и `models/*.joblib`. Если решим коммитить часть результатов — нужно
добавить ignore-паттерны для:

```text
reports/holdout_30d_bakery_compare.csv
reports/holdout_30d_bakery_charts/
reports/holdout_sku_compare.csv
reports/holdout_sku_charts/
data/processed/product_lookup.csv
```

А скрипты и ноутбуки безопасно коммитить как есть.

## Open Threads / Next Steps

Из этой ветки естественно вытекают три задачи. Они **не привязаны** к
другой ветке (VibeCode redeploy), решаются отдельно.

1. **Sparse-profile filter** (highest leverage). Перед умножением
   `bakery_day × sku_share` отсекать SKU с
   `recent_n_days_sold < N` или `zero_share_rate > X`. Эти колонки уже
   есть в `sku_hour_share_profile.csv` (см. `recent_n_days`,
   `zero_share_rate`). Гипотеза: уберёт большую часть «мёртвых корзин»
   и автоматически перебросит ~500K ед на живые SKU. Должно дать
   заметное улучшение SKU WMAPE без переобучения bakery-модели.

2. **Profile freshness**. Текущий профиль построен на длинной истории
   (видно по `n_days` в профиле). Для ассортиментных изменений нужен
   short-window профиль (последние 30-60 дней) с приоритетом перед
   long-history. Альтернатива (1).

3. **Догрузить SKU-факт за 2026-05-13..05-31** из ClickHouse
   (`mart_zero_sales_60d` или аналогичный mart, который использовался
   при сборке `actual_sku_daily_clickhouse_eval30d.csv`). Это
   расширит SKU-окно с 11 до 30 дней, даст более устойчивые цифры по
   мёртвым корзинам и dow-паттернам. Не критично для выбора (1)/(2).

4. **Flagman SKU under-forecast** (b=30 case). На крупных позициях
   модель недодаёт в разы. Профиль раздаёт долю «слишком справедливо»,
   не учитывая что в конкретной пекарне один SKU доминирует. Это
   отдельный класс ошибки от мёртвых корзин — здесь профиль занижает
   долю, а не раздаёт лишнее.

## Useful Commands For Resuming

Пересобрать bakery CSV:
```powershell
.venv\Scripts\python.exe scripts\build_holdout_30d_compare.py
```

Пересобрать SKU CSV (требует `bakery_hour_profile.csv` и оба
`sku_hour_share_profile*.csv`; занимает ~30 сек памяти ~3 GB):
```powershell
.venv\Scripts\python.exe scripts\build_holdout_sku_compare.py
```

Headless-прогон обоих ноутбуков (для регенерации PNG без Jupyter):
```powershell
.venv\Scripts\python.exe -c "import json,io,contextlib,matplotlib; matplotlib.use('Agg'); ..."
```

См. fix-pattern в этом репо (запускается `exec(combined)` с заменой
`plt.show() -> plt.close()` и `display() -> print()`).

# Техническая реализация плана выпекания

Последнее обновление: 2026-07-06

Бизнес-правила — в `baking_plan_rules.md`. Этот документ описывает **как именно** это реализовано в коде.

---

## 1. Компоненты системы

```
ClickHouse
  sku_forecast_hour_embedded   ← почасовой прогноз SKU
  bakeable_products            ← разрешённый ассортимент выпекания
  mart_sales_60d               ← факт продаж (источник для ассортимента)

apps/forecast_embedded/app/
  routers/ui.py                ← HTTP endpoint /bakery/{id}/baking-plan.xlsx
  services/baking_plan.py      ← генерация Excel (алгоритм окон)
  services/bakery.py           ← get_bakeable_products()
  assets/baking_plan_template.xlsx          ← базовый шаблон
  assets/baking_plan_individual/{id}.xlsx   ← шаблоны конкретных пекарен
```

---

## 2. Алгоритм выбора окон выпекания

### 2.1 Входные данные

- **sku_hour_rows** — почасовой прогноз на дату плана (из `sku_forecast_hour_embedded`)
- **next_day_sku_hour_rows** — почасовой прогноз следующего дня (для дефростов)
- **sku_meta** — метаданные SKU из листа "комментарии" шаблона
- **sheet_windows** — доступные окна выпекания из строки 5 шаблона

### 2.2 Разбор листа "комментарии" — `parse_comments_sheet(wb)`

Лист "комментарии" присутствует во всех шаблонах (base + индивидуальных).  
Колонки:

| Колонка | Содержимое |
|---|---|
| B | Название SKU (или название группы — заголовок секции) |
| C | Кратность выпуска (число или "по запросу") |
| D | "двухдневка" если SKU является двухдневкой |

Строка считается **заголовком группы** если col C пустая или равна "кратность выпуска".  
Все последующие SKU до следующего заголовка принадлежат этой группе.

Результат: `dict[sku_name → SkuMeta]`

```python
@dataclass
class SkuMeta:
    dough_group: str    # тесто-группа из "комментарии"
    kratnost: int       # кратность выпуска (или 1 если "по запросу")
    is_two_day: bool    # признак двухдневки
    is_on_demand: bool  # признак "по запросу"
```

**Нормализация имён:** все имена приводятся к нижнему регистру, убираются двойные пробелы — для устойчивого fuzzy-матчинга.

### 2.3 Агрегация профиля по тесто-группе — `_build_group_hourly()`

Для каждой тесто-группы суммируем прогноз по часам по всем входящим SKU:

```python
group_hourly: dict[group_name → dict[hour → qty]]
```

### 2.4 Определение пиков — `_detect_peaks(hourly)`

Час H считается значимым пиком при выполнении трёх условий:
1. Локальный максимум: `hourly[H] > hourly[H-1]` и `> hourly[H+1]`
2. Prominence ≥ 1.2: `hourly[H] / min(left_trough, right_trough) ≥ 1.2`
3. Доля ≥ 5% суточного объёма

Если значимых пиков нет — берём час с максимальным значением.

### 2.5 Кластеризация пиков — `_cluster_peaks(peaks)`

Пики объединяются в кластер если зазор между ними ≤ 3 часа (`_PEAK_MERGE_HOURS = 3`).  
Центроид кластера = среднее арифметическое часов входящих пиков.

### 2.6 Выбор окна по центроиду — `_centroid_to_window(centroid, windows)`

Из доступных окон шаблона выбирается то, чей `end_hour ≤ centroid`, ближайшее снизу.  
Логика: изделие должно выйти из печи до пика продаж.

### 2.7 Расписание группы — `_compute_group_windows()`

Для каждой тесто-группы: один или два кластера → одно или два окна.  
Результат: `dict[group_name → list[BakingWindow]]`

### 2.8 Расписание SKU — `_build_sku_schedule()`

```python
def _build_sku_schedule(sku_name, sku_meta, group_windows, sheet_windows, has_next_day):
```

1. Находим `meta = sku_meta[sku_name]` (нормализованный матч)
2. Берём окна группы: `bake_windows = group_windows[meta.dough_group]`
3. Если SKU — двухдневка И есть прогноз следующего дня:
   - Добавляем дефрост в **последнее окно** шаблона (конец смены)
   - Это окно исключается из списка обычных окон выпечки
4. Для каждого bake_window добавляем `ScheduledColumn(is_defrost=False)`
5. Возвращаем список `ScheduledColumn`

**Фоллбэк:** если SKU не найден в sku_meta ИЛИ его группа не имеет прогнозных данных — расписание берётся из предзаполненных ячеек шаблона (`read_row_schedule()`).

### 2.9 Расчёт количества — `allocate_template_row()`

Для каждого окна в расписании:
- Покрываемые часы = от конца предыдущего окна до конца текущего
- Дефрост: покрываемые часы = ранние часы следующего дня (6–11)
- `qty = sum(forecast[hour] for hour in covered_hours)`
- Округление вверх до `kratnost` (из sku_meta)

### 2.10 Фильтрация по ассортименту

После построения расписания каждая строка шаблона проверяется через `resolve_assortment_product()`.  
Если SKU отсутствует в `bakeable_products` для данной пекарни — строка пропускается (не попадает в итоговый файл).

---

## 3. Ассортимент выпекания (bakeable_products)

### 3.1 Структура таблицы

```sql
CREATE TABLE bakeable_products (
  city         LowCardinality(String),
  product_id   String,
  product_name String,
  category_name String,
  is_bakeable  UInt8,
  source       LowCardinality(String),
  source_file  String,
  scope        LowCardinality(String),   -- 'city' или 'bakery'
  bakery_id    Nullable(Int64),          -- NULL для city-строк
  valid_from   Date,
  valid_to     Nullable(Date),
  is_active    UInt8,
  loaded_at    DateTime64(3),
  comment      String
)
ENGINE = ReplacingMergeTree(loaded_at)
ORDER BY (city, product_id, scope, coalesce(bakery_id, -1), valid_from);
```

### 3.2 Два слоя ассортимента

**scope='city' — городской шаблонный ассортимент**
- Продукты, проданные в ≥ 80% пекарен города за последние 7 дней
- Стабильный core: не исчезает из-за одного дня без продаж в отдельной точке

**scope='bakery' — пекарня-специфичный ассортимент**
- Продукты, проданные в конкретной пекарне за последние 7 дней
- Только те, которых НЕТ в city-слое
- Улавливает уникальные позиции, новинки, сезонные эксперименты

### 3.3 Категорийный фильтр

Оба слоя фильтруются по категории (подстрочный матч, case-insensitive):
- `"пирог"` → Пироги сытные, Пироги сладкие
- `"выпечка"` → Выпечка сытная, Выпечка сладкая
- `"фастфуд"` → Фастфуд

Покупные позиции (кофе, напитки, суши и т.д.) автоматически исключаются.

### 3.4 Источник данных

Факт продаж: `mart_sales_60d` (ClickHouse)  
Скрипт: `scripts/build_city_assortment_from_sales.py`

```
mart_sales_60d (last 7 days)
  → агрегация: (city, bakery_id, product_id) с quantity > 0
  → city-слой: product sold in >= 80% of bakeries per city
  → bakery-слой: остаток (не вошедшие в city)
  → фильтр по категории
  → INSERT INTO bakeable_products
```

### 3.5 Период обновления

Пересчитывается еженедельно в рамках `refresh_production_datasets()` — до запуска инференса модели на VM.  
`valid_from` = дата пересчёта; новые строки не удаляют старые — ReplacingMergeTree дедуплицирует по `(city, product_id, scope, bakery_id, valid_from)` при FINAL-запросах.

### 3.6 Чтение ассортимента — `get_bakeable_products(city, date, bakery_id=None)`

```python
# ui.py вызывает с bakery_id:
bakeable_products = bakery_service.get_bakeable_products(city, date, bakery_id=bakery_id)
```

Запрос возвращает UNION:
- все `scope='city'` строки для города
- `scope='bakery'` строки где `bakery_id = <текущая пекарня>`

Параметр `valid_from` — берётся максимальный на дату прогноза.  
Если список пуст — выброс `RuntimeError` (фейл-закрытый режим, не пропускаем покупные).

---

## 4. Шаблоны

### 4.1 Базовый шаблон

`apps/forecast_embedded/app/assets/baking_plan_template.xlsx`

Содержит 4 листа (уровни выручки пекарни):
- `до 1,5 млн` — минимальный набор окон
- `до 2,5 млн`
- `от 2,5 млн`
- `от 3млн` — максимальный набор окон
- `комментарии` — метаданные SKU (группы, кратности, двухдневки)

Лист выбирается по выручке пекарни из `get_month_revenue_bucket()`.

### 4.2 Индивидуальные шаблоны

`apps/forecast_embedded/app/assets/baking_plan_individual/{bakery_id}_{name}.xlsx`

Применяются для пекарен, чей ассортимент или расписание существенно отличается от базового шаблона. Приоритет над базовым. Содержат тот же лист "комментарии".

Текущие индивидуальные шаблоны:
- `20_mira_45.xlsx` — Мира 45 (Дербышки)
- `21_parkovaya_7.xlsx` — Парковая 7
- `22_sibirskiy_trakt_25.xlsx` — Сибирский тракт 25

### 4.3 Порядок выбора шаблона

```python
template_path_for_bakery(bakery_id)
  → если существует individual/{bakery_id}_*.xlsx → взять его
  → иначе → базовый шаблон, лист по выручке
```

---

## 5. Пайплайн генерации файла

```
GET /bakery/{bakery_id}/baking-plan.xlsx?date=YYYY-MM-DD

ui.py:
  1. get_bakery_day()              → информация о пекарне (city, выручка)
  2. get_sku_hour()                → почасовой прогноз на дату
  3. get_sku_hour(date+1)          → прогноз следующего дня (для дефростов)
  4. get_month_revenue_bucket()    → уровень выручки → выбор листа шаблона
  5. get_bakeable_products(city, date, bakery_id)  → ассортимент выпекания
  6. template_path_for_bakery()    → выбор шаблона

baking_plan_service.build_baking_plan_workbook():
  7. parse_comments_sheet()        → SkuMeta для всех SKU
  8. _sheet_windows()              → доступные окна из строки 5 шаблона
  9. _build_group_hourly()         → профиль по тесто-группам
  10. _compute_group_windows()     → окна для каждой группы
  11. для каждой строки шаблона:
      a. _build_sku_schedule()     → список окон для SKU (или фоллбэк)
      b. resolve_assortment_product() → проверка по ассортименту
      c. allocate_template_row()   → расчёт количеств
      d. write_row_to_sheet()      → запись в Excel
  12. return .xlsx bytes
```

---

## 6. Параметры алгоритма

| Константа | Значение | Смысл |
|---|---|---|
| `_PEAK_MERGE_HOURS` | 3 | Максимальный зазор для объединения пиков в один замес |
| `_PEAK_MIN_SHARE` | 0.05 | Минимальная доля пика от суточного объёма (5%) |
| `_PEAK_MIN_PROMINENCE` | 1.2 | Минимальное превышение пика над соседними впадинами (20%) |
| `_FORECAST_HOURS` | 6–22 | Рабочие часы для агрегации прогноза |
| `ASSORTMENT_WINDOW_DAYS` | 7 | Окно факта продаж для ассортимента |
| `ASSORTMENT_CITY_THRESHOLD` | 0.80 | Минимальная доля пекарен для city-слоя |

---

## 7. Известные ограничения

- **Нет проверки ёмкости окна** — алгоритм выбирает окна по пикам, но не проверяет загрузку пекарей (кол-во шт × минут). Реализовано в бизнес-правилах, не в коде.
- **Числовые артефакты в "комментарии"** — группа "Проект пиццы большой" содержит строки с числовыми значениями в col B (цены/коды), которые парсятся как имена SKU и не матчатся ни с чем.
- **city-порог 80% не учитывает новые пекарни** — пекарня, открытая на прошлой неделе, занижает знаменатель и может убрать позиции из city-слоя. Пока не критично из-за редкости открытий.
- **Альметьевск** — не вошёл в `assortment_city_products.csv`, остаётся на legacy markup_xlsx данных в ClickHouse. Обновление вручную при необходимости.

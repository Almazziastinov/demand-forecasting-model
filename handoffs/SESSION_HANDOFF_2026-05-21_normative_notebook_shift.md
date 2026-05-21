# Session Handoff - 2026-05-21 - Normative Notebook Shift

## Goal

Сдвинуть работу по normative demand из тяжелых full-run экспериментов в более интерпретируемый notebook-first режим, сократить область анализа и подготовить удобную площадку для быстрых гипотез.

## What Was Done

### 1. Added exp76 baseline

Создан `src/experiments_v2/76_normative_v1_v2/run.py`.

Логика:
- `normative_v1 = trend + static weekday profile`
- `normative_v2 = trend + adaptive weekday amplitude`
- `normative_candidate` выбирается по сегменту

Добавлены тесты:
- `tests/test_normative_v1_v2.py`

Результат full run по panel:
- артефакты в `src/experiments_v2/76_normative_v1_v2/`
- вывод: `stable` выглядит рабочим, `bakery_driven` и часть других сегментов требуют отдельной логики

### 2. Added anchor analysis

Создан `src/analysis/normative_anchor_analysis.py`.

Назначение:
- оценить, чем потенциально должен держаться норматив
- сравниваются 4 опоры:
  - `self_pattern`
  - `release`
  - `bakery_scale`
  - `category_role`

Добавлены тесты:
- `tests/test_normative_anchor_analysis.py`

Артефакты full run:
- `reports/normative_anchor_analysis/anchor_profile_by_pair.csv`
- `reports/normative_anchor_analysis/anchor_summary_by_segment.csv`
- `reports/normative_anchor_analysis/anchor_dominance_by_segment.csv`
- `reports/normative_anchor_analysis/metrics.json`

Ключевой смысл:
- `stable` ближе к release-aware pattern
- `bakery_driven` часто держится на bakery scale
- `amplitude_unstable` и `intermittent` часто тяготеют к `category_role`

### 3. Added exp77 segmented constructors

Создан `src/experiments_v2/77_segmented_normative_constructors/run.py`.

Текущая реализация:
- `stable -> stable_release_weekday`
- `bakery_driven -> bakery_total_x_sku_share`

Добавлены тесты:
- `tests/test_segmented_normative_constructors.py`

Важно:
- скрипт был ужат по памяти
- читает только нужные колонки
- ориентирован только на `stable` и `bakery_driven`

### 4. Added exploratory notebooks

Созданы ноутбуки:
- `notebooks/normative_segmented_constructors_exp77.ipynb`
- `notebooks/normative_sitnaya_sample_analysis.ipynb`

#### `normative_segmented_constructors_exp77.ipynb`

Ноутбук для интерактивного прогона логики exp77.

По ходу работы выяснилось:
- full panel неудобен для ручной интерпретации
- часть `bakery_driven` пар имеет слишком короткую историю
- смотреть недельные ряды бессмысленно для содержательной оценки норматива

#### `normative_sitnaya_sample_analysis.ipynb`

Это новый рабочий фокус.

Настройки:
- категория: `Выпечка сытная`
- выборка: `35` пекарен
- фильтр по минимальной длине истории
- несколько простых нормативов в одном ноутбуке:
  - `norm_sales_weekday`
  - `norm_release_weekday`
  - `norm_blend_50_50`
  - `norm_bakery_share`
  - `norm_category_share`

Идея:
- не плодить отдельные `expXX`
- быстро проверять простые и интерпретируемые гипотезы на ограниченной выборке
- смотреть таблицы и графики прямо в notebook

## Key Findings

### About exp77

По сохраненным артефактам `reports/normative_segmented_constructors_exp77/`:

- `stable`:
  - форма конструктора выглядит осмысленной
  - корреляция с фактом и выпуском приличная
  - но уровень заметно завышен

- `bakery_driven`:
  - идея bakery-anchor подтверждается частично
  - но масштаб уходит слишком низко
  - дополнительно выяснилось, что много пар слишком короткие для уверенного анализа

### Important methodological shift

Пользователь явно сместил исследование в сторону:
- маленькой выборки вместо полного panel
- notebook-first instead of experiment-first
- простых аналитических гипотез вместо тяжелых вычислительных веток

Это разумный разворот. Следующую работу лучше продолжать именно из `notebooks/normative_sitnaya_sample_analysis.ipynb`.

## Recommended Next Step

Продолжать завтра из `notebooks/normative_sitnaya_sample_analysis.ipynb`:

1. Прогнать выборку `Выпечка сытная` / `35` пекарен.
2. Посмотреть, какой из 5 простых нормативов лучше держит:
   - форму
   - связь с выпуском
   - связь с bakery total
3. Выбрать 5-10 характерных пар.
4. Смотреть графики и вручную решать, какие механики действительно имеют смысл:
   - release-based
   - bakery-share
   - category-share
   - blend

## Git Scope For This Session

Релевантные новые файлы:
- `handoffs/SESSION_HANDOFF_2026-05-21_normative_notebook_shift.md`
- `notebooks/normative_segmented_constructors_exp77.ipynb`
- `notebooks/normative_sitnaya_sample_analysis.ipynb`
- `src/analysis/normative_anchor_analysis.py`
- `src/experiments_v2/76_normative_v1_v2/run.py`
- `src/experiments_v2/77_segmented_normative_constructors/run.py`
- `tests/test_normative_anchor_analysis.py`
- `tests/test_normative_v1_v2.py`
- `tests/test_segmented_normative_constructors.py`

Нерелевантные или тяжелые untracked файлы, которые лучше не коммитить автоматически:
- `data/raw/*.csv`
- измененные пользовательские ноутбуки `notebooks/bakery_day_backtest.ipynb`, `notebooks/hourly_sales_day.ipynb`
- большие generated csv/json, если не будет отдельного решения держать их в git

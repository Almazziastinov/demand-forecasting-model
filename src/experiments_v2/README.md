# Experiments V2

Основной исследовательский контур проекта для почековых данных за 8 месяцев
(август 2025 - апрель 2026).

## Данные

- Источник: `data/raw/sales_hrs_all.csv` (~30M строк)
- Основные колонки: дата продажи, дата времени чека, вид события по кассе,
  касса, торговая точка, категория, номенклатура, свежесть, цена, количество

## Структура

```text
experiments_v2/
  common.py              -- общие утилиты (загрузка, метрики, обучение)
  01_baseline_8m/        -- базовая модель на 8 месяцах
    run.py               -- скрипт эксперимента
    metrics.json         -- результаты (MAE, WMAPE, ...)
    predictions.csv      -- fact vs pred
  02_experiment_name/
    ...
```

## Запуск

```bash
.venv/Scripts/python.exe src/experiments_v2/01_baseline_8m/run.py
```

## Общие утилиты (`common.py`)

- `load_checks()` - загрузка 30M чеков
- `aggregate_daily()` - агрегация до дневного уровня
- `wmape()`, `print_metrics()`, `print_category_metrics()`
- `train_lgbm()`, `predict_clipped()`
- `save_results(exp_dir, metrics, predictions)` - сохранение артефактов эксперимента

## Эксперименты

| # | Название | Описание | MAE | WMAPE |
|---|---|---|---|---|
| 01 | baseline_8m | Базовая модель LightGBM на 8 месяцах | - | - |

## Research Notes

- [HYBRID_RESEARCH_PLAN.md](HYBRID_RESEARCH_PLAN.md) - план исследования гибридной модели, сегментации SKU и router-подхода


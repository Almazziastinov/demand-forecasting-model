# Experiments V2

Эксперименты на почековых данных (8 месяцев, авг 2025 — апр 2026).

## Данные
- Источник: `data/raw/sales_hrs_all.csv` (~30M строк)
- 9 колонок: Дата продажи, Дата время чека, Вид события по кассе, Касса.Торговая точка, Категория, Номенклатура, Свежесть, Цена, Кол-во

## Структура
```
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

## Общие утилиты (common.py)
- `load_checks()` — загрузка 30M чеков
- `aggregate_daily()` — агрегация до дневного уровня
- `wmape()`, `print_metrics()`, `print_category_metrics()`
- `train_lgbm()`, `predict_clipped()`
- `save_results(exp_dir, metrics, predictions)` — сохранение в папку эксперимента

## Эксперименты

| # | Название | Описание | MAE | WMAPE |
|---|----------|----------|-----|-------|
| 01 | baseline_8m | Базовая модель LightGBM на 8 мес | — | — |

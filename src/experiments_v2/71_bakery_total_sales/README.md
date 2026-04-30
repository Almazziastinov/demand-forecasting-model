# Experiment 71: Bakery Total Sales

Эксперимент на уровне `Пекарня x Дата` с таргетом `Продано`.

Сравниваются три семейства моделей:

- `global_lgbm` — одна общая модель на все пекарни
- `per_bakery_lgbm` — отдельная LightGBM-модель на каждую пекарню
- `per_bakery_prophet_holidays` — отдельная Prophet-модель на каждую пекарню

Prophet-конфигурация взята из `68_prophet_benchmark`:

- `prophet_holidays`
- `changepoint_prior_scale = 0.05`
- `seasonality_prior_scale = 10.0`
- `seasonality_mode = additive`
- `weekly_seasonality = True`
- `yearly_seasonality = False`

Источник данных:

- `data/processed/daily_sales_8m_demand.csv`

Ряд агрегируется из товарного уровня до суммарных продаж точки за день.

Оценка по умолчанию:

- holdout = последние `30` дней
- в `predictions_*.csv` сохраняются дневные предикты по каждой пекарне на тестовом месяце

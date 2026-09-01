# Demand Forecasting Model

Исследовательский репозиторий для прогноза спроса на продукцию пекарен.

Проект сейчас находится не в стадии продуктовой стабилизации, а в стадии проверки гипотез. Поэтому в репозитории сосуществуют несколько контуров с разным статусом зрелости.

## Текущий статус

- Production-модель: Direct alpha=.25, `model_version=direct_alpha_025_v1`,
  active run pattern `prod_direct_alpha_025_YYYYMMDD_h14`.
- Каноническое описание живого состояния: `docs/ops/CURRENT_STATE.md`.
- Основной исследовательский контур: `src/experiments_v2/`
- Legacy baseline pipeline: `src/preprocessing.py`, `src/models/train_and_save.py`, `run_pipeline.py`
- Demo UI: `web/`

## Active production forecast

The current production forecast uses the bakery-day LightGBM volume followed
by Direct daily allocation across mature SKUs. It does not inherit old category
totals or use the legacy hourly SKU profile for allocation. Predictive uplift,
Core-SKU protection, alpha `.25`, adaptive floor and a causal tail cap are part
of the selected model; cold-start SKUs are handled by an independent path.
Hourly values are derived only after the daily SKU forecast is finalized.

Do not treat `base_norm_recent` as the current model: its nightly run is an
inactive source stage for Direct. Operational details, verification and
rollback are in [`docs/ops/CURRENT_STATE.md`](docs/ops/CURRENT_STATE.md) and
[`docs/ops/RUNBOOK.md`](docs/ops/RUNBOOK.md).

Подробное разделение ролей описано в [PROJECT_STATUS.md](PROJECT_STATUS.md).

## Что считать актуальным

Если задача связана с текущими гипотезами, качеством модели, feature engineering или сравнением результатов, ориентироваться нужно в первую очередь на `src/experiments_v2/`.

Если задача связана с историческим baseline или старым пайплайном подготовки данных, смотреть на `src/` и `run_pipeline.py`, но считать их вспомогательной legacy-веткой.

Если задача связана с интерфейсом и демонстрацией сценария использования, смотреть на `web/`, но не трактовать этот код как production-ready или как источник истины по архитектуре проекта.

## Основные точки входа

- Legacy pipeline: `python run_pipeline.py`
- Отдельные исследовательские запуски: `python src/experiments_v2/<experiment>/run.py`
- Demo web: `python web/app.py`

## Структура высокого уровня

- `src/` - базовые модули, preprocessing, monitoring, training, старые эксперименты
- `src/experiments_v2/` - текущий основной исследовательский трек
- `web/` - демонстрационный интерфейс для просмотра прогнозов
- `tests/` - инфраструктурные и частично legacy-тесты
- `models/` - сохраненные модели и метаданные
- `reports/` - отчеты, prediction dumps, summaries

## Как использовать этот репозиторий сейчас

- Для новых ML-гипотез: продолжать работу в `src/experiments_v2/`
- Для сравнения с историческим baseline: использовать legacy pipeline как контрольную точку
- Для демонстрации интерфейса: использовать `web/` без требования полного соответствия исследовательскому контуру

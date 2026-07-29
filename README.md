# Demand Forecasting Model

Исследовательский репозиторий для прогноза спроса на продукцию пекарен.

Проект сейчас находится не в стадии продуктовой стабилизации, а в стадии проверки гипотез. Поэтому в репозитории сосуществуют несколько контуров с разным статусом зрелости.

## Текущий статус

- Основной исследовательский контур: `src/experiments_v2/`
- Legacy baseline pipeline: `src/preprocessing.py`, `src/models/train_and_save.py`, `run_pipeline.py`
- Demo UI: `web/`

## Active pilot baking-plan correction

The daily Bitrix24 baking-plan publisher for the 10-bakery pilot applies two
temporary, category-neutral SKU correction layers:

- own-sales cold start for forecast-immature products `11573` and `11574`;
- adaptive persistent-bias correction for mature bakery/SKU pairs.

The layers preserve bakery/category totals and run before previous-day stock
subtraction and kratnost rounding. The combined 28-day walk-forward backtest
improved WAPE from `25.7551%` to `25.0720%`. Implementation, deployment,
rollback, and verification details are documented in
[`docs/pilot_sku_corrections_20260729.md`](docs/pilot_sku_corrections_20260729.md).

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

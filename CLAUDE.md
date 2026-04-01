# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

```bash
# Install
pip install -r requirements-dev.txt

# Lint
ruff check src/ web/ tests/ --select=E,F,W

# Test (all)
pytest tests/ -v

# Test (single)
pytest tests/test_preprocessing.py::test_preprocess_returns_dataframe -v

# Full pipeline: preprocess → weather → train
python run_pipeline.py

# Train only (skip preprocessing/weather)
python run_pipeline.py --skip-preprocess --skip-weather

# Web app
python web/app.py

# Docker
docker build -t demand-forecast . && docker run -p 5000:5000 demand-forecast
```

## What This Is

Bakery demand forecasting for the Beigl chain (Tatarstan/Chuvashia, Russia). Predicts daily sales per product per store using LightGBM. Data columns and comments are in Russian.

## Architecture

**Data flow:** Raw Excel → `preprocessing.py` (feature engineering) → `fetch_weather.py` (Open-Meteo enrichment) → `train_and_save.py` (LightGBM training + versioned save) → `web/app.py` (Flask serving)

**`src/config.py` is the single source of truth** for feature lists (`FEATURES`), model hyperparameters (`MODEL_PARAMS`), target column (`TARGET = "Продано"`), test split (`TEST_DAYS = 7`), and all file paths. Always import from here rather than hardcoding values.

**Experiment system:** Each experiment lives in `src/experiments/exp_{a..k}_*.py`, imports shared utilities from `src/experiments/common.py` (`load_data()`, `wmape()`, `train_lgbm()`, `save_predictions()`), and logs results via `src/tracking.log_experiment()` to `reports/experiment_log.jsonl`.

**Model versioning:** `train_and_save.py` saves latest model to `models/` and an archived copy to `models/archive/YYYYMMDD_HHMMSS/` with `metrics.json`.

## Important Conventions

- **Use `src/logger.py`** (`get_logger(name, log_file)`) instead of `print()` for any script output. Logs go to both console and `logs/` directory.
- **Russian column names** are used everywhere: Дата, Пекарня, Номенклатура, Категория, Город, Продано, Выпуск, Остаток, ДеньНедели. See AGENTS.md for full translations.
- **Windows console limitation:** avoid Unicode characters outside cp1251 in console output.
- **Tests use synthetic data** — they don't depend on real data files existing. Use `tmp_path` fixture for temp files.
- **Git-ignored:** `data/processed/`, `models/*.pkl`, `logs/`. Raw data (`data/raw/*.xlsx`) IS tracked.
- **User runs `.venv/Scripts/python.exe`** — the project uses a local virtual environment on Windows.

## See Also

`AGENTS.md` contains detailed ML infrastructure docs, data column reference, code style guidelines, and typical development scenarios.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Start Here For Production Or Ops Work

Before changing code or production state, read `docs/ops/CURRENT_STATE.md`
and `docs/ops/SERVICES.md`. For deployment or incident work, also read
`docs/ops/RUNBOOK.md` and `docs/ops/LLM_WORKFLOW.md`. `docs/ops/` is the
current operational source of truth, kept up to date after every production
change — `handoffs/` is historical session log only and must not override it
without fresh verification.

This repo now has two parts:

- **Legacy local ML pipeline** (`src/`, `run_pipeline.py`, `web/app.py`) —
  the LightGBM training/experiment codebase described below. No production
  role today (see `docs/ops/SERVICES.md`).
- **Live production system** — a VM (`root@201.51.7.24`,
  `/opt/demand-forecasting-model`) is the only forecast writer, publishing
  to ClickHouse on a nightly systemd timer
  (`pipelines/forecast_publish/`). A read-only embedded FastAPI/UI
  (`apps/forecast_embedded/`) serves Bitrix24 users from VibeCode/Blackhole
  and must never generate forecasts itself. `apps/baking_plan/` is a
  standalone package (MILP-based baking-window planner) mounted in-process
  into the embedded app — see its `README.md` and
  `docs/baking_plan_implementation.md` for the business-rule spec.

Do not infer current production state (active run, deployed scenario, timer
status) from this file or from memory — always check `docs/ops/CURRENT_STATE.md`
or live systems.

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

- `AGENTS.md` contains detailed ML infrastructure docs, data column reference, code style guidelines, and typical development scenarios for the legacy pipeline.
- `docs/ops/` contains live production state, service ownership, runbooks, data contracts, and durable architecture decisions for the VM/ClickHouse/Blackhole system.
- `docs/baking_plan_implementation.md` is the canonical business-rule spec for the `apps/baking_plan/` MILP allocator.
- `docs/dev_environment.md` explains the `_dev`-suffixed ClickHouse dev environment for the embedded app.

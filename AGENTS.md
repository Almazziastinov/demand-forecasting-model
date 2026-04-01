# AGENTS.md - Development Guidelines for Demand Forecasting Model

## Overview
This is a Python-based demand forecasting project for bakery products using LightGBM. The codebase is primarily in Russian (data columns, comments), but code follows Python conventions.

## Build, Lint, and Test Commands

### Install Dependencies
```bash
pip install -r requirements-dev.txt
```

### Lint (Ruff)
```bash
ruff check src/ web/ tests/ --select=E,F,W
```

### Run All Tests
```bash
pytest tests/ -v
```

### Run Single Test
```bash
pytest tests/test_preprocessing.py::test_preprocess_returns_dataframe -v
pytest tests/test_config.py::test_config_values -v
pytest tests/test_web.py::test_index_route -v
```

### Run Tests with Coverage
```bash
pytest tests/ --cov=src --cov-report=term-missing
```

## Code Style Guidelines

### Imports
- Standard library first, then third-party, then local
- Use explicit relative imports for project modules: `from src import config`
- Avoid `import *`

### Formatting
- Line length: 120 characters (ruff default)
- Use 4 spaces for indentation
- Use Black-style formatting where applicable
- Sort imports with `ruff check --select=I --fix`

### Type Hints
- Use type hints for function arguments and return types
- Example: `def preprocess_data(file_path: str) -> pd.DataFrame:`

### Naming Conventions
- Variables/functions: `snake_case` (e.g., `preprocess_data`, `sales_lag1`)
- Classes: `PascalCase` (e.g., `DataQualityMonitor`)
- Constants: `UPPER_SNAKE_CASE`
- File names: `snake_case.py`

### Error Handling
- Use try/except with specific exceptions
- Always log errors before raising
- Use `logging` module (see `src/logger.py`)

### Data Columns (Russian)
The following column names are used throughout the codebase:
- `Дата` - Date
- `Пекарня` - Bakery
- `Номенклатура` - Product
- `Категория` - Category
- `Город` - City
- `Продано` - Sold quantity (target)
- `Выпуск` - Production
- `Остаток` - Stock/Inventory
- `ДеньНедели` - Day of week

### Project Structure
```
src/
├── config.py           # Configuration (paths, features, model params)
├── preprocessing.py    # Data loading and feature engineering
├── logger.py           # Logging utility
├── tracking.py         # Experiment tracking
├── models/             # Model training and tuning
├── experiments/        # Experiment scripts (exp_a, exp_b, etc.)
├── analysis/           # Analysis scripts (EDA, error analysis, etc.)
├── monitoring/         # Data quality and drift monitoring
web/
└── app.py              # Flask web application
tests/
├── test_preprocessing.py
├── test_config.py
└── test_web.py
models/
├── demand_model.pkl    # Latest model
├── model_meta.pkl      # Latest metadata
└── archive/             # Versioned models (by date)
```

### Configuration
- All configuration in `src/config.py`
- Model params are stored in `MODEL_PARAMS` dict
- Feature lists defined in `FEATURES` list

### Testing Guidelines
- Tests should be independent and not rely on real data files
- Use `tmp_path` fixture for temp file creation
- Create synthetic data for testing (see `tests/test_preprocessing.py`)
- Run lint before committing: `ruff check src/ tests/ --select=E,F,W`

### Git Conventions
- Commit messages in English
- Format: `type: description` (e.g., `feat: add new feature`, `fix: resolve issue`)
- Test files mirror source structure: `tests/test_<module>.py`

## ML Infrastructure

### 1. Config — единый источник правды
All parameters in one place:
```python
from src.config import FEATURES, MODEL_PARAMS, TARGET, PROCESSED_DATA_PATH
```

### 2. Logging
Instead of `print()`, use:
```python
from src.logger import get_logger
logger = get_logger("my_script", log_file="my_script.log")
logger.info("Загрузка данных...")
logger.warning("Мало данных!")
```
Output goes to both console and `logs/my_script.log` with timestamp.

### 3. Experiment Tracking
```python
from src.tracking import log_experiment
log_experiment(
    name="exp_new_features",
    metrics={"mae": 3.25, "wmape": 22.1},
    params=MODEL_PARAMS,
    notes="Добавил фичу X",
)
```
Results in `reports/experiment_log.jsonl`.

### 4. Pipeline
```bash
# Full cycle: preprocessing → weather → train
python run_pipeline.py

# Merge two data files + full cycle
python run_pipeline.py --merge

# Data ready, only train
python run_pipeline.py --skip-preprocess --skip-weather

# Weather already loaded
python run_pipeline.py --skip-weather
```

### 5. Model Versioning
On each `train_and_save.py` run:
```
models/
  demand_model.pkl          ← latest
  model_meta.pkl            ← latest
  archive/
    20260330_154500/         ← version by date
      demand_model.pkl
      model_meta.pkl
      metrics.json           ← metrics for comparison
```

### 6. Monitoring
```bash
# Data quality check before training
python -m src.monitoring.data_quality

# Prediction drift check
python -m src.monitoring.drift_check
```

### 7. Web Application
```bash
python web/app.py
```
- Logs in `logs/web.log`
- Each `/predict` logged to `logs/predictions.csv`

### 8. Docker
```bash
docker build -t demand-forecast .
docker run -p 5000:5000 demand-forecast
```

## Data Files and Paths

### Raw Data
- `data/raw/beigl_data.xlsx` - Main data file
- `data/raw/3_month_data.xlsx` - Recent 3-month data

### Processed Data
- `data/processed/preprocessed_data_3month_enriched.csv` - Preprocessed 3-month with features
- `data/processed/preprocessed_data_merged_enriched.csv` - Merged historical + recent data

### Model Files
- `models/demand_model.pkl` - Trained LightGBM model
- `models/model_meta.pkl` - Model metadata (feature names, encoders, etc.)

### Output
- `reports/` - Experiment predictions and summaries
- `logs/` - Application logs

## Working with Data

### Preprocessing Pipeline
1. Raw data loaded from Excel files with smart header detection
2. Negative values clipped to zero
3. Temporal features: day of week, day, month, weekend flag
4. Aggregation to Date-Bakery-Product level
5. Lag features: lag1, lag2, lag3, lag7, lag14, lag30
6. Rolling stats: mean3, mean7, mean14, mean30, std7
7. Stock features: stock_lag1, stock_sales_ratio, stock_deficit

### Data Enrichment
- Weather data fetched via Open-Meteo API (cached)
- Holiday calendar features
- Payday/week features

## Model Development

### Training
- Use `src/models/train_and_save.py` for training + saving with versioning
- Use `src/models/tune_model.py` for hyperparameter tuning with Optuna
- Test set: last 7 days (configurable via `TEST_DAYS`)

### Experiments
- Run experiments from `src/experiments/` directory
- Naming: `exp_a_*, exp_b_*` for sequential experiments
- Results saved to `reports/` as CSV files
- Track with `src.tracking.log_experiment()`

## Key Modules
- `src/config.py` - All configuration constants
- `src/preprocessing.py` - Data loading and feature engineering
- `src/logger.py` - Logging utility
- `src/tracking.py` - Experiment tracking
- `src/models/train_and_save.py` - Training with versioning
- `src/monitoring/data_quality.py` - Data quality checks
- `web/app.py` - Flask web application

## Typical Scenarios

**New data:**
```bash
python run_pipeline.py --merge
```

**New experiment:**
```python
from src.config import FEATURES, MODEL_PARAMS
from src.tracking import log_experiment
# ... training ...
log_experiment(name="my_exp", metrics={...})
```

**Deploy:**
```bash
docker build -t demand-forecast . && docker run -p 5000:5000 demand-forecast
```

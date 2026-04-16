# Experiment 68 Status

## Purpose

Benchmark Prophet-only variants on the same SKU-level benchmark contract:

- `prophet_base`
- `prophet_holidays`
- `prophet_calendar`
- `prophet_calendar_holidays`
- `prophet_safe_regressors`

## Current State

- shortlist generation by `--r2-threshold` is working
- Prophet benchmark runner is implemented and gated on the `prophet` package
- full training still requires installing `prophet` in the local environment
- benchmark bakeries shared with experiment 67

## Planned Artifacts

- `run.py`
- `metrics_*.csv`
- `predictions_*.csv`
- `summary_best_by_sku.csv`
- `summary_by_model.csv`
- `selected_sku_subset.csv`

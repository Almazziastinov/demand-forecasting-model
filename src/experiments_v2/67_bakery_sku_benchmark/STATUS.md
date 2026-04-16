# Experiment 67 Status

## Purpose

Benchmark multiple model families on a fixed set of bakeries and SKUs:

- current best global model
- per-SKU model
- per-bakery model
- per-SKU Prophet
- 2-week average baseline

## Current State

- runner implemented
- benchmark bakeries selected
- benchmark artifacts generated in smoke mode

## Model Families

- `global_best`
- `sku_local`
- `bakery_local`
- `prophet_sku`
- `two_week_avg`

## Planned Artifacts

- `run.py`
- `metrics_*.csv`
- `predictions_*.csv`
- `summary_best_by_sku.csv`

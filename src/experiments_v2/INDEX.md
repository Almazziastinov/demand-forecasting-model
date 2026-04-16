# Experiments V2 Index

Фактический индекс директорий в `src/experiments_v2/`.

В отличие от `REGISTRY.md`, этот файл не пытается оценивать качество экспериментов. Он отвечает на более простой вопрос: какие каталоги реально существуют и насколько они заполнены артефактами.

## Правила статуса

- `completed` - есть `run.py`, `metrics.json` и `predictions.csv`
- `partial` - есть `run.py` и `metrics.json`, но нет `predictions.csv`
- `script-only` - есть `run.py`, но нет результатов
- `scaffold` - папка есть, но стандартных артефактов нет
- `non-experiment` - служебная директория

## Directory Index

| Directory | Status | run.py | metrics.json | predictions.csv | Notes |
|---|---|---:|---:|---:|---|
| `01_baseline_8m` | completed | yes | yes | yes | baseline experiment |
| `02_optuna_8m` | script-only | yes | no | no | tuning script without saved standard outputs |
| `03_demand_target` | completed | yes | yes | yes | demand-target experiment |
| `05_profiles` | completed | yes | yes | yes | profile features |
| `08_location_features` | completed | yes | yes | yes | location-based features |
| `09_price_features` | completed | yes | yes | yes | price features |
| `10_stacking` | completed | yes | yes | yes | stacking experiment |
| `11_13_tier3_ensembles` | completed | yes | yes | yes | combined tier-3 experiments |
| `20_per_bakery` | script-only | yes | no | no | per-bakery modeling script |
| `22_23_tier4_segments` | scaffold | no | no | no | placeholder or grouping directory |
| `23_cluster_models` | partial | yes | yes | no | results saved without predictions dump |
| `40_tweedie` | completed | yes | yes | yes | sales-target variant |
| `41_log_target` | completed | yes | yes | yes | sales-target variant |
| `42_asymmetric_loss` | completed | yes | yes | yes | sales-target variant |
| `43_quantile` | completed | yes | yes | yes | sales-target variant |
| `44_mixture_of_experts` | completed | yes | yes | yes | sales-target variant |
| `45_log_residual` | completed | yes | yes | yes | sales-target variant |
| `46_variance_weighted` | completed | yes | yes | yes | sales-target variant |
| `50_tweedie_demand` | completed | yes | yes | yes | demand-target variant |
| `51_log_target_demand` | completed | yes | yes | yes | demand-target variant |
| `52_asymmetric_loss_demand` | completed | yes | yes | yes | demand-target variant |
| `53_quantile_demand` | completed | yes | yes | yes | demand-target variant |
| `54_mixture_of_experts_demand` | completed | yes | yes | yes | demand-target variant |
| `55_log_residual_demand` | completed | yes | yes | yes | demand-target variant |
| `56_variance_weighted_demand` | completed | yes | yes | yes | demand-target variant |
| `60_baseline_v3` | completed | yes | yes | yes | v3 baseline |
| `61_censoring_behavioral` | partial | yes | yes | no | behavioral censoring |
| `62_assortment_availability` | partial | yes | yes | no | assortment availability |
| `63_combined_61_62` | partial | yes | yes | no | combined 61 and 62 |
| `64_high_demand_deep_dive` | partial | yes | yes | no | analysis-first experiment |
| `65_high_demand_model` | partial | yes | yes | no | high-demand focused model |
| `66_cluster_features` | completed | yes | yes | yes | exp63 + location/time-series clusters |
| `67_bakery_sku_benchmark` | completed | yes | yes | yes | per-bakery and per-SKU benchmark harness |
| `68_prophet_benchmark` | partial | yes | yes | no | Prophet-only benchmark harness on the shared SKU contract |
| `__pycache__` | non-experiment | no | no | no | Python cache |

## Summary

- `completed`: 24
- `partial`: 6
- `script-only`: 2
- `scaffold`: 1
- `non-experiment`: 1

## How To Use This File

- Если нужен рабочий готовый эксперимент с типовыми артефактами, начинать с `completed`
- Если нужен последний незавершенный исследовательский след, смотреть `partial`
- Если каталог в статусе `script-only` или `scaffold`, не предполагать, что по нему уже есть воспроизводимый результат

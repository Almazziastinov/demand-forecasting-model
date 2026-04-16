# Experiment 68: Prophet Benchmark

This experiment benchmarks Prophet variants on the same SKU-level evaluation contract used by experiment 67.

## Scope

- Compare Prophet configurations per SKU.
- Keep the evaluation aligned with the same bakeries and SKU rows used in the shared benchmark.
- Produce per-SKU metrics and predictions in the common benchmark schema.
- Support a configurable filter over the `global_best_r2` column from experiment 67.

## Expected Prophet Variants

- `prophet_base`
- `prophet_holidays`
- `prophet_calendar`
- `prophet_calendar_holidays`
- `prophet_safe_regressors`

## Selection

The runner accepts `--r2-threshold` and builds a shortlist from
`src/experiments_v2/67_bakery_sku_benchmark/summary_best_by_sku.csv`.
Rows where `global_best_r2 < threshold` are written to
`selected_sku_subset.csv` and used for the benchmark.

## Outputs

- Per-SKU metrics: `r2`, `mse`, `mae`, `wmape`.
- Per-SKU predictions.
- Model-level summary.
- Final best model per SKU.

## Notes

- Prophet tuning is intentionally isolated from boosting experiments.
- Only safe, forward-known regressors should be considered.
- Keep the same train/test split contract as the shared benchmark.
- If `prophet` is not installed, the runner still produces the shortlist and exits cleanly.

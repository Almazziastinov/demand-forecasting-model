# Experiment 67: Bakery and SKU Benchmark

This experiment is a comparison harness for a fixed set of bakeries and all SKUs inside them.

## Scope

- Compare the current best global model via inference only.
- Train one model per SKU.
- Train one model per bakery.
- Train one Prophet model per SKU.
- Use a simple 2-week average baseline per SKU.

## Selected Bakeries

- `Халтурина 8/20 Казань`
- `проспект Мусы Джалиля 20 Наб Челны`
- `Камая 1 Казань`
- `Дзержинского 47 Курск`
- `Фучика 105А Казань`

## Outputs

- Per-model metrics by SKU: `r2`, `mse`, `mae`, `wmape`.
- Per-model prediction files.
- A final table with the best `r2` per SKU.
- A model-level summary table.

## Notes

- The experiment will use the full available history.
- The selected bakeries should be fixed before running the benchmark.
- This directory is the working area for the benchmark implementation.

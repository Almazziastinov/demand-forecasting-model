# Session Handoff - 2026-06-23 - Pie Recent-Correction Cap and Audit

## Summary

Implemented a guard for costly pie categories in the ClickHouse SKU allocation path and generated a new pilot bakery pie audit for the production-aligned lead-1 window.

The protected categories are:

- `Пироги сытные`
- `Пироги сладкие`

The default behavior now prevents recent SKU correction from lifting those categories above the base profile forecast, and also applies a recent absolute average cap when recent sales are available. Any removed mass is redistributed to non-protected SKUs for the same bakery/date so the bakery-day total is preserved where possible.

## Code Changes

Changed files:

- `src/experiments_v2/apply_bakery_profiles_clickhouse.py`
- `pipelines/forecast_publish/run_production_inference.py`
- `tests/test_apply_bakery_profiles_clickhouse_recent.py`
- `scripts/build_new_pie_predictions_excel.py`

Key implementation details:

- Added `DEFAULT_RECENT_UPWARD_CAP_CATEGORY_PATTERN` for savory/sweet pie categories.
- Added `_apply_category_upward_cap(...)`.
- Wired cap arguments through:
  - `_build_recent_correction_targets(...)`
  - `apply_recent_sku_hour_correction(...)`
  - `allocate_from_clickhouse(...)`
  - `run_production_inference.py` CLI and summary JSON.
- Added tests for:
  - pie recent correction cannot lift above base
  - pie fallback to recent absolute cap
- Added `scripts/build_new_pie_predictions_excel.py` to rebuild the audit workbook from:
  - new local dev forecast CSV
  - previous audit facts JSON

## Validation

Commands run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_apply_bakery_profiles_clickhouse_recent.py -q
.venv\Scripts\python.exe -m ruff check src\experiments_v2\apply_bakery_profiles_clickhouse.py pipelines\forecast_publish\run_production_inference.py tests\test_apply_bakery_profiles_clickhouse_recent.py --select=E,F,W
.venv\Scripts\python.exe -m ruff check scripts\build_new_pie_predictions_excel.py --select=E,F,W
```

Results:

- Recent correction tests passed: 12 passed.
- Ruff passed for changed model/pipeline/test files.
- Ruff passed for the audit workbook generator.

## Dev Forecast Runs

First exploratory run:

- `dev_uplifted_bakery_norm_uplift_sku_20260601_h14`
- Horizon: `2026-06-01..2026-06-14`
- Not activated.

Production-aligned audit run:

- `dev_uplifted_bakery_norm_uplift_sku_20260616_h7`
- Horizon: `2026-06-16..2026-06-22`
- Command:

```powershell
.venv\Scripts\python.exe -m pipelines.forecast_publish.run_production_inference --env-file .env.dev --scenario uplifted_norm --horizon-days 7 --start-date 2026-06-16 --run-prefix dev --activate-run none --summary-path reports\dev_production_inference_summary_20260616_20260622.json --profile-table sku_hour_share_profile_smoothed_embedded_dev --uplift-table sku_hour_uplift_multiplier_embedded_dev --assortment-table assortment_city_products_dev --require-nonprod-tables
```

Run loaded to dev as draft only:

- `activated`: false
- `bakery_rows`: 1512
- `sku_day_rows`: 227355
- `sku_hour_rows`: 2354043

## Excel Audit

Final workbook:

```text
outputs/pie_audit_new_predictions_20260616_20260622/pie_audit_new_predictions_dev_uplifted_2026-06-16_2026-06-22.xlsx
```

Inputs:

- Forecast: `data/processed/sku_day_forecast_prod_uplifted_bakery_norm_uplift_sku.csv` from run `dev_uplifted_bakery_norm_uplift_sku_20260616_h7`
- Facts: `outputs/pie_audit_20260616_20260622/pie_audit_data_fixed_product_id.json`

Workbook verification:

- 11 sheets: summary + 10 pilot bakeries.
- 1816 data rows.
- 1156 formulas.
- No duplicate nomenclature after product-id normalization.
- No formula error tokens found.
- No mojibake samples found in workbook cells.

Totals in generated audit:

- Fact: `4501.0`
- Forecast: `4662.547181595681`

## Pilot Bakeries

- 20 - Мира 45 Дербышки Казань
- 21 - Парковая 7 Казань
- 22 - Сибирский Тракт 25 Казань
- 28 - Гудованцева 27 Казань
- 80 - Калинина 63 Казань
- 89 - Парина 6 Казань
- 107 - Четаева 46А Казань
- 221 - Салиха Батыева 15 Казань
- 222 - Габдуллы Тукая 62А Казань
- 257 - Ярмарочная 12 Чебоксары

## Notes

- Production was not touched.
- Dev active run was not changed.
- Generated `outputs/` and `reports/` files are local artifacts and are not intended to be committed unless explicitly needed.
- PowerShell sometimes displays UTF-8 JSON as mojibake, but workbook verification checked actual cell values and did not find mojibake.

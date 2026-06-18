# Session Handoff - 2026-06-18 - Assortment, Baking Plans, SKU Profiles

## Current Objective

Bring the forecast system in line with the actual bakery assortment, then rebuild SKU hourly share profiles and continue with baking plan template cleanup.

## Assortment State

- Source of truth for Tatarstan: director file `Асорт для Алмаза.xlsx`.
- Source of truth for Cheboksary: OCR/top files from the assortment folder.
- ClickHouse table design added in `apps/forecast_embedded/sql/assortment_schema.sql`.
- Active local export: `reports/required_assortment/assortment_city_products.csv`.
- Current active assortment shape:
  - total active rows: 2403
  - Tatarstan cities: 388 products per city
  - Cheboksary: 75 products
- Important decisions:
  - `Ватрушка в ассортменте` maps to `Ватрушка` and remains active where assortment allows.
  - `Пирожок/Булочка с яблоками (тесто ночное)` maps to `Пирожок яблоко`.
  - `Вишневый` is excluded everywhere.
  - `Капуста и курица` remains active only for `Чебоксары`.

## Production/Dev Implementation State

Implemented assortment-aware allocation:

- `src/experiments_v2/apply_bakery_profiles_clickhouse.py`
  - loads active `(city, product_id)` pairs from `assortment_city_products`;
  - filters exact and fallback SKU allocations by active city assortment;
  - includes filter stats in allocation summary;
  - supports disabling the filter.
- `pipelines/forecast_publish/run_production_inference.py`
  - added `--assortment-table`;
  - added `--disable-assortment-filter`.

Implemented assortment-aware profile rebuild:

- `src/experiments_v2/build_sku_hour_share_profile.py`
  - supports `--assortment-path`;
  - filters raw hourly SKU rows before share normalization;
  - adds city/product lookup fallback for older raw exports.
- Test added:
  - `tests/test_build_sku_hour_share_profile.py::test_assortment_filter_removes_sku_before_share_normalization`.

## VM Rebuild State

New VM:

- host tested: `root@201.51.7.24`
- hostname: `msk-1-vm-tpez`
- private key path locally: `tmp_remote_keys/new_prod_vm_ed25519`
- remote root: `/root/demand-forecasting-model`

Remote helper:

- `scripts/remote_rebuild_sku_profiles.ps1`
- Uses weekly ClickHouse export batches to avoid OOM.
- Installs remote dependencies:
  - `pandas==3.0.1`
  - `numpy==2.4.4`
  - `clickhouse-connect`
  - `joblib`
  - `lightgbm`
  - `scikit-learn`

Current running remote log:

- `/root/demand-forecasting-model/logs/rebuild_sku_profiles_20260618_175910.log`

Latest observed VM state before handoff:

- process is alive;
- command running:
  - `python scripts/export_clickhouse_checks.py --env-file .env.dev --sql-template scripts/clickhouse_export_template.sql --date-from 2026-01-01 --date-to 2026-06-16 --batch-mode weekly --output data/raw/sales_hrs_all_clickhouse.csv`
- latest log showed export at `[16/24]`;
- note: this rerun started export again instead of skipping the existing `data/raw/sales_hrs_all_clickhouse.csv`.

Previous VM run:

- completed ClickHouse export successfully:
  - rows: 29,921,146
  - output: `data/raw/sales_hrs_all_clickhouse.csv`
- then failed before profile build due missing `joblib`;
- dependencies were added and VM was restarted.

Next VM action:

1. Check whether `rebuild_sku_profiles_20260618_175910.log` finished.
2. If it restarts export again in future, patch `remote_rebuild_sku_profiles.ps1` to test for remote file existence reliably before launching export.
3. If profile build completes, verify row counts in dev tables:
   - `sku_hour_share_profile_smoothed_embedded_dev`
   - `sku_hour_uplift_multiplier_embedded_dev`
4. Run dev inference and compare before activating anything on prod.

## Baking Plan Template Audit

Template folder analyzed:

- `C:\Users\dns\Desktop\План выпикания шаблоны`

Generated audit:

- `reports/baking_plan_templates/template_assortment_audit.csv`
- `reports/baking_plan_templates/template_assortment_audit.xlsx`

Workbook sheets:

- `summary`
- `by_product`
- `problem_rows`
- `all_rows`

Audit totals:

- total parsed template rows: 3578
- `OK`: 1839
- `Нет в актуальном ассортименте города`: 855
- `Нет в dim_products`: 611
- `В dim_products выведено`: 110
- `Город вне текущей таблицы ассортимента`: 109
- `Не определен город шаблона`: 54

For the three Kazan bakeries:

- `План выпекания Сибирский тракт 25.xlsx`
- `План выпекания Мира 45.xlsx`
- `План выпекания Парковая 7.xlsx`

Confirmed:

- `Ватрушка в ассортменте` -> `Ватрушка`: OK.
- `Пирожок/Булочка с яблоками (тесто ночное)` -> `Пирожок яблоко`: OK.
- `Булочка с вишней (тесто ночное)`: not in actual Kazan assortment.
- `Вишневый`: not in actual Kazan assortment.
- `Мандариновый пай`: not in actual Kazan assortment.
- `Пирожок капуста курица` / `Капуста курица` / `Пирог Капуста курица`: maps to `Капуста и курица`, not active for Kazan.
- `Хуплу` / `Пирог Хуплу`: not active for Kazan.

New script:

- `scripts/audit_baking_plan_templates_assortment.py`

Verification:

- `ruff check scripts/audit_baking_plan_templates_assortment.py --select=E,F,W` passed.
- `ruff check scripts/export_clickhouse_checks.py --select=E,F,W` passed.

## Known Issues / Next Plan

1. Finish VM SKU profile rebuild and load dev profile tables.
2. Fix remote script so repeat runs do not re-export ClickHouse data when the CSV already exists.
3. Use `template_assortment_audit.xlsx` as the basis for baking plan cleanup.
4. Decide how to handle generic template rows such as:
   - `Ассортимент пирогов РТ`
   - `Ассортимент пирогов Чебоксары`
   - note/instruction rows accidentally parsed as products.
5. After profile rebuild, run dev inference and inspect known bad cases:
   - pies with overforecast;
   - hourly spikes like 05:00/22:00 in screenshots;
   - three Kazan bakeries from the partner feedback.
6. Do not activate prod until dev comparison is reviewed.

## Useful Commands

Check VM:

```powershell
C:\Windows\System32\OpenSSH\ssh.exe -o ConnectTimeout=10 -i tmp_remote_keys\new_prod_vm_ed25519 root@201.51.7.24 "tail -n 120 /root/demand-forecasting-model/logs/rebuild_sku_profiles_20260618_175910.log"
```

Check VM process:

```powershell
C:\Windows\System32\OpenSSH\ssh.exe -o ConnectTimeout=10 -i tmp_remote_keys\new_prod_vm_ed25519 root@201.51.7.24 "ps -eo pid,etime,cmd | grep -E 'build_sku_hour_share_profile|smooth_sku_profiles|load_sku_profiles|export_clickhouse_checks|rebuild_sku_profiles' | grep -v grep"
```

Run baking plan audit:

```powershell
.venv\Scripts\python.exe scripts\audit_baking_plan_templates_assortment.py
```

Run relevant lint:

```powershell
.venv\Scripts\python.exe -m ruff check scripts\audit_baking_plan_templates_assortment.py scripts\export_clickhouse_checks.py --select=E,F,W
```

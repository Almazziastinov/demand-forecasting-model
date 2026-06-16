# Handoff: dev environment for forecast pilot

Date: 2026-06-16
Repo: `C:\Users\dns\Desktop\Projects\demand-forecasting-model`
Branch: `master`

## Goal

Create a usable dev environment before pilot work:

- separate dev front;
- runnable script for dev forecast generation;
- same ClickHouse database as production, but isolated dev serving tables;
- no accidental writes/reads to production serving tables.

## Implemented

### Dev environment config

Added template:

```text
deploy/vm/forecast.dev.env.example
```

Expected local file:

```text
.env.dev
```

Important values:

```text
APP_ENV=dev
PORT=3001
FORECAST_TABLE_SUFFIX=_dev
FORECAST_RUN_PREFIX=dev
FORECAST_ACTIVATE_RUN=none
CLICKHOUSE_DATABASE=Svezhar
CLICKHOUSE_PORT=8443
```

`.env.*` is gitignored, so `.env.dev` is local-only.

### Dev front

Added:

```text
scripts/dev_run_embedded_api.ps1
```

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\dev_run_embedded_api.ps1
```

URL:

```text
http://127.0.0.1:3001
```

The embedded UI now shows a visible environment badge for non-prod:

```text
DEV _dev
```

This is rendered in `apps/forecast_embedded/app/templates/layout.html` and styled
in `apps/forecast_embedded/app/static/app.css`.

### Dev forecast runner

Added:

```text
scripts/dev_run_inference.ps1
```

Run and activate dev run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\dev_run_inference.ps1 -ActivateRun uplifted_norm
```

The runner refuses to run if:

- `APP_ENV=prod`;
- `FORECAST_TABLE_SUFFIX` is empty;
- `FORECAST_TABLE_SUFFIX` has invalid characters.

### Table isolation

Production and dev use the same ClickHouse database, but dev forecast-serving
tables use the `_dev` suffix:

```text
forecast_runs_embedded_dev
bakery_forecast_day_embedded_dev
forecast_day_context_embedded_dev
sku_forecast_day_embedded_dev
sku_forecast_hour_embedded_dev
bakery_forecast_day_snapshots_dev
sku_forecast_day_snapshots_dev
sku_forecast_hour_snapshots_dev
bitrix_user_bakery_access_embedded_dev
bakery_month_revenue_embedded_dev
```

Shared source/reference tables remain unsuffixed:

- `Svezhar.fct_check_lines`;
- `dim_management`;
- `dim_bakeries`;
- `sku_hour_share_profile_smoothed_embedded`;
- `sku_hour_uplift_multiplier_embedded`;
- `mart_sales_60d`.

Implementation:

```text
apps/forecast_embedded/app/table_names.py
pipelines/forecast_publish/table_names.py
```

`load_schema()` rewrites embedded table names to suffixed names when
`FORECAST_TABLE_SUFFIX` is set.

### Env loading and port bug

Fixed env loading so `ENV_FILE=.env.dev` is supported by the embedded app.

Fixed a critical bug in ClickHouse connection settings:

- before: publish scripts could take `PORT=3001` as ClickHouse port;
- now: `CLICKHOUSE_PORT=8443` has priority, while `PORT=3001` is only for the dev front.

Touched:

```text
apps/forecast_embedded/app/settings.py
apps/forecast_embedded/app/db.py
pipelines/forecast_publish/load_forecast_run.py
pipelines/forecast_publish/activate_run.py
scripts/export_clickhouse_checks.py
```

## Current dev run

The user successfully ran:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\dev_run_inference.ps1 -ActivateRun uplifted_norm
```

Active dev run:

```text
dev_uplifted_bakery_norm_uplift_sku_20260601_h14
```

Summary:

```text
reports/dev_production_inference_summary.json
```

Loaded rows:

```text
bakery_rows: 3038
context_rows: 154
sku_day_rows: 555420
sku_hour_rows: 5222976
bakery_snapshot_rows: 3038
sku_day_snapshot_rows: 555420
sku_hour_snapshot_rows: 5222976
```

## Docs

Added:

```text
docs/dev_environment.md
```

Also updated:

```text
apps/forecast_embedded/README.md
deploy/vm/README.md
```

## Verification

Passed:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_run_production_inference.py tests/test_forecast_embedded_access.py tests/test_forecast_publish_load_run.py -v
```

Result:

```text
19 passed
```

Passed:

```powershell
.venv\Scripts\python.exe -m ruff check ... --select=E,F,W,I
```

PowerShell script syntax checked for:

```text
scripts/dev_run_inference.ps1
scripts/dev_run_embedded_api.ps1
```

## Known working commands

Create local env file:

```powershell
Copy-Item deploy\vm\forecast.dev.env.example .env.dev
```

Run forecast into dev tables and activate:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\dev_run_inference.ps1 -ActivateRun uplifted_norm
```

Run dev front:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\dev_run_embedded_api.ps1
```

Open:

```text
http://127.0.0.1:3001
```

## Pilot context for next work

User confirmed dev environment works. Next task is pilot bakery audit.

Pilot pool from screenshot:

- `Салиха Батыева 15 Казань`
- `Габдуллы Тукая 62А Казань`
- `Парина 6 Казань`
- `Четаева 46А Казань`
- `Парковая 7 Казань`
- `Гудованцева 27 Казань`
- `Калинина 53 Казань`
- `Мира 45 Дербышки Казань`
- `Ярмарочная 12 Чебоксары`
- `Сибирский Тракт 25 Казань`

Problem bakeries named by user:

- `Габдуллы Тукая 62А Казань` - bad;
- `Парковая 7 Казань` - not liked;
- `Мира 45 Дербышки Казань` - bad;
- `Сибирский Тракт 25 Казань` - bad.

Need next:

1. Map these to `bakery_id`.
2. Pull fact vs forecast for recent/dev run and historical holdout.
3. Split error into bakery-total error vs SKU-allocation error.
4. Check stockout/availability/recent assortment signals.
5. Decide per bakery: pilot as-is, pilot with manual correction/caps, or exclude.


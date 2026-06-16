# Dev environment

Dev uses the same ClickHouse database as production, but all forecast serving
tables are suffixed with `FORECAST_TABLE_SUFFIX`, normally `_dev`.

## First setup

```powershell
Copy-Item deploy\vm\forecast.dev.env.example .env.dev
```

Fill `CLICKHOUSE_*` in `.env.dev`. `CLICKHOUSE_DATABASE` can be the production
database. Keep:

```text
APP_ENV=dev
PORT=3001
FORECAST_TABLE_SUFFIX=_dev
FORECAST_RUN_PREFIX=dev
FORECAST_ACTIVATE_RUN=none
```

## Run forecast into dev tables

```powershell
.\scripts\dev_run_inference.ps1
```

This creates the schema and writes forecast rows into tables such as
`forecast_runs_embedded_dev`, `bakery_forecast_day_embedded_dev`,
`sku_forecast_day_embedded_dev`, and `sku_forecast_hour_embedded_dev`.

To activate the dev run inside dev tables:

```powershell
.\scripts\dev_run_inference.ps1 -ActivateRun uplifted_norm
```

## Run dev front

```powershell
.\scripts\dev_run_embedded_api.ps1
```

Open `http://127.0.0.1:3001`. The header shows `DEV` and the table suffix, for
example `_dev`.

## Safety

The dev scripts refuse to run when:

- `APP_ENV=prod`;
- `FORECAST_TABLE_SUFFIX` is empty;
- `FORECAST_TABLE_SUFFIX` contains anything except letters, digits, and `_`.

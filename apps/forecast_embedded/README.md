# Forecast Embedded App

Embedded Bitrix24 viewer for bakery-driven forecast runs stored in ClickHouse.

## Purpose

This app serves already calculated forecast runs from ClickHouse and renders:

- bakery daily totals
- top SKU per bakery/day
- hourly bakery totals
- hourly breakdown for a SKU

The app does not train models and does not build forecasts on demand.

## Environment

Required application environment variables:

- `PORT` (default `3000`)
- `APP_ENV` (default `dev`)
- `APP_TITLE` (default `Bakery Forecast Embedded`)

Optional:

- `ACTIVE_RUN_ID` - force a specific run instead of reading the active one
- `BITRIX_EMBED_MODE` - `true` or `false`
- `CLICKHOUSE_HOST`
- `CLICKHOUSE_PORT`
- `CLICKHOUSE_USER`
- `CLICKHOUSE_PASSWORD`
- `CLICKHOUSE_DATABASE`
- `CLICKHOUSE_SECURE` (default `true`)
- `CLICKHOUSE_VERIFY` (default `false`)

For local development, ClickHouse connection can still fall back to the repository `.env`:

- `HOST`
- `PORT`
- `USER`
- `PASSWORD`
- `DATABASE`

## Local Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 3000 --reload
```

## Storage

Serving tables are created from [sql/schema.sql](sql/schema.sql):

- `forecast_runs_embedded`
- `bakery_forecast_day_embedded`
- `sku_forecast_day_embedded`
- `sku_forecast_hour_embedded`

## Key Routes

- `/health`
- `/`
- `/bakery/{bakery_id}?date=YYYY-MM-DD`
- `/api/v1/runs/active`
- `/api/v1/dates`
- `/api/v1/bakeries?date=YYYY-MM-DD`
- `/api/v1/bakeries/{bakery_id}/summary?date=YYYY-MM-DD`
- `/api/v1/bakeries/{bakery_id}/sku-hour?date=YYYY-MM-DD&product_id=...`

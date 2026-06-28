# Data Contracts

Last updated: 2026-06-28

## Forecast Run Contract

Production forecast outputs are grouped by `run_id`.

Current active run pattern:

```text
prod_uplifted_bakery_norm_uplift_sku_YYYYMMDD_h14
```

The active run must be represented in `forecast_runs_embedded` with
`status = 'active'`. There should be one intended active run for the production
scenario.

## Current Scenario Contract

- Scenario: `uplifted_norm`
- Horizon days: `14`
- Active horizon observed on 2026-06-28: `2026-06-23` through `2026-07-06`
- Recent correction mode: `runner_city_prior_soft_weekpart`
- Recent correction days: `30`
- Recent sales table: `mart_sales_60d`

## Serving Tables

The embedded app reads ClickHouse serving tables. Known active tables:

- `forecast_runs_embedded`
- `bakery_forecast_day_embedded`
- `forecast_day_context_embedded`
- `sku_forecast_day_embedded`
- `sku_forecast_hour_embedded`

## Snapshot Tables

The production verification script checks snapshot tables for active runs:

- `bakery_forecast_day_snapshots`
- `sku_forecast_day_snapshots`
- `sku_forecast_hour_snapshots`

Expected rows for active run
`prod_uplifted_bakery_norm_uplift_sku_20260623_h14`:

- bakery day snapshots: `3024`
- SKU day snapshots: `480206`
- SKU hour snapshots: `5138172`

## Local Files On Production VM

Observed fresh artifacts for the 2026-06-28 production run:

- `data/processed/bakery_day_forecast_prod_uplifted_norm.csv`
- `data/processed/sku_day_forecast_prod_uplifted_bakery_norm_uplift_sku.csv`
- `data/processed/sku_hour_forecast_prod_uplifted_bakery_norm_uplift_sku.csv`

Local files are useful for debugging, but ClickHouse active run state is what
the embedded app serves.

## Access And Secrets

Do not commit or print:

- `.env`
- ClickHouse credentials
- VibeCode API keys
- Bitrix tokens
- VM SSH keys

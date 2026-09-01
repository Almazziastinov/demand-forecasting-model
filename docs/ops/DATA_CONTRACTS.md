# Data Contracts

Last updated: 2026-09-01

`CURRENT_STATE.md` remains the operational source of truth for the exact live
run. This file defines the durable serving contract for the Direct model.

## Forecast Run Contract

Production forecast outputs are grouped by `run_id`.

Current active run pattern:

```text
prod_direct_alpha_025_YYYYMMDD_h14
```

The active run must be represented in `forecast_runs_embedded` with
`status = 'active'`. There should be one intended active run for the production
scenario.

## Current Scenario Contract

- Model version: `direct_alpha_025_v1`.
- Horizon: 14 days.
- Bakery-day volume source: inactive `prod_base_bakery_norm_recent_*` run.
- SKU allocation: direct bakery-day-to-SKU prediction over mature assortment;
  no inherited category totals and no hourly SKU-profile allocation.
- Volume/guard layers: predictive expected-loss uplift, Core-SKU protection,
  alpha `.25`, adaptive floor and causal tail cap.
- Cold-start SKU forecasts are produced independently of mature-SKU
  normalization.
- Hourly rows are a downstream timing split and must conserve finalized
  SKU-day totals.
- The active run, and only the active run, is served through ClickHouse.

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

Observed rows for the verified active run
`prod_direct_alpha_025_20260831_h14`:

- bakery day snapshots: `2478`
- SKU day snapshots: `149526`
- SKU hour snapshots: `2484338`

Lead-1 historical backfills use separate draft run ids:

```text
backfill_uplifted_bakery_norm_uplift_sku_YYYYMMDD_h1
```

These runs provide `lead_days = 1` rows for fact-vs-forecast history and must
not be activated as the current production forecast. As of 2026-06-29,
lead-1 snapshots for `2026-06-24` through `2026-06-28` are loaded in all three
snapshot tables.

## Local Files On Production VM

Observed fresh artifacts for the 2026-06-28 production run:

- `data/processed/bakery_day_forecast_prod_uplifted_norm.csv`
- `data/processed/sku_day_forecast_prod_uplifted_bakery_norm_uplift_sku.csv`
- `data/processed/sku_hour_forecast_prod_uplifted_bakery_norm_uplift_sku.csv`

Lead-1 backfill artifacts are date-specific and may be present as:

- `data/processed/bakery_day_forecast_prod_lead1_YYYYMMDD.csv`
- `data/processed/sku_day_forecast_prod_lead1_YYYYMMDD.csv`
- `data/processed/sku_hour_forecast_prod_lead1_YYYYMMDD.csv`

Local files are useful for debugging, but ClickHouse active run state is what
the embedded app serves.

## Access And Secrets

Do not commit or print:

- `.env`
- ClickHouse credentials
- VibeCode API keys
- Bitrix tokens
- VM SSH keys

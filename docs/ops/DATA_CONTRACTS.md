# Data Contracts

Last updated: 2026-07-14

**Note:** this file had drifted out of date (last content update 2026-06-29
describing the `uplifted_norm` scenario) while `CURRENT_STATE.md` kept
moving — always trust `CURRENT_STATE.md`'s "Active Forecast" section over
this file if they disagree. Refreshed now as part of the 2026-07-14
pilot-uplift reconfiguration.

## Forecast Run Contract

Production forecast outputs are grouped by `run_id`.

Current active run pattern (as of 2026-07-14):

```text
prod_base_bakery_raw_uplift_sku_YYYYMMDD_h14
```

The active run must be represented in `forecast_runs_embedded` with
`status = 'active'`. There should be one intended active run for the production
scenario.

## Current Scenario Contract

- Scenario: `base_raw_uplift` (base bakery-day model + raw SKU-hour uplift
  multiplier, all bakeries — switched 2026-07-14 for the pilot, see
  `docs/ops/DECISIONS.md`)
- Horizon days: `14`
- Active horizon observed on 2026-07-14: `2026-07-14` through `2026-07-27`
- Recent correction mode: `runner_city_prior_soft_weekpart`
- Recent correction days: `30`
- Recent sales table: `mart_sales_60d` (VM production writer); pilot publisher
  uses `fct_check_lines` directly since 2026-08-14 (mart ETL outage)
- SKU-hour uplift multiplier: `sku_hour_uplift_multiplier_embedded`,
  `profile_version=weekly_20260714` (not renormalized — SKU-hour sums can
  exceed the bakery-hour total, see `CURRENT_STATE.md`)

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

Observed rows for active run
`prod_base_bakery_raw_uplift_sku_20260714_h14`:

- bakery day snapshots: `2954`
- SKU day snapshots: `445822`
- SKU hour snapshots: `5017688`

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

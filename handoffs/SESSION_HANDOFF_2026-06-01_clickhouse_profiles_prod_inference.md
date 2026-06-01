# Session Handoff - 2026-06-01 - ClickHouse Profiles and Prod Inference

## Context

We moved the forecasting pipeline toward a lighter production setup where large SKU profiles are stored in ClickHouse instead of on the app VM.

User preference:

- Front/API VM can be small.
- Do not rebuild raw/profile/model every day at first if it makes infra heavy.
- Store SKU profiles in ClickHouse.
- Support two production forecast variants:
  1. Base bakery model + raw SKU uplift multiplier allocation.
  2. Uplifted bakery model + normalized SKU uplift allocation.

Current date context: forecasts built from history through `2026-05-31`, horizon `2026-06-01..2026-06-14`.

## Important Commits

Latest relevant commits on `master`:

- `2900cd9 feat: support clickhouse-backed sku profiles`
- `91beb8d feat: support raw uplift sku allocation`
- `fe21633 feat: add uplifted bakery daily target builder`
- `85ca249 fix: use current check lines export source`
- `3ff105c feat: preserve uplift audit in smoothed share profiles`

## Data Source Correction

The current raw source should be:

```sql
Svezhar.fct_check_lines
```

Not:

```sql
Svezhar.fct_check_lines_new
```

Updated file:

```text
scripts/clickhouse_export_template.sql
```

Important details:

- No `FINAL`, because `fct_check_lines` is MergeTree and does not support `FINAL`.
- Uses `ANY LEFT JOIN` to dimensions to avoid row duplication.
- Uses hex filter for sales event:

```sql
hex(fcl.cash_event_type) = 'D09FD180D0BED0B4D0B0D0B6D0B0'
```

## Current Local Data Artifacts

Generated files exist locally and are intentionally not committed:

```text
data/raw/sales_hrs_increment_2026-05-13_2026-05-31.csv
data/raw/sales_hrs_all_clickhouse_2026-05-31.csv
```

The merged raw file covers history through `2026-05-31`.

Rebuilt processed artifacts include:

```text
data/processed/bakery_daily_sales.csv
data/processed/bakery_hour_profile.csv
data/processed/sku_hour_share_profile.csv
data/processed/sku_hour_share_profile_smoothed.csv
data/processed/sku_hour_share_profile_daily_smoothed.csv
data/processed/bakery_daily_sales_uplifted.csv
```

Key summaries:

```text
bakery_daily_sales:
date_min = 2025-01-15
date_max = 2026-05-31
rows = 87,715
bakeries = 217

sku_hour_share_profile_smoothed:
profile_rows = 3,856,275
applied_rows = 36,365,329
bakeries = 217
products = 1,331
mean_uplifted_row_rate = 0.436985
mean_share_uplift_raw = 0.005242
```

Uplifted bakery target summary:

```text
mean_base_bakery_sales = 939.706915
mean_uplifted_bakery_sales = 1132.142688
mean_uplift_rate = 0.209726
p95_uplift_rate = 0.35464
max_uplift_rate = 2.317313
```

## Uplift Logic Agreed With User

For training uplifted bakery model:

```text
base bakery = 200

base shares:
0.3 + 0.4 + 0.3 = 1.0

uplift shares:
0.3 + 0.5 + 0.3 = 1.1

bakery_uplift = 200 * 1.1 = 220
```

For SKU allocation from uplifted bakery total:

```text
normalized uplift shares:
0.3 / 1.1 = 0.2727
0.5 / 1.1 = 0.4545
0.3 / 1.1 = 0.2727

SKU:
220 * 0.2727 = 60
220 * 0.4545 = 100
220 * 0.2727 = 60
```

Therefore:

- Base bakery + raw uplift SKU: SKU total can be greater than bakery total.
- Uplifted bakery + normalized uplift SKU: SKU total equals uplifted bakery total.

## Forecast Runs Produced

Two current draft production runs were loaded to ClickHouse:

### 1. Base bakery + raw uplift SKU

Current valid run:

```text
prod_base_bakery_raw_uplift_sku_v2_20260601_h14
```

Do not use older mistaken run:

```text
prod_base_bakery_raw_uplift_sku_20260601_h14
```

Summary:

```text
bakery_total = 2,659,349.371407
sku_total = 3,213,188.114292
allocation_ratio = 1.208261
rows: bakery=3,038, sku_day=425,604, sku_hour=3,096,400
```

### 2. Uplifted bakery + normalized uplift SKU

Run:

```text
prod_uplifted_bakery_norm_uplift_sku_20260601_h14
```

Summary:

```text
bakery_total = 3,225,853.455795
sku_total = 3,225,853.455795
allocation_ratio = 1.0
rows: bakery=3,038, sku_day=425,604, sku_hour=3,096,400
```

Comparison saved:

```text
reports/compare_prod_base_raw_v2_vs_uplifted_norm_20260601_h14.json
```

Neither run was activated in this session.

## ClickHouse Profile Storage

Added schema:

```text
apps/forecast_embedded/sql/schema.sql
```

Existing profile table:

```text
sku_hour_share_profile_smoothed_embedded
```

New multiplier table:

```text
sku_hour_uplift_multiplier_embedded
```

Loaded into ClickHouse:

```text
sku_hour_share_profile_smoothed_embedded: 3,856,275 rows
sku_hour_uplift_multiplier_embedded: 27,598 rows
profile_version = sku_uplift_20260601
```

`sku_hour_uplift_multiplier_embedded` uses:

- normal rows: `bakery_id x dow x hour`
- fallback rows: `dow = -1`, for `bakery_id x hour`

This preserves local fallback behavior.

## New/Updated Code

### Uplifted target builder

```text
src/experiments_v2/build_uplifted_bakery_daily_dataset.py
```

Builds:

```text
data/processed/bakery_daily_sales_uplifted.csv
data/processed/bakery_daily_sales_uplifted_summary.json
```

### Raw uplift allocation support

```text
src/experiments_v2/apply_bakery_profiles.py
```

Added options:

```text
--sku-share-col
--no-normalize-sku-shares
--uplift-multiplier-path
```

The stable raw uplift variant should use normalized profile shares plus hour-level uplift multiplier, not `mean_sku_share_in_hour` directly.

### ClickHouse profile store

```text
pipelines/forecast_publish/sku_hour_profile_store.py
```

Modes:

```text
--mode load
--mode export
--mode load-uplift-multipliers
```

Useful commands:

```powershell
.venv\Scripts\python.exe pipelines\forecast_publish\sku_hour_profile_store.py `
  --mode load `
  --profile-path data\processed\sku_hour_share_profile_smoothed.csv `
  --table sku_hour_share_profile_smoothed_embedded `
  --chunk-size 200000 `
  --truncate
```

```powershell
.venv\Scripts\python.exe pipelines\forecast_publish\sku_hour_profile_store.py `
  --mode load-uplift-multipliers `
  --applied-path data\processed\sku_hour_share_profile_daily_smoothed.csv `
  --uplift-table sku_hour_uplift_multiplier_embedded `
  --profile-version sku_uplift_20260601 `
  --chunk-size 1000000 `
  --truncate
```

### ClickHouse-backed allocation

```text
src/experiments_v2/apply_bakery_profiles_clickhouse.py
```

Normalized uplift SKU allocation:

```powershell
.venv\Scripts\python.exe -m src.experiments_v2.apply_bakery_profiles_clickhouse `
  --bakery-forecast-path data\processed\bakery_day_forecast_future_uplifted_smoothed_bias_adj.csv `
  --bakery-hour-profile-path data\processed\bakery_hour_profile.csv `
  --forecast-col bakery_day_forecast_bias_adj `
  --output-dir data\processed `
  --output-suffix future_uplifted_smoothed_bias_adj_ch
```

Base bakery + raw uplift SKU allocation:

```powershell
.venv\Scripts\python.exe -m src.experiments_v2.apply_bakery_profiles_clickhouse `
  --bakery-forecast-path data\processed\bakery_day_forecast_future_smoothed_bias_adj.csv `
  --bakery-hour-profile-path data\processed\bakery_hour_profile.csv `
  --forecast-col bakery_day_forecast_bias_adj `
  --output-dir data\processed `
  --output-suffix future_base_bakery_raw_uplift_sku_ch `
  --use-raw-uplift-multiplier `
  --uplift-profile-version sku_uplift_20260601
```

Validation:

```text
normalized local = 3,225,853.455795
normalized CH = 3,225,853.455795

raw local = 3,213,188.114292
raw CH = 3,213,188.114292
```

## Infra Direction

If profiles stay in ClickHouse and app VM does not rebuild data/profile/model:

```text
small VM is feasible:
1-2 vCPU
1-2 GB RAM
20-40 GB disk
```

Recommended safer MVP:

```text
2 vCPU / 2 GB RAM
```

This VM should:

- run API/frontend
- run inference if needed
- read profiles from ClickHouse
- write forecast runs to ClickHouse

Heavy rebuilds can remain manual/off-box for now.

## Next Steps

1. Integrate `apply_bakery_profiles_clickhouse.py` into a production inference script.
2. Build one command that:
   - runs base bakery forecast,
   - runs uplifted bakery forecast,
   - applies ClickHouse-backed allocation for both scenarios,
   - loads both runs to ClickHouse,
   - optionally activates selected default run.
3. Avoid local SKU profile CSV on prod VM.
4. Build minimal frontend:
   - run/model switcher,
   - bakery selector,
   - date selector,
   - daily forecast chart,
   - selected bakery/day hourly chart.
5. Later: automate profile rebuild and profile upload once infra is ready.

## Current Git/Workspace Notes

Tracked code is pushed through:

```text
2900cd9 feat: support clickhouse-backed sku profiles
```

Known untracked local generated files:

```text
data/raw/sales_hrs_all_clickhouse_2026-05-31.csv
data/raw/sales_hrs_increment_2026-05-13_2026-05-31.csv
tests/_tmp_smooth_profiles/
```

Do not commit large raw/generated files unless explicitly requested.

# Session Handoff - 2026-06-29 - Prod Recovery, Weather Refresh, Lead-1 Backfill

## Scope

Production embedded app recovery and data freshness work:

- fixed `Forecast run not found`;
- restarted and repaired the Blackhole embedded app;
- fixed app `Internal Server Error` caused by a missing ClickHouse table;
- refreshed production forecast from current ClickHouse data and real weather;
- started lead-1 backfill for the missing fact-vs-forecast dates.

## Production Ownership

Production forecast generation must run only on the VM:

- host: `root@201.51.7.24`
- path: `/opt/demand-forecasting-model`
- timer: `forecast-production.timer`

VibeCode/Blackhole is read-only app/API:

- server id: `82bb03a8-c356-4225-97a4-a1540cdc29e6`
- app URL: `https://app-8613ac40f10d.vibecode.bitrix24.tech`
- forecast timers on Blackhole must remain disabled/inactive.

## Current Active Forecast

Active production run after manual refresh:

```text
prod_uplifted_bakery_norm_uplift_sku_20260629_h14
```

Horizon:

```text
2026-06-29..2026-07-12
```

Rows:

- bakery snapshots: `2842`
- SKU-day snapshots: `460708`
- SKU-hour snapshots: `5014812`
- context rows: `126`

Weather context check:

- date range: `2026-06-29..2026-07-12`
- default-weather rows: `0`

VM `.env` was updated so the timer refreshes datasets and weather:

```text
FORECAST_REFRESH_DATASETS=1
FORECAST_REFRESH_WEATHER=1
```

## Fixes Applied

### Active run missing

Symptom:

```text
{"detail":"Forecast run not found"}
```

Cause: production `forecast_runs_embedded` had no active run.

Temporary repair: copied the known good `prod_uplifted_bakery_norm_uplift_sku_20260623_h14`
from `_dev` serving/snapshot tables into production and activated it.

Later replaced by the fresh `20260629` run.

### App Internal Server Error

Symptom: embedded app returned 500.

Cause: app queried nonexistent `Svezhar.dim_management`.

Fix:

- local: `apps/forecast_embedded/app/services/bakery.py`
- production Blackhole hotfix: switched open-bakery lookup to `dim_bakeries`

Verification:

- `/`
- `/health`
- `/api/v1/bakeries?date=2026-06-29`

all returned `200`.

### Date list mixed stale snapshots

Symptom: UI date list showed old dates through `2026-06-23` after the active run
had moved to `2026-06-29..2026-07-12`.

Cause: `apps/forecast_embedded/app/services/runs.py::get_run_dates()` queried
`bakery_forecast_day_snapshots where lead_days = 1` without filtering by
`source_run_id`.

Fix:

```sql
where lead_days = 1
  and source_run_id = %(run_id)s
```

Hotfix deployed on Blackhole and `app.service` restarted. `/api/v1/dates` then
returned only active run dates `2026-06-29..2026-07-12`.

## Data Refresh

ClickHouse data was verified present:

- `mart_sales_60d`: `2026-06-01..2026-06-29`
- `Svezhar.fct_check_lines`: `2025-12-01..2026-06-29`

Manual VM refresh command used:

```bash
.venv/bin/python -m pipelines.forecast_publish.run_production_inference \
  --env-file .env \
  --refresh-datasets \
  --history-start-date 2025-12-01 \
  --activate-run uplifted_norm \
  --scenario uplifted_norm \
  --notes 'manual refresh after ClickHouse data recovery 2026-06-29'
```

Assortment and bakeable products were rebuilt and loaded after the fresh run:

```bash
scripts/build_city_assortment_from_forecast.py --env-file .env
scripts/build_bakeable_products_table.py
scripts/load_city_assortment_to_clickhouse.py --env-path .env --replace-current
scripts/load_bakeable_products_to_clickhouse.py --env-path .env --replace-current
```

Loaded:

- `assortment_city_products`: `1910` rows
- `bakeable_products`: `570` rows

## Lead-1 Backfill

Problem: facts exist through `2026-06-29`, but lead-1 forecast snapshots were
missing for `2026-06-24..2026-06-28`.

Added script:

```text
scripts/build_prod_lead1_model_backfill.py
```

Purpose: for dates with no bakery-level lead-1 snapshot, build a one-day
forecast from history before that date, allocate through ClickHouse SKU/hour
profiles, and load a draft h1 run.

Run ids:

```text
backfill_uplifted_bakery_norm_uplift_sku_YYYYMMDD_h1
```

Do not activate these runs.

Command used:

```bash
.venv/bin/python scripts/build_prod_lead1_model_backfill.py \
  --env-file .env \
  --date-from 2026-06-24 \
  --date-to 2026-06-28 \
  --uplift-profile-version prod_allowlist_22_222_old_else_20260617 \
  --replace-existing
```

The SSH connection closed during the first run, but the process continued
through `2026-06-26`. A second background run was started for `2026-06-27..2026-06-28`:

```bash
nohup .venv/bin/python scripts/build_prod_lead1_model_backfill.py \
  --env-file .env \
  --date-from 2026-06-27 \
  --date-to 2026-06-28 \
  --uplift-profile-version prod_allowlist_22_222_old_else_20260617 \
  --replace-existing \
  > logs/prod_lead1_backfill_20260627_20260628.log 2>&1 &
```

Final status:

- `2026-06-24`: loaded
- `2026-06-25`: loaded
- `2026-06-26`: loaded
- `2026-06-27`: loaded
- `2026-06-28`: loaded

Loaded rows observed:

| date | bakery snapshots | SKU-day snapshots | SKU-hour snapshots |
| --- | ---: | ---: | ---: |
| `2026-06-24` | `202` | `32509` | `353367` |
| `2026-06-25` | `203` | `32557` | `354420` |
| `2026-06-26` | `203` | `32695` | `358125` |
| `2026-06-27` | `203` | `32750` | `355058` |
| `2026-06-28` | `203` | `33324` | `355353` |

## Code And Docs Changed Locally

Code:

- `apps/forecast_embedded/app/services/bakery.py`
- `apps/forecast_embedded/app/services/runs.py`
- `pipelines/forecast_publish/production_dataset_refresh.py`
- `scripts/export_clickhouse_bakery_daily.py`
- `scripts/build_prod_lead1_model_backfill.py`
- tests updated for the app/date/export fixes

Docs:

- `docs/ops/CURRENT_STATE.md`
- `docs/ops/DATA_CONTRACTS.md`
- `docs/ops/RUNBOOK.md`
- this handoff

## Verification Already Done

Local:

```bash
pytest tests/test_forecast_embedded_access.py tests/test_export_clickhouse_bakery_daily.py tests/test_run_production_inference.py -q
ruff check apps/forecast_embedded/app/services/bakery.py apps/forecast_embedded/app/services/runs.py pipelines/forecast_publish/production_dataset_refresh.py scripts/export_clickhouse_bakery_daily.py tests/test_forecast_embedded_access.py tests/test_export_clickhouse_bakery_daily.py --select=E,F,W
python -m py_compile scripts/build_prod_lead1_model_backfill.py
```

Results before adding the backfill script:

- focused pytest: `22 passed`
- ruff: passed
- backfill script py_compile: passed

Production:

- `scripts.verify_prod_deploy --env-file .env`: `VERIFY OK`
- `/`, `/health`, `/api/v1/dates`, `/api/v1/bakeries?date=2026-06-29`: OK after hotfixes
- lead-1 snapshots verified in ClickHouse for `2026-06-24..2026-06-28`

## Immediate Next Steps

1. Decide how the embedded date list should expose historical lead-1 dates:
   the current app date endpoint filters snapshots by active `source_run_id`,
   while the backfill runs intentionally have their own draft run ids.

Verification query:

```bash
cd /opt/demand-forecasting-model
.venv/bin/python - <<'PY'
from pipelines.forecast_publish.load_forecast_run import create_client
c = create_client(".env")
for table in [
    "bakery_forecast_day_snapshots",
    "sku_forecast_day_snapshots",
    "sku_forecast_hour_snapshots",
]:
    df = c.query_df(f"""
    select forecast_date, count() rows, uniqExact(source_run_id) runs,
           groupArrayDistinct(source_run_id) run_ids
    from {table}
    where lead_days = 1
      and forecast_date between '2026-06-24' and '2026-06-28'
    group by forecast_date
    order by forecast_date
    """)
    print("\\n" + table)
    print(df.to_string(index=False))
PY
```

## Do Not Do

- Do not run production forecast generation on Blackhole.
- Do not enable Blackhole forecast timers.
- Do not activate `backfill_*_h1` runs as the main run.
- Do not print `.env`, ClickHouse credentials, VibeCode keys, or SSH private keys.

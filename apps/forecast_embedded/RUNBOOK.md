# Embedded Forecast Runbook

## Scope

This runbook covers:

- deploying the embedded app to Blackhole
- loading forecast runs into ClickHouse
- comparing a candidate run to the active run
- activating a new run without changing UI code

The embedded app serves forecast data only. It does not train models and does not build forecasts on demand.

## App Location

- app directory: `apps/forecast_embedded`
- serving tables:
- `forecast_runs_embedded`
- `bakery_forecast_day_embedded`
- `sku_forecast_day_embedded`
- `sku_forecast_hour_embedded`
- `sku_hour_share_profile_smoothed_embedded`

## Required Runtime Configuration

Set these environment variables on the Blackhole server:

```env
APP_ENV=prod
APP_TITLE=Bakery Forecast Embedded
PORT=3000
BITRIX_EMBED_MODE=true

CLICKHOUSE_HOST=...
CLICKHOUSE_PORT=8443
CLICKHOUSE_USER=...
CLICKHOUSE_PASSWORD=...
CLICKHOUSE_DATABASE=...
```

Optional:

```env
ACTIVE_RUN_ID=
```

Leave `ACTIVE_RUN_ID` empty in normal production mode so the app reads the latest `active` run automatically.

## Blackhole Deploy

Recommended server:

- provider: `bitrix-cloud`
- runtime: `python311`
- plan: `bc-medium`

Deploy only the contents of `apps/forecast_embedded`:

- `app/`
- `requirements.txt`
- `README.md`
- `RUNBOOK.md`

Install command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 3000
```

## Smoke Test After Deploy

Check these endpoints:

1. `GET /health`
2. `GET /api/v1/runs/active`
3. `GET /api/v1/dates`
4. `GET /api/v1/bakeries?date=2026-05-18`
5. `GET /api/v1/bakeries/60/summary?date=2026-05-18`

Then check the UI:

1. `GET /`
2. `GET /bakery/60?date=2026-05-18`

## Forecast Run Lifecycle

### 1. Load a New Run

Load forecast artifacts from current CSV outputs into ClickHouse:

```bash
.venv\Scripts\python.exe pipelines\forecast_publish\load_forecast_run.py --run-id my_new_run
```

Default inputs:

- `data/processed/bakery_day_forecast.csv`
- `data/processed/sku_day_forecast_future_smoothed_bias_adj.csv`
- `data/processed/sku_hour_forecast_future_smoothed_bias_adj.csv`

### 2. Compare Candidate vs Active

Compare the candidate run with the current active run:

```bash
.venv\Scripts\python.exe pipelines\forecast_publish\compare_runs.py --candidate-run-id my_new_run
```

Optional explicit base run:

```bash
.venv\Scripts\python.exe pipelines\forecast_publish\compare_runs.py --base-run-id old_run --candidate-run-id my_new_run
```

What to review:

- total bakery forecast delta
- mean absolute bakery delta
- mean absolute SKU delta
- top bakery deltas
- top SKU deltas

### 3. Activate the New Run

If the comparison looks acceptable:

```bash
.venv\Scripts\python.exe pipelines\forecast_publish\activate_run.py --run-id my_new_run
```

After activation, the embedded app will read the new run automatically.

## Safe Update Pattern

Use this sequence for every forecast update:

1. build new forecast artifacts
2. load new run into ClickHouse
3. compare candidate run to active run
4. activate only after review

Do not overwrite old rows in place.

## Automated Nightly Mode

There is now a repository-level orchestration script for the serving pipeline:

```bash
.venv\Scripts\python.exe pipelines\forecast_publish\nightly_refresh.py
```

Default nightly behavior:

1. export raw sales from ClickHouse from `2025-01-01` through `yesterday`
2. rebuild `data/processed/bakery_daily_sales.csv`
3. run bakery forecast from `today` for `7` days ahead
4. apply existing bakery-hour and smoothed SKU-hour profiles
5. publish the new run into ClickHouse
6. activate the new run immediately

Operational assumptions:

- model retraining is **not** part of the nightly job
- profile rebuild is **not** part of the nightly job
- large SKU profile delivery to Blackhole should use ClickHouse storage, not deploy upload
- the job uses the existing:
  - `models/bakery_day_model.joblib`
  - `models/bakery_day_meta.joblib`
  - `reports/bakery_day_model_bias_by_bakery.csv`
  - `data/processed/bakery_hour_profile.csv`
  - `sku_hour_share_profile_smoothed_embedded` in ClickHouse

One-time profile load into ClickHouse:

```bash
.venv\Scripts\python.exe pipelines\forecast_publish\sku_hour_profile_store.py --mode load --truncate
```

If the nightly job runs with `--profile-source clickhouse`, it will export the profile
to a local cache file only when the cache is missing or when `--refresh-profile-cache`
is passed explicitly.

Useful flags:

```bash
.venv\Scripts\python.exe pipelines\forecast_publish\nightly_refresh.py --skip-export --skip-publish
```

This is useful for a local dry-run on already exported raw data.

On Windows, the repository includes a helper to register a daily scheduled task:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\register_nightly_forecast_task.ps1
```

## Model Updates vs UI Updates

### Model update

Examples:

- new features
- new bias adjustment
- new allocation logic
- new smoothing logic

Operationally:

1. generate new artifacts
2. load new run
3. compare
4. activate

No UI redeploy is required if the serving contract stays the same.

### UI update

Examples:

- new tables
- new charts
- new export view
- new optional metrics

Operationally:

1. deploy new app code
2. keep current active run unchanged

No new forecast run is required unless the UI depends on new fields that do not exist yet.

## Rollback

Rollback is done by re-activating the previous run:

```bash
.venv\Scripts\python.exe pipelines\forecast_publish\activate_run.py --run-id previous_good_run
```

No app redeploy is required for data rollback.

## Operational Notes

- The embedded app is read-only with respect to forecast data.
- ClickHouse is the source of truth for serving.
- The app should always read the active run unless a fixed `ACTIVE_RUN_ID` is explicitly set.
- Use preview/private access first, then attach the app to Bitrix placements after the app and data are verified.

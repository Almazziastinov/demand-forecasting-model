# Session Handoff - 2026-06-02 - VM Production and Bitrix Embedded App

## Context

We moved the forecast serving setup from local/manual artifacts to a working
production-style split:

1. A small VM runs scheduled production inference and writes forecast runs to
   ClickHouse.
2. A VibeCode / Bitrix Cloud Blackhole app serves the read-only API/UI from the
   same ClickHouse tables.

The user chose **option A** for now:

- keep production inference on the created VM;
- deploy frontend/read-only API on VibeCode so Bitrix24 can open it over the
  managed HTTPS app URL;
- do not expose the VM through nginx for the Bitrix24 UI path.

## Latest Relevant Commits

Pushed to `origin/master`:

```text
a086c1b fix: list forecast runs without nested aggregation
be0ca51 feat: serve forecast runs from vm api
dd687dc fix: replace existing forecast runs on publish
a1288ac fix: clear stale active forecast runs
970dc9d feat: add vm production inference deployment
```

## Production VM

VM resources:

```text
2 vCPU
2 GB RAM
40 GB disk
```

Project path:

```text
/opt/demand-forecasting-model
```

Linux user:

```text
forecast
```

The repository on the VM was pulled through:

```text
a086c1b fix: list forecast runs without nested aggregation
```

Runtime note:

- The VM venv currently uses Python `3.14.4`.
- LightGBM required system package `libgomp1`; installed successfully.
- The stack works, but for future hardening Python `3.12` would be preferable.

## VM Artifacts Copied

These non-git artifacts were copied to the VM:

```text
data/processed/bakery_daily_sales.csv                         ~50 MB
data/processed/bakery_daily_sales_uplifted.csv                ~70 MB
data/processed/bakery_hour_profile.csv                        ~3.6 MB
models/bakery_day_model.joblib                                ~22 MB
models/bakery_day_meta.joblib
models/bakery_day_model_uplifted.joblib                       ~22 MB
models/bakery_day_meta_uplifted.joblib
models/bakery_day_bias.json
models/bakery_day_bias_uplifted.json
```

Large SKU profile CSV is **not** required on the VM. SKU profiles are read from
ClickHouse:

```text
sku_hour_share_profile_smoothed_embedded
sku_hour_uplift_multiplier_embedded
```

## VM Environment

VM `.env` contains ClickHouse credentials and production defaults.

Important forecast defaults:

```text
FORECAST_SCENARIO=both
FORECAST_HORIZON_DAYS=14
FORECAST_UPLIFT_PROFILE_VERSION=sku_uplift_20260601
FORECAST_ACTIVATE_RUN=uplifted_norm
```

Do not commit `.env` or print secrets in handoffs.

## Production Inference

Entrypoint:

```text
pipelines/forecast_publish/run_production_inference.py
```

It supports:

```text
--scenario both
--scenario base_raw_uplift
--scenario uplifted_norm
```

The script now defaults to replacing existing rows for the same run id before
publishing, preventing duplicate forecast rows on repeated scheduled runs.

Successful manual/systemd run produced two ClickHouse runs:

```text
prod_base_bakery_raw_uplift_sku_20260601_h14
prod_uplifted_bakery_norm_uplift_sku_20260601_h14
```

Counts after clean publish:

```text
prod_uplifted_bakery_norm_uplift_sku_20260601_h14
  forecast_runs_embedded         2   # draft + active metadata rows
  bakery_forecast_day_embedded   3038
  sku_forecast_day_embedded      425604
  sku_forecast_hour_embedded     3096400

prod_base_bakery_raw_uplift_sku_20260601_h14
  forecast_runs_embedded         1   # draft metadata row
  bakery_forecast_day_embedded   3038
  sku_forecast_day_embedded      425604
  sku_forecast_hour_embedded     3096400
```

Active/default run:

```text
prod_uplifted_bakery_norm_uplift_sku_20260601_h14
```

Active row verification showed only this run with `status = 'active'`.

## VM systemd

Forecast service:

```text
/etc/systemd/system/forecast-production.service
```

Timer:

```text
/etc/systemd/system/forecast-production.timer
```

Timer enabled:

```text
NEXT Wed 2026-06-03 03:30:00 UTC
```

That is `06:30` Moscow time.

Manual systemd run succeeded:

```text
PRODUCTION INFERENCE COMPLETE
uplifted_norm: prod_uplifted_bakery_norm_uplift_sku_20260601_h14 active
memory peak: ~960.6 MB
wall clock: ~1min 54s
```

Read-only API service was also installed on VM:

```text
forecast-embedded-api.service
```

It listens on:

```text
127.0.0.1:3000 / 0.0.0.0:3000
```

VM local checks passed:

```text
/health
/api/v1/runs/active
/api/v1/runs
```

This VM API is not the final Bitrix24 frontend route; it is useful for local
debugging and can stay running.

## Embedded API/UI Code

App path:

```text
apps/forecast_embedded
```

Important changes:

- API reads ClickHouse env names directly:
  - `CLICKHOUSE_HOST`
  - `CLICKHOUSE_PORT`
  - `CLICKHOUSE_USER`
  - `CLICKHOUSE_PASSWORD`
  - `CLICKHOUSE_DATABASE`
  - `CLICKHOUSE_SECURE`
  - `CLICKHOUSE_VERIFY`
- Added `/api/v1/runs` for run/model switching.
- `/api/v1/dates` accepts optional `run_id`.
- Bakery endpoints accept optional `run_id`.
- Export endpoint accepts optional `run_id`.
- Minimal UI has a run selector.

Key API endpoints:

```text
GET /health
GET /api/v1/runs
GET /api/v1/runs/active
GET /api/v1/dates
GET /api/v1/dates?run_id=...
GET /api/v1/bakeries?date=YYYY-MM-DD
GET /api/v1/bakeries?date=YYYY-MM-DD&run_id=...
GET /api/v1/bakeries/{bakery_id}/summary?date=YYYY-MM-DD
GET /api/v1/bakeries/{bakery_id}/sku-day?date=YYYY-MM-DD
GET /api/v1/bakeries/{bakery_id}/sku-hour?date=YYYY-MM-DD&product_id=...
```

## VibeCode / Bitrix Cloud Deploy

Existing VibeCode server:

```text
id: 82bb03a8-c356-4225-97a4-a1540cdc29e6
name: bakery-forecast-embedded
provider: bitrix-cloud
plan: bc-medium
resources: 2 vCPU / 4 GB RAM / 40 GB SSD
status: running
blackholeStatus: CONNECTED
accessPolicy: PORTAL
localPort: 3000
appUrl: https://app-8613ac40f10d.vibecode.bitrix24.tech
```

Deployment decision:

- Deploy only the read-only FastAPI/UI app to VibeCode.
- Keep forecast generation on the VM.
- VibeCode app reads the same ClickHouse serving tables.

VibeCode deploy succeeded via `/v1/infra/servers/:id/deploy`.

Deploy used:

```text
source.url: https://github.com/Almazziastinov/demand-forecasting-model/archive/refs/heads/master.tar.gz
app dir: /opt/app/demand-forecasting-model-master/apps/forecast_embedded
serviceName: forecast-embedded
port: 3000
healthPath: /health
```

Runtime issue:

- VibeCode `runtime=python311` failed because package `python3.11` was not
  available on the image.
- Workaround used: omit runtime and install system Python packages in
  `preStart`:

```text
apt-get update -qq && apt-get install -y python3 python3-venv python3-pip
```

Install command:

```text
cd /opt/app/demand-forecasting-model-master/apps/forecast_embedded &&
python3 -m venv .venv &&
.venv/bin/python -m pip install --upgrade pip &&
.venv/bin/python -m pip install -r requirements.txt
```

Start command:

```text
cd /opt/app/demand-forecasting-model-master/apps/forecast_embedded &&
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 3000
```

Deploy response:

```text
success: true
serviceName: forecast-embedded
status: running
appUrl: https://app-8613ac40f10d.vibecode.bitrix24.tech
healthcheck: ok
tunnel_routing: ok
```

Browser check:

- User opened the VibeCode app URL and confirmed it works.

Curl without authentication returns Blackhole login / `BH_LOGIN_REQUIRED`, which
is expected for `accessPolicy=PORTAL`.

For external smoke tests, mint an API bearer token:

```text
POST /v1/infra/servers/:id/access-tokens { "mode": "api-bearer" }
```

Then call the app URL with:

```text
Authorization: Bearer <token>
```

## Current Architecture

```text
Production VM
  systemd timer
  run_production_inference.py
  ClickHouse forecast publish

ClickHouse
  forecast_runs_embedded
  bakery_forecast_day_embedded
  sku_forecast_day_embedded
  sku_forecast_hour_embedded
  sku_hour_share_profile_smoothed_embedded
  sku_hour_uplift_multiplier_embedded

VibeCode Blackhole App
  FastAPI embedded API/UI
  HTTPS app URL for Bitrix24
  reads ClickHouse only
```

## Important Operational Notes

1. Do not publish the same run id manually with older code; use latest
   `run_production_inference.py`, which replaces existing run rows by default.
2. `forecast_runs_embedded` may have both `draft` and `active` rows for the
   active run id. This is expected. Forecast data tables should not be
   duplicated.
3. `activate_run.py` was fixed to delete stale `active` rows after archiving,
   leaving a single current active row.
4. VibeCode app is protected by Blackhole gateway. Anonymous curl will not hit
   the app unless an access token is supplied or the request is authenticated
   through VibeCode/Bitrix24.
5. Keep ClickHouse and VibeCode API keys out of git and handoffs.

## Next Steps

1. Bind the VibeCode app URL into Bitrix24 as an application/placement.
2. Decide where the app should appear:
   - left menu / app page;
   - CRM detail tab;
   - another Bitrix24 placement.
3. If user-specific permissions are needed, add support for
   `X-Vibe-Authorization` from Blackhole gateway and call VibeCode
   `/v1/users/me`.
4. Improve frontend UX beyond the current minimal table UI:
   - model/run switcher;
   - date selector;
   - bakery search;
   - daily total chart;
   - hourly chart;
   - SKU table with export.
5. Optionally disable or keep VM `forecast-embedded-api.service`; it is not
   required for Bitrix24 once VibeCode app is serving UI/API.

## Commands Useful For Resuming

VM status:

```bash
systemctl status forecast-production.timer --no-pager
systemctl status forecast-production.service --no-pager
systemctl status forecast-embedded-api.service --no-pager
```

VM active run check:

```bash
cd /opt/demand-forecasting-model
sudo -u forecast .venv/bin/python - <<'PY'
from pipelines.forecast_publish.load_forecast_run import create_client

client = create_client(".env")
df = client.query_df("""
select run_id, status, horizon_start, horizon_end, generated_at
from forecast_runs_embedded
where status = 'active'
order by generated_at desc
""")
print(df)
PY
```

VibeCode app URL:

```text
https://app-8613ac40f10d.vibecode.bitrix24.tech
```

VibeCode server id:

```text
82bb03a8-c356-4225-97a4-a1540cdc29e6
```


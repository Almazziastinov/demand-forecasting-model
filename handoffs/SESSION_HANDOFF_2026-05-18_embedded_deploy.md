# SESSION HANDOFF 2026-05-18 - EMBEDDED VIEWER / BLACKHOLE DEPLOY

## Summary

Built and deployed a standalone embedded-viewer app for the current bakery-driven forecasting pipeline.

Status:
- Blackhole server created and running
- FastAPI viewer app deployed successfully
- ClickHouse-backed serving layer works on the server
- app is accessible by direct URL
- Vibe OAuth app created and published
- Bitrix24 left-menu placement is still **not attached**; app does not appear in portal menu yet

## Deployed Runtime

Blackhole server:
- server id: `82bb03a8-c356-4225-97a4-a1540cdc29e6`
- name: `bakery-forecast-embedded`
- app URL: `https://app-8613ac40f10d.vibecode.bitrix24.tech`
- runtime: `python311`
- plan: `bc-medium`
- access policy: `PORTAL`

App service:
- systemd service name: `bakery-forecast-embedded`
- app listens on port `3000`
- health endpoint: `/health`

## New App Code

New serving/UI app:
- `apps/forecast_embedded/`

Main files:
- `apps/forecast_embedded/app/main.py`
- `apps/forecast_embedded/app/settings.py`
- `apps/forecast_embedded/app/db.py`
- `apps/forecast_embedded/app/routers/*`
- `apps/forecast_embedded/app/services/*`
- `apps/forecast_embedded/sql/schema.sql`
- `apps/forecast_embedded/README.md`
- `apps/forecast_embedded/RUNBOOK.md`

Forecast run ops:
- `pipelines/forecast_publish/load_forecast_run.py`
- `pipelines/forecast_publish/activate_run.py`
- `pipelines/forecast_publish/compare_runs.py`

## Serving Storage

Serving tables in ClickHouse:
- `forecast_runs_embedded`
- `bakery_forecast_day_embedded`
- `sku_forecast_day_embedded`
- `sku_forecast_hour_embedded`

Loaded active run:
- `run_id = first_embedded_run`

Validated:
- active run lookup works
- dates endpoint works
- bakery list endpoint works
- app can read live serving data on the deployed server

## Important Runtime Fixes

### 1. Blackhole Python install

The `python311` runtime did not have ready-to-use `pip`.
Deploy was fixed by:
- installing `python3.12-venv`
- creating local `/opt/app/.venv`
- installing app dependencies inside that virtualenv

### 2. ClickHouse TLS

Server-side ClickHouse connection initially failed with:
- `SSLCertVerificationError`
- `self-signed certificate in certificate chain`

Fix applied in app config:
- `CLICKHOUSE_SECURE=true`
- `CLICKHOUSE_VERIFY=false`

Code updated to support:
- `clickhouse_secure`
- `clickhouse_verify`

## Smoke Checks Completed

Verified on deployed server:
- `GET /health` -> `{"ok":true}`
- `GET /api/v1/runs/active`
- `GET /api/v1/dates`
- `GET /api/v1/bakeries?date=2026-05-18`

Known real sample:
- bakery `60`
- `Мусина 68 Казань`
- `forecast_final = 3381.698914631409`

## Vibe / Bitrix App State

OAuth app created:
- title: `Bakery Forecast Embedded`
- app id: `40e205b0-a0bf-4f5a-bfee-80c109b5948a`
- client id: `local.6a0b1a3ee359e3.69188261`

Published state:
- app has `appUrl = https://app-8613ac40f10d.vibecode.bitrix24.tech`
- app scope: `placement`
- category chosen in UI: `Меню в Битрикс24`

Problem:
- app does **not** appear in Bitrix24 left menu
- `/v1/apps` shows app record correctly
- `/v1/placements` still returns an empty placements list for this app

Likely blocker:
- Vibe OAuth / placement install flow did not complete in a way that produces a registered `REST_APP_URI`
- direct `publish` succeeded, but placement registration is still absent
- manual `oauth/authorize` link flow was unreliable for the user

## Current Recommended Usage

Use the app directly by URL for now:
- `https://app-8613ac40f10d.vibecode.bitrix24.tech`

This is the current working delivery path until placement binding is resolved.

## Suggested Next Steps

### Option A - Continue with direct URL for now

Use the deployed viewer as-is and postpone Bitrix menu embedding.

### Option B - Finish Bitrix embedding later

Continue investigating the Vibe OAuth app installation / placement bind flow.

Things to check next:
- whether Vibe UI has a hidden or secondary "install in portal" step after publish
- whether `REST_APP_URI` bind requires a valid `vibe_session_*` Bearer session that was never created
- whether support needs to manually inspect why placements remain empty after publish

## Support Draft

If escalation to VibeCode support is needed, use this:

```text
Created and published OAuth app "Bakery Forecast Embedded" with placement scope and "Меню в Битрикс24".
Blackhole app URL: https://app-8613ac40f10d.vibecode.bitrix24.tech
App ID: 40e205b0-a0bf-4f5a-bfee-80c109b5948a
Client ID: local.6a0b1a3ee359e3.69188261

Problem:
- publish succeeds
- appUrl is saved
- app does not appear in Bitrix24 left menu
- API placements remain empty
- OAuth authorize flow did not produce a reliable placement install/bind result

Need help finishing REST_APP_URI menu binding for this app.
```

## Git Scope For This Session

Intended commit scope:
- `apps/forecast_embedded/**`
- `pipelines/forecast_publish/**`
- this handoff file

Do not mix with unrelated dedup work in:
- `src/experiments_v2/build_bakery_daily_dataset.py`
- `tests/test_build_bakery_daily_dataset.py`
- `src/experiments_v2/raw_sales_dedup.py`

# Session Handoff — 2026-05-19 — Blackhole Nightly Setup

## Scope

This handoff records the server-side setup work for moving the forecast pipeline onto the Blackhole host that already serves the embedded viewer.

Server:

- name: `bakery-forecast-embedded`
- server id: `82bb03a8-c356-4225-97a4-a1540cdc29e6`
- mode: `BLACKHOLE`

## What Was Implemented Locally

The repository now contains a nightly orchestration path and related server helpers:

- `pipelines/forecast_publish/nightly_refresh.py`
- `pipelines/forecast_publish/sku_hour_profile_store.py`
- `tests/test_nightly_refresh.py`
- `scripts/build_blackhole_forecast_bundle.py`
- `scripts/register_nightly_forecast_task.ps1`

Related updates were also made in:

- `pipelines/forecast_publish/load_forecast_run.py`
- `pipelines/forecast_publish/activate_run.py`
- `scripts/export_clickhouse_checks.py`
- `apps/forecast_embedded/sql/schema.sql`
- `apps/forecast_embedded/RUNBOOK.md`

Important design decision:

- the large `sku_hour_share_profile_smoothed.csv` is no longer expected to be uploaded to Blackhole
- instead it is stored in ClickHouse table `sku_hour_share_profile_smoothed_embedded`
- `nightly_refresh.py` can read it via `--profile-source clickhouse`

## What Was Applied On Blackhole

Using Deploy API `exec`, the following were set up on the server:

1. Repository clone at `/opt/forecast_job`
2. Overlay of the new/changed forecast pipeline files from the local workspace
3. Forecast-specific virtualenv at `/opt/forecast_job/.venv`
4. Forecast dependencies installed and import-verified:
   - `clickhouse_connect`
   - `joblib`
   - `lightgbm`
   - `numpy`
   - `pandas`
   - `sklearn`
5. Forecast env file at `/opt/forecast_job/.env`
6. `systemd` unit installed:
   - `/etc/systemd/system/bakery-forecast-nightly.service`
7. `systemd` timer installed and enabled:
   - `/etc/systemd/system/bakery-forecast-nightly.timer`

## Nightly Schedule

The timer is active and scheduled for Moscow midnight:

- `OnCalendar=*-*-* 00:00:00 Europe/Moscow`

The server itself runs in UTC, so the next trigger was verified as:

- `2026-05-19 21:00:00 UTC`

That corresponds to:

- `2026-05-20 00:00:00 Europe/Moscow`

Regular nightly unit behavior:

- `nightly_refresh.py --profile-source clickhouse`

This means the normal daily run is intended to:

1. export sales history through yesterday
2. rebuild `bakery_daily_sales.csv`
3. reuse existing model artifacts if present
4. reuse existing bakery hour profile if present
5. read SKU hour profile from ClickHouse
6. publish a new run to ClickHouse
7. activate it

## Bootstrap Run

A one-time bootstrap job was started separately to create the initial server-side artifacts:

- transient unit: `bakery-forecast-bootstrap.service`
- command:
  - `nightly_refresh.py --profile-source clickhouse --rebuild-bakery-hour-profile --train-model`

At start time, the bootstrap unit was confirmed running and had begun the raw export phase:

- first observed log line:
  - `[1/17] Querying 2025-01-01 .. 2025-01-31`

## Current Unknown

The final completion state of the bootstrap run was **not** confirmed in this session.

Reason:

- Blackhole `exec` became unstable during the long-running bootstrap process
- later status/log polling was interrupted before a final confirmation step completed

At the last successful ClickHouse check, the latest visible run in `forecast_runs_embedded` was still:

- `first_embedded_run`

So at this handoff point there are two possibilities:

1. bootstrap was still running and had not published yet
2. bootstrap failed before publish and needs a post-check

## Practical Next Step

When resuming, do this first:

1. check `bakery-forecast-bootstrap.service` status/logs on Blackhole
2. check whether a new `nightly_...` run appeared in `forecast_runs_embedded`
3. if bootstrap failed, rerun it once
4. if bootstrap succeeded, no more setup is required and the nightly timer should take over automatically

## Operational Notes

- Deploy API on this Blackhole environment has a practical payload limit and rate limit
- large file delivery must be chunked or avoided
- repeated `exec` bursts can return `429 Too Many Requests`
- long `exec`/log polling can also terminate even while the underlying server process continues running


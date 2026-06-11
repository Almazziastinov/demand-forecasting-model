# Session Handoff - 2026-06-11 - Embedded History Snapshots

## Scope

Embedded VibeCode/Bitrix frontend API now reads historical forecast values from ClickHouse forecast snapshot tables.

This complements the production VM work that writes:

- `bakery_forecast_day_snapshots`
- `sku_forecast_day_snapshots`
- `sku_forecast_hour_snapshots`

## Commit

Pushed to `origin/master`:

```text
711973e feat: read embedded history from lead snapshots
```

Changed files:

```text
apps/forecast_embedded/app/services/bakery.py
apps/forecast_embedded/app/services/runs.py
tests/test_forecast_embedded_access.py
```

## Behavior

The embedded service source queries now combine:

- active-run tables for the current forecast horizon;
- latest snapshot rows with `lead_days = 1` for historical dates.

If both active-run and snapshot rows exist for the same date/key, active-run rows win via a higher sort priority.

`get_run_dates()` now returns the union of active-run bakery forecast dates and `bakery_forecast_day_snapshots where lead_days = 1`, so the UI date selector can see historical snapshot dates as they accumulate.

Important limitation: snapshot history starts from the date when snapshot publishing was introduced. Dates before that do not have true `lead_days = 1` historical forecasts unless backfilled separately.

## Local Validation

```text
.venv\Scripts\python.exe -m pytest tests\test_forecast_embedded_access.py -v
8 passed

.venv\Scripts\python.exe -m ruff check apps\forecast_embedded\app\services\bakery.py apps\forecast_embedded\app\services\runs.py tests\test_forecast_embedded_access.py --select=E,F,W
All checks passed
```

## VibeCode Deployment

Target:

```text
server id: 82bb03a8-c356-4225-97a4-a1540cdc29e6
name: bakery-forecast-embedded
appUrl: https://app-8613ac40f10d.vibecode.bitrix24.tech
```

Deployment method:

- VibeCode `/v1/infra/servers/:id/exec?stream=false`
- downloaded GitHub `master` tarball;
- copied `apps/forecast_embedded/app` to `/opt/app/app`;
- copied `requirements.txt`, `README.md`, `RUNBOOK.md` to `/opt/app`;
- ran `py_compile` for changed service files;
- kept duplicate `forecast-embedded.service` disabled;
- restarted `app.service`.

Runtime verification:

```text
snapshot_marker=yes
lead_marker=yes
app_service=active
dup_service=inactive
health=200
health_body={"ok":true}
```

Internal API smoke:

```text
GET /api/v1/bakeries?date=2026-06-11
status=200
items=186
first_bakery=89
first_forecast=3642.5770192372943
```

Historical smoke for `2026-06-10` returned HTTP 200 but `items=0`, which is expected because true `lead_days = 1` snapshots were not being stored before the current rollout.


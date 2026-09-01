# Blackhole Pilot UI Release Candidate — 2026-08-20

Status: deployed and verified on 2026-08-20.

## Boundary

This release updates only pilot-management presentation and report-reading
logic on Blackhole. It does not deploy forecast generation, change ClickHouse
schemas, rebuild reports, or modify the pilot publisher.

## Files To Deploy

| Local source | Blackhole target |
| --- | --- |
| `apps/forecast_embedded/app/routers/pilot_management.py` | `/opt/app/app/routers/pilot_management.py` |
| `apps/forecast_embedded/app/static/app.js` | `/opt/app/app/static/app.js` |
| `apps/forecast_embedded/app/templates/layout.html` | `/opt/app/app/templates/layout.html` |
| `apps/forecast_embedded/app/templates/pilot_management.html` | `/opt/app/app/templates/pilot_management.html` |
| `apps/forecast_embedded/app/templates/pilot_bakery.html` | `/opt/app/app/templates/pilot_bakery.html` |
| `apps/forecast_embedded/app/templates/pilot_bakery_week.html` | `/opt/app/app/templates/pilot_bakery_week.html` |
| `src/pilot_management_service.py` | `/opt/src/pilot_management_service.py` |

## Explicitly Excluded

- `/opt/app/app/main.py`: Blackhole has a deployment-layout-specific version;
  replacing it with the workstation file could break `/opt/src` imports.
- `/opt/app/app/db.py`: Blackhole has connection recovery behavior not present
  in the current workstation file. Preserve the deployed version.
- Files whose hashes already match Blackhole: auth, settings, pilot config,
  UI router, bakery service, index/config/SKU templates, and the remaining
  `src/pilot_*` modules.
- `scripts/publish_pilot_forecast.py` and all production writer files.

## Access Contract

All pilot-management routes require `AuthContext.is_pilot_user`. This permits
admins and user ids configured through `PILOT_USER_IDS`; other users receive
HTTP 403.

## Verification Completed Locally

- Ruff E/F/W: passed.
- Pilot/access test selection: 70 passed.
- Local Uvicorn smoke test: `/health` HTTP 200 and `/pilot` HTTP 200.
- Development health response confirmed `app_env=dev` and table suffix `_dev`.

## Blackhole Recovery Point

Full pre-release copy:

`/opt/backups/codex_20260820_before_pilot_ui`

It contains `/opt/app`, `/opt/src`, `/opt/app/.env`, and the current
`/opt/reports/pilot_management_summary`. File-count and `.env` hash checks
ended with `BACKUP_VERIFY_OK`.

## Post-Deploy Checks

1. Restart only `app.service`.
2. Confirm `/health` returns HTTP 200 with `app_env=prod` and empty suffix.
3. Confirm both Blackhole forecast timers remain disabled and inactive.
4. Confirm all pilot routes remain registered.
5. Confirm an admin and a real `PILOT_USER_IDS` user can open `/pilot`.
6. Confirm an unrelated portal user receives HTTP 403.
7. Check application logs for ClickHouse, template, and import errors.

## Rollback

Restore the seven deployed targets from
`/opt/backups/codex_20260820_before_pilot_ui`, restart `app.service`, and repeat
the health, route, timer, and access checks.

# Services

Last updated: 2026-07-11

## Service Ownership Matrix

| Service | Environment | Role | May write forecast runs? | Status |
| --- | --- | --- | --- | --- |
| Production forecast VM | `201.51.7.24` | Generates and publishes forecasts | Yes | Active |
| ClickHouse | External database | Serving tables and snapshots | N/A | Active |
| VibeCode/Blackhole app | `bakery-forecast-embedded` | Embedded read-only API/UI | No | Active |
| Baking plan package | In-process, mounted on Blackhole app | Generates per-bakery baking-window Excel plan | No | Active (MILP rebuild deployed 2026-07-11) |
| Legacy Flask app | `web/app.py` | Local/demo legacy app | No prod role | Legacy |

## Production Forecast VM

- SSH target: `root@201.51.7.24`
- Repo path: `/opt/demand-forecasting-model`
- Python env: `/opt/demand-forecasting-model/.venv`
- Env file: `/opt/demand-forecasting-model/.env`
- Primary command:

```bash
.venv/bin/python -m pipelines.forecast_publish.run_production_inference \
  --env-file .env
```

The systemd unit expands the production settings from `.env` and command-line
flags. See `CURRENT_STATE.md` for the current scenario and verification command.

## VibeCode / Blackhole

- Server id: `82bb03a8-c356-4225-97a4-a1540cdc29e6`
- Server name: `bakery-forecast-embedded`
- App URL: `https://app-8613ac40f10d.vibecode.bitrix24.tech`
- Runtime role: serve the embedded FastAPI/UI from `/opt/app`.
- Historical forecast job path: `/opt/forecast_job`

The historical `/opt/forecast_job` tree may still exist. It must not be treated
as the production writer. Forecast timers there must stay disabled.

## Baking Plan Package

- Repo path: `apps/baking_plan/` (sibling package to `apps/forecast_embedded/app`,
  not a subpackage of it — see `apps/baking_plan/README.md`).
- Runtime role: generates the per-bakery baking-window Excel plan, mounted
  in-process into the Blackhole `app.service` via
  `apps/forecast_embedded/app/main.py` (`baking_plan.router.router`).
- Not a separate process/port. Rebuilt from scratch 2026-07-09, deployed
  2026-07-11; see `DECISIONS.md` for the package-vs-service rationale and
  `CURRENT_STATE.md` for deploy details.
- On the Blackhole VM the sibling-package layout is mirrored as `/opt/app`
  (= local `apps/forecast_embedded/`) and `/opt/baking_plan` (= local
  `apps/baking_plan/`), both directly under `/opt` so `app/main.py`'s
  `sys.path` insert of its grandparent directory resolves `import
  baking_plan` correctly.
- Requires `scipy` (MILP solver, `scipy.optimize.milp`) — declared in
  `apps/forecast_embedded/requirements.txt`, must be installed into
  `/opt/app/.venv` on any redeploy.
- Deploy: any Blackhole deploy touching this feature must replace both
  `apps/forecast_embedded/app/*` (→ `/opt/app/app`) and `apps/baking_plan/*`
  (→ `/opt/baking_plan`) — there is no dedicated deploy script yet, uploads
  have been manual (full `git archive`/tarball of `master` + directory
  replace, see `CURRENT_STATE.md`).

## ClickHouse

ClickHouse is the production serving store. The forecast writer publishes run
metadata and forecast snapshots there. The embedded app reads from those tables.

Known serving/snapshot tables include:

- `forecast_runs_embedded`
- `bakery_forecast_day_embedded`
- `forecast_day_context_embedded`
- `sku_forecast_day_embedded`
- `sku_forecast_hour_embedded`
- `bakery_forecast_day_snapshots`
- `sku_forecast_day_snapshots`
- `sku_forecast_hour_snapshots`

## Local Development

Use the repo root on the workstation for code changes. Do not infer production
state from local generated files without checking VM and ClickHouse.

Useful commands:

```bash
ruff check src/ web/ tests/ --select=E,F,W
pytest tests/ -v
```

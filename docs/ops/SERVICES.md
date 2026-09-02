# Services

Last updated: 2026-09-02

## Service Ownership Matrix

| Service | Environment | Role | May write forecast runs? | Status |
| --- | --- | --- | --- | --- |
| Production forecast VM | `201.51.7.24` | Generates and publishes forecasts | Yes | Active |
| Pilot management report job | Production forecast VM | Builds validated CSV statistics and publishes them atomically to Blackhole | No | Active; daily 05:00 UTC |
| ClickHouse | External database | Serving tables and snapshots | N/A | Active |
| VibeCode/Blackhole app | `bakery-forecast-embedded` | Embedded read-only API/UI | No | Active |
| Baking plan package | In-process, mounted on Blackhole app | Generates per-bakery baking-window Excel plan | No | Active — MILP-based, deployed 2026-07-21 (HTTP 200 smoke-tested, see `CURRENT_STATE.md`) |
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

The main systemd command refreshes datasets and builds an inactive
`prod_base_bakery_norm_recent_*` source run. The drop-in
`/etc/systemd/system/forecast-production.service.d/direct-alpha.conf` runs
`pipelines.forecast_publish.direct_alpha_production` as `ExecStartPost`; only
that successful post-process activates the served
`prod_direct_alpha_025_YYYYMMDD_h14` run (`model_version=direct_alpha_025_v1`).
The source run is an implementation input, not the current production model.
See `CURRENT_STATE.md` for the authoritative model description and live run.

For Direct production, `.env` sets `FORECAST_PROFILE_MAX_AGE_DAYS=-1`. This
disables only the retired hourly SKU-profile age check; assortment freshness,
dataset refresh, Direct post-processing, activation, and final verification
remain required.

### Pilot management report job

- Timer: `pilot-management-report.timer`, daily at `05:00 UTC` (`08:00 MSK`).
- Service: `pilot-management-report.service`, user/group `forecast`.
- Entry point:
  `/opt/demand-forecasting-model/scripts/run_pilot_management_report_job.py`.
- Source interval: fixed pilot start `2026-07-23` through Moscow yesterday.
- Destination: Blackhole `/opt/reports/pilot_management_summary`.
- The job reads ClickHouse and publishes report files only. It does not create
  or activate forecast runs. Deployment is validated and atomic, with a
  pre-swap backup under `/opt/backups`.

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
- Not a separate process/port. Rebuilt from scratch 2026-07-09, deployed as
  a MILP allocator 2026-07-11, reverted to template-driven 2026-07-14,
  **restored to MILP and redeployed 2026-07-21** (commit `84557b6`, see
  `CURRENT_STATE.md` for deploy detail).
- Current mode: **MILP-based**. `service.py` calls `demand_milp.build_sku_demand`
  → `algorithms/milp.allocate_milp_detailed` → `rendering_milp.render_workbook`
  (builds xlsx from scratch, no template mutation). The Excel template is
  used only to read the bakery's window time-slot structure.
- On the Blackhole VM the sibling-package layout is mirrored as `/opt/app`
  (= local `apps/forecast_embedded/`) and `/opt/baking_plan` (= local
  `apps/baking_plan/`), both directly under `/opt` so `app/main.py`'s
  `sys.path` insert of its grandparent directory resolves `import
  baking_plan` correctly.
- Requires `scipy==1.17.1` (`scipy.optimize.milp`, HiGHS backend) — already
  installed in the Blackhole venv since the 2026-07-11 MILP deploy.
- Required ClickHouse tables: `baking_sku_meta`, `baking_capacity_config`,
  `baking_category_molding_minutes` — all present since 2026-07-11.
- Deploy: replace both `apps/forecast_embedded/app/*` (→ `/opt/app/app`) and
  `apps/baking_plan/*` (→ `/opt/baking_plan`) via GitHub tarball. No
  `pip install` needed (all dependencies already installed). See
  `CURRENT_STATE.md` for the full step-by-step.
- Rollback: `/opt/app/app_backup_20260721_milp` and
  `/opt/baking_plan_backup_20260721_milp` on the Blackhole VM.

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

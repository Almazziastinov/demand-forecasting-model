# Current Project State

Last updated: 2026-07-13

## Summary

The production forecast writer is the VM only. VibeCode/Blackhole is a
read-only embedded UI/API over ClickHouse and must not run forecast generation.

## Production Source Of Truth

- Production VM: `root@201.51.7.24`
- VM path: `/opt/demand-forecasting-model`
- VM hostname observed: `msk-1-vm-tpez`
- VM systemd timer: `forecast-production.timer`
- VM timer schedule: daily `03:30:00 UTC`
- VM repo state observed: behind origin by docs/handoff only; production code
  was effectively current during the 2026-06-28 audit.
- **Known issue (2026-07-13):** `git pull` on the VM currently fails —
  `docs/ops/*.md` are owned by `root:root` (the `forecast` user can't
  unlink them), and the working tree also has uncommitted baking-plan
  drift unrelated to this VM's own job (files were placed directly,
  bypassing git, presumably from Blackhole-deploy tooling being pointed
  at the wrong host). Neither has been fixed — the 2026-07-13 rolling-bias
  deploy below worked around it with a targeted SFTP file copy instead of
  `git pull`. Whoever owns the baking-plan deploy tooling should confirm
  this VM was an intentional target and either commit+clean up the drift
  or stop touching this host; `chown` on the docs/ops files needs a
  decision on why they went root-owned before just reverting it.

## Embedded App

- VibeCode server id: `82bb03a8-c356-4225-97a4-a1540cdc29e6`
- VibeCode server name: `bakery-forecast-embedded`
- VibeCode app URL: `https://app-8613ac40f10d.vibecode.bitrix24.tech`
- Mode: `BLACKHOLE`
- Role: read-only FastAPI/UI for Bitrix24 users.
- Forecast generation on VibeCode/Blackhole is forbidden.

## Active Forecast

- Active run on 2026-07-01: `prod_base_bakery_no_sku_uplift_20260701_h14`
- Scenario: `base_no_sku_uplift` (new scenario, added 2026-07-01)
  - Bakery-day model: **base** (`bakery_day_model.joblib`, no bakery-level uplift)
  - SKU-hour allocation: raw `sku_hour_share_profile_smoothed_embedded`
    (no floor-uplift, see below)
  - SKU-hour uplift multiplier: **disabled** (`use_raw_uplift_multiplier=False`)
- `.env` on the VM updated: `FORECAST_SCENARIO=base_no_sku_uplift`,
  `FORECAST_ACTIVATE_RUN=base_no_sku_uplift` (nightly timer will keep using
  this scenario going forward)
- Horizon days: `14`
- Recent correction mode: `runner_city_prior_soft_weekpart`
- Recent correction days: `30`
- Recent sales table: `mart_sales_60d`
- Dataset refresh: enabled on the VM (`FORECAST_REFRESH_DATASETS=1`)
- Weather refresh: enabled on the VM (`FORECAST_REFRESH_WEATHER=1`)
- Bakery-day bias correction: **rolling** (trailing 7-day window,
  recomputed every run), not the old static one-time snapshot — see
  "Rolling Bakery-Day Bias Correction Deployed" below.

Previous scenario (`uplifted_norm`, active through 2026-06-29..2026-06-30) is
still defined in `SCENARIOS` for rollback if needed.

Observed active snapshot rows after the 2026-06-29 refresh:

- `bakery_forecast_day_snapshots`: `2842`
- `sku_forecast_day_snapshots`: `460708`
- `sku_forecast_hour_snapshots`: `5014812`

Observed active weather context after the 2026-06-29 refresh:

- `forecast_day_context_embedded`: `126` rows
- Date range: `2026-06-29` through `2026-07-12`
- Default-weather rows (`temp_mean=10`, `precipitation=0`,
  `is_bad_weather=0`): `0`

## Current Timers

Must be enabled and active:

- VM `forecast-production.timer`

Must remain disabled and inactive:

- Blackhole `forecast-production.timer`
- Blackhole `forecast-production.service`
- Blackhole `bakery-forecast-nightly.timer`
- Blackhole `bakery-forecast-nightly.service`

## Important Incident Fixed On 2026-06-28

The active ClickHouse run was being overwritten after the VM job by an old
Blackhole timer. The stale writer ran from VibeCode/Blackhole host
`fhmab3h2o3lo0jqd552k`, path `/opt/forecast_job`, and loaded:

- stale run: `prod_uplifted_bakery_norm_uplift_sku_20260601_h14`
- source IP in ClickHouse query log: `84.201.174.223`

Action taken:

- Re-activated fresh run `prod_uplifted_bakery_norm_uplift_sku_20260623_h14`.
- Disabled Blackhole `forecast-production.timer`.
- Verified VM timer remains active and ClickHouse active run is consistent.

## Active Run Repair On 2026-06-29

The embedded app returned `Forecast run not found` because production
`forecast_runs_embedded` had no `status = 'active'` row. The expected run was
present and active in the `_dev` serving tables, while the production table only
contained archived/draft runs.

Action taken:

- Verified VM `forecast-production.timer` was still enabled and active.
- Copied run `prod_uplifted_bakery_norm_uplift_sku_20260623_h14` from `_dev`
  serving/snapshot tables into production serving/snapshot tables.
- Activated that run through `pipelines.forecast_publish.activate_run`.
- Verified `scripts.verify_prod_deploy --env-file .env` ends with
  `VERIFY OK: env, summary, and active run are consistent`.

## Fresh Data And Weather Refresh On 2026-06-29

ClickHouse data availability was verified:

- `mart_sales_60d`: `2026-06-01` through `2026-06-29`
- `Svezhar.fct_check_lines`: `2025-12-01` through `2026-06-29`

The production VM was manually refreshed from ClickHouse data through
`2026-06-28`, producing and activating
`prod_uplifted_bakery_norm_uplift_sku_20260629_h14`.

Action taken:

- Ran production inference with dataset refresh from `2025-12-01`.
- Refreshed weather features through `2026-07-12`.
- Rebuilt and loaded dynamic `assortment_city_products` and `bakeable_products`
  from the new active run.
- Enabled `FORECAST_REFRESH_DATASETS=1` and `FORECAST_REFRESH_WEATHER=1` on the
  VM so the timer refreshes data/weather automatically.
- Patched the production refresh default history start to `2025-12-01` and made
  the bakery-day exporter tolerate empty ClickHouse windows.

## Lead-1 Backfill On 2026-06-29

The active forecast run starts on `2026-06-29`, but facts exist in ClickHouse
through `2026-06-29`. The gap for historical fact-vs-forecast comparison was
missing lead-1 snapshots for `2026-06-24` through `2026-06-28`.

Action taken:

- Added `scripts/build_prod_lead1_model_backfill.py` for gaps where no
  bakery-level lead-1 snapshot exists yet.
- The script builds each date independently using only history before that
  date, the uplifted bakery model, real weather features, ClickHouse SKU
  profiles, current assortment filter, and
  `runner_city_prior_soft_weekpart` recent correction.
- Backfill runs are loaded as draft runs named
  `backfill_uplifted_bakery_norm_uplift_sku_YYYYMMDD_h1`.
- These runs must not be activated as the main production run.

Observed ClickHouse lead-1 snapshot status at 2026-06-29 after completion:

- `2026-06-24`: loaded in bakery/SKU-day/SKU-hour snapshots
- `2026-06-25`: loaded in bakery/SKU-day/SKU-hour snapshots
- `2026-06-26`: loaded in bakery/SKU-day/SKU-hour snapshots
- `2026-06-27`: loaded in bakery/SKU-day/SKU-hour snapshots
- `2026-06-28`: loaded in bakery/SKU-day/SKU-hour snapshots

Observed loaded rows:

| date | bakery snapshots | SKU-day snapshots | SKU-hour snapshots |
| --- | ---: | ---: | ---: |
| `2026-06-24` | `202` | `32509` | `353367` |
| `2026-06-25` | `203` | `32557` | `354420` |
| `2026-06-26` | `203` | `32695` | `358125` |
| `2026-06-27` | `203` | `32750` | `355058` |
| `2026-06-28` | `203` | `33324` | `355353` |

## Verification Commands

On the VM:

```bash
cd /opt/demand-forecasting-model
systemctl is-enabled forecast-production.timer
systemctl is-active forecast-production.timer
systemctl list-timers --all --no-pager | grep forecast-production
.venv/bin/python -m scripts.verify_prod_deploy --env-file .env
```

Expected final line:

```text
VERIFY OK: env, summary, and active run are consistent
```

## Base-Raw Variant Evaluation (2026-06-30) — Resolved 2026-07-01

A lead-1 dev backfill (`dev_base_raw_YYYYMMDD_h1`) was run for pilot bakeries
`[20, 21, 22, 28, 80, 89, 107, 221, 222, 257]` using scenario `base_raw_uplift`
(base bakery model + raw uplift multiplier).

Initial 7-day results (2026-06-22..2026-06-28, 10 pilot bakeries):

| metric | prod (uplifted_norm) | base_raw_uplift |
| --- | ---: | ---: |
| bias% | +11.9% | +6.6% |
| wMAPE% | 72.2% | 35.2% |

The extended 21-day backfill (2026-06-01..2026-06-21) completed successfully
(21/21 days). The follow-up 28-day comparison
(`analyze_variants_comparison.py --start 2026-06-01 --end 2026-06-28`) produced
numbers that look broken (bias% swings to +216.7% for prod / -73.3% for
base_raw, far outside the 7-day pilot range, with base_raw row counts ~4x
lower than expected) — **do not trust that specific run's output**; the
discrepancy was not root-caused before the decision below was made.

**Decision (2026-07-01):** based on the 7-day pilot signal and separate
manual review, switched prod to base bakery-day model. However, the
SKU-hour uplift multiplier itself was independently rejected the same day
(see "SKU-Hour Share Profile Floor Removed" below) as unjustified, so
`base_raw_uplift` (which bundles base model + raw uplift multiplier) was not
deployed as-is. Instead, added a new scenario `base_no_sku_uplift` (base
bakery model, raw SKU-hour profile, no SKU-hour uplift multiplier) and
deployed that. See `DECISIONS.md` for the full rationale.

This replaces the active run for ALL bakeries. There is currently no
per-bakery override mechanism in the embedded app.

## SKU-Hour Share Profile Floor Removed (2026-07-01)

`smooth_sku_hour_share_profile.py` previously applied
`adjusted_share = max(raw_share, mean_share)` — a floor that lifted any
hourly share below the historical mean up to the mean. Investigation this
session (censoring/dip-depth/intraday signal analysis, category-floor
formula attempts) could not establish that low hourly shares reflect
shelf-absence (stockout) rather than genuine low demand — the floor was
therefore an unjustified upward distortion.

Action taken:

- Removed the floor; `smooth_sku_hour_share_profile.py` now passes raw
  shares through unchanged (still does the chunked renormalize/rebuild).
- The per-group Python `for` loop in `build_sku_hour_share_profile()` was
  vectorized into `groupby().agg()` — the old loop was OOM-killing the VM
  (16GB RAM) when building the profile over the full ~10-month/61M-row
  history; the vectorized version completes the same step in ~10 minutes
  instead of hanging for hours.
- Fixed two `weekly_profile_refresh.py` bugs found during the first
  successful end-to-end run: wrong `--mode` value for the uplift-multiplier
  load step, and wrong `--applied-path` (was pointing at the raw daily file
  instead of the smoothed daily file that has the `sku_share_in_hour_adj*`
  columns).
- Reloaded ClickHouse tables `sku_hour_share_profile_smoothed_embedded`
  (3,291,510 rows) and `sku_hour_uplift_multiplier_embedded`
  (`profile_version=weekly_20260701`, 26,937 rows) with `--truncate`.
- `median_sku_share_in_hour` in the profile table is still overwritten with
  `mean_sku_share_in_hour` during the smoothing rebuild (pre-existing,
  unrelated bug) — this column is dead weight; only
  `mean_sku_share_in_hour_norm` is actually consumed downstream
  (`apply_bakery_profiles.py`), so it was left as-is.

## Baking Plan + Assortment Deploy (2026-07-06)

Задеплоено на Blackhole (`82bb03a8`, `/opt/app`):

- `baking_plan.py` — data-driven алгоритм окон по профилю пекарни (parse_comments_sheet, peak detection, cluster→window)
- `bakery.py` — `get_bakeable_products()` принимает `bakery_id`, возвращает city + bakery слои
- `ui.py` — передаёт `bakery_id` в `get_bakeable_products`
- `baking_plan_template.xlsx` + индивидуальные шаблоны 20, 21, 22 — добавлен лист "комментарии"

ClickHouse:
- `bakeable_products` — мигрирована: добавлены колонки `scope`, `bakery_id`, ORDER BY обновлён
- Бэкап старой таблицы: `bakeable_products_backup_20260706_165145`

Новый скрипт: `scripts/build_city_assortment_from_sales.py` (city + bakery слои из `mart_sales_60d`)
Миграция: `scripts/migrate_bakeable_products_add_scope.py`
Пересчёт ассортимента встроен в `production_dataset_refresh.refresh_production_datasets()`

Документация: `docs/baking_plan_implementation.md`
Коммиты: `c087857` (план выпекания), `71465a1` (ассортимент)

## Bakery-Day Model Retrain (2026-07-06)

New model trained on `data/processed/stg_daily_v1/bakery_daily_sales.csv`
(stg_check_lines, Jan 2025 – Jul 2026, 94 456 rows, 219 bakeries).

Key change: added `bakery_sales_lag365` as a feature — YoY signal that
captures same-bakery sales ~1 year ago. CV showed consistent MAE improvement
(delta ≈ −0.003, importance 2–3% gain). Three files modified:
- `src/experiments_v2/build_bakery_daily_dataset.py` — lag list `[1,2,3,7,14,30,365]`
- `src/experiments_v2/bakery_day_forecast.py` — BASE_FEATURES, numeric_fill_cols, recursive_forecast
- `pipelines/forecast_publish/production_dataset_refresh.py` — DEFAULT_HISTORY_START_DATE `2025-12-01` → `2025-06-01`

History start extended to 2025-06-01 so VM dataset covers ≥13 months;
lag365 coverage will be ~50–60% for July 2026 rows, growing over time.

Model metrics on holdout (Jun 2026):
- MAE: 67.2, WMAPE: 7.4%, Bias: −22.2 (overforecast, −2.7%)
- 160/188 bakeries overforecast (desired), 28 underforecast

Deployed artifacts:
- `models/bakery_day_model.joblib` — SCP'd to VM 2026-07-06
- `models/bakery_day_meta.joblib` — SCP'd to VM 2026-07-06
- `models/bakery_day_bias.json` — updated from new holdout, SCP'd to VM 2026-07-06
- Code: git `2c38e80` pulled to VM via `deploy.sh --no-run`

Status: code and model files on VM; service will run tomorrow (2026-07-07)
when nightly timer fires with a fresh run_id. Today's run_id
`prod_base_bakery_no_sku_uplift_20260706_h14` was already consumed by the
morning timer (03:30 UTC), causing a ClickHouse delete-timeout on the
afternoon redeploy. The morning run (old model) remains active today.

## Embedded Hour Discrepancy UI Deploy (2026-07-07)

Deployed to Blackhole (`82bb03a8`, `/opt/app`) as a read-only embedded app
change:

- Bakery hourly profile now marks high fact-vs-forecast discrepancy hours.
- All hour cards are clickable.
- `/api/v1/bakeries/{bakery_id}/hour-discrepancy` returns top SKU contributors
  for a selected bakery/date/hour.

Deploy details:

- Backed up `/opt/app/app` to `app_backup_ui_discrepancy_20260707_071254`.
- Uploaded only embedded app files under `apps/forecast_embedded/app`.
- Ran `python3 -m py_compile` for changed Python modules.
- Restarted `app.service`.

Post-deploy verification on Blackhole:

- `app.service`: `active`
- `http://localhost:3000/health`: OK
- Active run: `prod_base_bakery_no_sku_uplift_20260707_h14`
- Dates endpoint: `14` dates
- Smoke with admin headers:
  `/api/v1/bakeries/{bakery_id}/hour-discrepancy?date=2026-07-07&hour=14`
  returned OK with `items=3`.
- Blackhole forecast timers remained disabled/inactive.

## Baking Plan Torn Down And Restructured (2026-07-09)

The previous baking-plan implementation (deployed 2026-07-06, see
"Baking Plan + Assortment Deploy" above) was torn down and is being rebuilt
from scratch as its own package.

Removed:

- `apps/forecast_embedded/app/services/baking_plan.py` (996-line algorithm:
  peak detection, window clustering, template allocation)
- `apps/forecast_embedded/app/assets/baking_plan_template.xlsx` and
  `baking_plan_individual/{20,21,22}_*.xlsx`
- The `/bakery/{id}/baking-plan.xlsx` route, its "Выгрузить план выпекания"
  button in `bakery.html`, and its JS special-case in `app.js`
- Dead code left orphaned in `app/services/bakery.py`:
  `get_bakeable_products`, `get_city_assortment`, `get_month_revenue_bucket`,
  `get_historical_hourly_profile`, and the ClickHouse table constants only
  those used
- `docs/baking_plan_implementation.md`,
  `scripts/audit_baking_plan_templates_assortment.py`,
  `config/baking_plan_template_overrides.csv`, and their tests

Added: `apps/baking_plan/` — a new standalone package (not a subpackage of
`apps/forecast_embedded/app`) that owns the baking-plan feature end to end.
See `apps/baking_plan/README.md` for the package boundary contract. Layout:

```
apps/baking_plan/
  service.py    -- public entrypoint: build_baking_plan_workbook(...)
  router.py      -- GET /bakery/{bakery_id}/baking-plan.xlsx
  windows.py       -- peak detection / window-selection algorithm
  assortment.py       -- bakeable-products allowlist (city + bakery scope)
  templates.py            -- xlsx template selection + "комментарии" parsing
  data.py                    -- ClickHouse reads specific to this feature
  assets/, assets/individual/  -- xlsx templates (currently empty)
```

Wiring: `apps/forecast_embedded/app/main.py` inserts `apps/` onto `sys.path`
and mounts `baking_plan.router.router`. This is a code-organization change
only — still one process, one deploy target (Blackhole `app.service`), no new
port or systemd unit. See `DECISIONS.md` (2026-07-09 entry) for the
service/package boundary rationale.

Status: scaffolding only. Every function in `apps/baking_plan/` raises
`NotImplementedError`. The route is mounted and importable but not
functional — the export button was removed from the UI until it works.
Assortment and window-selection logic need a fresh design, not a port of the
removed code (the old peak-detection/clustering approach and the SKU-hour
floor-uplift it depended on were already flagged as unreliable — see the
2026-07-01 decisions below).

Deploy note: Blackhole deploys have historically uploaded only
`apps/forecast_embedded/app/*` manually (see the 2026-07-07 entry below).
Any future Blackhole deploy touching baking-plan must also upload
`apps/baking_plan/*`.

## Baking Plan MILP Rebuild Deployed (2026-07-11)

The `apps/baking_plan/` rebuild (torn down and restructured 2026-07-09, see
below) was committed (`8e3e79f`, `c8eedac`) and deployed to Blackhole
(`82bb03a8`, host `fhmab3h2o3lo0jqd552k`).

Pre-deploy fix: `algorithms/milp.py` imports `scipy.optimize.milp` at module
load time, and that import is unconditional from `app.main` (mounts
`baking_plan.router` on startup). `scipy` was missing from
`apps/forecast_embedded/requirements.txt` — deploying without it would have
crashed the whole embedded app on boot, not just the baking-plan route.
Added `scipy==1.17.1` to requirements before deploying.

Deploy method (no dedicated script exists yet):

- Fetched a tarball of `origin/master` on the server via
  `curl .../archive/refs/heads/master.tar.gz` (VibeCode exec API).
- Backed up `/opt/app/app` → `/opt/app/app_backup_20260710_230211`.
- Replaced `/opt/app/app` wholesale from the tarball's
  `apps/forecast_embedded/app` (previous surgical file-by-file deploys had
  already drifted — `templates/bakery.html` had uncommitted-looking changes
  that a partial file list would have missed; full-directory replace avoids
  that class of bug).
- Created `/opt/baking_plan` (new, sibling to `/opt/app`, both directly
  under `/opt`) from the tarball's `apps/baking_plan`.
- Copied `apps/forecast_embedded/requirements.txt` → `/opt/app/requirements.txt`
  and ran `/opt/app/.venv/bin/pip install -r requirements.txt` (installs
  `scipy`; other pins were already satisfied, no-op).
- Ran a preflight `cd /opt/app && python -c "import app.main"` *before*
  restarting the service — only restarted `app.service` after that import
  succeeded, so a bad deploy would have left the old process running
  instead of taking the app down.
- `systemctl restart app.service`; verified `http://localhost:3000/health`
  → `{"ok":true,...}` and `systemctl is-active app.service` → `active`.

Post-deploy smoke test: `GET /bakery/21/baking-plan.xlsx?date=2026-07-10`
(bakery 21 = Парковая 7, Казань) with admin auth headers and an explicit
`run_id` returned `HTTP 200` with a well-formed `.xlsx` (valid `PK` zip
signature, `xl/worksheets/sheet1.xml` / `styles.xml` / `workbook.xml`
present) generated from the live active run
`prod_base_bakery_no_sku_uplift_20260710_h14`. A request without an
explicit `run_id`/admin role returned `404 Bakery forecast not found` —
expected existing access-control behavior for a synthetic non-portal test
user, not a regression.

Rollback: `/opt/app/app_backup_20260710_230211` and (if needed)
`/opt/baking_plan_backup_20260710_230211` on the server.

## Baking Plan MILP Redesign Deployed (2026-07-13)

Merged дефрост/двухдневка into the same MILP as regular production
(previously three separate tray-variable families) — see
`docs/baking_plan_implementation.md` and `apps/baking_plan/algorithms/milp.py`
module docstring for the model. Also added: molding-pace floor (54s/3:30)
with automatic retry, per-window capacity-shortage recommendation text on
the rendered plan, red/orange Итого highlighting for unfulfilled SKUs, and
crediting yesterday's overnight defrost batch back out of today's demand
via `sku_forecast_hour_snapshots` (`lead_days = 1`).

Deploy method: same manual tarball-replace pattern as 2026-07-11 (no
dedicated deploy script yet), this time via the VibeCode `/v1/infra/servers/:id/exec`
API directly (server id `82bb03a8-c356-4225-97a4-a1540cdc29e6`) rather than
a prior session's access path:

- Committed and pushed only the 8 baking_plan-related files (repo working
  tree had unrelated uncommitted changes from other sessions — left
  untouched) — commit `3b18eac`.
- Staged verification *before* touching `/opt/app` or `/opt/baking_plan`:
  fetched the `origin/master` tarball into `/tmp/deploy_src`, mirrored the
  `/opt/app/app` + `/opt/baking_plan` sibling-package layout under
  `/tmp/deploy_stage`, ran a dependency-free Python script (no `pip
  install`, reused the existing `/opt/app/.venv` read-only) exercising the
  same invariants as the local test suite — mandatory-always-wins-over-
  higher-priority-regular, no gratuitous overproduction, defrost window
  consolidation, clean-integer tail splitting, floor-pace constants — all
  7 checks passed.
- Only after that: backed up `/opt/app/app` → `/opt/app/app_backup_20260713_072134`
  and `/opt/baking_plan` → `/opt/baking_plan_backup_20260713_072134`,
  replaced both from the tarball, re-ran the plain `import app.main`
  preflight in the live location, then `systemctl restart app.service`.
- Post-deploy: `systemctl is-active app.service` → `active`,
  `http://localhost:3000/health` → `{"ok":true,...}`,
  `GET /bakery/21/baking-plan.xlsx?date=2026-07-10&run_id=prod_base_bakery_no_sku_uplift_20260710_h14`
  with admin headers → `HTTP 200`, valid `.xlsx` (8338 bytes, correct zip
  structure), clean service logs.

No `requirements.txt` changes this deploy (no new dependencies).

Rollback: `/opt/app/app_backup_20260713_072134` and
`/opt/baking_plan_backup_20260713_072134` on the server.

## Baking Plan Night Storage Rules Deployed (2026-07-13)

Deployed commit `6e27bd9` (`fix: account for night storage in baking plan`)
to Blackhole (`82bb03a8`, host `fhmab3h2o3lo0jqd552k`).

Code changes:

- Added direct overnight-stock limits from the freezer/refrigerator
  night-storage PDFs dated 15.05.2026
  (`NIGHT_STORAGE_DIRECT_UNITS_BY_SKU`).
- Capped both tomorrow's extra overnight batch and today's lead-1 defrost
  credit by those PDF quantities.
- Added prep-only night-storage labor reductions for `Жар Киш ...` and
  smetannik SKUs (`NIGHT_PREP_LABOR_MINUTES_BY_SKU`).
- Added same-SKU label swapping so a physically identical regular batch and
  defrost batch can exchange labels, placing `"ночная дефр"` later without
  changing capacity usage.
- Added `scripts/analyze_baking_plan_fact_night_storage.py` for fact-based
  diagnostics against the night-storage scenarios.

Deploy method:

- Local tests before commit: focused baking-plan pytest suite
  (`44 passed`) and ruff over `apps/baking_plan`, the diagnostics/compare
  scripts, and focused baking-plan tests (`All checks passed`).
- Pushed `6e27bd9` to `origin/master`.
- Via the VibeCode exec API, fetched the GitHub tarball for exact commit
  `6e27bd90c8312bd384f521de2ccb6abfcb9463b9` into `/tmp/deploy_src`,
  staged `/tmp/deploy_stage/opt/app/app` and
  `/tmp/deploy_stage/opt/baking_plan`, and ran import/compile preflight using
  the existing `/opt/app/.venv`.
- Backed up `/opt/app/app` to `/opt/app/app_backup_20260713_144022` and
  `/opt/baking_plan` to `/opt/baking_plan_backup_20260713_144022`, replaced
  both live directories from the staged tarball, ran live import, then
  restarted `app.service`.

Post-deploy verification:

- `app.service`: `active`.
- `http://localhost:3000/health`: `{"ok":true,"app_env":"prod","table_suffix":""}`.
- Blackhole forecast timers remained disabled/inactive:
  `forecast-production.timer` disabled/inactive and
  `bakery-forecast-nightly.timer` disabled/inactive.
- Smoke export:
  `GET /bakery/16/baking-plan.xlsx?date=2026-07-13&run_id=prod_base_bakery_no_sku_uplift_20260713_h14`
  with admin headers returned `HTTP 200`, valid `.xlsx` (8340 bytes).
  In the exported workbook, `Киш грибы курица` has regular `10` in
  `10:00-11:00` and `10 (ночная дефр)` in `11:00-12:00`, confirming the
  same-SKU defrost-label swap is active in production.

Rollback: `/opt/app/app_backup_20260713_144022` and
`/opt/baking_plan_backup_20260713_144022` on the server.

## Rolling Bakery-Day Bias Correction Deployed (2026-07-13)

`models/bakery_day_bias.json` was a one-time snapshot of mean(actual -
forecast) per bakery from the June holdout, applied unconditionally to
every forecast forever. It never refreshed, so after the 2026-07-06
bakery-day model retrain (`bakery_sales_lag365` added) it went stale and
was actively pulling several pilot bakeries' forecasts in the wrong
direction — e.g. Парина 6 (bakery 89) got a constant `-125.6`/day
correction computed in June that no longer matched the retrained model's
July behaviour, deepening a live underforecast users were seeing in the
embedded app (reported by the user against Парковая 7 / Парина 6,
2026-07-06..11).

Root-caused via live ClickHouse `forecast_base` vs `forecast_final` on the
already-active prod run (not a backtest reconstruction) — confirmed
`forecast_final = forecast_base + bias.json[bakery_id]`, i.e. the static
file, not the retrained model itself, was the dominant driver of the
Парина 6 error.

Fix: `pipelines/forecast_publish/rolling_bakery_bias.py` — recomputes the
same style of per-bakery correction from a trailing 7-day window of live
lead-1 `forecast_base` vs `mart_sales_60d` on every run (falls back to the
static snapshot for bakeries with `< 3` days of recent history). Wired
into `run_production_inference.py` as the default (opt out with
`--no-rolling-bias-correction`); same `bias_clip_pct=0.15` safety cap as
before.

Validated on dev (`.env.dev`, `_dev`-suffixed tables) via an 11-day
walk-forward lead-1 backfill (2026-07-01..11, all 10 pilot bakeries,
`scripts/build_prod_lead1_model_backfill.py --use-rolling-bias`), rebuilt
with real Open-Meteo weather (the first pass used stale/default weather
and overstated the win — flagged and rerun before trusting the result):

| variant | wMAPE | bias% |
| --- | ---: | ---: |
| static (prod as of 2026-07-13 morning) | 8.1% | -1.2% |
| no correction (raw `forecast_base`) | 5.7% | -1.6% |
| rolling (this fix) | 5.6-5.8% | -0.2% to -1.8% |

Static is worse than every alternative for 8/10 pilot bakeries. Rolling
vs no-correction is close in aggregate but rolling clearly wins for
bakeries with a persistent (non-weather, non-noise) bias, e.g. Парковая 7
(bakery 21): wMAPE 6.7% (no correction) vs 4.6% (rolling).

Deploy method: pushed commit `0dcb638` to `origin/master`. A concurrent
session was mid-deploy of unrelated baking-plan changes on this same VM
(`/opt/demand-forecasting-model` working tree had uncommitted baking-plan
drift, plus `docs/ops/*.md` are root-owned and block `forecast`-user
`git pull`) — rather than force a full `git pull` through that, SFTP'd
only the 3 changed files (`rolling_bakery_bias.py`,
`run_production_inference.py`, `build_prod_lead1_model_backfill.py`)
directly to their paths, `chown forecast:forecast`, verified they import
cleanly under the VM's venv, then `systemctl start
forecast-production.service` to regenerate and activate a fresh run
immediately rather than waiting for tomorrow's 03:30 UTC timer. VM git
history is therefore not fast-forwarded to `0dcb638` yet — file contents
are correct and live, but `git log` on the VM will look stale until
someone resolves the docs/ops ownership + baking-plan working-tree drift
and pulls cleanly.

Post-deploy verification: `scripts.verify_prod_deploy` → `VERIFY OK`.
New active run `prod_base_bakery_no_sku_uplift_20260713_h14`
(generated `2026-07-13 18:33:59+03:00`). Confirmed the new correction is
live by reading `forecast_final - forecast_base` directly from
`bakery_forecast_day_snapshots` for this run: bakery 21 now gets a
constant `+114.3`/day adjustment (vs the old near-zero static value, which
was insufficient), bakery 89 gets `-5.2`/day (vs the old `-125.6`).

## Do Not Do

- Do not run production forecast generation from VibeCode/Blackhole.
- Do not enable Blackhole forecast timers.
- Do not treat `handoffs/` as current truth without checking this file first.
- Do not manually change active ClickHouse runs except through the documented
  activation script and only after verifying the intended run id.
- Do not print secrets from `.env`, ClickHouse config, VibeCode API keys, or
  VM SSH keys.

## When This File Must Be Updated

Update this file after any change to:

- production writer ownership;
- VM host, path, timer, or schedule;
- VibeCode/Blackhole role;
- ClickHouse active run contract;
- forecast scenario, horizon, correction mode, or source tables;
- emergency production state changes.

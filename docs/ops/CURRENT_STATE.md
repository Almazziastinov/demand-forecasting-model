# Current Project State

Last updated: 2026-07-06

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

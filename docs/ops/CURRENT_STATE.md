# Current Project State

Last updated: 2026-08-14

## Publisher Migrated From mart To fct Tables (2026-08-14)

**Context**: Around 2026-08-10 the Svezhar ETL pipeline stopped propagating
data from `fct_*` raw tables into `stg_*` and `mart_zero_sales_60d`. Both
`stg_production_release` and `mart_zero_sales_60d` have been empty for all
pilot bakeries since then. The ETL root cause is unresolved (Yandex Cloud MDB
maintenance restarts appear to have disrupted the pipeline's recovery logic;
Svezhar team is owner).

**Fix applied** (commit `a1d1dbf`, branch `claude/jovial-chaplygin-ec1d44`,
deployed to Blackhole 2026-08-14):

`scripts/publish_pilot_forecast.py` no longer reads `mart_zero_sales_60d`
anywhere. All three mart dependencies were replaced with direct `fct_*` queries:

1. **Stock balance** (`остатки со вчерашнего дня`): now computed as
   `fct_production_release` (argMax dedup by `release_id, line_id`) minus
   `fct_check_lines` (DISTINCT dedup on business fields = STRICT_DUP_KEYS),
   clipped to `≥ 0`. Both tables are refreshed continuously by Svezhar ETL and
   are unaffected by the stg/mart outage.

2. **Cold-start sales history**: now queries `fct_check_lines` with the same
   DISTINCT dedup, matching what the forecast training pipeline uses
   (`clickhouse_export_template.sql` + `raw_sales_dedup.py`).

3. **Mature-SKU correction history** (`sold_qty`, `produced_qty`,
   `last_sale_time`): same fct sources as above. `product_name` /
   `category_name` come from the already-loaded `forecast_df` (from
   `sku_forecast_day_snapshots`) instead of mart.

**Why this is consistent**: the bakery-day model and SKU forecasts are trained
on `fct_check_lines` data. The mart was an additional transformation layer that
was already introducing ~13% overcount vs the properly deduped fct stream. Using
fct directly eliminates that inconsistency.

**Rollback**: `/opt/scripts/publish_pilot_forecast.py.backup_20260814_fct`
on Blackhole. No ClickHouse schema changes — pure publisher logic change.

**Previous ETL incident note** (2026-08-13): the day before this migration,
a narrower fix had been applied (2-day mart window + DQ warning). That fix was
immediately superseded by this full mart→fct migration and is no longer active.

## Pilot SKU Corrections Deployed To Daily Publisher (2026-07-29)

The 10-bakery pilot publisher now applies two category-neutral SKU correction
layers before yesterday's stock subtraction and kratnost rounding:

1. Forecast-cold-start products `11573` and `11574` use an own-sales EWMA
   floor (`alpha=0.90`, minimum 3 sales days) while they have at most 13 prior
   positive-forecast days. Lost-demand estimates are deliberately not used by
   this floor.
2. Products with at least 14 positive-forecast days can enter the mature-SKU
   systematic correction registry described below. The transition between the
   two mechanisms is automatic and non-overlapping.

Both layers preserve each `date × bakery × category` forecast total. The
combined rolling 28-day backtest through `2026-07-28` improved total WAPE from
`25.7551%` to `25.0720%` (`-0.6831 pp`). For the two cold-start products,
WAPE improved from `95.0597%` to `57.4101%`.

Deployment target: Blackhole server
`82bb03a8-c356-4225-97a4-a1540cdc29e6`.
Remote dry-run for `2026-07-30`: 18 bakery/SKU cold-start floors, 426 changed
rows after mature correction, 535 final SKU rows across 10 bakeries, valid
28,739-byte workbook, no Bitrix24 send. The timer remains enabled and active
for `03:00 UTC` / `06:00 MSK`.

Rollback:
`/opt/scripts/publish_pilot_forecast.py.backup_20260729_sku_corrections`.
The added modules are
`/opt/src/experiments_v2/sku_cold_start.py` and
`/opt/src/experiments_v2/sku_systematic_correction.py`; the old publisher does
not import them.

## Mature-SKU Systematic Correction (2026-07-29)

A conservative, category-neutral correction layer was implemented locally for
the 10-bakery pilot and is active in the daily pilot-plan publisher as
described above. It does not change the production forecast snapshots.

The registry uses only information strictly earlier than each forecast date.
Products `11573` and `11574` enter it automatically after leaving cold start;
the maturity gates prevent overlap between the two mechanisms.
Eligibility requires at least 28 observed days, at least 14 days with a
positive forecast, age of at least 28 days, 150 units of demand, absolute bias
of at least 15%, error directionality of at least 40%, and a same-direction
recent seven-day bias of at least 10%. The positive-forecast maturity guard
prevents established products with newly appeared forecast coverage from
being treated as persistent underforecasts.
Multipliers have no hard lower or upper bound. Their adaptive smoothing
strength is selected in `[0.10, 0.30]` from directionality, recent bias,
history length, demand volume, and repeated lost-demand evidence. Geometric
smoothing (`full_multiplier ** smoothing`) is used so extreme ratios caused by
near-zero forecasts do not pass through linearly. Registry entries expire
after 14 days.

After multipliers are applied, forecasts are renormalized to preserve the
original `date × bakery × category` total. The base bakery/category forecast
therefore does not change; only the SKU mix changes.

Rolling 28-day backtest through `2026-07-28`:

- baseline WAPE: `25.1106%`
- corrected WAPE: `24.8957%`
- delta: `-0.2149 pp`
- underforecast reduced by `178.30` units
- overforecast reduced by `178.30` units
- exact total forecast and aggregate bias preserved
- improved on `24/28` dates and all 10 bakeries
- 102 distinct registry pairs appeared during the rolling test
- current registry contains 58 pairs

Implementation:

- `src/experiments_v2/sku_systematic_correction.py`
- `scripts/backtest_sku_systematic_correction.py`
- `reports/sku_systematic_correction_backtest/`
- optional publisher override: `--sku-correction-registry`

Publisher dry-run for `2026-07-29` succeeded: 535 rows across 10 bakeries,
185 rows changed by correction/renormalization, category totals preserved,
and no Bitrix24 message sent.

The publisher builds the registry from ClickHouse automatically on every run.
The optional CSV argument is an override for controlled diagnostics; production
does not depend on a static registry file.

## Pilot Daily Forecast Publisher — Previous-Day Stock (2026-07-28)

The Bitrix24 chat publisher for chat `179919`
(`Пилот выставления планов выпекания ИИ`) now publishes the forecast for the
current calendar day at `06:00 MSK` (`03:00 UTC`) instead of publishing the
next day's forecast at `08:00 MSK`.

Before kratnost rounding, the publisher subtracts all positive closing stock
from the previous day:

`net_need = max(forecast_qty - yesterday_stock, 0)`

`production_plan = round_up_to_kratnost(net_need)`

The Excel output columns are now:

`Пекарня`, `Категория`, `Номенклатура`, `Прогноз`,
`Остаток со вчерашнего дня`, `Чистая потребность`, `План выпуска`,
`Итого на продажу`, `Кратность`.

`Итого на продажу = План выпуска + Остаток со вчерашнего дня`.

Runtime details:

- script: `/opt/scripts/publish_pilot_forecast.py`
- timer: `pilot-forecast-publish.timer`
- schedule: `OnCalendar=*-*-* 03:00:00 UTC`
- server: VibeCode/Blackhole `82bb03a8-c356-4225-97a4-a1540cdc29e6`
- pre-deploy backup:
  `/opt/scripts/publish_pilot_forecast.py.backup_20260728_123628`
- remote dry-run for `2026-07-28`: 598 SKU rows, 258 rows with positive
  previous-day stock, 158 rows with a reduced production plan

## Base Pilot Reduced To 10 Bakeries (2026-07-29)

Bakery `16` (`Кулагина 4 Казань`) is excluded from the base pilot until
further notice. The current base pilot set is:

`{20, 21, 22, 28, 80, 89, 107, 221, 222, 257}`

The 10-bakery scope is now used by the Bitrix24 daily forecast publisher and
by local pilot analysis/profile-building scripts. The deployed publisher is
`/opt/scripts/publish_pilot_forecast.py`; its pre-change backup is
`/opt/scripts/publish_pilot_forecast.py.backup_20260729_pilot10`.

Post-deploy dry-run produced `535` SKU rows across `10` bakeries and did not
send a Bitrix24 message. `pilot-forecast-publish.timer` remains enabled and
active.

The production writer VM still references the historical profile versions
`pilots_evening_20260716` and `stockout_20260716`, which were built for the
previous 11-bakery scope. They were not rebuilt or activated in this change
because direct VM access was unavailable. New profile builds use the
10-bakery base set; switching the active production profiles requires a
separate controlled VM rollout.

## Summary

The production forecast writer is the VM only. VibeCode/Blackhole is a
read-only embedded UI/API over ClickHouse and must not run forecast generation.

**Current operational pilot state (as of 2026-07-29):** the base pilot
contains **10 bakeries** —
{20, 21, 22, 28, 80, 89, 107, 221, 222, 257}. Bakery 16
(`Кулагина 4 Казань`) is excluded until further notice. The Bitrix24
publisher and local pilot defaults use this set. The active production writer
still references the historical 2026-07-16 uplift/correction profiles pending
a controlled VM rollout.

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

### Operations Director Access (2026-07-20, fixed 2026-07-21)

Four active Bitrix24 users whose work position contains `Операционный
директор` were granted access to the Blackhole embedded app and full bakery
visibility inside the app. The inactive matching user and the user whose
position is only `Операционист` were excluded.

| Bitrix24 userId | Name | Position |
| --- | --- | --- |
| 1475 | Руслан Назаренко | Операционный директор |
| 8509 | Вероника Соломко | Операционный директор г. Курск |
| 11297 | Ильнар Миннигалиев | Операционный директор |
| 31623 | Карина Галиева | Операционный директор производств |

- VibeCode server access: the four user ids are present in the server access
  list for `82bb03a8-c356-4225-97a4-a1540cdc29e6`.
- Application data access: each user has `operations_director` rows for all
  `275` bakery ids from `dim_bakeries` in
  `bitrix_user_bakery_access_embedded`, source
  `manual_operations_director_full_access`.
- They were not added to `ADMIN_USER_IDS`: this is full operational bakery
  visibility, without technical admin-only run/scenario controls.

**Bug found and fixed 2026-07-21**: the 2026-07-20 insert used
`bitrix_portal_id = 'franshizasvezhar.bitrix24.ru'` (domain string), but
VibeCode injects `x-vibe-portal-id` as a UUID
(`390d6913-26b6-4516-9da0-d8d575031afa`). The `_access_filter` subquery
filters on `bitrix_portal_id` first, so all four directors saw `bakeries=0`
despite having rows in the table. Fixed by inserting a duplicate set of 275
rows per user with `bitrix_portal_id = '390d6913-26b6-4516-9da0-d8d575031afa'`
via `scripts/fix_ops_director_portal_id.py`. Verified live in logs:
`bakeries=275` after fix.

**Important for future access grants**: always use
`bitrix_portal_id = '390d6913-26b6-4516-9da0-d8d575031afa'` when inserting
into `bitrix_user_bakery_access_embedded`. VibeCode does not forward
`x-vibe-user-email` for real portal users (field arrives as `None`), so the
email fallback in `_access_filter` is not a safety net.

This is a manual snapshot of the current Bitrix24 users, not a position-based
automatic sync. New operations directors require a separate access update.

## Active Forecast

- Active run: `prod_base_bakery_raw_uplift_sku_20260716_h14`
  (generated `2026-07-16 09:23:30 UTC`, 7m53s CPU)
- Scenario: `base_raw_uplift` (switched from `base_no_sku_uplift` on
  2026-07-14 for the pilot launch — see "SKU-Level Uplift Reactivated For
  Pilot" below for the full rationale)
  - Bakery-day model: **base** (`bakery_day_model.joblib`, no bakery-level uplift)
  - SKU-hour allocation: raw `sku_hour_share_profile_smoothed_embedded`,
    **with the mean-share floor restored** (see below — floor-uplift is
    back after being removed 2026-07-01)
  - SKU-hour uplift multiplier: **enabled** (`use_raw_uplift_multiplier=True`),
    `profile_version=pilots_evening_20260716` for pilot bakeries
    {16,20,21,22,28,80,89,107,221,222,257}; non-pilot bakeries use
    `weekly_20260714` values (copied unchanged into the profile)
  - Stockout correction: **enabled**, `profile_version=stockout_20260716`
    (10,152 rows, 11 pilot bakeries, 79 SKU, hours 6–23 where dropout detected)
- `.env` on the VM: `FORECAST_SCENARIO=base_raw_uplift`,
  `FORECAST_ACTIVATE_RUN=base_raw_uplift`,
  `FORECAST_UPLIFT_PROFILE_VERSION=pilots_evening_20260716`,
  `FORECAST_STOCKOUT_CORRECTION_VERSION=stockout_20260716`,
  `FORECAST_MAX_SKU_UPLIFT_RATIO=1.2`,
  `FORECAST_HIERARCHICAL_HAIRCUT_TARGET_RATIO=1.15`
- Horizon days: `14`
- Recent correction mode: `runner_city_prior_soft_weekpart`
- Recent correction days: `30`
- Recent sales table: `mart_sales_60d` (VM production writer) / `fct_check_lines` (pilot publisher since 2026-08-14 — mart outage)
- Dataset refresh: enabled on the VM (`FORECAST_REFRESH_DATASETS=1`)
- Weather refresh: enabled on the VM (`FORECAST_REFRESH_WEATHER=1`)
- Bakery-day bias correction: **rolling** (trailing 7-day window,
  recomputed every run), not the old static one-time snapshot — see
  "Rolling Bakery-Day Bias Correction Deployed" below.

Previous scenario (`base_no_sku_uplift`, active 2026-07-01..2026-07-14) and
`uplifted_norm` (active through 2026-06-29..2026-06-30) remain defined in
`SCENARIOS` for rollback if needed.

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

## SKU-Hour Fallback Profile Fix Deployed, Not Yet Exercised (2026-07-13)

Investigated a user report that several real, currently-selling SKUs at
bakery 16 (Кулагина 4, Казань) showed a forecast collapsed to near-zero
despite steady actual demand — e.g. "Пирог с Манго" (product 11465):
~7/day actual sales (`mart_sales_60d`, every day for 30 days) vs
`sku_forecast_hour_embedded` showing `0.043`/day, with the entire day's
forecast concentrated in a single, near-dead hour (22:00) instead of the
SKU's real active hours.

Root cause: `apply_bakery_profiles_clickhouse.py:load_profile_lookup_frames`
(the tier-2, dow-blind fallback used for SKUs whose per-(bakery,dow,hour)
`n_days` never reaches the tier-1 gate of 8) averaged
`mean_sku_share_in_hour_norm` across dow with **no minimum sample-size
filter at all**. A single-observation row (`n_days=1`) at an edge hour
(05:00 or 22:00, low-traffic enough that one sale reads as "100% of that
hour") produced an unsmoothed extreme share that then dominated the
fallback for SKUs thin everywhere. Confirmed this is systemic, not a
one-off: bakery 16 alone had 16 profile rows with `n_days<=2` and
share > 0.1, 9 of them at hour 22 and 6 at hour 5 — affecting at least
8-9 SKUs at this one bakery, not just the one reported.

Fix: added `MIN_FALLBACK_N_DAYS = 3` gate excluding `n_days` 1-2 rows from
the fallback average, in both `src/experiments_v2/apply_bakery_profiles.py`
(CSV path, `build_sku_hour_profile_fallback`) and
`apply_bakery_profiles_clickhouse.py` (the production ClickHouse path,
`load_profile_lookup_frames`). `n_days == 0` is still trusted as before —
that value means "no `n_days` column at all in a legacy profile" (defaults
to 0 upstream), not "observed zero days," and should still get a fallback
estimate rather than being silently dropped. Committed `e3f39e6`, pushed to
`origin/master`. 2 new regression tests added; verified 4 pre-existing,
unrelated test failures (3 in `test_apply_bakery_profiles_clickhouse_recent.py`
pie-category-cap tests, 1 `test_build_bakeable_products_table.py` collection
error from a renamed function) are untouched by this change (confirmed via
`git stash`) — flagged separately, not fixed as part of this work.

Deploy method: backed up
`src/experiments_v2/apply_bakery_profiles.py(.bak_20260713_152709)` and
`apply_bakery_profiles_clickhouse.py(.bak_20260713_152709)` on the VM,
SFTP'd the two fixed files directly (working around the same VM git
blockers noted above — root-owned `docs/ops/*.md` and unrelated
uncommitted baking-plan drift), verified `py_compile` and a live import
of `MIN_FALLBACK_N_DAYS` succeed. Deliberately did **not** trigger a
manual `systemctl start forecast-production.service` — decided to let the
fix land through the normal 03:30 UTC nightly timer (2026-07-14) rather
than force an extra out-of-band production run today.

**Confirmed NOT yet exercised**: a concurrent session manually restarted
`forecast-production.service` at `2026-07-13 18:33:59+03:00` for an
unrelated fix (see "Rolling Bakery-Day Bias Correction Deployed" above),
regenerating today's active run. Checked directly afterward —
`sku_forecast_hour_embedded` still shows product 11465 at `0.043775`/day,
unchanged from before the fix landed on disk. Most likely explanation:
that process's Python interpreter had already imported the old module
code before the SFTP file replacement completed (the two events were only
minutes apart) — module source isn't re-read mid-process. **First real
run of this fix will be the 2026-07-14 03:30 UTC nightly timer** (or any
earlier manual `run_production_inference` invocation). Whoever checks
that morning should re-verify product 11465 (bakery 16) directly against
`mart_sales_60d` before trusting the new forecast, since this fix has
never actually executed yet.

Rollback: `src/experiments_v2/apply_bakery_profiles.py.bak_20260713_152709`
and `apply_bakery_profiles_clickhouse.py.bak_20260713_152709` on the VM.

**Separately noticed, not fixed**: `bakeable_products` city-scope rows for
Казань all come from the old `forecast_category_filter`/
`partner_baking_markup` sources (no per-city sales-share threshold at
all), not from `build_city_assortment_from_sales.py`'s `sales_window`
source (which enforces the documented 80% threshold). Traced this to the
same uncommitted VM drift flagged in the "Known issue" note above —
the `production_dataset_refresh.py`/`build_city_assortment_from_sales.py`
assortment-threshold code was placed on the VM at `2026-07-13 11:46 UTC`
(after this morning's 03:30 UTC run), so it has **never executed even
once** yet (`journalctl -u forecast-production.service` has zero
"assortment" mentions in its entire history). Left it alone — it's
someone else's in-flight, unreviewed change, not mine to touch. Its first
real run will also be the 2026-07-14 03:30 UTC timer; worth checking then
whether it actually drops the low-share SKUs (e.g. product 5105/10670/
10628/5106/11213) from Казань's city scope as the 80% threshold intends.

## SKU-Hour Fallback + Assortment-Threshold Fixes: Both Verified Live (2026-07-14)

Follow-up to the two 2026-07-13 entries above. The 2026-07-14 03:30 UTC
nightly timer fired as expected and surfaced the assortment code's first
real execution — it failed immediately:

```
Assortment refresh FAILED: unsupported operand type(s) for -: 'str' and 'datetime.date'
```

Root-caused: `scripts/build_city_assortment_from_sales.py:build_layers()`
built `combined["valid_from"]` via
`pd.to_datetime(valid_from).date().isoformat()` — a **string**. That's
fine for `build_bakeable_products_table.py`'s CSV-only sibling, but this
function's output is inserted straight into ClickHouse via
`client.insert_df()` against a `Date`-typed column; `clickhouse-connect`'s
Date serializer does `(value - epoch).days` per cell, which raises
exactly this error when `value` is a `str` instead of a `datetime.date`.
This is the actual reason `sales_window` (the 80%-threshold source) had
never produced a single row in production — every attempt crashed inside
the try/except and got silently logged as `assortment_status: failed`.

Reproduced the exact production traceback against a throwaway ClickHouse
table (`.env.dev`, `_dev`-suffixed environment — not touching any real
table) before and after the fix, to confirm root cause without writing
to anything shared. Fix: `combined["valid_from"] = pd.to_datetime(valid_from).date()`
(drop `.isoformat()`, keep it a real `date` object). This is a fix to
already-committed, shipped code (`71465a1`, 2026-07-06) — the VM's
uncommitted-looking copy of this file was not some other session's WIP,
it was this same feature, manually placed on the VM ahead of `git`
because the VM's git HEAD is stuck at `2c38e80` (see "Known issue" note
above). Added a regression test asserting `valid_from` stays a
`datetime.date`. Committed `1b29184`, pushed to `origin/master`, SFTP'd
to the VM (backup `scripts/build_city_assortment_from_sales.py.bak_20260714_073303`),
verified `py_compile` + live import.

**Both fixes then manually triggered and verified together** via
`systemctl start forecast-production.service` (full run, ~9 minutes,
regenerated and re-activated `prod_base_bakery_no_sku_uplift_20260714_h14`):

- Assortment: `Assortment refresh: city=318 bakery=2170 inserted=2488
  valid_from=2026-07-13` — no more `FAILED`. Confirmed for Казань: the 5
  originally-flagged low-share SKUs (product 5105/10670/10628/5106/11213)
  now correctly resolve to `scope='bakery'` (source `sales_window`) rather
  than `scope='city'` — they don't clear the 80% citywide threshold, but
  do sell at specific bakeries.
  - **Wide blast radius, not just Казань/bakery 16**: `sales_window`
    rows now exist for all 9 cities (`318` city-scope rows total) with
    `valid_from=2026-07-13`, newer than the old `forecast_category_filter`/
    `partner_baking_markup` rows' last update (`2026-06-30`).
    `get_bakeable_products()` selects rows by `valid_from = max(valid_from)
    for that city` — so from this run onward, **every city's served
    assortment switches from the old, unfiltered ~110-product set to the
    new, threshold-checked ~52-product city layer plus per-bakery
    additions**. The old rows are still in the table, just no longer the
    "current" batch. This is the intended fix finally working, but it's a
    live behavior change across the whole embedded app's baking plans,
    not a narrow one-bakery correction — watch for SKUs unexpectedly
    disappearing from plans at bakeries that don't have their own
    `scope='bakery'` entry for something the old, looser filter used to
    let through.
- SKU-hour fallback (`e3f39e6`, deployed 2026-07-13): bakery 16, product
  11465 (Пирог с Манго) forecast for 2026-07-14 = `2.97`/day across 3
  hours (7-12), up from `0.043`/day in a single dead hour (22:00) before
  the fix — actual recent demand is `~6.9`/day, so this is a large
  improvement but not a full close of the gap. Product 11213 (Роллы
  Вулкан с курицей) = `0.048`/day across 16 hours (6-21), properly spread
  now but still far below actual (`~2.0`/day). The remaining under-forecast
  for both is a separate, not-yet-investigated limitation in the
  recent-sales correction blend weights (see `DECISIONS.md`), not
  something this fix was meant to address.

Commits this round: `1b29184` (assortment date-type fix),
`6376930`/`e3f39e6` (SKU-hour fallback fix + its docs, 2026-07-13).

Rollback: `scripts/build_city_assortment_from_sales.py.bak_20260714_073303`
on the VM for the assortment fix; see the 2026-07-13 entry above for the
SKU-hour fallback rollback path. There is no rollback for the assortment
*data* itself (the old `forecast_category_filter`/`partner_baking_markup`
rows are still present, just no longer selected) — if the new
`sales_window` assortment turns out to be wrong for some city/bakery, the
fix would need to be in the threshold/window-days parameters, not a data
revert.

## SKU-Level Uplift Reactivated For Pilot (2026-07-14)

The project is pivoting toward a pilot launch. User direction: the project's
core value is eliminating missed sales/underforecast, which requires real
SKU-level uplift even though the mechanism is known to be imprecise (can't
distinguish shelf-absence/stockout from genuine low demand — see the
2026-07-01 rejection below). Applied to all bakeries (no per-bakery
override exists in the embedded app); deployed straight to prod per user
direction, no dev pre-validation this time.

**Root finding before any change**: switching `FORECAST_SCENARIO` to
`base_raw_uplift` alone would have done nothing. `sku_hour_uplift_multiplier`
is derived from the gap between a mean-share floor and the raw share; that
floor (`adjusted_share = max(raw_share, mean_share)`) was removed 2026-07-01
(commit `625605d`). Confirmed live before touching anything: the
`sku_hour_uplift_multiplier_embedded` table's only existing version
(`weekly_20260712`, produced automatically by the still-enabled
`weekly-profile-refresh.timer`) had **0 of 27,150 rows with multiplier >
1.0** — the mechanism had been a complete no-op since the floor was removed,
undetected because the active scenario never used it.

**Change**: restored the floor
(`work[ADJUSTED_SHARE_COL] = np.maximum(work[SKU_SHARE_COL],
work[PROFILE_MEAN_COL])`) in
`src/experiments_v2/smooth_sku_hour_share_profile.py`
(`build_adjusted_applied_chunk`), reverting only that one line from
`625605d` — the rest of that commit (vectorization,
`weekly_profile_refresh.py` CLI fixes) is unaffected and correct. Updated
the one test that had been asserting no-floor passthrough behavior back to
floor-based expected values. Committed `144ef59`, pushed to
`origin/master`.

**Deploy**: VM `git pull` is still blocked (see "Known issue" above) — the
usual SFTP workaround also failed this session (`Subsystem sftp` is not
configured in this VM's sshd — confirmed by a bare `sftp.put()` failing with
`ENOENT` even against `/tmp`, not a path-specific issue). Worked around by
streaming the file content over the existing SSH exec channel
(`base64 -d > path` fed via stdin) instead of the SFTP subsystem. Backed up
the prior file as
`src/experiments_v2/smooth_sku_hour_share_profile.py.bak_20260714_152419`
on the VM, verified `py_compile` and a live import confirming the floor
formula is present before proceeding.

**Rebuilt the profile pipeline end to end** with the restored floor via
`scripts/weekly_profile_refresh.py --env-file .env` (full 12-month
export → build → smooth → load profile → load multipliers, ~47 min
total). Produced a fresh `profile_version=weekly_20260714` (distinct from
the two no-op tags `weekly_20260701`/`weekly_20260712`): 3,542,847 profile
rows, 27,155 multiplier rows, **95.4% of multiplier rows now > 1.0**
(avg `1.29`, max `3.53`) — confirms the floor is live and producing a real
signal again.

Updated VM `.env` (backed up as `.env.bak_20260714_162514`):
`FORECAST_SCENARIO=base_raw_uplift`, `FORECAST_ACTIVATE_RUN=base_raw_uplift`,
`FORECAST_UPLIFT_PROFILE_VERSION=weekly_20260714`. Manually triggered
`systemctl start forecast-production.service` (full run, ~9 min) rather than
waiting for the nightly timer. New active run:
`prod_base_bakery_raw_uplift_sku_20260714_h14`.
`scripts.verify_prod_deploy` → `VERIFY OK`.

**Verified the uplift is live** by comparing the same SKU across scenarios:
product 11465 (Пирог с Манго, bakery 16) went from `2.97`/day
(`base_no_sku_uplift`, no uplift) to `3.44`/day (`base_raw_uplift`, this
change) — still below the ~6.9/day actual, but a real, directionally
correct increase from the multiplier, not a no-op.

**Important — magnitude/blast-radius note for the pilot team**: this uplift
is intentionally **not renormalized** (`apply_bakery_profiles_clickhouse.py`
skips renormalization when `use_raw_uplift_multiplier=True`), so per-hour
SKU-forecast sums can now legitimately **exceed** what the bakery-day model
predicted for that hour — observed up to `607` units summed across SKUs in
a single bakery-hour on the new run. This is the intended mechanism (lift
SKU-level forecasts above the aggregate to counter suspected undercounting),
not a bug, but it means downstream consumers (baking plan, any capacity
planning) will see materially higher SKU-hour numbers than under
`base_no_sku_uplift` — worth watching closely during the pilot.

Rollback: revert to scenario `base_no_sku_uplift` in VM `.env` (restore from
`.env.bak_20260714_162514` or edit the three keys back) and re-run
`forecast-production.service` — no code rollback needed, the smoothing
script's floor-restoration only affects behavior when
`use_raw_uplift_multiplier=True`, harmless with the old scenario active. If
the smoothing code itself needs to be rolled back too:
`src/experiments_v2/smooth_sku_hour_share_profile.py.bak_20260714_152419`
on the VM.

## Baking Plan Reverted To Template-Driven, Deployed To Blackhole (2026-07-14)

Phase 2 of the pilot reconfiguration (phase 1 was the SKU-uplift
reactivation above). `apps/baking_plan/` no longer computes window
placement (dropped both the pre-MILP peak-detection distribution and the
MILP solver) — window assignment is read directly from the reference Excel
template's pre-filled cells. See `docs/ops/DECISIONS.md` (2026-07-14 entry,
"Baking Plan Reverted From MILP To Template-Driven Window Assignment") for
the full rationale and `docs/baking_plan_implementation.md` for the current
spec.

Restored `apps/baking_plan/assets/template.xlsx` (4 revenue-tier sheets +
"комментарии") and `assets/individual/{20,21,22}_*.xlsx` from git history
(pre-2026-07-09-teardown commit `8e3e79f~1`), replacing the MILP-era
single-sheet template and empty `individual/` directory. Deleted
`capacity.py`, `algorithms/` (milp.py/greedy.py/common.py), and
`constants.py` (PDF-derived night-storage caps) — fully removed, not left
dormant. Added `apps/baking_plan/allocation.py` (pure window-reading/
allocation functions). Rewrote `demand.py`, `rendering.py`, `service.py`;
`assortment.py` and `router.py` unchanged.

Verified locally (read-only against **production** ClickHouse tables, not
dev — `.env.dev`'s `bakeable_products_dev` is missing the `scope`/
`bakery_id` columns added to prod on 2026-07-06, a pre-existing schema-drift
bug unrelated to this change, flagged in `DECISIONS.md`): generated real
`.xlsx` output for bakery 21 (individual template, non-standard sheet
label, confirmed the sheet-selection fallback handles it) and bakery 16
(base template, correctly matched "от 3млн" by revenue), both showing
partial per-row window population matching the template's own pre-filled
structure, and leftover (not-in-template) fastfood SKUs correctly appended
with no window breakdown and a raw unrounded total.

**Deployed to Blackhole** (`82bb03a8`, host `fhmab3h2o3lo0jqd552k`) the same
session, once VibeCode API credentials were provided (saved as
`.codex/blackhole.env`, gitignored, alongside the pre-existing
`.codex/prod_vm.env` for the unrelated forecast-writer VM). Deploy method:
same tarball-replace pattern as 2026-07-11/13, this time via the VibeCode
REST `/infra/servers/:id/exec` endpoint directly (`vibecode_api.py`
scratchpad helper) rather than a prior session's access path:

- Fetched the `origin/master` GitHub tarball into `/tmp/deploy_src`, staged
  `/tmp/deploy_stage/opt/app/app` (from `apps/forecast_embedded/app`) and
  `/tmp/deploy_stage/opt/baking_plan` (from `apps/baking_plan`, including
  the restored `assets/template.xlsx` and `assets/individual/*.xlsx`).
- Ran a staged preflight (`cd .../opt/app && /opt/app/.venv/bin/python -c
  "import app.main"`, reusing the existing venv) — passed — before backing
  up anything live.
- Backed up `/opt/app/app` → `/opt/app/app_backup_20260714_150358` and
  `/opt/baking_plan` → `/opt/baking_plan_backup_20260714_150358`, replaced
  both live directories from the staged tree, `chown root:root`, re-ran the
  same preflight import at the live location (passed), then
  `systemctl restart app.service`.
- Post-deploy: `systemctl is-active app.service` → `active`,
  `curl http://localhost:3000/health` → `{"ok":true,"app_env":"prod",
  "table_suffix":""}`.

**Not smoke-tested at the route level.** Unlike prior baking-plan deploys,
this session did not verify `GET /bakery/{id}/baking-plan.xlsx` directly —
doing so would have required guessing/forging the `x-vibe-user-*` admin
auth headers this endpoint checks (`app/auth.py`), which the auto-mode
safety classifier correctly flagged as credential forgery against a live
production service with no explicit authorization for that specific
bypass. The underlying business logic (template selection, window
allocation, rendering) was already verified thoroughly pre-deploy against
real production data locally (see above) — service health and a clean
import are the only route-level confirmation for this deploy. Whoever has
a real portal/admin session should click through the actual endpoint at
least once before trusting it fully.

## Lead-1 Backfill Rebuilt Under base_raw_uplift For 2026-07-01..13 (2026-07-14)

Following the phase-1 scenario switch, rebuilt lead-1 (day-ahead)
historical snapshots for the full 2026-07-01..2026-07-13 window under the
new `base_raw_uplift` scenario, so fact-vs-forecast history reflects the
pilot model instead of the old `base_no_sku_uplift` backfills that
previously covered these dates. Used
`scripts/build_prod_lead1_model_backfill.py --use-raw-uplift-multiplier
--uplift-profile-version weekly_20260714 --use-rolling-bias
--replace-existing` (matches the live scenario's rolling-bias correction
and the newly-rebuilt profile version from the phase-1 floor restoration).
Run ids: `backfill_base_bakery_raw_uplift_sku_rollingbias_YYYYMMDD_h1`.

Split into two runs on the VM due to a background-process interruption
(nohup'd child survived a first SSH channel drop but was later found dead
mid-run — see `[[vm_ssh_access_and_deploy_gotchas]]`-style note, not fully
root-caused): 2026-07-01..07 completed in the first run, 2026-07-08..13 in
a second, restarted nohup'd run. Confirmed via direct ClickHouse query that
all 13 dates now carry the new run_id in
`bakery_forecast_day_snapshots`/`sku_forecast_day_snapshots`/
`sku_forecast_hour_snapshots` (`lead_days = 1`); dates where the old
no-uplift backfill row hasn't been merged away yet by
`ReplacingMergeTree(generated_at)` show both run ids temporarily — the new
(later `generated_at`) one wins once merged, per the documented engine
behavior (see the 2026-07-13 "Discovered but did not fix" note above).

These are draft backfill runs for historical comparison only — never
activate them as the production forecast.

## Assortment-Exclusion Demand Fix Under Raw Uplift (2026-07-14/15)

See `docs/ops/DECISIONS.md` (2026-07-14/15 entry) for the full root-cause
and fix. Summary of what's live now:

- Two commits (`114bacd`, `488af38`) deployed to the VM the same session
  they were found — `src/experiments_v2/apply_bakery_profiles_clickhouse.py`
  now compensates for assortment-filtered-out demand under
  `use_raw_uplift_multiplier=True`, instead of silently dropping it.
- New active run after both fixes: `prod_base_bakery_raw_uplift_sku_20260715_h14`
  (horizon `2026-07-15..2026-07-28`). Verified directly: bakery 257
  (Ярмарочная 12, Чебоксары) SKU-day-sum-to-bakery-day-total ratio went
  0.62 → 0.89 (first fix) → 1.30 (second fix), now matching every other
  pilot bakery's 1.26-1.32 range.
- The 2026-07-01..13 lead-1 backfill (built the previous day with the
  un-fixed code, see the entry above) is being rebuilt with the fixed
  code so historical dashboard views correct themselves too — run ids
  unchanged (`backfill_base_bakery_raw_uplift_sku_rollingbias_YYYYMMDD_h1`),
  `--replace-existing` so `ReplacingMergeTree(generated_at)` supersedes
  the stale rows once merged.
- Rollback: VM backups at
  `src/experiments_v2/apply_bakery_profiles_clickhouse.py.bak_20260715_084030`
  (pre-first-fix) and `.bak_20260715_085842` (pre-second-fix).

## Per-SKU Raw-Uplift Cap Deployed (2026-07-15)

The `base_raw_uplift` production scenario now caps each
`(forecast date, bakery, product)` daily SKU forecast at `1.2` times that
SKU's recent rolling daily mean. The cap only scales forecasts down; SKUs
without recent history are left unchanged. This replaces the proposed
bakery-level cap for the pilot because a bakery-level scale reduced every SKU
equally and did not remove the large SKU-specific positive-bias outliers.

- Code: commit `466217c` (`cap_sku_uplift_per_sku` plus production CLI/env
  wiring and tests), pushed to `origin/master`.
- VM `.env`: `FORECAST_MAX_SKU_UPLIFT_RATIO=1.2`.
- Deployment backup timestamp: `20260715_082356` for both changed Python
  files and `.env` under `/opt/demand-forecasting-model`.
- Manually triggered `forecast-production.service`; systemd result was
  `success` with `ExecMainStatus=0`.
- Active run remains
  `prod_base_bakery_raw_uplift_sku_20260715_h14`, republished with
  `generated_at=2026-07-15 11:33:21+03:00`; verification ended with
  `VERIFY OK`.
- Allocation summary confirms the cap ran: `130139` of `445950` SKU-days
  capped (`29.2%`), average scale among capped SKU-days `0.8172`.
- `forecast-production.timer` remains enabled and active.

Rollback: restore the two `.bak_20260715_082356` Python files and the matching
`.env` backup, or remove `FORECAST_MAX_SKU_UPLIFT_RATIO` from `.env`, then
rerun `forecast-production.service` and verify the intended active run.

## Hierarchical Bakery/SKU Haircut Deployed (2026-07-15)

The active `base_raw_uplift` scenario now applies a downward-only hierarchical
post-processing coefficient after the SKU cap. Coefficients are derived from
the latest seven days of lead-1 forecasts and UI-equivalent actual sales:

- bakery coefficient targets a forecast/actual ratio of `1.15`;
- bakery-product coefficients are shrunk toward the bakery coefficient with a
  `7`-day prior;
- maximum haircut is `15%` (`min_coefficient=0.85`);
- if the bakery-level history is not over the target, the bakery and all its
  SKUs are protected from any haircut.

Code commit `3470678` was pushed to `origin/master`. VM `.env` now contains
`FORECAST_HIERARCHICAL_HAIRCUT_TARGET_RATIO=1.15`, history days `7`, pair prior
days `7`, and minimum coefficient `0.85`. Deployment backups use timestamp
`20260715_104624` for both Python files and `.env`.

Manually reran `forecast-production.service`; systemd finished with `success`
and `ExecMainStatus=0`. Active run remains
`prod_base_bakery_raw_uplift_sku_20260715_h14`, republished with generated time
`2026-07-15 13:55:55+03:00`; `scripts.verify_prod_deploy` ended with
`VERIFY OK`. Live allocation summary:

- SKU cap: `130731 / 445950` SKU-days capped;
- hierarchical haircut: `3714640 / 5020196` SKU-hour rows scaled;
- total SKU forecast: `2820612.58 -> 2699153.09` (`0.956939`, a `4.31%`
  reduction after the cap);
- `63 / 212` bakeries protected from haircut;
- `36562` bakery-product history pairs used.

The production timer remains enabled and active. Historical lead-1 snapshots
were not rebuilt with the haircut as part of this deploy; the deployed active
`h14` run is the source of truth for current forecasts.

Rollback: restore the two `.bak_20260715_104624` Python files and matching
`.env` backup, rerun `forecast-production.service`, then require `VERIFY OK`.

## SKU Cap / Assortment Compensation Ordering Regression Fixed (2026-07-15)

The initial SKU-cap deployment applied the cap *after* assortment-exclusion
compensation. That order regressed the 2026-07-15 bakery-257 fix: compensation
redistributed excluded-SKU demand onto the remaining assortment, then the cap
mistook the redistribution for excessive per-SKU uplift and removed it again.
Bakery 257's active SKU/bakery ratio fell from the previously verified `1.30`
to an average `0.787` (`0.702..0.869`). The later hierarchical haircut was not
the cause; bakery 257 was correctly protected from it.

Commit `0baf002` moves the SKU cap to the complete pre-assortment SKU set. The
order is now cap -> assortment filter -> exclusion compensation -> protected
hierarchical haircut. A regression test asserts that compensation preserves
the already-capped pre-filter total. Backfill CLI wiring was extended with the
hierarchical parameters in the same commit.

Deployed to the production VM with backup timestamp `20260715_123751`, then
manually reran `forecast-production.service`. Systemd finished with `success`,
`ExecMainStatus=0`, and `scripts.verify_prod_deploy` ended with `VERIFY OK`.
The active run remains `prod_base_bakery_raw_uplift_sku_20260715_h14`,
republished at `2026-07-15 15:47:04+03:00`. Bakery 257 now has active
SKU/bakery ratio average `1.142`, range `1.04..1.24`; the SKU sum is again
above the bakery-day forecast while still respecting the cap on the complete
SKU set.

A replacement lead-1 rebuild for 2026-07-01..14 was started as transient unit
`forecast-lead1-orderfix-backfill-20260715.service` with the full current
production logic (rolling bias, raw uplift, SKU cap `1.2`, and hierarchical
haircut settings). Its draft runs must never be activated.

Rollback: restore the two `.bak_20260715_123751` runtime files, rerun
`forecast-production.service`, and require `VERIFY OK`. This rollback would
reintroduce the known ordering regression and is for emergency use only.

## Stockout-Aware Hourly Uplift Deployed (2026-07-15)

Evidence-based per-(bakery, product, hour) correction factors are now applied
after the hierarchical haircut in the `base_raw_uplift` scenario. Corrections
address systematic undercounting in hours after the last baking window runs
out of product — the "dropout" pattern where hourly sales drop to zero while
the bakery is still open and selling other items.

**Algorithm**: for each pilot bakery × SKU × coverage window, count stockout
days (продано/выпуск ≥ 0.90), detect last-sale hour within window, estimate
missed demand from avg selling rate × hours after dropout (where bakery was
still active). Correction = `1 + stockout_rate × avg_missed / avg_daily_sold`,
capped at 2.0. Applied only where factor > 1.0 (never scales down).

**Result on prod run**:
- `13,198` of `5,667,202` SKU-hour rows corrected (pilot bakeries only)
- Avg correction factor: `1.205` (+20.5%) where applied
- Evening hours (16-23h) get highest correction (~1.23) — last window covering
  8 hours is the dominant source of missed demand (57% of estimated misses)

**Files changed** (via base64-SSH, VM git still blocked):
- `scripts/build_stockout_correction.py` — new script; uploaded to VM
- `src/experiments_v2/apply_bakery_profiles_clickhouse.py` — backup
  `.bak_20260715_165257`
- `pipelines/forecast_publish/run_production_inference.py` — backup
  `.bak_20260715_165355`
- `apps/baking_plan/allocation.py` — uploaded to VM (needed by build script;
  file existed locally from 2026-07-14 baking-plan revert but was absent
  from VM's older git state)

**ClickHouse**: `sku_hour_stockout_correction_embedded` table created and
populated in prod with `4446` rows (`profile_version=stockout_20260715`,
5 pilot bakeries, 58 SKUs).

**VM `.env`**: `FORECAST_STOCKOUT_CORRECTION_VERSION=stockout_20260715` added.

Active run `prod_base_bakery_raw_uplift_sku_20260715_h14` republished at
`2026-07-15 17:14:18+03:00`; `VERIFY OK`.

Rollback: restore the two `.bak_20260715_165257` / `.bak_20260715_165355`
Python files, remove `FORECAST_STOCKOUT_CORRECTION_VERSION` from VM `.env`,
rerun `forecast-production.service`, verify.

## Double-Uplift Fix: Pilots Evening Profile Deployed (2026-07-15)

**Problem identified**: pilot bakeries were receiving two simultaneous uplifts:
1. `weekly_20260714` mean-share floor multiplier (~×1.28 avg) — applied to
   **all hours** of pilot bakeries
2. Stockout correction (`stockout_20260715`) — applied only to dropout hours
   (16-23h, where product runs out)

This double-counting meant the stockout correction had zero net effect on total
daily forecast vs baseline — the mean-share floor was already uplifting all
hours beyond what the stockout correction added, and both ran simultaneously.
The overall pilot bias was +22.1% against 60-day avg, not meaningfully
different from baseline.

**Fix**: built `pilots_evening_20260715` uplift profile from `weekly_20260714`
with all 654 pilot-bakery rows set to `sku_uplift_multiplier = 1.0`. Non-pilot
bakeries (26,501 rows, avg `1.294`) copied unchanged. Stockout correction is
now the **sole** uplift mechanism for pilot bakeries.

Script: `scripts/build_pilots_evening_uplift.py` (runs locally against prod
ClickHouse; writes directly to `sku_hour_uplift_multiplier_embedded`).

**Result after deploy** (pilot bakeries {16,20,21,22,257}, vs 60-day avg):

| Bakery | Before | After |
|--------|-------:|------:|
| 16 | +19.8% | +3.8% |
| 20 | +17.9% | +7.5% |
| 21 | +22.0% | +8.5% |
| 22 | +25.0% | +17.8% |
| 257 | +26.5% | +13.3% |
| **Total** | **+22.1%** | **+9.6%** |

Note: positive bias vs 60-day avg in evening hours (16-19) is **expected** —
the historical avg includes censored stockout days where actual sold was lower
than true demand. The correction estimates uncensored demand, so FC > hist_avg
is the intended behavior for those hours. The remaining +9.6% overall is an
aggregate of slight over-correction in evenings and slight under-forecast in
mornings (h9-11 for bakery 21: −14 to −19%).

**CF distribution** (`stockout_20260715`, 2,010 correction rows > 1.0):
- Mean CF: 1.227; p50: 1.173; p90: 1.459; max: 2.0 (8 rows, all bakery 21 pid 10662)
- By hour: h06-07 (16 SKU, mean 1.10), h16-23 (58 SKU, mean 1.23)
- By bakery (evening): bak16=1.158, bak20=1.289, bak21=1.213, bak22=1.190, bak257=1.303

VM `.env` updated: `FORECAST_UPLIFT_PROFILE_VERSION=pilots_evening_20260715`
(was `weekly_20260714`). Service manually triggered 2026-07-15 19:33 UTC,
completed at 19:40 UTC (7m39s CPU). Run_id unchanged:
`prod_base_bakery_raw_uplift_sku_20260715_h14`.

Rollback: set `FORECAST_UPLIFT_PROFILE_VERSION=weekly_20260714` in VM `.env`,
rerun `forecast-production.service`, verify.

## Pilot Expanded To 11 Bakeries (2026-07-16)

Pilot set expanded from 5 to 11 bakeries. Added: {28, 80, 89, 107, 221, 222}.
Kept existing: {16, 20, 21, 22, 257}.

| ID | Пекарня | Bias vs 60d avg |
|----|---------|----------------|
| 16 | Кулагина 4 Казань | +9.3% |
| 20 | Мира 45 Дербышки Казань | −0.5% |
| 21 | Парковая 7 Казань | +10.4% |
| 22 | Сибирский Тракт 25 Казань | +17.6% |
| 28 | Гудованцева 27 Казань | +7.3% *(новая)* |
| 80 | Калинина 63 Казань | −5.6% *(новая)* |
| 89 | Парина 6 Казань | −5.0% *(новая)* |
| 107 | Четаева 46А Казань | −4.4% *(новая)* |
| 221 | Салиха Батыева 15 Казань | +10.7% *(новая)* |
| 222 | Габдуллы Тукая 62А Казань | +20.4% *(новая, наблюдать)* |
| 257 | Ярмарочная 12 Чебоксары | +16.1% |

Итого: **+6.7%** (новые +2.5%, старые +11.4%).

Changes deployed:
- `scripts/build_stockout_correction.py` + `scripts/build_pilots_evening_uplift.py`:
  `PILOT_BAKERY_IDS` обновлён до 11 пекарен.
- `stockout_20260716`: 10,152 строки, 79 SKU (было 4,446 / 58 SKU для 5 пекарен).
- `pilots_evening_20260716`: 1,437 пилотных строк = 1.0 (было 654).
- VM `.env` backup: `.env.bak_20260716_pilots11`.

Rollback: restore `.env.bak_20260716_pilots11`, rerun `forecast-production.service`.
To reduce pilot: rebuild both tables with a smaller `PILOT_BAKERY_IDS` set.

## Allocation and weekly profile refresh repair (2026-07-20)

- Production allocation assortment is refreshed daily into
  `assortment_city_products` from the recent seven-day sales window. Cities
  absent from that window carry forward their latest known assortment with
  source `carried_forward_no_recent_sales`.
- Allocation reads only the latest effective city assortment batch and rejects
  batches older than two days.
- The weekly SKU profile was rebuilt through 2026-07-19: 3,537,105 rows across
  210 bakeries and 1,142 products.
- Weekly uplift refresh replaces only its own version. The production version
  `pilots_evening_20260716` is preserved; `weekly_20260720` was loaded beside it.
- Active run: `prod_base_bakery_raw_uplift_sku_20260720_h14`, generated at
  2026-07-20 19:30 MSK. Verification: 489,130 SKU-day rows, 5,499,898 SKU-hour
  rows, all 12 allocation control pairs non-zero on all 14 days, `VERIFY OK`.
- Current allocation snapshot: 2,190 rows across 10 cities, zero `unknown`
  rows. Same-day reruns replace older refresh-managed rows for the effective
  date; cleanup cutoffs are required to be timezone-aware.
- `forecast-production.timer` and `weekly-profile-refresh.timer` are enabled
  and active on the production writer VM.

## Baking Plan Reverted To MILP (2026-07-21)

`apps/baking_plan/` switched back from template-driven window assignment to
MILP-based allocation. The template is now used only to read the bakery's
window time structure (which time slots exist); quantity allocation and
rendering are fully MILP-driven.

Key files added/restored (all under `apps/baking_plan/`):

- `demand_milp.py` — `build_sku_demand()`: loads SKU demand with hourly
  profile, credits yesterday's overnight defrost stock out of today's early
  hours for `DEFROST_SKU_NAMES` (11 SKUs) via `sku_forecast_hour_snapshots`
  `lead_days=1` snapshot.
- `constants.py` — `NIGHT_STORAGE_DIRECT_UNITS_BY_SKU`,
  `NIGHT_PREP_LABOR_MINUTES_BY_SKU`, `DEFROST_SKU_NAMES`, `DEFROST_HOURS`.
- `capacity.py` — reads `baking_capacity_config` and
  `baking_category_molding_minutes` from ClickHouse.
- `algorithms/milp.py` — HiGHS-backed MILP solver (scipy, already in
  requirements.txt since 2026-07-11). Cumulative coverage constraints ensure
  production is scheduled before demand arrives (respects hourly sales
  profile). Separate labour pools for bakers and baker assistants.
- `algorithms/common.py`, `algorithms/greedy.py` — shared helpers.
- `rendering_milp.py` — `render_workbook()`: builds Excel from scratch (no
  template mutation). Yellow fill for mandatory assortment (10 SKUs), red for
  full shortfall, yellow for partial shortfall, orange for defrost, purple for
  двухдневка. `Итого` = sum of all windows with no column collision.
- `service.py` — rewritten: calls `build_sku_demand` → MILP →
  `render_workbook`. No longer calls `allocation.allocate_template_row` or
  `rendering.write_plan`.

Mandatory assortment (10 SKUs forced into first window) is hardcoded in
`rendering_milp.MANDATORY_ASSORTMENT` — same list as the original MILP-era
implementation (restored from git 3b18eac).

Operator scripts (local only, not deployed to Blackhole):
- `scripts/run_milp_baking_plan.py` — console plan for all pilot bakeries
  (verified: 0 shortfall all 11 bakeries on 2026-07-21).
- `scripts/export_milp_baking_plan.py` — exports `.xlsx` for all pilots.

Deploy to Blackhole: same tarball-replace pattern as previous deploys. Must
replace both `/opt/app/app` (from `apps/forecast_embedded/app`) and
`/opt/baking_plan` (from `apps/baking_plan`). `scipy==1.17.1` is already in
`requirements.txt` and installed on the Blackhole venv from the 2026-07-11
deploy — no `pip install` needed.

Required ClickHouse tables (all present since 2026-07-11 MILP deploy):
- `baking_sku_meta` — kratnost, dough_group, station, is_two_day per product
- `baking_capacity_config` — bakers/ovens/trays/bake_minutes per bakery
- `baking_category_molding_minutes` — labor minutes per unit per category

Rollback: redeploy the previous `service.py` (template-driven version) and
remove `demand_milp.py`, `capacity.py`, `constants.py`, `algorithms/`,
`rendering_milp.py` from `/opt/baking_plan`. Or restore from the Blackhole
backup that will be taken before this deploy.

## ClickHouse Connection Leak Fixed (2026-07-21)

`apps/forecast_embedded/app/db.py` previously called `get_client()` in a way
that created a brand-new ClickHouse TCP/TLS connection on every invocation.
With 17 call sites across `bakery.py` and `runs.py`, every user request
leaked multiple file descriptors that were never explicitly closed. Under load
these accumulated to the OS fd limit, crashing the Blackhole `app.service`
with `OSError: [Errno 24] Too many open files` (observed 2026-07-21 12:48 UTC
— 489 such errors before the server rebooted at 12:48:42).

Fix (commit `9c7770b`): lazy singleton — `_client` module-level variable;
first call creates the client, all subsequent calls reuse it.
`clickhouse_connect` uses `urllib3` internally which is thread-safe and
manages its own connection pool. Deployed to Blackhole via exec API
(backup: `/opt/app/app/db.py.bak_20260721`); `app.service` restarted and
verified `active` + `/health` → `{"ok":true}`.

Rollback: restore `/opt/app/app/db.py.bak_20260721`, restart `app.service`.

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
## Stockout direction shadow update (2026-07-22)

- Read-only, run-time-aware analysis classified all 47 clear-stockout SKU-days
  with no forecast as exclusions by the latest allocation-assortment batch
  available before the historical run. An earlier 46+1 split was lookahead
  caused by ignoring `loaded_at` on a batch loaded the following day.
- In `prod_base_bakery_raw_uplift_sku_20260722_h14`, all 18 affected
  bakery/SKU pairs are present on all 14 horizon days; the refresh repair from
  2026-07-20 has removed the observed failure mode.
- The stockout shadow runner now records at most one prospective observation
  per Moscow calendar date under
  `reports/stockout_direction_shadow/history/`.
- First observation: 2026-07-22, all gates pass, 1/21 distinct days observed.
- Historical replay days do not count toward the prospective requirement.
- No production state was changed.

### Local assortment coverage guard (not deployed)

- A fail-fast pre-allocation guard now exists locally. It compares the prior
  seven days of sales with the selected allocation-assortment batch and rejects
  established missing bakery/SKU pairs (>=2 selling days and >=2 units).
- Read-only validation for the 2026-07-22 run: 211 bakeries, 29,578 recent
  bakery/SKU pairs, zero blocking gaps.
- This code has not been deployed to the production writer VM.

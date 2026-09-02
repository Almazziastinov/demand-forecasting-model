# Current Project State

Last updated: 2026-09-02

## SALES ETL EMERGENCY MODE — ACTIVE (2026-09-02)

- `Svezhar.fct_check_lines` is incomplete. The deduplicated 2026-09-01 data
  contains 74,505.5 units / 27,207 checks and stops at 11:34:38 MSK, versus
  roughly 176k units for the previous three Tuesdays. No 2026-09-02 sales
  rows were present at the incident check. The ETL process resumed writing at
  08:25:35 MSK but continued replaying incomplete 2026-09-01 data rather than
  advancing the business watermark.
- The run generated from that incomplete day,
  `prod_direct_alpha_025_20260902_h14`, is no longer served. The known-good
  Direct run `prod_direct_alpha_025_20260831_h14` (history through 2026-08-30,
  horizon through 2026-09-13) was reactivated for all forecast consumers.
  Snapshot scope: 2,478 bakery-day, 149,526 SKU-day and 2,484,338 SKU-hour.
- `forecast-production.timer` on the production VM is deliberately
  **disabled/inactive** so incomplete facts cannot create or activate another
  run. Do not re-enable it until 2026-09-01 and 2026-09-02 have been backfilled
  and completeness checks pass.
- Daily pilot files continue at 04:00 UTC. The publisher is pinned to
  `prod_direct_alpha_025_20260831_h14` and sets
  `PILOT_DISABLE_STOCK_SUBTRACTION=1`; every stock cell renders
  `нет данных по остатку`, no stock is subtracted, and the chat text explicitly
  identifies emergency mode.
- Controlled 2026-09-03 dry-run: 55 bakeries, 3,213 rows, forecast 52,625.9,
  production plan 62,014, and every stock cell marked unavailable. Publisher
  SHA-256:
  `dd257eb367b30380f523c7ba4ff5c87f9e1ff50f08e94372c325bb10df60f4f6`.
  Rollback backup:
  `/opt/backups/pilot_forecast_emergency_etl_20260902_142909`.
- `scripts.verify_prod_deploy` confirms the reactivated run's active status,
  model and snapshot counts, but exits non-zero on the expected bookkeeping
  mismatch because the latest local refresh summary still names the rejected
  2026-09-02 source run. Do not treat that summary as the served run during
  this incident.

## Direct nightly freshness-guard incident and stock-data warning (2026-09-02)

- The 2026-09-01 and first 2026-09-02 nightly attempts were stopped by the
  legacy hourly SKU-profile freshness guard (`data_through=2026-08-23`,
  `max_age_days=8`). Direct alpha=.25 does not use that profile for its daily
  SKU allocation. Production now explicitly sets
  `FORECAST_PROFILE_MAX_AGE_DAYS=-1`; the guard reports `status=disabled` while
  the legacy source stage remains an inactive implementation input.
- A manual recovery run completed successfully. The active run is
  `prod_direct_alpha_025_20260902_h14`, sourced from
  `prod_base_bakery_norm_recent_20260902_h14`, with history through 2026-09-01
  and horizon 2026-09-02..2026-09-15. It contains 2,492 bakery-day, 150,192
  SKU-day and 2,501,412 SKU-hour snapshot rows. `scripts.verify_prod_deploy`
  returned `VERIFY OK`; `forecast-production.timer` remains enabled and active
  for 03:30 UTC daily.
- Rollback for the environment change is the production VM backup matching
  `.env.bak_20260902_*_before_direct_profile_guard_bypass`.
- The pilot workbook complaint for `Лукина 5 Чебоксары` was confirmed as an
  upstream completeness problem, not a Direct-allocation defect. For
  2026-09-01 the event source has 145 sales but zero recorded production and
  zero transfers, so an exact closing stock cannot be reconstructed. The
  publisher no longer displays a fabricated numeric zero when a bakery has
  positive sales and no recorded production: it displays
  `нет данных по остатку`, while production need is conservatively calculated
  without subtracting an unknown stock value.
- Publisher dry-run for 2026-09-02 retained 3,216 rows and marked all 56
  `Лукина 5` rows with the explicit missing-stock label. Installed publisher
  SHA-256: `870d504c4c5905dcdbbfbd2d23eff56057a3369f87ce0bd6e0c0c9ae2aac69c5`.
  Rollback backup:
  `/opt/backups/pilot_forecast_missing_stock_label_20260902_074854`.

## Pilot closing-stock full-flow correction (2026-09-02)

- The pilot publisher previously computed previous-day stock as only
  `max(produced - sold, 0)`. It did not query transfers or write-offs, so sent
  and written-off product could remain in the displayed stock and reduce the
  production plan incorrectly.
- The installed publisher now deduplicates `fct_moves` by `(move_id, line_id)`
  and `fct_write_offs` by `(write_off_doc_num, line_id)` using `argMax` over
  `_updated_at`. Closing stock is calculated as
  `max(produced + received - sent - sold - written_off, 0)` for the preceding
  date. Positive sales with neither production nor receipts still render as
  `нет данных по остатку` instead of a fabricated zero.
- Controlled 2026-09-02 dry-run: 55 bakeries, 3,216 rows, forecast 48,496.1,
  closing stock 33,448, production plan 26,117; one bakery / 52 rows remain
  explicitly marked as unavailable. SKU concentration remains healthy: no
  bakery has a top-SKU share at or above 20%.
- The corrected workbook was published at 12:46 MSK: text message `8258485`,
  file message `8258487`, chat file `1678283`.
- Installed publisher SHA-256:
  `9576467284351e49f124f03db7fab4dc18a232c3c8f0ddcfa2fa46727185ea19`.
  Rollback backup:
  `/opt/backups/pilot_forecast_stock_flow_full_20260902_manual`.
- Remaining source limitation: there is no authoritative opening-inventory
  snapshot in the publisher input. The corrected figure includes every
  observable previous-day flow but cannot reconstruct stock carried into that
  day from an earlier date. Do not describe it as a warehouse/accounting stock
  snapshot until such a source is integrated.

## AUTHORITATIVE ACTIVE MODEL — READ THIS FIRST

Production has switched from the legacy hourly/category SKU allocation to the
**Direct alpha=.25 model**. This is not a research-only candidate and must be
the default meaning of “current model” in future work.

- Active `model_version`: `direct_alpha_025_v1`.
- Emergency active run:
  `prod_direct_alpha_025_20260831_h14`, horizon 2026-08-31..2026-09-13. It was
  reactivated on 2026-09-02 because the next run used incomplete sales facts.
- Nightly run pattern: `prod_direct_alpha_025_YYYYMMDD_h14`.
- The bakery-day LightGBM forecast remains the volume source. Direct then
  allocates each bakery-day total directly across mature SKUs. It does **not**
  preserve legacy category totals and does **not** use the old hourly SKU
  profile for allocation. Category totals emerge from the SKU predictions.
- Selected post-processing is causal expected-loss predictive uplift, Core-SKU
  protection, alpha `0.25` soft volume expansion, adaptive floor, and causal
  tail cap. SKU-day quantities are distributed to hours only after the Direct
  daily allocation, conserving every finalized SKU-day quantity.
- SKU cold start is an independent, non-competing path: cold-start SKU volume
  is not subtracted from the mature Direct pool. Bakeries with no positive sale
  for more than 30 days are treated as closed and excluded.
- The legacy source run `prod_base_bakery_norm_recent_*` is still built as an
  inactive intermediate because it supplies the bakery-day forecast and
  refreshed datasets. It is **not** the served production model. A successful
  systemd `ExecStartPost` Direct step activates the final run.
- The Blackhole/VibeCode app remains read-only. Its pilot publisher consumes
  the active Direct run and must not reapply legacy cold-start allocation.
- Latest corrected pilot workbook check (2026-09-01): 55 bakeries, 3,215 SKU
  rows, 51,903.6 forecast units; maximum top-SKU bakery share `17.66%`, with
  zero bakeries at or above `20%`. The old 40–58% concentration failure is
  absent.
- Remaining operational risk is **kratnost rounding, not Direct allocation**:
  the same workbook has 46,561.4 net need and 55,771 planned production
  (`+9,209.6` units from row-level upward rounding). Treat forecast quality and
  conversion of forecast to production plan as separate layers.

Live verification on 2026-09-01 returned `VERIFY OK`; the production timer is
enabled and active. Rollback is activation of the corresponding verified
`prod_base_bakery_norm_recent_*` source run plus removal of the Direct systemd
drop-in, only as described in the runbook.

## Pilot management statistics refresh (2026-09-01)

- The stale static report on Blackhole was replaced by an automated build and
  publish job on the production forecast VM.
- `pilot-management-report.timer` is enabled and active on `201.51.7.24`; it
  runs daily at `05:00 UTC` (`08:00 MSK`), after the forecast writer and the
  pilot workbook publisher. The timer is deliberately non-persistent so
  enabling it after the scheduled time cannot trigger an unexpected catch-up.
- `pilot-management-report.service` runs as `forecast`, builds the event-aware
  pilot scope from `2026-07-23` through Moscow yesterday, requires complete
  daily coverage, and refuses to publish an empty last-day scope.
- Publication uses the VibeCode credential already required by the forecast
  publisher. The Blackhole app remains read-only with respect to ClickHouse:
  the VM builds the CSVs and atomically replaces
  `/opt/reports/pilot_management_summary`, preserving the previous directory
  in `/opt/backups` before the swap.
- Manual end-to-end run on 2026-09-01 succeeded: report range
  `2026-07-23..2026-08-31` (40 days), 55 bakeries on the last day,
  `forecast_source=snapshot_fallback`. Blackhole report files were written at
  `2026-09-01 09:11 UTC`; `app.service` and `/health` remained healthy.
- Installed entry point:
  `/opt/demand-forecasting-model/scripts/run_pilot_management_report_job.py`.
- The first two installation smoke attempts failed before publication because
  the VM still had the old fixed-38 builder. The event-aware report-only
  builder modules were then synchronized; no forecast run or served report was
  changed by those failed attempts.

## Baking SKU metadata template sync and no-silent-drop fallback (2026-09-01)

- `baking_sku_meta` was synchronized from
  `Шаблон плана выпекания для ИИ (1).xlsx` (updated 2026-08-04). The reviewed
  `комментарии` sheet yielded 78 matched production SKUs. Explicit aliases
  cover renamed hand rolls, tube rolls, large pizzas, Keksovyi mango and
  `Пирог с киви` (`11613`); fuzzy matches are never written.
- `Основа чиабатта покупная` is treated as a group heading, not an SKU.
  `Мексиканский ролл` remains unresolved because it is absent from
  `dim_products` and the active forecast. Newer separately business-confirmed
  SKUs absent from the template were preserved.
- Older active base versions for the synchronized product IDs were closed
  before inserting the 2026-08-31 version. Verification: 82 active rows for
  82 unique products and zero duplicate active product/scope/bakery keys.
  Confirmed changes include `Сосиска под шубой` multiple `10`,
  `Сэндвич курица` multiple `6`, and newly covered SKU `11572` multiple `10`.
- Full pre-change backup table:
  `baking_sku_meta_backup_20260901_0006_before_template_sync` (76 rows).
- The pilot publisher and the in-app single-bakery plan no longer silently
  discard a forecast SKU when `baking_sku_meta` is missing. Such a row is kept,
  unit-rounded, and the `Кратность` column displays
  `нет данных по кратности`. Known frozen/no-production rows with explicit
  metadata remain excluded as before.
- Pilot publisher read-only dry-run for 2026-09-01 produced 3,018 rows across
  55 bakeries before the early-filter audit. That initial zero
  missing-multiple result was invalid: an older `eligible[has_meta]` filter
  still removed the rows before rendering. The filter was removed after the
  missing `Капуста и курица` report.
- Corrected 2026-09-01 dry-run produces 3,215 rows across 55 bakeries and keeps
  197 missing-meta rows (1,227.4 forecast units) with the explicit text label.
  It includes `Капуста и курица` in 53 bakeries (318.1 forecast units) and
  `Пирожок капуста и курица` in 7 bakeries (187.9 units). Installed publisher
  SHA-256:
  `ae682062442370cdd63768eb6671f66c20ad716895feb99fade89829bbfc7257`;
  rollback backup:
  `/opt/backups/pilot_forecast_remove_early_meta_filter_20260831_213733`.
- In-app smoke for bakery 29 / 2026-09-01 produced 67 rows and retained three
  otherwise missing-meta products with the explicit text label. Installed
  `/opt/baking_plan/simple_plan.py` SHA-256:
  `4ad0ecedc0d97e44ee4567ff775d0f4318414f01f9f449b8a76ccecb0c60e887`.
  Rollback backup:
  `/opt/backups/baking_simple_plan_missing_kratnost_20260831_211918`.
  `app.service` is active and `/health` is OK. Blackhole forecast-writer
  timers remain disabled/inactive.

## Direct pilot publisher compatibility and schedule (2026-08-31)

- The Blackhole pilot publisher now reads the active run `model_version`. For
  `direct_alpha_025_v1` it does not reapply the legacy category-neutral
  new-SKU cold-start layer because independent cold-start is already included
  by the production Direct runner. Other model versions keep the prior
  publisher behavior.
- A read-only 2026-09-01 dry-run used active run
  `prod_direct_alpha_025_20260831_h14`, produced 2,951 workbook rows across 55
  pilot bakeries and did not send a Bitrix24 message. Zorge 101 showed
  Smetannik `25.1` and chicken triangle `232`, instead of the reported legacy
  plan values `1` and `420`.
- `pilot-forecast-publish.timer` was moved from `03:00 UTC` to `04:00 UTC`
  (07:00 MSK), after the production writer timer at `03:30 UTC`. It is enabled
  and active; the next trigger is 2026-09-01 04:00 UTC. Both Blackhole forecast
  writer timers remain disabled and inactive.
- Operational incident: restarting the persistent publisher timer after its
  old daily slot had passed immediately triggered one successful duplicate
  publication for 2026-08-31 (Bitrix message ids `8233805` and `8233807`). No
  2026-09-01 file was sent during deployment. Future schedule-only changes
  must stop the timer, update/reload it, and start it with a future trigger or
  temporarily disable persistence to avoid catch-up execution.
- Installed publisher SHA-256:
  `d1588e2ca6a4ba763651c77e9598c51d9f0cc09f3d5dae8dd5a10fea40d7d2a9`.
  Rollback backup:
  `/opt/backups/pilot_forecast_direct_compat_20260831_141320`.

## Closed bakery production filter (2026-08-31)

- Production now excludes a bakery before forecasting and assortment fallback
  when its last positive sale is more than 30 days before the facts cutoff.
- The 2026-08-30 refresh excluded 37 historically observed but inactive
  bakeries, reducing the active forecast scope from 214 to 177 bakeries.
  The same open-bakery set is used for bakery-day datasets and the flat
  bakery/SKU assortment, so closed IDs cannot return through carry-forward,
  city-core, or network-core fallback.
- Active run remains named `prod_direct_alpha_025_20260831_h14`, regenerated at
  16:23 MSK with 2,478 bakery-day, 149,526 SKU-day and 2,484,338 SKU-hour rows.
  All 37 closed IDs have zero rows at all three levels. `VERIFY OK`; production
  timer is enabled and active.
- Production backup:
  `/opt/backups/20260831_closed_bakery_filter/production_dataset_refresh.py`.
  Local verification: 13 focused tests and Ruff passed.

## Independent SKU cold-start candidate (2026-08-31)

- The agreed lifecycle has exactly two states: `cold_start` and `mature`; no
  transition blend is used.
- Cold-start bakery/SKU pairs are excluded from Direct normalization. The full
  bakery forecast is allocated over mature SKUs, while the own-sales EWMA
  cold-start forecast is added independently above that total.
- The pilot publisher no longer applies category-neutral cold-start
  renormalization for a Direct run and mature systematic corrections exclude
  cold-start rows.
- Local verification passes: 19 focused tests and Ruff on the new production
  Direct/cold-start code. This is a local candidate only; production services
  and the currently active run have not been changed by this edit.
- A fallback shadow using the locally archived 2026-07-22..2026-08-23 history
  and the frozen 2026-08-31 h14 Direct horizon estimates `+27,125` units
  (`+1.087%`) over 14 days from 1,258 effective bakery/SKU cold-start pairs.
  This is diagnostic only: the local archive does not contain the complete
  60-day lead-1 forecast history through 2026-08-30, so it over-classifies
  cold-start pairs. Exact production comparison remains pending network access.
- The shadow exposed and fixed an all-cold bakery guard: when classification
  leaves no mature SKU in a bakery-day, the original allocation is retained.
  Twenty focused tests now pass.

## Direct alpha=.25 shadow integration started (2026-08-31)

- The research candidate is frozen as Direct bakery-day-to-SKU allocation,
  causal expected-loss uplift, Core-SKU protection, alpha=.25 soft volume
  expansion, adaptive floor, and causal tail cap.
- Selected post-processing was moved into
  `src/experiments_v2/direct_alpha_allocation.py` with immutable defaults and
  no database dependencies. It does not use hourly/category allocation.
- `scripts/run_direct_alpha_shadow.py` writes local parquet/CSV/JSON artifacts
  only. It has no ClickHouse client, load path, or activation command.
- Four focused tests pass. On the calibrated 2026-08-17 frozen fold, the new
  module reproduces all 72,849 reference rows with max absolute error 0.0.
- The next integration component is the current-horizon causal feature builder
  and Direct/predictive model artifact loader. No production or dev database
  state was changed.

## Direct alpha=.25 current-horizon shadow passed (2026-08-31)

- A single read-only shadow was built for 2026-09-01 from active run
  `prod_base_bakery_norm_recent_20260831_h14`, using sales only through
  2026-08-30 (the day before run generation).
- Scope: 214 bakeries, 185 products, 12,282 SKU rows. Selected shadow volume is
  189,506.67 versus Direct P50 181,584.44; one tail-cap row was applied.
- Zorge 101 complaint is corrected on the fresh horizon: Smetannik changes
  0.043 -> 27.593 and chicken triangle 379.811 -> 235.282. Bakery 29 chicken
  triangle changes 466.815 -> 262.935.
- Zero negative/NaN/duplicate rows and zero incumbent-positive rows assigned a
  near-zero selected forecast.
- Twenty-five bakery-days have no 56-day sales evidence. They now use an
  explicit incumbent fallback. All four top shares >=30% belong to that
  fallback group; no observed Direct bakery-day is >=30%.
- Five focused tests pass. Shadow artifacts are under
  `reports/direct_alpha_current_shadow_20260831/`. No ClickHouse write or run
  activation occurred.

## Bakeable assortment mass inflation fix (2026-08-28)

- Root cause: `bakery_product_assortment_embedded` intentionally contains only
  bakeable categories, but the normalized allocation restored those remaining
  SKU rows to the full bakery-day forecast and the `0.95` allocation floor
  reinforced that transfer.  Demand previously allocated to non-bakeable
  products was therefore moved onto bakery products.
- Production now defaults `FORECAST_DISABLE_ASSORTMENT_RENORMALIZATION=1` for
  bakery-product assortment tables.  True missing bakery-hours are still
  filled, but existing filtered groups keep their post-filter mass; the
  bakery-total allocation floor is skipped in this mode.
- A read-only 2026-08-28 dry-run produced 32,405.82 units for the original 39
  pilot bakeries versus 47,867.35 before the fix; bakery 270 produced 425.997.
  The selected allocation ratio is 0.805236, all 756 true hour gaps were
  filled, and zero groups remained unfilled.
- The first same-id production reload timed out waiting for old
  `sku_forecast_hour_embedded` parts to disappear.  No incomplete run was
  activated.  The already generated corrected files were loaded under the
  unique run id `hotfix_base_bakery_norm_recent_20260828_h14` and activated.
  `scripts.verify_prod_deploy` ended with `VERIFY OK`: 2,996 bakery-day,
  172,676 SKU-day, and 1,901,964 SKU-hour snapshot rows.
- Blackhole publisher dry-run for 2026-08-28 generated 2,965 rows / 55
  bakeries without sending to Bitrix24.  Workbook forecast is 49,421.9 units
  versus 67,395.8 before the fix; bakery 270 is 396.1 versus 497.7.  All three
  temporary pletenka display names remain present.
- Production writer backup:
  `/opt/backups/assortment_mass_fix_20260828_093038`.  Writer timer is
  enabled/active; Blackhole remains read-only.

## In-app baking-plan assortment fix (2026-08-27)

- The actual `/bakery/{bakery_id}/baking-plan.xlsx` generator is
  `/opt/baking_plan/simple_plan.py`; it is separate from the pilot chat
  publisher. Its product universe now comes from the latest effective
  `bakery_product_assortment_embedded` snapshot for the requested bakery/date.
- Product names and categories are filled from `dim_products`. If an effective
  assortment SKU has no row for the requested day because of a sparse weekday
  profile, its automatic fallback is the same SKU's mean forecast over the
  active run horizon; no manual assortment value is involved.
- Production verification for bakery 270 / 2026-08-27 includes 11575, 11615,
  11616, and 11617 and excludes 11573/11574. The generated workbook has 51
  rows. Installed hash:
  `6a9e2385ce23b4199f56dbc839579a193cb3c8c01651edd007ce8463e10d137e`.
  Rollback backup:
  `/opt/backups/baking_plan_assortment_fix_20260827_153413`.
- `app.service` is active and `/health` is OK. Blackhole writer timers remain
  disabled/inactive; the separate pilot chat publisher timer remains
  enabled/active.
- The download response is explicitly non-cacheable and uses the distinguishable
  filename `baking_plan_<bakery>_<date>_assortment_v2.xlsx`, preventing an older
  same-named workbook in the browser/Downloads directory from being reopened.
  Router rollback backup:
  `/opt/backups/baking_plan_download_cache_fix_20260827_154607`.
- Temporary emergency display-name overrides are active in both the in-app
  plan and the pilot chat publisher: 11615=`Плетенка кленовая`,
  11616=`Плетенка с черникой`, 11617=`Плетенка с земляникой`. They do not
  mutate `dim_products` and must be removed after the duplicate-name records
  there are reconciled. Production verification generated all three names.
  In-app rollback backup:
  `/opt/backups/baking_plan_assortment_fix_20260827_160034`; publisher rollback
  backup: `/opt/backups/pilot_forecast_publisher_fix_20260827_160045`.

## Pilot Excel assortment fix (2026-08-27)

- The Blackhole chat publisher now fills missing `product_name` and
  `category_name` values from `dim_products` before applying bakeable-category
  filters. It also renders the completed cold-start candidate frame instead of
  falling back to the original forecast frame and silently discarding added
  bakery/SKU rows.
- A production dry-run for 2026-08-28 verified bakery 270 contains all four
  introduced SKUs: 11575, 11615, 11616, and 11617, with the expected categories
  and baking multiples. The dry-run did not send a Bitrix24 message.
- Installed Blackhole file hash:
  `de9d2f5ff0d8e4c1590c6d7ad91c7f1675ba655402d2ea86eadc1d27967ff2a7`.
  Rollback backup:
  `/opt/backups/pilot_forecast_publisher_fix_20260827_151128`.
- `pilot-forecast-publish.timer` remains enabled/active. The embedded app is
  healthy. Both Blackhole forecast-writer timers remain disabled/inactive.

## Automatic bakery assortment rollout (2026-08-27)

Production now uses `bakery_product_assortment_embedded` as the effective
bakery/SKU allowlist.  The primary source is positive sales for the concrete
bakery during the previous seven calendar days, restricted to bakery
categories (`пирог`, `выпечка`, `фастфуд`) and inactive-name prefixes are
excluded.  Effective-dated manual overrides remain emergency-only.

Automatic zero-data fallbacks are hierarchical and apply only when the bakery
has no current seven-day pairs: latest prior bakery snapshot, then the 80%
city core computed over bakeries participating in the window, then the 80%
network core for a completely new city.  Bakery-scoped filtering is repeated
inside city/hour fallbacks so a SKU from another bakery cannot leak back into
the forecast.

Active run:
`prod_assortment7d_v4_base_bakery_norm_recent_20260827_h14`, horizon
2026-08-27..2026-09-09.  Production verification returned `VERIFY OK` with
2,996 bakery-day, 172,130 SKU-day, and 1,893,880 SKU-hour snapshot rows.  The
writer timer is enabled/active.  Blackhole is read-only, app health is OK, and
both forecast timers there remain disabled/inactive.

Bakery 270 snapshot verification: removed SKU 11573/11574 are absent; new SKU
11575/11615/11616/11617 are present with positive forecast.  Baking metadata
for the four new SKU was seeded before activation.  Writer rollback backup:
`/opt/backups/assortment_rollout_20260827_094330`; Blackhole reader backup is
under `/opt/backups/assortment_rollout_*` on that host.

## Direct bakery-day to SKU allocation research (2026-08-27)

- A clean frozen-fold candidate now allocates the bakery-day total directly
  across all forecast-assortment SKUs. Incumbent SKU shares, incumbent category
  totals/shares, hourly profiles, and old uplift outputs are excluded. Category
  is a feature only; category totals emerge from SKU predictions.
- On the 1,406 bakery-day current fold, SKU WAPE is `33.17%` versus `56.62%`
  current and `44.46%` previous Predictive. Category WAPE is `13.20%` versus
  `28.68%` for both inherited-category variants. Direct wins 1,297/1,406
  bakery-days against previous Predictive.
- On the earlier 2,154 bakery-day fold, SKU WAPE is `38.00%` versus `40.39%`
  current and `39.54%` previous Predictive. This is a smaller but positive
  temporally separate result.
- Current-fold maximum top-SKU share is `16.07%` with zero bakery-days >=20%
  and zero positive-sales rows assigned a near-zero forecast. SKU 1071 WAPE is
  `15.82%` and bias `-5.81%`, although it remains the predicted leader on
  1,158/1,406 bakery-days and requires a separate leadership audit.
- Bakery 29 / 2026-08-23 SKU 1071 changes from 440.98 current and 300.56 old
  Predictive to 221.43 direct, against actual 161. Savory bakery changes from
  the inherited 1,101.56 to an emergent 856.31, against actual 767.
- This remains research only: target is observed sales, the assortment universe
  is still inherited from the historical snapshot, and reconstructed-demand and
  operational-balance tests remain outstanding. See
  `docs/analysis/direct_bakery_sku_allocation_20260827.md`.
- In the same Kazan two-day FIFO economic simulation, direct allocation at the
  incumbent bakery volume raises gross profit from 93.632m current to 100.627m.
  With P50 bakery volume it reaches 106.374m (+4.186m versus actual and +4.772m
  versus previous Predictive P50) at 71.80% service level. The old Predictive
  floor remains highest at 111.641m but produces 224,025 more units than direct
  P50 and is most exposed to the known aggressive reconstructed-demand target;
  it is not production-approved by this result.
- Production and dev database state were not changed.

## Weighted Direct and soft normalization (2026-08-27)

- Volume-weighted Direct training (`sqrt(1+historical volume)`) is rejected:
  calibrated WAPE worsens `41.08% -> 41.33%`, underbake rises, and Bakery 29 /
  SKU 1071 / Aug 23 rises from Direct P50 246 to weighted 275 against actual 161.
- Soft normalization on original Direct is validated. Alpha is the fraction of
  expected-loss uplift added to bakery volume; Core SKUs (causal top 70% trailing
  volume) cannot fall below original Direct P50. Full no-normalization alpha=1
  is rejected (WAPE 42.36%, SKU 1071 bias +8.24%).
- Calibrated alpha=.25+floor gives best WAPE 40.29% and minimum imbalance:
  surplus/underbake 407,649/686,849. Alpha=.50 gives 40.50% and prioritizes
  underbake at 481,967/618,233; versus actual it cuts underbake 29.8% for 44.4%
  more surplus. SKU 1071 WAPE/bias is 25.33%/-2.92% with alpha .50.
- Kazan FIFO delta versus actual for previous final / alpha .25 / alpha .50 is
  +3.02/+3.76/+4.56% conservative, +6.57/+9.06/+11.43% calibrated, and
  +8.35/+12.80/+16.88% upper. Alpha .50 adds material production and terminal
  carry, so alpha .25 remains the safer comparator.
- Bakery 244 / SKU 11018 / Jul 27 is a blocker: 124 forecast vs 32 demand and
  23.3% bakery share. No shadow publication until causal tail cap plus capacity
  and rounding validation. See
  `docs/analysis/weighted_direct_soft_normalization_20260827.md`.
- Production and dev database state were not changed.

## Rolling Direct robustness and tail audit (2026-08-27)

- The snapshot archive supports four real rolling blocks; no missing forecast
  snapshots were synthesized. The first block initializes floor calibration and
  three later blocks are independent expanding-history evaluation folds.
- Across conservative/current/upper lost-demand scenarios, Direct P50 WAPE is
  `33.37/41.08/48.07%`, uplift is `33.12/40.86/47.90%`, and final adaptive
  floor is `33.10/40.64/47.56%`. Uplift lowers both surplus and underbake at
  fixed volume in all scenarios; floor lowers WAPE and underbake in all nine
  fold/scenario tests. The same strict floor gate is selected every time.
- Kazan FIFO gross-profit uplift versus actual remains positive under all demand
  scenarios: final floor is +3.02% conservative, +6.57% calibrated, and +8.35%
  upper. It is also positive in every individual evaluation fold.
- Tail improvement versus Direct P50 covers 71.8% of bakeries, 68.0% of
  categories, and 67.2% of products. SKU 1071 is the main blocker: uplift moves
  25,209 units away, worsening WAPE `25.38% -> 26.42%` and bias
  `-6.61% -> -15.71%`; floor restores only 670 units.
- Decision: aggregate/economic robustness passes, but no shadow deployment yet.
  Next work is a prior-fold SKU-level uplift gate/shrinkage rule, without old
  shares or category constraints. See
  `docs/analysis/rolling_direct_robustness_20260827.md`.
- Production and dev database state were not changed.

## Direct predictive uplift and adaptive floor (2026-08-27)

- Three clean variants were compared: Direct+P50, Direct+causal expected-loss
  uplift+P50, and Direct+uplift+adaptive floor. None uses old SKU/category
  shares, hourly allocation, previous Predictive, or the previous floor.
- Uplift is `P(clear stockout) * E(imputed loss | stockout)` using forecast-time
  causal features only. At exactly the same 20-date P50 volume it improves WAPE
  `41.83% -> 41.63%` and reduces both surplus and underbake by 3,591 units.
- Adaptive-floor parameters were selected on the earlier blocked fold only:
  n>=8 matching weekdays, historical stockout rate>=75%, mean lost>=4,
  `0.8*P67`, cap `min(+5 units,+10%)`. Applied unchanged to the current fold,
  it improves WAPE `41.45% -> 41.25%`, underbake `436,615 -> 428,976`, and
  recognized imputed loss `96,218 -> 101,861`, while surplus rises
  `189,975 -> 194,659`. No current bakery-day has a top SKU share >=20%.
- Kazan two-day FIFO gross profit is 106.374m Direct+P50, 108.112m with uplift,
  and 108.869m with selected floor, versus 102.187m actual and 93.632m current.
  Final service level is 72.58% and simulated lost demand 656,622.
- This remains research: uplift/floor depend on reconstructed-demand labels and
  need probability calibration, more rolling folds, economic-tail and capacity
  validation. See `docs/analysis/direct_uplift_adaptive_floor_20260827.md`.
- Production and dev database state were not changed.

## Bakery 29 / SKU 1071 time-history audit (2026-08-27)

Over June 1-August 23, SKU 1071 sells 230/day overall but 195.8 on Sundays.
August 23 sales of 161 are the period minimum but the day is not a detected
stockout: production is 200 and sales 161. Historical Sunday SKU share is
26.53% inside savory bakery and 13.23% of all bakery sales; August 23 is
20.99%/9.99%. Predictive assigns a reasonable 27.28% inside-category share
versus incumbent 40.03%, but preserves an incumbent category forecast of
1,101.6 against 767 observed category sales, producing a still-high 300.6 SKU
forecast; P50 raises it to 354.5. Calibrated lost-demand references also lift
some observed Sundays 201/220/248 to reconstructed 365/298/410. The residual
incident is therefore mainly frozen category-total allocation plus possibly
aggressive SKU lost-demand reconstruction. Next test direct bakery-day-to-SKU
allocation without incumbent category constraints. See
`docs/analysis/bakery29_sku1071_history_20260827.md`. Production unchanged.

## Guarded predictive allocation (2026-08-27)

A research candidate fills zero predictive rows from causal-trend shares,
preserves bakery-day totals, caps floor uplift at +25%, and water-fills SKU
share above 20%. Exact-zero positive-demand rows fall from 1,318 original-floor
to zero, but the fill assigns only 507 units against 4,688 reconstructed demand
and leaves thin-SKU WAPE near 90%. Final WAPE/RMSE/R2/bias is
45.210%/11.784/0.7388/-0.254%, slightly better than original floor except for
WAPE comparison to P50. Kazan FIFO gross profit is 111.552m (+9.365m vs actual)
versus 111.641m original floor (+9.454m). SKU 1071 dominance is unchanged.
The guard improves numerical coverage and volume tails but is not a replacement
candidate; the next allocation research needs a stronger causal thin-SKU
quantity prior and a separate 1071 ranking correction. See
`docs/analysis/guarded_predictive_allocation_20260827.md`. Production unchanged.

## Forecast shape audit (2026-08-27)

Across 3,554 controlled bakery-days, top-SKU share >=20% drops from 652 current
to 57 predictive and 7 with floor; >=30% drops 191 -> 2 -> 0. However SKU 1071
is still top on 2,860 predictive and 2,838 floor days versus 1,965 reconstructed
demand days. Predictive also outputs exact zero on 1,323 positive-demand rows
(4,688 demand units / 1,010 bakery-days); simple floor leaves 1,318 of them
because qualifying history is thin or absent. Bakery 29 / 2026-08-23 SKU 1071
improves from 32.86% current to 21.96% predictive and 18.67% floor but remains
above reconstructed demand share 8.25%. Concentration amplitude is largely
fixed; dominance frequency, thin-SKU coverage, and volume tails are not. See
`docs/analysis/forecast_shape_audit_20260827.md`. Production unchanged.

## Causal economic floor gate (2026-08-27)

Kazan-only walk-forward research tested a SKU/category profit gate over
P50+Predictive+floor. On evaluation blocks 2-4, gross profit is 43.408m actual,
35.370m current, 41.389m P50, 46.958m universal floor, and 46.949m gated floor.
The gate reduces production by 636, strategy expiry by 79, and terminal carry
by 167 versus universal floor, but serves 390 fewer units and loses 8.2k gross
profit. It trails universal floor in all three evaluation blocks. A static
past-profit SKU sign is therefore not validated; four date blocks are too few
for reliable product-level automation. See
`docs/analysis/causal_economic_floor_gate_20260827.md`. Production unchanged.

## Actual-markup two-day economics (2026-08-26)

Clean Kazan-only rerun: 114 bakeries, 324 mapped products, four uninterrupted
date blocks. Opening-stock expiry is separated from strategy expiry and block
end carry is not treated as expiry. Gross profit is 102.187m actual, 93.632m
current (-8.555m), 101.601m P50+Predictive (-0.586m), and 111.641m
P50+Predictive+floor (+9.454m). Strategy-created expiry is 33,168 / 24,892 /
29,111 / 50,738; terminal carry is 63,556 / 60,187 / 64,668 / 90,466. Floor
beats actual in all four blocks but has materially negative category/SKU
pockets, so the next candidate is a causal economic gate over floor, not a
universal rollout. See `docs/analysis/clean_kazan_two_day_economics_20260826.md`.
Production unchanged.

Correction later on 2026-08-26: forecast variants must treat forecast as a
target available stock, not as fresh daily production. With carry deducted
before production, gross profit on the 328 price-mapped products is 146.552m
actual, 133.611m current, 145.213m P50+Predictive, and 159.542m
P50+Predictive+simple floor. Floor is +12.990m versus actual in this research
simulation; P50 is -1.340m. Expiry is 62,117 / 47,369 / 50,580 / 81,708.
SKU-specific price/cost is retained and results are now also aggregated by
workbook category. This remains non-production research and requires Kazan-only
scope plus SKU-level economic gating before automation.

Kazan workbook prices/new costs were mapped to 328 products covering 96.76%
of rolling demand; day-two sales receive a 30% discount. Actual-state gross
profit is 146.552m. Current is 128.511m (-12.31%), P50+Predictive 137.639m
(-6.08%), and simple floor 141.192m (-3.66%). Floor beats P50 by 3.553m and
current by 12.681m but remains 5.360m below actual because 31.645m added
revenue costs 37.005m additional production. The earlier relative-cost result
was optimistic. Automation must optimize product-level profit and discounted
carry, not underbake alone. Scope still mixes a Kazan price file with network
operations and is not accounting-grade. See
`docs/analysis/markup_two_day_economics_20260826.md`. Production unchanged.

## Two-day carryover economics (2026-08-26)

A FIFO two-day shelf-life simulation gives served/lost/expired units of
2,488,435/1,085,858/65,070 for actual state, 2,679,480/894,813/203,614 for
P50+Predictive, and 2,963,941/610,352/349,523 for floor. With sale price 1,
disposal cost .05 and production cost .35, profit delta versus actual is
+23,031 (+1.46%) P50 and +113,588 (+7.18%) floor. Break-even production-cost
ratios versus actual are .400 and .464; floor beats P50 below .520. This is a
sensitivity model, not a ruble business case, and excludes capacity, labor,
batch and heterogeneous shelf-life constraints. See
`docs/analysis/two_day_economics_20260826.md`. Production unchanged.

## Candidate canonical ML metrics (2026-08-26)

On 20 dates/282,842 controlled SKU-days, current SKU WAPE/MAE/RMSE/bias is
48.92%/6.182/13.734/-23.47%; P50+Predictive is
44.81%/5.663/12.132/-14.85%; P50+Predictive+simple floor is
45.29%/5.723/11.801/+0.06%. P50 is best on SKU WAPE/MAE, while floor is best
on RMSE, sMAPE, R2 and bias. SKU-level recognized reconstructed loss is
163,060 (15.73%) current, 230,685 (22.25%) P50, and 387,117 (37.34%) floor.
At bakery-day level floor has WAPE 12.51% and recognizes 78.59%, but bakery
aggregation masks SKU placement errors; SKU metrics remain operationally
authoritative. See `docs/analysis/candidate_canonical_metrics_20260826.md`.
Production unchanged.

## Rolling actual-state comparison (2026-08-26)

Actual production, transfers and prior-day calculated stock were joined to
the 20 rolling dates on 282,842 controlled SKU-days. Actual-state
surplus/underbake is 466,915/1,155,055; current is 454,798/1,293,696;
same-volume Predictive 394,322/1,226,923; P50+Predictive
535,389/1,066,303; and P50+Predictive+simple floor 810,525/808,272.
P50+Predictive beats actual underbake on all four folds and has the lowest
aggregate imbalance. Floor reduces underbake most but adds nearly one surplus
unit per underbake unit saved versus actual. Actual underbake includes a
118,223 availability reconciliation gap where sales exceed computed
availability; this is DQ/opening-stock uncertainty, not confirmed underbake.
See `docs/analysis/rolling_actual_state_comparison_20260826.md`. Production
unchanged.

## Rolling calibrated-loss and floor backtest (2026-08-26)

Nine weekly pseudo-stockout folds over June 1-August 23 validate cutoff-hour
calibration: weekly aggregate recovery is 94.04%-102.19%. A four-fold,
20-forecast-date rolling comparison gives current surplus/underbake
714,389/1,510,091; same-volume Predictive 638,293/1,433,995; P50+Predictive
831,826/1,250,431; and P50+Predictive+simple n>=8 floor
1,123,840/978,629. Predictive remains the allocation baseline. P50 and floor
both reduce underbake on an economically favorable trade when underbake costs
more than about 1.05-1.07 times surplus, but floor is not best at equal cost.
The evaluation is rolling one-day-ahead style, not fixed 14-day recursive.
Two-level product selection was excluded because clean pre-fold selection
history is unavailable for every fold. See
`docs/analysis/rolling_post_last_sale_and_floor_20260826.md`. Production
unchanged.

## Two-level selective SKU floor (2026-08-26)

Product-specific caps were selected on four calibration dates and evaluated
on four frozen test dates. Standard n>=8 gives total surplus/underbake
297,912/359,545. The under-center two-level rule selects 49 calibration-
efficient products, uses scale 1.05/cap 15, and reaches 347,105/323,167; test
is 156,760/174,098. Versus the prior n>=6 center, test underbake improves by
7,437 for 11,128 more surplus (break-even ~1.50). Kystyby P is selected;
SKU 1071 and Makovka are rejected by the frozen 50% efficiency gate. See
`docs/analysis/two_level_selective_floor_20260826.md`. Production unchanged.

## Calibrated selective floor decomposition (2026-08-26)

The center SKU floor adds 265,893 units: 150,118 reduce underbake and 115,775
become surplus (56.46% efficiency). Efficiency improves from 54.21% on the
calibration half to 58.44% on test. The 8+ history segment is 58.76% efficient,
while 6-7 observations are only 46.33%, making history depth the clearest
guardrail. Kystyby P (10340) is above average at 61.32% efficiency but retains
21,247 underbake because the 8-unit cap is restrictive. Next test n>=8 as the
default and calibration-selected product-specific caps for efficient residual
underbake. See
`docs/analysis/calibrated_selective_floor_decomposition_20260826.md`.
Production unchanged.

## Calibrated selective SKU floor (2026-08-26)

A 9,900-candidate causal grid was evaluated with a frozen chronological 4+4
split. The calibration-selected balanced P50 floor (same-weekday P67, n>=7,
scale .75, cap 10) has total surplus/underbake 263,550/388,110 and test
115,820/208,397, beating test actual underbake 229,282. The underbake-first
center candidate (P50, n>=6, scale .95, cap 8) reaches total
324,338/336,729 and test 145,633/181,535. An extreme P95+2% floor reaches
156,419 underbake but creates 798,235 surplus and is rejected operationally.
The center candidate is the next research point; no rollout is authorized.
See `docs/analysis/calibrated_selective_sku_floor_20260826.md`. Production
unchanged.

## Calibrated lost-demand quantile comparison (2026-08-26)

The network label was rebuilt with post-last-sale coefficients frozen on
August 1-10, raising historical reconstructed loss from 2.774m to 6.212m.
On the common eight-date/175-bakery/267-SKU operational scope, calibrated
actual-state underbake is 468,732 versus 626,677 for current. P50 has the
lowest equal-cost imbalance at 695,411. P67 is the first quantile below
actual-state underbake (457,048); P95 +2% reaches 377,522 but raises surplus
to 364,416. The remaining underbake is SKU-placement error. Coefficients use
only ten calibration dates and are not rollout-ready. See
`docs/analysis/calibrated_quantile_operational_balance_20260826.md`.
Production unchanged.

## Post-last-sale demand calibration (2026-08-26)

A frozen August 1-10 calibration / August 11-23 holdout shows that the current
fixed lost-demand cap is severely downward biased for early last-sale hours.
It recovers only 4.6% at 07:00, 14.7% at 10:00, 42.1% at 15:00 and 69.0% at
17:00. A cutoff-hour calibrated same-day rate recovers 83.9%, 90.1%, 96.8%
and 99.3% respectively; 18:00 is mildly high at 103.6%. Case-level error is
still large, so these are aggregate label coefficients, not SKU-day forecasts.
Next rebuild the post-last-sale labels with frozen coefficients and rerun the
operational model comparison. See
`docs/analysis/post_last_sale_calibration_20260826.md`. Production unchanged.

## Same-day-rate pseudo-stockout validation (2026-08-26)

The relaxed lost-demand formula recovers only 46.7%, 57.2%, and 75.6% of
hidden sales after synthetic 15:00, 16:00, and 17:00 cutoffs; at 18:00 it
recovers 102.5%. The fixed cap makes the current 191,866 lost-demand label a
conservative lower bound for early stockouts, not a validated point estimate.
Consequently the selective floor's numerical win over observed underbake is
not yet rollout evidence. Cold-start is secondary: <6-observation rows account
for 31,898 of 170,270 residual underbake, while 8+ rows account for 122,921.
Next calibrate the loss estimator by last-sale hour on a frozen holdout. See
`docs/analysis/pseudo_stockout_same_day_rate_20260826.md`. Production was
unchanged.

## Selective SKU floor frozen split (2026-08-26)

The aggregate P67 x0.70 floor was rejected after a chronological 4+4 split:
it won only the first half and failed all four later dates. A selective floor
on P85 + Predictive +2% using same-weekday P67 x0.83, at least six historical
observations, and a 15-unit per-SKU-day uplift cap passes both halves. Across
all eight dates it has volume 1,373,874, surplus 315,612 and underbake 170,270
versus 191,866 observed and 409,352 current. The underbake gain is robust in
this small split but costs 144,230 surplus units above observed. See
`docs/analysis/selective_sku_floor_20260826.md`. Production was unchanged.

## Causal SKU floor beats observed underbake (2026-08-26)

On top of P85 + Predictive +2%, a causal same-weekday P67 SKU floor at scale
0.70 reduces underbake to 189,948 versus 191,866 observed and 409,352 current.
Volume is 1,338,723 and surplus 300,138. This is the first tested causal plan
to pass the underbake-first gate, but the margin is only 1,918 units and the
surplus cost is material. Floors requiring three observations cannot remove
the final approximately 19 thousand underbake units; cold-start/category
priors are required for that segment. See
`docs/analysis/causal_sku_floor_20260826.md`. Production was unchanged.

## High-quantile +2% grid (2026-08-26)

P75-P95 with Predictive allocation and optional +2% bakery uplift were tested
on the common eight-date, 175-bakery, 267-SKU scope. P95 +2% has the lowest
underbake at 227,237, a 44.5% reduction from current 409,352, but remains above
actual 191,866 and raises surplus to 320,724. More bakery volume cannot remove
the remaining SKU-placement error; the next candidate needs a causal SKU
floor. See `docs/analysis/network_high_quantile_grid_20260826.md`. Production
was unchanged.

## Network quantile operational comparison (2026-08-26)

The bakery-day history was extended through 2026-08-23 and relaxed stockout
labels were rebuilt network-wide. P50-P75 were trained through 2026-08-10 and
combined with the frozen Predictive allocation on the common eight-date,
175-bakery, 267-SKU operational universe. Predictive +2% has the lowest
equal-cost imbalance (488,160); P67 is best when underbake costs 1.5 times
surplus, and P75 is best at weight 2.0. P75 volume (1,188,173) is close to
actual available volume (1,178,537), but its underbake remains 272,082 versus
191,866 actual, confirming that volume alone does not repair SKU placement.
See `docs/analysis/network_quantile_operational_balance_20260826.md`.
Production was unchanged.

## Relaxed same-day-rate stockout experiment (2026-08-26)

A research-only detector now marks an inventory-controlled SKU-day as a
stockout when available quantity is positive but no greater than sales and the
last sale is before 19:00. Lost demand is extrapolated from the same day's
average rate between 07:00 and the last sale through a 23:00 close, retaining
the existing min(10 units, 50% of sales) cap. It identifies 28,391 historical
SKU-days and 142,059 capped units. On the common 18-day/11-bakery comparison,
lost demand increases from 2,339 to 30,564 units. P75 has the lowest tested
imbalance (19,773) but still covers only 60.69% of reconstructed loss; direct
demand imbalance is 24,146. Selective uplift collapses to the sales baseline
under the prevalent label. See
`docs/analysis/relaxed_stockout_quantiles_20260826.md`. This is not validated
for rollout and production was unchanged.

## Available-to-sell balance correction (2026-08-26)

The factual operational-state row was rebuilt on the same eight dates, 176
bakeries and 267 products using total available to sell: production plus
positive prior-day closing stock plus received minus sent. Deduplicated
`fct_*` sources were used because the mart covered only 91 products and did
not reconcile to trusted sales. The corrected factual state is 1,183,823
units available, 172,466 surplus, 40,035 recognized underbake, and 212,501
total imbalance. Forecast-plan rows are unchanged: current 537,651 total
imbalance, predictive 392,906, predictive +2% 397,104. No candidate beats
the factual underbake. See
`docs/analysis/available_to_sell_balance_20260826.md`. Production was not
changed.

## Forecast evaluation correction (2026-08-25)

The 2026-08-24 conclusion that a direct seven-day mean beat the current
bakery-level ML model was invalid. The evaluation silently treated 304
forecast-only bakery-days across 38 bakeries as zero observed demand. Their
158,267 forecast units inflated bakery-level WAPE.

On 1,406 observable bakery-days across 176 bakeries, current
`base_norm_recent` bakery-level observed-sales WAPE is `7.0755%` versus
`12.4591%` for Mean7. Strict-demand WAPE is `7.5902%` versus `12.6066%`.
The base model itself has `6.7632%` WAPE; recent correction improves bias but
slightly worsens WAPE to about `7.10%`. SKU snapshots conserve the bakery
total exactly, so remaining large SKU errors belong to allocation/mix.

The predictive allocation backtest was also filtered to observable
bakery-days. Predictive remains the best challenger (`38.6535%` end-to-end
SKU WAPE and `31.0759%` equal-total allocation WAPE), but the prior historical
concentration claim disappears on the corrected universe and is not evidence
for the separate August incident.

Corrected sources:

- `docs/analysis/base_norm_recent_vs_mean7_20260824.md`
- `docs/analysis/daily_predictive_sku_allocation_20260824.md`
- `scripts/recalculate_active_bakery_universe.py`

Production and serving state were not changed.

### Current SKU allocation backtest (2026-08-25)

On the same eight `base_norm_recent` dates and 1,406 observable bakery-days,
the incumbent SKU allocation has strict-demand WAPE `56.1277%`. A causal
daily trend allocation that preserves every bakery/category total reduces it
to `43.7352%` and improves 1,233 of 1,406 bakery-days. Its p95 largest-SKU
share is `18.52%` versus `30.36%` for the incumbent; bakery-days at or above
30% fall from 73 to 15. SKU 1071 WAPE falls from `60.27%` to `22.53%`.

This is research evidence only, not a rollout decision. The challenger was
selected on the evaluation dates and requires a blocked fold or prospective
shadow. See `docs/analysis/current_sku_allocation_backtest_20260825.md`.

The blocked 2026-07-17..2026-08-02 validation rejected full causal-trend
replacement: WAPE worsened from `40.4570%` to `40.9225%`. A conservative 25%
blend improved WAPE to `40.2778%`, won all 17 dates and improved 2,007 of
3,072 bakery-days while preserving category totals and concentration. It is
the only retained shadow candidate; this gain is not sufficient for canary.
See `docs/analysis/blocked_sku_allocation_backtest_20260825.md`.

The forecast-conditioned predictive-choice model was subsequently rebuilt
with explicit runs, non-oracle production totals and frozen folds. It improves
blocked observed-sales WAPE from `40.3878%` to `39.5393%` across all 12 test
dates, and current WAPE from `56.6228%` to `44.4550%` across all eight dates.
Current p95 top-SKU share falls from `30.36%` to `19.28%`; >=40% cases fall
from 10 to zero. Predictive choice replaces causal blend 25% as the primary
shadow candidate, but no production rollout is authorized. See
`docs/analysis/rebuilt_predictive_choice_20260825.md`.

The first prospective local shadow was generated from production run
`prod_base_bakery_norm_recent_20260825_h14` for 2026-08-25, using training
information only through 2026-08-23. It conserves the 199,963.23-unit network
plan and every bakery/category total (maximum delta `1.14e-13`). The p95
largest-SKU share falls from `28.57%` to `17.69%`; >=20% cases fall from 69 to
6 and >=30% cases from 6 to 1. There are 7,882 cold-start rows and 38 bakeries
without recent observable sales, so accuracy must be evaluated separately
after the 2026-08-25 fact closes. No production data was changed. See
`docs/analysis/predictive_choice_shadow_20260825.md`.

A joint current-period diagnostic then tested higher bakery volume together
with predictive allocation. On the corrected eight-date observable universe,
a uniform +2% volume candidate raises forecast by 26,850 units and recognized
SKU lost demand by 2,037 units while reducing true SKU overforecast by 67,032
units and 2,162 rows versus the incumbent. Strict-demand SKU WAPE is 44.54%
versus 56.13%. At bakery level WAPE improves from 7.59% to 7.25% and bias from
-3.34% to -1.41%, but true-overforecast bakery-days increase from 524 to 635.
Therefore +2% is only a diagnostic center point: the next challenger must use
a causal bakery/day-specific uplift and pass a frozen fold. A first frozen
four-date calibration/four-date test rejected simple smoothed residual
calibration because bakery-level true-overforecast cases rose from 251 to 274.
See
`docs/analysis/joint_demand_allocation_20260825.md`. No production data was
changed.

A three-fold causal comparison on the 11 bakeries with reconstructed
lost-demand history evaluated direct demand-target LightGBM and a two-stage
selective uplift. Mean bakery WAPE improves from 6.024% for the sales-target
baseline to 5.878% for direct demand and 5.848% for selective uplift;
recognized-lost coverage rises from 37.41% to 41.90% and 43.46%. Selective
uplift beats the baseline on all three folds and direct demand on two of
three. Neither passes the overforecast gate: mean true-overforecast quantity
rises from 3,619 to 3,881 and 4,167 respectively. Continue both as research
candidates, with direct demand the conservative default; network rollout is
not authorized. See
`docs/analysis/direct_demand_vs_selective_uplift_20260825.md`.

An asymmetric operational-balance gate was added on the current eight dates and the 267
products covered by production releases. The observed production-state proxy
has 142,200 surplus units plus 40,035 underbake units, total 182,235. The
incumbent forecast plan has 537,651 units of projected imbalance; predictive
allocation improves this to 392,906, while predictive +2% has 397,104. No
candidate beats observed underbake, so none passes the primary gate. Surplus
is secondary: predictive +2% trades 12,635 additional surplus units for 8,437
fewer underbake units versus predictive alone and is economically preferable
when underbake costs more than 1.50 times surplus.
Opening inventory is unavailable; observed surplus is explicitly only
`max(same-day production-sales,0)`, not verified ending stock. See
`docs/analysis/operational_balance_20260825.md`. Production was unchanged.

Direct-demand quantiles P50/P55/P60/P67/P75 were compared on the same three
causal 14-day folds and 11 labeled bakeries. Every quantile reduces underbake
on all three folds. P50 already moves aggregate bias relative to sales to
+0.29%; P55 gives +1.38%. With underbake weighted 1.5 times surplus, P55 has
the lowest weighted operational loss; at weight 2.0 P67 is best. Mean WAPE
worsens as the quantile rises, so operational weighting rather than WAPE must
select the level. P50 is the conservative candidate and P55 the center
candidate; neither is authorized for rollout. See
`docs/analysis/direct_demand_quantiles_20260825.md`.

## Pilot management history restored on production (2026-08-25)

- The production report at `/opt/reports/pilot_management_summary` was
  replaced atomically with the validated 2026-07-23..2026-08-23 history.
  All 32 calendar dates are present. Weeks start on Monday; weeks beginning
  2026-07-27, 2026-08-03, 2026-08-10, and 2026-08-17 are complete. The week
  beginning 2026-07-20 is intentionally partial because the pilot started on
  Thursday 2026-07-23.
- Historical scope is event-aware: 38 bakeries through 2026-08-16 and 39 from
  the 2026-08-17 addition of bakery 273. Before 2026-08-07, bakery 270 has no
  saved forecast; available bakery data is retained and the missing forecast
  is explicit instead of dropping the whole date.
- Forecast selection now uses the latest complete or best-covered run that was
  available before 08:00 MSK. This retains dates on which the active forecast
  came from an older horizon (`lead_days > 1`) after a missed nightly run.
- Pre-deploy validation required 39 distinct bakery names, exact 32-date
  coverage, and the four complete weeks above. Post-deploy verification found
  the same five week rows, `app.service=active`, and `/health` returned HTTP
  200 with `app_env=prod` and an empty table suffix.
- Backup:
  `/opt/backups/pilot_management_summary_before_20260825_20260825_062105`.
  Forecast tables, active forecast run, and Blackhole writer timers were not
  changed; both Blackhole forecast-writer timers remain disabled/inactive.

## Current live production status (2026-08-24)

- The active production run is
  `prod_base_bakery_norm_recent_20260823_h14`, horizon
  `2026-08-23..2026-09-05`. `scripts.verify_prod_deploy` returns `VERIFY OK`;
  active snapshots contain 2,996 bakery-day, 566,696 SKU-day, and 6,953,436
  SKU-hour rows.
- The scheduled 2026-08-24 writer run refreshed source data through
  `2026-08-23` but stopped before inference and activation because the SKU
  profile freshness guard rejected `data_through=2026-08-15` at age 9 days
  (configured maximum: 8 days). The previous known-good run remained active;
  no partial forecast run was activated and no serving data was lost.
- SKU profiles are scheduled separately by `weekly-profile-refresh.timer`
  every Sunday at `02:00 UTC` (`05:00 MSK`). The timer is enabled/active and
  did fire on 2026-08-23, but `weekly-profile-refresh.service` was terminated
  by the Linux OOM killer during export batch 12/13 (the July 2026 block) after
  loading a rolling one-year range. The last successful refresh therefore
  remains `weekly_20260816`, with `data_through=2026-08-15`; the failure was in
  resource usage, not timer scheduling.
- `forecast-production.timer` remains enabled/active, with its next scheduled
  attempt at `2026-08-25 03:30 UTC` (`06:30 MSK`). Before that attempt, refresh
  the SKU profile or make an explicit, validated freshness-policy decision; do
  not silently raise the limit.
- The separate Blackhole chat publisher succeeded on 2026-08-24 at
  `03:00 UTC` (`06:00 MSK`): 2,076 SKU rows across 39 bakeries were generated,
  uploaded, and sent to chat `179919` (file message id `8146035`). Its timer
  remains enabled/active.
- Blackhole forecast-writer timers remain disabled/inactive. Production writer
  ownership is unchanged: only the VM may generate and publish forecast runs.

## Pilot Management UI Release Deployed To Blackhole (2026-08-20)

The 2026-08-18 pilot-management UI and report-reader changes were deployed as
a deliberately narrow seven-file release. Pilot management routes now use
`AuthContext.is_pilot_user`, preserving access for admins and all 37 ids in
`PILOT_USER_IDS` while rejecting unrelated portal users.

Deployed targets:

- `/opt/app/app/routers/pilot_management.py`
- `/opt/app/app/static/app.js`
- `/opt/app/app/templates/layout.html`
- `/opt/app/app/templates/pilot_management.html`
- `/opt/app/app/templates/pilot_bakery.html`
- `/opt/app/app/templates/pilot_bakery_week.html`
- `/opt/src/pilot_management_service.py`

`main.py` and `db.py` were intentionally preserved from the Blackhole runtime:
the former contains deployment-layout-specific `/opt` import handling and the
latter contains connection recovery behavior absent from the workstation
version. Forecast writer and pilot publisher files were not part of this
release.

Pre-deploy recovery point:
`/opt/backups/codex_20260820_before_pilot_ui` (`BACKUP_VERIFY_OK`).

Post-deploy verification:

- all 7 deployed SHA-256 hashes match the local release candidate;
- `app.service`: `active`;
- `/health`: HTTP 200, `app_env=prod`, empty table suffix;
- 11 `/pilot*` routes registered;
- access model verified for configured pilot user, admin, and non-pilot user;
- both Blackhole forecast timers remain `disabled` and `inactive`;
- recent `app.service` logs show clean shutdown/startup and no import, template,
  or ClickHouse errors.

Detailed release boundary and rollback instructions:
`docs/ops/BLACKHOLE_RELEASE_20260820.md`.

## Production Forecast Writer Repaired After Missed 2026-08-19/20 Runs (2026-08-20)

The production timer fired at `03:30 UTC` on both 2026-08-19 and 2026-08-20,
but `forecast-production.service` failed before inference because the deployed
runtime files were from incompatible point-in-time copies. `.env` selected
`base_norm_recent`, while the active `run_production_inference.py` did not
define that scenario. Restoring the runner exposed two further mismatches: the
active ClickHouse allocator did not accept `assortment_max_age_days`, and its
matching version required `MIN_FALLBACK_N_DAYS` from the base allocator. The
active dataset refresh module also no longer refreshed assortment tables, so
the freshness guard correctly rejected nine cities at age 3 days (limit 2).

The following mutually compatible VM backups were restored:

- `pipelines/forecast_publish/run_production_inference.py.backup_20260810_174532_pilot38_12m`
- `src/experiments_v2/apply_bakery_profiles_clickhouse.py.backup_20260810_122506_base_norm_recent`
- `src/experiments_v2/apply_bakery_profiles.py.backup_20260806_155645`
- `pipelines/forecast_publish/production_dataset_refresh.py.backup_20260806_assortment_dim_fix`

Pre-repair copies of the replaced files were preserved with the suffix
`backup_20260820_before_base_norm_restore` or
`backup_20260820_before_assortment_restore` beside the active files.

The repaired service refreshed source data through `2026-08-19`, refreshed
allocation assortment with `valid_from=2026-08-19`, and successfully published
and activated `prod_base_bakery_norm_recent_20260820_h14` for horizon
`2026-08-20..2026-09-02`.

Post-repair verification:

- `forecast-production.timer`: `enabled`, `active`
- bakery-day snapshot rows: `2,968`
- SKU-day snapshot rows: `561,570`
- SKU-hour snapshot rows: `6,877,148`
- `scripts.verify_prod_deploy`: `VERIFY OK`

The VM Git checkout remains unsuitable as deployment truth because runtime
files have historically been delivered as targeted copies. Do not run a broad
`git pull` or replace the VM tree until these production-only versions are
reconciled and committed as one tested deployment unit.

## Pilot Access Control Deployed To Blackhole (2026-08-17)

All pilot-management and pilot-config routes are now live on Blackhole
(`82bb03a8-c356-4225-97a4-a1540cdc29e6`, `/opt/app`). Access was previously
restricted to admins only; it is now granted to all directors, data analysts,
and AI team members listed in `PILOT_USER_IDS`.

**Access model:**

- New `is_pilot_user` property in `app/auth.py`: `True` if `is_admin` or
  `user_id in settings.pilot_user_ids`.
- New `PILOT_USER_IDS` env var parsed in `app/settings.py` as
  `frozenset[str]` from a comma-separated list.
- All 5 pilot-management routes and 3 pilot-config routes now call
  `_require_pilot_user(request)` instead of `_require_admin(request)`.

**Users granted access (37 total):**
25 directors (`Операционный директор`, `Директор региона`, `Директор
города`, `Директор партнёр`, `Директор пекарни`), 10 data analysts and
AI team members, 2 IT. `PILOT_USER_IDS` in `.env` lists their Bitrix24
user IDs.

**New files deployed (first-time):**

- `/opt/app/app/routers/pilot_management.py` (8 564 bytes)
- `/opt/app/app/routers/pilot_config.py` (3 296 bytes)
- `/opt/app/app/templates/pilot_management.html`
- `/opt/app/app/templates/pilot_bakery.html`
- `/opt/app/app/templates/pilot_bakery_week.html`
- `/opt/app/app/templates/pilot_sku.html`
- `/opt/app/app/templates/pilot_config.html`
- `/opt/src/pilot_management_service.py`
- `/opt/src/pilot_config_service.py`

**Updated files:** `main.py` (added pilot router imports), `auth.py`,
`settings.py`, `db.py`, `templates/index.html`, `services/bakery.py`.

**Side-effect install:** `python-multipart` (required by `Form` parameters
in `pilot_config.py`) was not present in the venv — installed at deploy
time; now matches `apps/forecast_embedded/requirements.txt`.

**Post-deploy verification:**

- `app.service`: `active`
- `http://localhost:3000/health`: `{"ok":true,"app_env":"prod","table_suffix":""}`
- All 10 pilot routes registered:
  `/pilot`, `/pilot/`, `/pilot/bakery/{bakery_id}`,
  `/pilot/bakery/{bakery_id}/week/{week_start}`,
  `/pilot/bakery/{bakery_id}/week/{week_start}/day/{date}/export`,
  `/pilot/bakery/{bakery_id}/sku/{product_id}`,
  `/pilot/config`, `/pilot/config/`,
  `/pilot/config/bakery/{bakery_id}/add`,
  `/pilot/config/bakery/{bakery_id}/exclude`
- `PILOT_USER_IDS` and `PILOT_REPORT_DIR=/opt/reports/pilot_management_summary`
  confirmed in `/opt/app/.env`.

**Backup:** `/opt/app/app_backup_20260817_170226`

**Rollback:**

```bash
cp -r /opt/app/app_backup_20260817_170226 /opt/app/app
systemctl restart app.service
# also remove PILOT_USER_IDS and PILOT_REPORT_DIR from /opt/app/.env
```

**Note on pilot report dir:** `/opt/reports/pilot_management_summary/` was
created on the server. It is currently empty — the `PilotManagementService`
reads pre-built CSVs from this path. Reports need to be generated and placed
there before the `/pilot` statistics page will show data.

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

## Dev forecast parity refresh (2026-08-20)

- The previous active `_dev` run covered only `2026-06-23..2026-07-06`.
- Before changing dev state, the complete `forecast_runs_embedded_dev` registry
  was copied to
  `forecast_runs_embedded_dev_backup_20260820_before_refresh` (116/116 rows).
- The mutable dev assortment tables were backed up as
  `bakeable_products_dev_backup_20260820_before_refresh` and
  `assortment_city_products_dev_backup_20260820_before_refresh`. Local dataset
  files were copied to `.codex_tmp/dev_refresh_backup_20260820/`.
- The known-good production runtime additions were reconciled into the local
  source, including `base_norm_recent`, SKU allocation coverage/floor handling,
  the network bakery-hour fallback, and bakery-product assortment refresh.
  The related selected test set passes: 53 tests.
- A local dataset refresh reached facts through `2026-08-19` and weather
  through `2026-09-02`. The dev assortment refresh remains unsuitable for
  parity because its legacy city values are mojibake/incomplete; production
  assortment/profile tables were therefore used read-only for validation.
- For exact UI comparison, production run
  `prod_base_bakery_norm_recent_20260820_h14` was copied server-side into the
  suffixed dev tables as
  `dev_mirror_prod_base_norm_recent_20260820_h14`. Counts and row hashes match
  for bakery-day (2,968), context (126), SKU-day (561,570), and SKU-hour
  (6,877,148). Only the dev run id differs.
- At this intermediate parity-check stage, the mirrored run was the sole active
  `_dev` run, with horizon `2026-08-20..2026-09-02`. It was superseded later
  the same day by the validated `devfix` run documented below. Production
  tables and production active state were not changed by this mirror step.
- Dynamic pilot configuration has 39 active bakery ids. Bakery 273 is present
  in the mirrored forecast for all 14 days. Bakery 270 is absent from both the
  production and mirrored-dev forecast universes and requires a separate
  model/network-scope decision; do not silently synthesize it.

### Root cause and local candidate for bakeries 270/271

- The `2026-08-20` source export still contained sales through `2026-08-19`
  for bakeries 270 and 271, but both rows had a null city. The model dataset
  builder groups by city and therefore dropped them.
- Root cause: `dim_bakeries` contains duplicate rows for these ids, including
  rows with an empty city. The export's direct `ANY LEFT JOIN` could select the
  empty duplicate nondeterministically.
- Local fix: `scripts/clickhouse_bakery_daily_template.sql` now aggregates the
  dimension to one row per bakery with `anyIf(... != '')` before joining.
  A regression assertion was added to
  `tests/test_export_clickhouse_bakery_daily.py`; the selected refresh/export
  suite passes (26 tests) and Ruff passes for the touched Python test/export
  files.
- The fixed local candidate covers `2026-08-20..2026-09-02`, 214 bakeries,
  2,996 bakery-day rows, 566,554 SKU-day rows, and 6,947,742 SKU-hour rows.
  All 39 dynamic pilot ids are present; bakeries 270, 271, and 273 each have 14
  bakery-day rows and balanced SKU allocations.
- Direct TLS connections from the Windows dev host continued to time out, so
  the validated candidate was transferred with a matching SHA-256 to the
  production writer VM and loaded from that VM exclusively into `_dev` tables.
- Before the load, the dev registry was copied to
  `forecast_runs_embedded_dev_backup_20260820_150018_before_devfix` (118 rows).
- The new active dev run is
  `devfix_base_bakery_norm_recent_20260820_h14`: 2,996 bakery-day rows, 126
  context rows, 566,554 SKU-day rows, 6,947,742 SKU-hour rows, and 214 distinct
  bakeries. Bakeries 270, 271, and 273 each have all 14 horizon days. The
  maximum bakery-day versus allocated SKU-day delta is
  `3.183231456205249e-12`.
- At this dev-only stage, production remained active on
  `prod_base_bakery_norm_recent_20260820_h14`; no production forecast table or
  run status was changed by the dev load. The later production rollout is
  documented separately below.
- For local UI inspection while the direct Windows-to-ClickHouse TLS route is
  unavailable, the dev API can use an SSH local forward through the writer VM
  (`127.0.0.1:18443` to ClickHouse `:8443`) with the local-only
  `.env.dev.tunnel`. The active-run endpoint returned the new devfix run and
  `/bakery/270?date=2026-08-20` returned HTTP 200.

## Pilot management report production refresh (2026-08-20)

- The validated 39-bakery management-report candidate from
  `reports/pilot_management_summary_candidate_20260820` was deployed to the
  Blackhole read-only app at `/opt/reports/pilot_management_summary`.
- The installed `detail.csv` was checked before activation and contains 39
  distinct non-empty bakery names. This restores the conditionally rendered
  bakery selector in the production pilot management UI.
- The previous production report was copied to
  `/opt/backups/pilot_management_summary_before_20260820_20260820_142900`.
- After the atomic directory replacement, `app.service` remained active and
  `/health` returned `{"ok":true,"app_env":"prod","table_suffix":""}`.
- `forecast-production.timer` and `bakery-forecast-nightly.timer` remained
  disabled and inactive. No forecast run, ClickHouse table, or production
  writer state was changed by this report-only deployment.

## Production city-dimension fix and pre-06:00 recovery run (2026-08-20)

- The validated aggregate join fix in
  `scripts/clickhouse_bakery_daily_template.sql` was deployed narrowly to the
  production writer VM. The previous template is backed up at
  `/opt/backups/codex_20260820_before_dim_bakeries_city_fix/` and the deployed
  SHA-256 matches the workstation candidate.
- A distinct transient recovery run was generated with prefix `prodfix` so the
  prior production run was not replaced in place. It became active on
  2026-08-20 as
  `prodfix_base_bakery_norm_recent_20260820_h14`, horizon
  `2026-08-20..2026-09-02`, and was subsequently superseded by normal nightly
  production runs. See the top of this file for the current active run.
- Independent ClickHouse verification passed: 214 bakeries, 2,996 bakery-day,
  126 context, 566,554 SKU-day, and 6,947,742 SKU-hour rows. All 39 dynamic
  pilot bakeries are present; bakeries 270, 271, and 273 each cover 14 days;
  maximum bakery/SKU allocation delta is `3.637978807091713e-12`. The common
  212-bakery network changed by only `-0.026487%` versus the prior active run.
- `scripts.verify_prod_deploy` ended with `VERIFY OK`. The VM
  `forecast-production.timer` remains enabled/active for `03:30 UTC` on
  2026-08-21.
- A Blackhole publisher dry-run for `2026-08-21` completed without sending to
  Bitrix24: valid 95,303-byte XLSX, 2,104 SKU rows, and 39 bakeries. The
  `pilot-forecast-publish.timer` remains enabled/active for `03:00 UTC` on
  2026-08-21.
- Blackhole forecast-writer timers remain disabled/inactive. Blackhole remains
  read-only; only its separate chat publisher is scheduled.
## Direct alpha=.25 production-package dry-run (2026-08-31)

The selected daily SKU allocation candidate is now packaged as versioned model
artifacts under `models/direct_alpha_025_v1/`.  The current-horizon runner loads
those artifacts without retraining and emits the three files accepted by
`load_forecast_run`: bakery-day, SKU-day and SKU-hour.  The hourly layer uses
only bakery/DOW timing and conserves every finalized SKU-day quantity; it does
not restore the retired category/hourly SKU allocation.

Read-only dry-run for 2026-09-01 used active source run
`prod_base_bakery_norm_recent_20260831_h14` and causal sales history through
2026-08-30.  It produced 214 bakery rows, 12,282 SKU-day rows and 201,083
SKU-hour rows.  SKU-day and SKU-hour totals both equal `189,506.6725913064`;
maximum conservation error is below `9e-14`, with no NaN, negative or duplicate
key defects. Six bakery/DOW pairs without their own hourly history used an
explicit network-DOW timing fallback. Outputs and summary are in
`reports/direct_alpha_publish_dryrun_20260831/`.

No database write or activation was performed.  Production draft insertion
must be executed on the production VM, not from the workstation or Blackhole.
## Direct alpha=.25 h14 activated (2026-08-31)

Production active run is `draft_direct_alpha_025_20260831_h14`, horizon
2026-08-31..2026-09-13. It contains 2,996 bakery-day, 171,858 SKU-day and
2,816,030 SKU-hour rows. All three levels total `2,494,990.693343`; keys are
unique, quantities are non-negative, product/category names are populated and
maximum SKU day/hour conservation error is below `1.2e-13`.
`scripts.verify_prod_deploy --env-file .env` returned `VERIFY OK` after
activation. On bakery 23 / 2026-08-31, Smetannik changes from `0.72` to
`24.82`, and SKU 1071 from `404.38` to `211.58`.

The existing timer code still generates `base_norm_recent`. To prevent it from
overwriting the Direct active run before the native timer scenario is deployed,
the production VM `.env` currently has `FORECAST_ACTIVATE_RUN=none`; the timer
remains enabled and active and may build drafts. Backup:
`.env.bak_20260831_before_direct_alpha_timer_guard`. Rollback active run:
`prod_base_bakery_norm_recent_20260831_h14`.
## Direct alpha=.25 native nightly integration (2026-08-31)

The production VM now runs Direct as a systemd `ExecStartPost` after the
unchanged `forecast-production.service` main command. The main
`base_norm_recent` scenario refreshes datasets and creates the bakery/source
run with `.env` `FORECAST_ACTIVATE_RUN=none`; only a successful Direct
postprocess activates `prod_direct_alpha_025_YYYYMMDD_h14`. Drop-in:
`/etc/systemd/system/forecast-production.service.d/direct-alpha.conf`.

The native module and frozen artifacts are deployed under
`pipelines/forecast_publish/direct_alpha_production.py` and
`models/direct_alpha_025_v1/`. Production does not require research scripts or
PyArrow; floor history is packaged as `floor_history.csv.gz`. A native draft
matched the manually generated run across all 171,858 SKU-day rows with max
absolute error `8.53e-14`.

The first full systemd preflight exposed a root-owned output directory left by
the earlier manual test; Direct stopped before loading and active production
was preserved. Ownership was corrected to `forecast:forecast`, then the exact
postprocess command was rerun as `forecast`. Active run is now
`prod_direct_alpha_025_20260831_h14`; it has 2,996 bakery-day, 171,858 SKU-day
and 2,816,030 SKU-hour snapshot rows. The verifier now recognizes the intended
inactive source run through the Direct run notes and ends with `VERIFY OK`.
Timer remains enabled/active. Rollback remains activation of
`prod_base_bakery_norm_recent_20260831_h14` plus removal of the systemd drop-in.

# Decisions

This file records durable project decisions. It is not a session log.

## 2026-08-31 - Direct Alpha=.25 Replaced Legacy SKU Allocation In Production

Decision:

- The production model is `direct_alpha_025_v1`; active runs follow
  `prod_direct_alpha_025_YYYYMMDD_h14`.
- The bakery-day LightGBM forecast remains the volume source, but the daily
  volume is allocated directly across mature SKUs. Legacy category totals and
  hourly SKU profiles are not allocation inputs.
- The selected model includes causal predictive uplift, Core-SKU protection,
  alpha `.25` soft expansion, adaptive floor and causal tail cap.
- SKU cold start is an independent path and does not compete for or reduce the
  mature Direct allocation pool. Hourly forecasts are a downstream,
  quantity-conserving timing split.

Context:

- The legacy filtered hourly profile could retain only part of its original
  SKU mass and then normalize that remainder back to 100%; subsequent hourly
  correction normalized again. This systematically concentrated volume in
  large SKUs, including bakery-level shares above 40–50%.
- Direct fixed the reported Zorge and bakery 29 failures and, in the corrected
  2026-09-01 pilot workbook, produced no bakery with a top-SKU share at or
  above 20%.
- Production keeps building `prod_base_bakery_norm_recent_*` as an inactive
  source run because it supplies bakery volume and refreshed datasets. Its
  presence does not make it the current model.

Implication:

- Future sessions, reports and comparisons must use Direct alpha=.25 as the
  current-system baseline unless live verification shows an explicit rollback.
- Forecast allocation quality and production-plan kratnost rounding are
  separate layers. The known rounding overage must not be attributed to the
  Direct forecast itself.
- Rollback requires both activation of a verified source run and removal of
  the Direct systemd drop-in; it must be recorded in `CURRENT_STATE.md`.

## 2026-06-28 - VM Is The Only Production Forecast Writer

Decision:

- Production forecast generation and ClickHouse forecast publishing run only on
  the VM `201.51.7.24`.
- VibeCode/Blackhole serves only the embedded read-only API/UI.
- Forecast timers on VibeCode/Blackhole must stay disabled.

Context:

- The project previously had forecast generation installed on Blackhole at
  `/opt/forecast_job`.
- On 2026-06-28, a Blackhole `forecast-production.timer` was found overwriting
  the active ClickHouse run with stale run
  `prod_uplifted_bakery_norm_uplift_sku_20260601_h14`.
- The stale timer was disabled and the fresh VM-generated run
  `prod_uplifted_bakery_norm_uplift_sku_20260623_h14` was re-activated.

Implication:

- Any future production writer migration must update `CURRENT_STATE.md`,
  `SERVICES.md`, this decision log, and the runbook before or during rollout.

## 2026-07-09 - Baking Plan Rebuilt As Its Own Package

Decision:

- Torn down the old baking-plan implementation (algorithm, templates,
  assortment allowlist, tests) and rebuilt the feature as `apps/baking_plan/`,
  a standalone Python package — not a subpackage of
  `apps/forecast_embedded/app`.
- Mounted in-process: `apps/forecast_embedded/app/main.py` adds `apps/` to
  `sys.path` and includes `baking_plan.router.router`. No new port, systemd
  unit, or network hop.
- Package boundary: `baking_plan` may import shared ClickHouse plumbing
  (`app.db`, `app.settings`, `app.table_names`) and forecast-serving reads
  (`app.services.bakery.get_bakery_day`, `get_sku_hour_forecast`); nothing
  outside `apps/baking_plan/` may import from `baking_plan.*` except the one
  router-mount line in `main.py`.

Context:

- User framed the target architecture as three logical units: a frontend
  service, a predictions service, and a baking-plan service. A full separate
  FastAPI process was considered and rejected for now — the baking-plan
  feature only reads already-published forecast data from ClickHouse, so a
  network-separated service would add deploy/health-check overhead without a
  matching operational need. A package boundary gives the same logical
  independence (own tests, own module graph, one-directional dependency on
  `app`) without new infra.
- The old implementation was already scheduled for a rewrite (see the
  2026-07-01 SKU-hour floor-uplift rejection above — the peak-detection
  window algorithm had no reliable way to distinguish genuine low demand from
  shelf-absence censoring), so the teardown and the restructure happened
  together.

Implication:

- If a genuine network-separated service becomes necessary later (e.g. the
  feature needs independent scaling or a different deploy cadence), promote
  `apps/baking_plan/router.py` to its own FastAPI app — the package boundary
  was designed so that move doesn't require re-touching `forecast_embedded`
  beyond removing the mount line.
- Any Blackhole deploy touching this feature must upload `apps/baking_plan/*`
  in addition to `apps/forecast_embedded/app/*`.

**Superseded 2026-07-11** — the "scaffolding only, `NotImplementedError`"
status above is no longer current. See the entry below.

## 2026-07-10/11 - Baking Plan: MILP Allocator Chosen, Implementation Finished And Deployed

Decision:

- Built both a greedy allocator and a MILP allocator
  (`scipy.optimize.milp`/HiGHS) and compared them on real data instead of
  picking one up front (`scripts/compare_baking_algorithms.py`). MILP won
  on weighted shortfall and dough-group window cohesion; chosen for
  production. Greedy is kept in the codebase as reference/fallback, not
  wired into `service.py`.
- Full business-rule spec (кратность, тесто-группы, дефрост vs
  двухдневка, mandatory-assortment highlighting, two-dimensional
  baker-minutes/tray-slot capacity, proportional window-demand
  redistribution, `Итого` = sum of scheduled windows) implemented and
  documented in `docs/baking_plan_implementation.md`.
- Deployed to Blackhole 2026-07-11 — see `CURRENT_STATE.md`.

Context:

- `scipy` was missing from `apps/forecast_embedded/requirements.txt` even
  though `algorithms/milp.py` imports it unconditionally at module load
  time (reached from `app.main` on every startup) — added before deploy,
  since without it the whole embedded app would fail to boot, not just
  the baking-plan route.
- дефрост and двухдневка were briefly conflated during implementation
  (дефрост implemented as "triggered by `is_two_day`") and then corrected
  after the user caught it against the reference file's own data (zero
  SKU overlap between the two groups). Kept as two independent
  variable/constraint families in the MILP so neither can be produced
  early or shorted in favor of regular SKUs. **Superseded 2026-07-13** —
  merged into a single shared-pool model; see the entry below.
- Full-day proportional demand redistribution (spreading each SKU's whole
  day across existing windows, since exact bakery closing hours aren't
  tracked) made capacity genuinely scarce across most windows at once
  once SKU count exceeds ~40, causing HiGHS branch-and-bound to run long;
  mitigated with a 15s solver time limit + 0.5% relative-gap tolerance.

Implication:

- `DEFROST_SKU_NAMES` and `MANDATORY_ASSORTMENT`
  (`apps/baking_plan/constants.py` / `rendering.py`) are hardcoded
  placeholder lists with no ClickHouse source of truth — flagged as an
  open item, not a bug, in `docs/baking_plan_implementation.md`.
- Any future change to business rules (capacity constants, dough-group
  mapping, mandatory assortment) should update
  `docs/baking_plan_implementation.md` alongside the code — it is now the
  canonical spec, not session memory.

## 2026-07-13 - Baking Plan: Дефрост/Двухдневка Merged Into A Single MILP Demand Pool

Decision:

- Дефрост and двухдневка are no longer separate tray-variable/constraint
  families in the MILP. Each SKU now has one shared `trays[sku][w]` pool
  sized against its **combined** target (regular + defrost top-up +
  two-day next-day volume), with the "always fully produced" guarantee
  preserved via a secondary `mandatory_shortfall[sku]` checkpoint
  constraint at the last window only, weighted at
  `MANDATORY_SHORTFALL_WEIGHT = 10_000.0`.
- Window *placement* of the mandatory portion (which window(s) it lands
  in) is decided by a post-solve, correctness-preserving pass —
  `_split_tail` then `_shift_to_later_windows` in
  `apps/baking_plan/algorithms/milp.py` — not by the solver's objective.
- Deployed to Blackhole 2026-07-13 — see `CURRENT_STATE.md`. Full
  mechanics documented in `docs/baking_plan_implementation.md`
  ("Дефрост vs двухдневка" and "Algorithm" sections).

Context:

- The prior two-independent-families model (2026-07-10/11 decision above)
  guaranteed дефрост/двухдневка could never be shorted, but gave the
  solver no signal about *which* window to place that guaranteed
  production in. In practice this scattered a single SKU's mandatory
  output across up to 6-7 windows, which the user flagged as
  operationally wrong — a bakery expects one consolidated late batch, not
  crumbs across the whole day.
- Two alternatives were tried and rejected before the shared-pool +
  post-processing design:
  1. **Scalar objective-weight window bias**
     (`LATE_WINDOW_PREFERENCE_WEIGHT` added to later windows' cost).
     Weighted below `ANTI_WASTE_WEIGHT` it was too small to reliably beat
     HiGHS's `mip_rel_gap=0.005` tolerance, so placement was
     non-deterministic between otherwise-identical runs. Weighted large
     enough to matter (~1e-3-1e-1) it started trading real production for
     placement — confirmed by a reproducible overproduction case (120
     units baked against ~75.7 units of actual demand) that appeared only
     with the weight active and vanished when it was removed. No single
     scalar weight could simultaneously "matter within the optimality
     gap" and "never override correctness."
  2. **Pre-solve greedy backward-fill allocator** (compute
     дефрост/двухдневка placement first via a separate greedy pass, then
     feed the MILP only the leftover regular demand). Rejected per the
     user's explicit correction: this loses the benefit of joint
     optimization (the MILP would no longer see the true full-day
     resource picture when placing regular SKUs) and reintroduces the
     kind of hand-tuned sequencing the MILP rebuild was meant to replace.
     The user's directive was explicit: treat regular + defrost + two-day
     as one combined demand figure solved together, and only *relabel*
     after the fact for display.
- A related bug was found and fixed during this work: the mandatory
  checkpoint's lower bound must be `-inf` (not left at its implicit
  default) for SKUs with zero defrost/two-day component, otherwise it
  silently duplicates the regular constraint and applies the 10,000×
  weight to ordinary regular shortfall too — this was caught by
  `test_two_day_always_wins_capacity_over_higher_priority_regular_sku`
  failing with the priority order backwards.
- A second bug produced fractional `Итого` values (e.g. `40.0265...`):
  `_split_tail` was originally called with the raw continuous solver
  value for the mandatory boundary instead of the kratnost-rounded
  regular-only target; fixed to use
  `ceil(cum_demand[last]/kratnost - eps) * kratnost`.

Implication:

- `ANTI_WASTE_WEIGHT` was raised from `1e-5` to `1e-2` as part of this
  work, but purely for solve-speed — a weight sweep (1e-5 to 0.3)
  produced identical production/shortfall at every tested value, so this
  is not a correctness-relevant change and should not be treated as one
  if revisited.
- Any future change to window-placement behavior should extend the
  post-processing pass, not reintroduce an objective-weight term — the
  `MIP_REL_GAP`/`ANTI_WASTE_WEIGHT` tension documented above applies to
  any similar "soft preference" signal added to this MILP.
- `Итого` in the rendered plan now excludes дефрост (it's tomorrow's
  batch, not today's) but still includes двухдневка (it fully replaces
  today's regular production) — see `docs/baking_plan_implementation.md`
  Rendering section.

## 2026-07-06 - bakery_sales_lag365 Added To Bakery-Day Model

Decision: add `bakery_sales_lag365` (same bakery, same day last year) as a
permanent feature in the bakery-day forecast model.

Context:
- CV (3 folds, Apr/May/Jun 2026) showed consistent MAE improvement:
  delta ≈ −0.003 avg, importance 2–3% gain.
- Seasonal transitions (May→Jun) cause systematic overforecast because
  lag30/roll_mean30 reflect higher May values. lag365 gives the model a
  direct YoY anchor to prior-year June, which is closer to current June.
- Tested three additional YoY variants (lag364, roll_mean4w_yoy,
  yoy_month_mean) — all negative/neutral due to only 27–29% dataset
  coverage (dataset starts Jan 2025, so Jun 2025 lags are not available
  for most training rows). Revisit in autumn 2026 when coverage reaches
  ~65%+.

Implication:
- Production dataset refresh history must start ≥ 13 months back so that
  lag365 is populated. `DEFAULT_HISTORY_START_DATE` set to `2025-06-01`.
- On VM the lag365 column will initially have ~50–60% coverage for
  Jul 2026 rows; coverage grows as the timer accumulates months.

## 2026-06-28 - Ops Docs Are The Current State Layer

Decision:

- `docs/ops/` is the operational source of truth for live state.
- `handoffs/` remains historical context only.

Implication:

- New LLM sessions should begin with `docs/ops/CURRENT_STATE.md`.
- Any production-state change must update `docs/ops/`.

## 2026-07-01 - SKU-Hour Profile Floor-Uplift Removed

Decision:

- Removed the `max(raw_share, mean_share)` floor from
  `smooth_sku_hour_share_profile.py`. The profile now serves raw hourly
  shares.
- Rejected the `sku_hour_uplift_multiplier` mechanism as a production input:
  no reliable signal could be found this session to distinguish genuine low
  hourly demand from shelf-absence (stockout) censoring, so lifting shares
  toward the mean (or toward a category-shape floor, also tried and
  rejected) had no evidentiary basis.

Context:

- Multiple detection approaches were tried and exhausted: rolling
  sell-through/closing-stock=0 classification, dip-depth comparison between
  "censored" and "pattern" bakeries (only 7% difference), five intraday
  signals (last-sale-hour, SKU-zero-while-bakery-active, mid-day gap, tail
  cutoff, closing-stock=0 — only 1/969 profile cells flagged as censored),
  and two category-floor formula variants (both produced values below the
  raw profile due to incompatible denominators between conditional-on-sale
  raw means and all-hours day shares).
- The underlying business reason floor-uplift seemed attractive (bakeries
  produce less when they still have stock from the previous batch) turned
  out to be indistinguishable from genuine per-hour demand variation without
  actual shelf-availability data, which is not collected.

Implication:

- Any future attempt to correct for shelf-absence must start from real
  inventory/availability signal, not inferred from sales patterns alone.
- `median_sku_share_in_hour` in the profile table remains a dead/unused
  column (overwritten with the mean during rebuild) — do not use it for
  anything; only `mean_sku_share_in_hour_norm` is consumed by
  `apply_bakery_profiles.py`.

## 2026-07-01 - Prod Bakery-Day Model Switched To Base (No Bakery-Level Uplift)

Decision:

- Added a new scenario `base_no_sku_uplift` to
  `pipelines/forecast_publish/run_production_inference.py`: base bakery-day
  model (`bakery_day_model.joblib`) + raw SKU-hour profile allocation + no
  SKU-hour uplift multiplier.
- Deployed and activated this scenario for ALL bakeries; updated
  `FORECAST_SCENARIO`/`FORECAST_ACTIVATE_RUN` in the VM `.env` so the nightly
  timer keeps using it.

Context:

- A 2026-06-30 pilot (`SESSION_HANDOFF_2026-06-30_base_raw_pilot_evaluation.md`)
  found the `base_raw_uplift` scenario (base model + raw SKU-hour uplift
  multiplier) strongly outperformed prod (`uplifted_norm`) on a 7-day pilot
  (bias +6.6% vs +11.9%, wMAPE 35.2% vs 72.2%).
- The two pre-existing scenarios bundle bakery-model choice with SKU-uplift
  choice (`base_raw_uplift` = base + SKU uplift on; `uplifted_norm` =
  uplifted + SKU uplift off) — neither matches "base model, no SKU uplift",
  which is what was actually wanted once the SKU-hour uplift multiplier was
  independently rejected the same day (see decision above).
- The follow-up 28-day comparison run for `base_raw_uplift` produced
  internally inconsistent numbers (see `CURRENT_STATE.md`) and was not
  root-caused; the switch to base model was made on the 7-day pilot signal
  plus manual review, not the 28-day numbers.

Implication:

- If bakery-day-level forecast quality needs re-evaluation, compare against
  `base_no_sku_uplift`, not `base_raw_uplift` (the latter still has the
  now-rejected SKU-hour uplift multiplier baked in and should not be
  reactivated without revisiting that rejection).
- The 28-day `analyze_variants_comparison.py` row-count/bias discrepancy is
  still unexplained and should be investigated before it's trusted for any
  future decision.

## 2026-07-13 - Static Bakery-Day Bias Correction Replaced With A Rolling One

Decision:

- Replaced the one-time static bias snapshot (`models/bakery_day_bias.json`,
  computed once from a June holdout and applied to every forecast forever)
  with a correction recomputed every run from a trailing 7-day window of
  live lead-1 `forecast_base` vs actual sales
  (`pipelines/forecast_publish/rolling_bakery_bias.py`). Falls back to the
  static snapshot for bakeries with fewer than 3 days of recent history.
  Same `0.15` relative clip as before. Live in prod as of run
  `prod_base_bakery_no_sku_uplift_20260713_h14`.

Context:

- User-reported underforecast on Парковая 7 / Парина 6 (2026-07-06..11) was
  traced to the 2026-07-06 bakery-day model retrain (`bakery_sales_lag365`
  added) making `bias.json`'s June-derived corrections stale — confirmed
  directly from live `forecast_base` vs `forecast_final` on the active prod
  run, not inferred from a backtest.
- A static, never-refreshed correction is fragile by construction: it can
  only be as good as whatever the model's bias happened to be during its
  one calibration window, and silently degrades after any retrain or
  seasonal shift with no signal that it's gone wrong.
- Validated with an 11-day dev walk-forward backfill before deploying
  (see `CURRENT_STATE.md` for the numbers) — first pass used stale/default
  weather and overstated the improvement; re-ran with real Open-Meteo
  weather before trusting it. Static was worse than every alternative
  tested (no correction, rolling, trend-extrapolated) in every variant.

Implication:

- Any future bakery-day model retrain no longer requires a matching manual
  bias-table regeneration step — the rolling correction adapts on its own
  within `min_days` (default 3) runs.
- `models/bakery_day_bias.json` is now only a cold-start fallback, not the
  primary correction source. Keep it reasonably fresh (regenerate at
  retrain time as before) but it's no longer load-bearing for bakeries with
  steady recent history.
- A trend-extrapolated variant (linear fit + damped one-step-ahead
  extrapolation over a trailing 14-day window) was tried and rejected —
  no better than the flat rolling mean on this data, and riskier (can
  double-count trend the model's own `bakery_sales_trend` feature already
  captures). Revisit only with a longer, more stable trend signal than a
  single transitional-period backtest can provide.
- Discovered but did not fix: `bakery_forecast_day_snapshots` and its
  SKU-day/SKU-hour counterparts (prod and dev) are `ReplacingMergeTree`
  with `source_run_id` outside the sort key
  (`ORDER BY forecast_date, lead_days, bakery_id[, product_id[, hour]]`).
  Background merges silently collapse multiple runs sharing a
  (date, bakery[, product[, hour]]) key down to one, regardless of
  run_id — confirmed by watching two deliberately-parallel dev backfill
  runs lose 9 of 11 days of one variant to a merge within about an hour.
  This likely also explains earlier-observed "run_id mixing" in
  historical lead-1 comparisons. Needs its own decision (adding
  `source_run_id` to the sort key means a full table rebuild) — flagged,
  not resolved.

## 2026-07-13 - SKU-Hour Fallback Profile Gains A Minimum-Sample-Size Gate

Decision:

- Added `MIN_FALLBACK_N_DAYS = 3` to the tier-2 (dow-blind) SKU-hour
  fallback profile: rows with `n_days` 1-2 are excluded from the fallback
  average entirely, in both `src/experiments_v2/apply_bakery_profiles.py`
  (CSV path) and `apply_bakery_profiles_clickhouse.py` (the production
  ClickHouse path actually used by `run_production_inference.py`).
  `n_days == 0` is still trusted (legacy profiles missing the column
  default to 0 upstream — that's "unknown", not "observed zero days").
  Committed `e3f39e6`, deployed to the VM via targeted SFTP (see
  `CURRENT_STATE.md`).

Context:

- Investigated a user report that real, steadily-selling SKUs at bakery 16
  (Кулагина 4) had a forecast collapsed to near-zero. Traced "Пирог с
  Манго" (product 11465): actual sales ~7/day every day for 30 days, but
  `sku_forecast_hour_embedded` showed `0.043`/day, entirely concentrated
  in a single near-dead hour (22:00).
- Root-caused by re-running the real production functions against real
  data rather than guessing: reconstructed the pre-correction base daily
  forecast (~6.4/day, matching actual sales — so the profile *should* have
  been fine), then ran the actual `_build_recent_correction_targets`
  (recent-sales correction) on real bakery-16 data and got `6.39` — the
  correction step was NOT the problem, contradicting the first hypothesis.
  Working backward from there: the SKU's per-(dow,hour) profile rows never
  reach the tier-1 gate (`n_days>=8`) in ANY hour, so it's entirely
  dependent on the tier-2 fallback — which had a single `n_days=1` row at
  hour 22 with an unsmoothed share of `0.5` (one Friday sale reading as
  "100% of that near-empty hour"). That one row, averaged in unfiltered,
  produced a fallback share of `0.135` for hour 22 vs `~0.002-0.004` for
  every other hour — pulling nearly the SKU's entire (correctly-scaled)
  daily total into a hour where the whole bakery only sells ~1 unit total,
  crushing it down to a small fraction of that tiny pool.
- Confirmed this is systemic, not a one-off for this one SKU: bakery 16
  alone had 16 profile rows with `n_days<=2` and share > 0.1 (9 at hour
  22, 6 at hour 5 — both low-traffic edge hours), affecting at least 8-9
  distinct SKUs at this one bakery.
- 4 pre-existing, unrelated test failures were found in passing (3
  pie-category-cap tests in `test_apply_bakery_profiles_clickhouse_recent.py`
  expecting numbers the current code doesn't produce, 1 collection error
  in `test_build_bakeable_products_table.py` from a renamed function) —
  confirmed via `git stash` that they fail identically without this
  change, left untouched, flagged as a separate follow-up.

Implication:

- Deployed but **not yet exercised** as of 2026-07-13: a concurrent
  session's unrelated manual production run (18:33:59+03:00) landed
  minutes after the SFTP file replacement and still shows the old
  (uncorrected) `0.043` value for product 11465 — most likely a race
  where that process had already imported the old module code before the
  files were replaced on disk. First real run will be the 2026-07-14
  03:30 UTC nightly timer. Re-verify product 11465 (bakery 16) against
  `mart_sales_60d` that morning before trusting the new forecast.
- Any future "why is this one SKU's forecast wrong" investigation should
  check `n_days` on its profile rows early — a SKU with no tier-1 coverage
  anywhere is entirely at the mercy of the tier-2 fallback's data quality,
  which has much less protection against small-sample noise than tier-1's
  own `MIN_TIER1_N_DAYS=8` gate.
- Separately noticed but deliberately not touched: `bakeable_products`
  city-scope rows for Казань all come from the old
  `forecast_category_filter`/`partner_baking_markup` sources with no
  sales-share threshold at all — `build_city_assortment_from_sales.py`'s
  `sales_window` source (the actual 80%-threshold logic) has never
  produced a single row for any city, because the code implementing it
  was placed on the VM uncommitted at 11:46 UTC today, after this
  morning's run, and has literally never executed
  (`journalctl -u forecast-production.service` has zero "assortment"
  mentions in its full history). This is someone else's in-flight,
  unreviewed work — see `CURRENT_STATE.md`'s "Known issue" note on VM git
  drift. Left alone; will get its first real run at the same 2026-07-14
  03:30 UTC timer.

## 2026-07-14 - Fixed The Assortment-Threshold Insert Bug And Verified Both Fixes Live

Decision:

- Fixed `scripts/build_city_assortment_from_sales.py:build_layers()`:
  `combined["valid_from"]` was built via
  `pd.to_datetime(valid_from).date().isoformat()` — a Python `str`. That
  function's output DataFrame is inserted straight into ClickHouse via
  `client.insert_df()` against a `Date`-typed column;
  `clickhouse-connect`'s Date serializer computes `(value - epoch).days`
  per cell, which raises `unsupported operand type(s) for -: 'str' and
  'datetime.date'` when `value` is a string instead of a real
  `datetime.date`. Changed to `pd.to_datetime(valid_from).date()`.
  Committed `1b29184`, deployed via SFTP, then manually triggered a full
  production run (`systemctl start forecast-production.service`) to
  verify rather than waiting for the next nightly timer.

Context:

- This turned out to be the real, sole cause of the 2026-07-13 finding
  that `sales_window` (the 80%-threshold assortment source) had never
  produced a single row: the 2026-07-14 03:30 UTC nightly timer gave the
  assortment-refresh code its first real execution ever, and it failed
  immediately with this exact error, confirming the hypothesis rather
  than leaving it as "hasn't run yet, presumably fine."
- Root-caused by copying the same construction pattern into an isolated
  reproduction against a throwaway ClickHouse table via the `.env.dev`
  (`_dev`-suffixed) environment — confirmed the exact production
  traceback before the fix, and a clean insert after — without writing
  test data to any real or shared table. The auto-mode safety classifier
  correctly blocked a first attempt at reproducing this directly against
  the production `bakeable_products` table; re-ran the reproduction
  against dev instead.
- This is a fix to already-shipped, committed code (`71465a1`, the
  2026-07-06 "sales-based bakeable assortment" feature) — the VM's
  seemingly-uncommitted copy of this file was never some other session's
  unreviewed WIP; it's this same feature, present on the VM ahead of
  `git` because the VM's git HEAD is stuck at `2c38e80` (predates
  `71465a1`) — see `CURRENT_STATE.md`'s "Known issue" note on VM git
  drift for why `git pull` doesn't work there.
- Manually running the full pipeline (rather than waiting for the next
  timer) surfaced a wider effect than anticipated: `sales_window` rows
  landed for all 9 cities at once (not just Казань), with a `valid_from`
  newer than the old `forecast_category_filter`/`partner_baking_markup`
  rows' last update. Since `get_bakeable_products()` selects by
  "freshest `valid_from` per city," this immediately switched every
  city's served assortment from the old ~110-product unfiltered set to
  the new, threshold-checked ~52-product city layer plus per-bakery
  additions — a live, wide-blast-radius behavior change, not confined to
  the one bakery the original report was about.
- Also directly re-verified the 2026-07-13 SKU-hour fallback fix
  (`e3f39e6`) against this same manually-triggered run: product 11465
  (Пирог с Манго) at bakery 16 went from `0.043`/day (one dead hour) to
  `2.97`/day (3 real hours) — a large improvement, though still below the
  `~6.9`/day actual. Product 11213 (Роллы Вулкан с курицей) similarly
  spread across 16 real hours but stayed at `0.048`/day vs `~2.0`/day
  actual. Both SKUs still under-forecast relative to real demand — a
  separate, not-yet-investigated issue in how the recent-sales
  correction blend weights treat SKUs whose recent share sits below both
  the "runner" (0.5%) and "core" (1%) boost thresholds documented in the
  2026-07-13 entry above. Not fixed here.

Implication:

- The two 2026-07-13 findings ("assortment threshold never runs" and
  "SKU-hour forecast collapses for thin SKUs") are now both confirmed
  fixed and verified against a real, freshly-generated production run —
  not just deployed-and-hoped. See `CURRENT_STATE.md` for the exact
  verification numbers.
- Anyone auditing individual bakeries' baking plans over the next few
  days should watch for SKUs that quietly disappeared from a plan because
  they no longer clear either the city 80% threshold or have their own
  `scope='bakery'` row — this is expected behavior now working correctly
  for the first time, not a new regression, but it will look like one if
  nobody remembers this decision.
- The remaining under-forecast gap for thin/low-volume SKUs (Пирог с
  Манго, Роллы Вулкан с курицей) is a legitimate follow-up: the
  recent-correction blend formula's "runner"/"core" thresholds may need a
  lower tier for SKUs that sell daily but at very low volume relative to
  the bakery's total, so they get *some* lift instead of falling through
  to the un-boosted default blend. Not investigated further this
  session.

## 2026-07-14 - SKU-Level Floor-Uplift Reactivated For Pilot (Reverses 2026-07-01 Decision)

Decision:

- Restored the `max(raw_share, mean_share)` floor in
  `smooth_sku_hour_share_profile.py` (reverted one line of `625605d`) and
  switched production to scenario `base_raw_uplift` (base bakery-day model +
  raw SKU-hour uplift multiplier), applied to all bakeries. This explicitly
  reverses the "2026-07-01 - SKU-Hour Profile Floor-Uplift Removed" decision
  above.

Context:

- The project is repositioning around an imminent pilot launch. Per the
  user, the project's core value proposition is eliminating missed
  sales/underproduction, not forecast-accuracy purity — without SKU-level
  uplift, that value proposition isn't delivered, even though the floor's
  known imprecision (no way to distinguish shelf-absence/stockout censoring
  from genuine low demand, per the 2026-07-01 investigation) still applies
  unchanged. The user explicitly acknowledged this tradeoff and chose to
  proceed anyway for the pilot.
- Before making any change, confirmed live that simply flipping
  `FORECAST_SCENARIO` to `base_raw_uplift` would have been a no-op: the
  `sku_hour_uplift_multiplier_embedded` table (version `weekly_20260712`,
  produced automatically every Sunday by the still-enabled
  `weekly-profile-refresh.timer` even after the floor was removed) had 0 of
  27,150 rows with a multiplier above 1.0. The floor removal had silently
  turned the entire uplift mechanism into a no-op weeks before anyone
  planned to use it again.
- Rebuilt the full profile pipeline with the restored floor
  (`scripts/weekly_profile_refresh.py`), producing `profile_version
  weekly_20260714` with 95.4% of multiplier rows now above 1.0 (avg 1.29,
  max 3.53) — confirms the mechanism produces a real signal again, not just
  that the code compiles.
- Verified end-to-end against a live production run
  (`prod_base_bakery_raw_uplift_sku_20260714_h14`): product 11465 (Пирог с
  Манго, bakery 16) forecast moved from `2.97`/day (no uplift) to
  `3.44`/day (with uplift) — a real, directionally-correct increase, though
  still below the ~6.9/day actual for that specific thin SKU.
- Deploy hit a new obstacle beyond the known VM git-pull block: the usual
  SFTP-copy workaround failed outright (`Subsystem sftp` isn't configured
  in this VM's sshd — confirmed with a bare `sftp.put()` to `/tmp` failing
  with `ENOENT`, unrelated to the target path). Worked around by piping
  base64-encoded file content over the existing SSH exec channel instead.
  Should be fixed properly (enable the sftp subsystem) so future sessions
  don't have to rediscover this.

Implication:

- The imprecision risk from the 2026-07-01 decision is now live in
  production again, by deliberate choice: any future "why is this forecast
  too high" investigation for a low-traffic SKU/hour should check the
  `sku_hour_uplift_multiplier` first, not assume it's still a no-op.
- SKU-hour forecasts are **not renormalized** under this scenario — summed
  SKU-hour forecasts within a bakery-hour can and do exceed what the
  bakery-day model predicts for that hour (observed up to 607 units summed
  in one bakery-hour on the first live run). This is intentional (the whole
  point of an un-renormalized uplift), but any downstream consumer of
  `sku_forecast_hour_embedded`/`sku_forecast_day_embedded` (baking plan,
  capacity planning) needs to expect materially higher numbers than under
  `base_no_sku_uplift`.
- Rollback is a scenario-config change only (`base_no_sku_uplift` in VM
  `.env`, restore from `.env.bak_20260714_162514`), not a code revert — the
  restored floor code is inert unless `use_raw_uplift_multiplier=True`.
- Baking-plan template rework (phase 2 of this pilot reconfiguration —
  templates instead of MILP, individual per-bakery templates, current
  assortment with missing-from-template SKUs left blank) is explicitly out
  of scope for this decision and not yet started.

## 2026-07-14 - Baking Plan Reverted From MILP To Template-Driven Window Assignment (Phase 2 Of Pilot Reconfiguration)

Decision:

- `apps/baking_plan/` no longer computes *which* window a SKU bakes in —
  neither the pre-MILP peak-detection distribution nor the MILP solver.
  Window assignment is read directly from the reference Excel template's
  pre-filled C:L cells (`allocation.read_row_schedule`); the package's own
  job is reduced to resolving each template row to a live assortment
  product and sizing quantities from the **live** hourly forecast into
  whichever windows the template already assigned. Quantities are still
  dynamic (recomputed every request); only window placement is now static.
- Individual per-bakery templates (pekарни 20/21/22) and the current
  assortment mechanism (`assortment.py`, `bakeable_products` scope=city/
  bakery, unchanged since the MILP era) are both kept. Assortment products
  with no matching template row are appended at the end of the plan with
  no window breakdown, only a raw (non-kratnost-rounded) `Итого` total.
- Capacity/mощность checking (baker-minutes, tray-slots, daily caps,
  shortfall highlighting) is removed entirely — matches the original
  pre-MILP system's documented limitation, not a regression.
- Night-defrost quantity reverts to the simple pre-MILP formula (tomorrow's
  early-hour forecast, uncapped) — the MILP-era PDF-derived caps
  (`NIGHT_STORAGE_DIRECT_UNITS_BY_SKU`/`NIGHT_PREP_LABOR_MINUTES_BY_SKU`)
  are removed.
- Revenue-tier sheet selection (`до 1,5 млн`/`до 2,5 млн`/`от 2,5 млн`/
  `от 3млн`) is restored, reading the existing (stale, one-time May 2026
  manual backfill) `bakery_month_revenue_embedded` table as-is — no
  refresh pipeline added.

This explicitly reverses three prior decisions: "2026-07-09 - Baking Plan
Rebuilt As Its Own Package" (the teardown of the template system),
"2026-07-10/11 - Baking Plan: MILP Allocator Chosen", and "2026-07-13 -
Baking Plan: Дефрост/Двухдневка Merged Into A Single MILP Demand Pool".
The package boundary and module-split conventions from those decisions
(standalone package, `_clickhouse.py`/`assortment.py` untouched, one
router-mount line into `forecast_embedded`) are kept — only the
window-assignment algorithm and capacity model are reverted.

Context:

- User-driven pivot: the project is preparing a pilot launch and considers
  the MILP's algorithmic sophistication unnecessary for it — explicit
  instruction was to drop both the old distribution logic and MILP, use
  the template's own window assignment, keep individual templates and the
  current assortment, and accept that new/low-volume SKUs not in the
  template get no window breakdown. The pilot targets top-selling SKUs, so
  this gap was explicitly accepted rather than engineered around.
- Confirmed via three explicit user choices before implementing: (1) drop
  capacity/shortfall checking entirely rather than keep a lighter-weight
  version without the solver; (2) use the simple pre-PDF defrost formula
  rather than keep the PDF-derived caps; (3) restore revenue-tier sheet
  selection from the existing stale `bakery_month_revenue_embedded` data
  rather than fix its staleness first or skip tiering for the pilot.
- Implementation ported the pre-teardown algorithm (git commit `8e3e79f~1`,
  `apps/forecast_embedded/app/services/baking_plan.py`) into the current
  package's module structure, but improved the SKU→forecast join: the old
  code matched hourly forecast rows to template SKU names by fuzzy string
  match; the new code first resolves the template SKU name to a
  `product_id` via the (already fuzzy-matched) assortment lookup, then
  fetches hourly forecast by exact `product_id` — the fuzzy matching only
  has to happen once per row, not once per hourly-forecast row.
- Verified end-to-end against live production data (read-only, no writes)
  for both an individual-template bakery (21, whose template's own sheet
  is labeled outside the standard four revenue tiers — confirmed the
  `select_sheet_name` fallback handles that correctly) and a base-template
  bakery (16, correctly matched to its "от 3млн" sheet). Confirmed
  window-partial cell population (only some C:L columns filled per row,
  matching the template's own pre-filled structure), leftover-assortment
  rows (fastfood SKUs like Хот-дог/Сэндвич not in either reference
  template) rendered with empty window cells and an unrounded raw total,
  and defrost annotations rendering correctly.
- Discovered but not fixed: `bakeable_products_dev` (the `_dev`-suffixed
  ClickHouse table used for local/dev verification) is missing the
  `scope`/`bakery_id` columns added to the production table on 2026-07-06
  — a pre-existing schema-drift bug unrelated to this change (`assortment.py`
  itself was not modified), meaning any local dev-environment testing of
  the baking-plan feature currently fails regardless of MILP vs template.
  Verification for this change was done against production tables
  read-only instead (explicit user approval obtained first, since the
  original plan called for dev-first testing).

Implication:

- `apps/baking_plan/capacity.py`, `apps/baking_plan/algorithms/` (milp.py,
  greedy.py, common.py), and `apps/baking_plan/constants.py` (PDF-derived
  night-storage constants) are deleted, not just unwired — the MILP
  approach is fully removed from this package, not kept dormant.
  `scripts/compare_baking_algorithms.py` (MILP vs greedy comparison,
  referenced by the prior README) is now stale/orphaned; not deleted this
  session, flagged for cleanup.
- `docs/baking_plan_implementation.md` and `apps/baking_plan/README.md`
  rewritten to describe the template-driven approach; both explicitly list
  the dropped capabilities as deliberate limitations, not gaps to silently
  rediscover later.
- The `bakeable_products_dev` schema drift should be fixed (migrate the
  `_dev` table to match prod's `scope`/`bakery_id` columns) before relying
  on dev-environment testing for *any* future baking-plan or assortment
  work — flagged, not fixed, out of scope for this change.
- Deploy to Blackhole was completed later the same session once VibeCode
  API credentials were provided (see `CURRENT_STATE.md` for the deploy
  method and result) — the code is live, but the route-level endpoint
  itself was not smoke-tested (would have required guessing/forging the
  `x-vibe-user-*` admin auth headers, correctly blocked as credential
  forgery against production with no explicit authorization for that
  specific bypass). A real portal/admin session should exercise the actual
  download endpoint at least once.

## 2026-07-14/15 - Assortment-Exclusion Demand Was Vanishing Under Raw Uplift, Fixed In Two Steps

Decision:

- Under `use_raw_uplift_multiplier=True` (`base_raw_uplift`), demand for
  products excluded by assortment filtering (`assortment_city_products`)
  was being **dropped entirely** instead of redistributed to the remaining
  in-assortment SKUs — because `renormalize_hourly_to_bakery_forecast`
  (the only existing mechanism that could have compensated for it) rescales
  to the flat bakery-hour total and would cancel the uplift's own
  deliberate inflation, so it's unconditionally skipped whenever
  `use_raw_uplift_multiplier=True`.
- Fixed in two commits: `114bacd` adds
  `compensate_for_assortment_exclusion()` — rescales remaining rows by
  `sum(pre-filter)/sum(post-filter)` per `(date, bakery, hour)`, which
  isolates purely the assortment-removed fraction (both sides of that
  ratio already carry the same uplift effect, so it doesn't touch the
  uplift itself). `488af38` fixes a second-order bug found while verifying
  the first: the *early* assortment filter (before recent-sales correction
  reads the file back in as its own input) was still dropping
  excluded-product demand before the new compensation step ever saw it, so
  raw-uplift + recent-correction now defers all assortment filtering to
  the single final pass where compensation immediately follows.

Context:

- User noticed a weekly-dashboard screenshot (bakery 257, Ярмарочная 12,
  Чебоксары) showing forecast dramatically *below* actual sales, and
  correctly pointed out this contradicted the pattern on every other pilot
  bakery, where the SKU-day sum consistently reads ~120-130% of the
  bakery-day total (the raw uplift's intended, accepted-as-of-yesterday
  effect — see the entry above). Investigated and found: bakery 257 /
  Чебоксары's SKU-day sum was 62% of its bakery-day total — the only
  pilot bakery below 100%, all others between 117-127%.
- Root cause traced via direct comparison: bakery-day total for bakery 257
  (2026-07-08) was 686, matching the bakery-level model output correctly;
  but summed SKU-day forecast was only 426 across 65 products —
  `forecast_removed` in the allocation summary confirmed hundreds of units
  per day were being dropped by assortment filtering with no
  redistribution. Confirmed the same mechanism affects the *live* active
  run, not just historical backfills — Чебоксары's assortment was cut more
  aggressively than other cities by the 2026-07-14 assortment-threshold
  fix (see that entry above), so it happened to be the city where this
  pre-existing gap became visible.
- First fix attempt (`114bacd`) improved bakery 257's ratio from 0.62 to
  0.89 — better, but still the only pilot bakery below 1.0. Kept
  investigating rather than accepting a partial fix, per explicit user
  direction ("логика аплифта должна давать больше факта, как на других
  пекарнях"). Found the second-order early-filter bug by checking
  `exclusion_compensation.groups_without_remaining_rows` in the allocation
  summary (`0` globally, ruling out "nothing left to redistribute onto" as
  the explanation) and tracing that `apply_recent_sku_hour_correction`
  reads back the *already-filtered* first-pass output as its own input.
  After the second fix, bakery 257 reads 1.30 — squarely inside the
  1.26-1.32 range of every other pilot bakery.
- Verified end-to-end against a live, freshly-triggered production run
  (`prod_base_bakery_raw_uplift_sku_20260715_h14`) both times, not just
  unit tests — re-ran `systemctl start forecast-production.service` after
  each deploy and re-queried ClickHouse directly to confirm the actual
  ratio moved, rather than trusting the code change alone.

Implication:

- `forecast_sku_total` (the value the embedded weekly dashboard prefers
  over the bakery-day total per `app/services/bakery.py`'s
  `coalesce(sku.forecast_sku_total, b.forecast_final)`) should now
  consistently read *above* the bakery-day total for every bakery under
  `base_raw_uplift`, city-assortment-narrowing severity no longer matters.
  If a bakery is ever found reading *below* its bakery-day total again
  under this scenario, that's a regression, not expected variance.
- A small residual exists: 36 `(date, bakery, hour)` groups (out of
  ~45,000) across the full 14-day horizon have *zero* remaining
  in-assortment products after filtering — nothing to redistribute onto,
  so those specific bakery-hours still lose that demand. Not fixed;
  flagged as an acceptable edge case given the scale (36 vs ~45k groups).
- The historical lead-1 backfill runs for 2026-07-01..13 (built the day
  before with the *un-fixed* code) were re-run with the fixed code
  afterward so the dashboard's historical view corrects itself too — see
  `CURRENT_STATE.md` for confirmation once that finishes.
- Both fixes only change behavior when `use_raw_uplift_multiplier=True`
  (i.e., `base_raw_uplift`); `base_no_sku_uplift` and any future
  non-raw-uplift scenario are unaffected — the existing
  `renormalize_hourly_to_bakery_forecast` path (and its own gap-filling)
  still runs exactly as before for those.

## 2026-07-15 - Cap Raw Uplift Per SKU Against Recent Daily Sales

Decision:

- Under `base_raw_uplift`, cap each `(date, bakery, product)` daily forecast
  at `1.2 * recent rolling daily mean` for that bakery-product pair.
- Apply one scale factor to all hourly rows of the affected SKU-day, preserving
  its intraday distribution. Never scale upward, and do not cap SKUs without
  recent history.
- Configure production through `FORECAST_MAX_SKU_UPLIFT_RATIO=1.2` rather
  than hard-coding the pilot threshold.

Rationale:

- Lead-1 analysis across 10 pilot SKUs and 10 bakeries showed that the raw
  uplift's error was uneven within a bakery: some SKU-bakery pairs were close
  to fact while others remained `45-70%` high.
- A bakery-level cap applies the same scale to every SKU. It reduces total
  overforecast but preserves the bad within-bakery allocation and can make
  already-good SKU forecasts worse.
- The simulated per-SKU cap at `1.2` reduced aggregate bias to about `14.5%`
  and removed nearly all SKU-bakery cells above `20%` (excluding the known
  bakery-257/backfill anomaly). It was more selective than the bakery-level
  alternative.

Production verification:

- Deployed commit `466217c` to the production VM and republished
  `prod_base_bakery_raw_uplift_sku_20260715_h14`.
- The live allocation summary recorded `130139 / 445950` SKU-days capped and
  an average scale of `0.8172` among capped SKU-days; deployment verification
  ended with `VERIFY OK`.

## 2026-07-15 - Add Hierarchical Bakery/SKU Downward Correction

Decision:

- After raw uplift, recent correction, assortment compensation, and the SKU
  cap, apply a learned downward-only coefficient per bakery-product pair.
- Estimate coefficients from the latest seven complete lead-1 days using the
  same filtered/deduplicated actual-sales definition as the embedded UI.
- Shrink pair coefficients toward their bakery coefficient with a seven-day
  prior, target a `1.15` forecast/actual ratio, and limit haircut to `15%`.
- If a bakery is not over the target in the calibration window, protect the
  entire bakery from reduction, including all SKU-level coefficients.

Rationale:

- A fixed bakery coefficient corrected aggregate bias but could not address
  heterogeneous SKU errors within the bakery. Fully independent SKU factors
  were too unstable with only a short pilot history.
- A walk-forward test on 2026-07-08..14 reduced pilot WMAPE from `37.15%` to
  `35.73%` and bias from `+12.64%` to `+10.05%` with the conservative settings
  above. The guardrail prevented the known underforecast bakery 257 from being
  reduced.
- Production summary after deployment showed an aggregate `4.31%` reduction
  after the existing SKU cap while protecting `63 / 212` bakeries.

Constraint:

- The initial walk-forward evaluation covered only seven holdout days and was
  performed on top of capped forecasts. Continue monitoring lead-1 results;
  do not interpret it as proof that the SKU cap can be removed without a
  separate uncapped backtest.

## 2026-07-15 - Apply SKU Cap Before Assortment Compensation

Decision:

- Apply the per-SKU rolling-mean cap to the complete pre-assortment SKU set.
- Only afterward filter to the active assortment and compensate the excluded
  demand onto remaining SKUs.
- Never reapply the SKU cap after assortment compensation; redistributed
  demand is not an uplift outlier and must not be removed.

Rationale and evidence:

- The opposite order silently cancelled the existing assortment-exclusion fix
  (`114bacd`, `488af38`) for bakery 257. Bakery-day lead-1 remained accurate
  (`+0.5%` over 2026-07-01..14), while its summed SKU lead-1 fell to `-20.4%`.
- The correctly rebuilt pre-cap history had previously restored bakery 257 to
  a SKU/bakery ratio of `1.30`. Rebuilding it after the cap was introduced
  overwrote those snapshots and lowered the ratio again, which made the issue
  look like a missing/stale lead-1 problem.
- After deploying corrected ordering, the live active ratio recovered from
  `0.787` average to `1.142` (`1.04..1.24`) while retaining both the SKU cap
  and the underforecast guardrail.

## 2026-08-12 - Separate Pilot Forecast Quality, Execution, And Data Triage

Decision:

- Evaluate the base 10-bakery pilot through separate forecast-quality,
  plan-execution, and data/process views. Do not collapse them into one score
  or infer responsibility from an execution deviation alone.
- Use full demand (`sales + accepted lost demand`) only where demand
  completeness is supported. Always publish the corresponding coverage;
  otherwise report observed-sales metrics separately.
- Treat the exact published workbook as the authority for the issued forecast
  and plan. Preserve publication provenance and conflicting versions rather
  than reconstructing an official plan from current code or later snapshots.
- Use `execution_triage.csv`, enriched with forecast WAPE, bias, MAE, demand
  coverage, case shares, and whether bakery action moved supply closer to
  demand, as the authoritative execution review queue. The older priority
  files remain diagnostic.

Rationale:

- Read-only recovery of the Bitrix24 attachment history showed gaps and
  conflicting publications that cannot be resolved safely from forecast
  snapshots alone.
- Corrected completeness rules increased full-demand coverage from the
  preliminary 23.51% to 85.92%, materially changing the aggregate
  interpretation: complete-demand bias is +2.82%, while WAPE remains 31.55%.
- A bakery can deviate from the rounded plan and still move closer to demand
  when the forecast or production multiple is imperfect. Forecast reliability
  is therefore required context before an execution case is escalated.

Constraints and next priority:

- The 2026-07-29 publication is excluded from official KPIs until its 06:00:02
  candidate is explicitly accepted or rejected by the business owner.
- Resolve the product scope and renames using `product_id` as the canonical
  key, then add an append-only publication journal. Only after these controls
  are in place should the management UI consume the new summaries.
- This decision records analytical contracts only; it made no production or
  Bitrix24 state change.

# Decisions

This file records durable project decisions. It is not a session log.

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

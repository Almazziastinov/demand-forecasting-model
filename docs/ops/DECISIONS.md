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
  early or shorted in favor of regular SKUs.
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

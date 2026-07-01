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

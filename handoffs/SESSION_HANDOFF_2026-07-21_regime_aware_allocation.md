# Session Handoff — 2026-07-21 — Regime-Aware Allocation

## Scope and boundary

Continues `SESSION_HANDOFF_2026-07-21_stockout_historical_shadow.md`. The user
approved the next allocation stage. All work remained offline/read only;
production was not changed. `docs/ops/` remains the operational source of
truth.

## Main finding

The failed earlier allocator used mismatched share denominators: forecast
share was normalized inside 53 screened SKU, while actual share used total
bakery sales. The replacement experiment loads the complete SKU forecast for
each historical bakery-day and preserves that complete total.

## Implemented

- `scripts/experiment_regime_aware_sku_allocation.py`
- full historical-run selection and complete SKU universe loading;
- prior-only smoothed residual and regime features;
- positive-only recipients;
- screened donor domain with trailing p90+0.5 floor;
- 0.25% bakery movement budget and 20% recipient cap;
- full-universe, normal-day, stockout, recurrent, top-5, and other-SKU gates;
- regime allocator added to the combined replay and local shadow runner;
- explicit stockout-risk overlay tested separately.

## Data coverage

- 539 bakery-days;
- 97,334 full-universe SKU-days;
- 7,145 matched screened rows / 92.61% coverage;
- 565 zero/missing-forecast screened rows;
- 47 zero/missing clear-stockout rows.

The last 47 are outside multiplicative allocation and belong to the
zero-allocation/assortment direction.

## Selected local shadow candidate

`positive_capacity_regime_q90m05_strength_1.00_budget_0.0025`

- stockout shortfall: −1.843 units;
- recurrent-pair shortfall: −0.772;
- normal MAE: −0.002058;
- full-universe MAE: −0.000319;
- full-universe shortfall: −15.534;
- 32.387 units shifted over 49 days;
- zero new underforecast in every evaluated segment;
- exact bakery-total preservation;
- no whole cases fixed by allocation alone.

Combined with demand preprocessing on the 397 confirmed misses:

- shortfall `1509.285 -> 1451.929`;
- 33 cases improved;
- 23 cases fixed;
- zero cases worsened.

## Rejected variants

- Symmetric residual correction: stockout shortfall +53.14, 31 new stockout
  and 25 new normal-day underforecasts.
- Explicit risk overlay: stockout shortfall −2.498 and no new underforecast,
  but normal-day MAE +0.001167; top-5 gain only 0.056.
- Previous LightGBM daily-share allocator remains rejected.

## Top-5 interpretation

The selected candidate does not improve the recurrent top-5 segment. One of
the four pairs (bakery 257 × product 10485) is a pure zero-forecast case. The
other three do not have a stable positive mean residual. Their stockouts are
better described as episodic variance/regime risk, which the current risk
signal cannot predict strongly enough without normal-day regression.

## Decision

Enable only the conservative regime-aware positive-capacity allocator in the
local read-only shadow. Do not deploy. Keep the risk overlay diagnostic. The
promotion gate remains at least 21 prospective days, zero new underforecasts,
manual review, and a materially stable gain.

## Verification

- Full end-to-end shadow runner completed successfully in 109.5 seconds.
- New allocator tests: 6 passed.
- Combined replay test: passed.
- Ruff clean for changed files.
- Production writes: none.

## Primary artifacts

- `docs/regime_aware_allocation_results_20260721.md`
- `reports/regime_aware_sku_allocation_experiment/summary.json`
- `reports/regime_aware_sku_allocation_experiment/scenario_comparison.csv`
- `reports/stockout_direction_combined_replay/summary.json`
- `reports/stockout_direction_shadow/manifest.json`

## Next step

Accumulate prospective shadow days. In parallel, separate zero-allocation
cases from allocation evaluation and investigate additional day-level risk
features; do not increase the current movement budget to manufacture gain.

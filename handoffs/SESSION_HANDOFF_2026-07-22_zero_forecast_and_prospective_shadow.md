# Session handoff — zero forecast causes and prospective shadow

## Completed

- Classified all 47 clear-stockout rows with missing/zero forecasts.
- Corrected the as-of logic to include `loaded_at` and exact latest-batch
  selection; all 47 cases are historical allocation-assortment exclusions.
- Added a read-only cause-analysis script and local CSV/JSON artifacts.
- Added a prospective journal keyed by Moscow calendar date.
- Integrated the journal into the complete stockout-direction shadow runner.
- Added tests preventing repeated same-day runs from inflating observed days.
- Completed a full read-only shadow run; the first prospective observation is
  2026-07-22 and all three gates pass.

## Prospective state

- observed distinct days: 1 of 21;
- same-day run count: 1;
- normal-day MAE delta: -0.0020583;
- new stockout underforecast cases: 0;
- combined cases worsened: 0;
- production proposal: not ready until 21 distinct prospective days exist.

Files are under `reports/stockout_direction_shadow/history/`.  Historical
replay days are deliberately not imported into this counter.

## Next work

1. Keep the existing two-day assortment freshness guard and monitor refresh
   completion before forecast allocation.
2. Run the shadow once on each new Moscow calendar day until 21 observations.
3. Review gates and only then prepare a separate production proposal.

## Correction after detailed trace

The originally reported single grid-drop was historical lookahead: its
assortment row became valid for 2026-07-19 but was loaded on 2026-07-20, after
the forecast run. Current run `prod_base_bakery_raw_uplift_sku_20260722_h14`
contains all 18 historically affected bakery/SKU pairs on 14/14 horizon days.
The 20 July refresh repair has therefore removed the observed failure mode.

## Assortment coverage guard

Added a fail-fast pre-allocation guard in
`apply_bakery_profiles_clickhouse.py`. It compares seven days of recent sales
with the selected assortment batch and blocks established missing pairs
(at least two selling days and two units). A current read-only audit passed:
211 bakeries, 29,578 recent pairs, zero blocking gaps. Production was not
changed. See `docs/assortment_coverage_guard_20260722.md`.

The publication boundary is covered by two isolated tests: simulated guard
failure results in zero `load_forecast_run` calls; guard success permits one
load with activation disabled. `tests/test_run_production_inference.py` passes
7/7, and the three targeted guard/freshness tests pass 3/3.

## Historical guard backtest

Replayed the guard across all 43 historical bakery/date contexts containing the
47 known no-forecast clear-stockout cases. Sales came from the local raw pilot
export; assortment batches were selected with both effective dates and
`loaded_at <= run_generated_at`. The guard caught 47/47 known cases. All 47
also pass a stricter 3 selling days / 3 units threshold.

The replay produced 4,174 blocking context rows, including 1,319 where an
applicable historical batch existed. This is not merely an artefact of missing
pre-2026-06-18 version history: 4,012 blockers sold on the forecast date or in
the following seven days. Among rows with a complete future window, 108 had no
subsequent sale and remain ambiguous. The conservative 2 days / 2 units rule
left 553 one-day/low-volume missing rows diagnostic-only, as intended.

Conclusion: retain the 2/2 threshold. Historical evidence supports the local
publication guard, but it has not been deployed. Full methodology and caveats:
`docs/assortment_coverage_guard_backtest_20260722.md`.

## Demand-adjusted profile prototype

Expanded demand preprocessing from model-underforecast rows to all 1,296 clear
stockout SKU-days. The inverse mechanism rule selected 591 cases not robustly
classified as allocation; 581 were reconstructed for 3,775 units. A temporal
A/B trained through 2026-07-05 and evaluated 2026-07-06..2026-07-19.

The experiment now mirrors the production `n_days >= 8` exact-profile gate and
fallback routing. Reconstruction added 90 tier-1 SKU rows in 66 existing exact
contexts and improved their clean SKU-day WAPE by 0.0729. One whole context was
promoted from fallback to exact and worsened sharply, so the guarded candidate
freezes exact/fallback routing to the observed-sales profile while allowing new
SKU members inside already-exact contexts.

With guarded routing, clean SKU-day WAPE improved from 0.8301 to 0.8259 and
clean bakery-day hourly WAPE improved from 1.2254 to 1.2240. Pair-level results
are mixed: underforecast decreased while overforecast increased. No production
state changed. Next: rolling-cutoff validation and a separate bakery-day target
experiment. See `docs/demand_adjusted_profile_experiment_20260722.md`.

## Demand-adjusted follow-up: rolling, bakery target, and shrinkage

Completed the planned non-production follow-up.

- Guarded demand-adjusted profiles improved clean SKU-day WAPE in all three
  rolling windows (mean delta -0.00212) and newly restored tier-1 contexts in
  all three (mean delta -0.11909).
- Directly adjusted pairs worsened in all three windows (mean delta +0.00361),
  with reduced underforecast outweighed by added overforecast.
- Retraining the global bakery-day model on the adjusted target won only 1/3
  windows. The baseline already overforecast reconstructed demand on classified
  demand-loss days by 3,300 / 968 / 479 units, so unconditional bakery uplift
  would double-count volume.
- Share blends at 25/50/75% did not repair the pair-level failure. This rules
  out simple shrinkage as the primary solution.

Decision: keep the guarded membership restoration as the promising component,
do not change the bakery-day target globally, and next test a pair-level
walk-forward eligibility gate with context renormalization. Production was not
touched.

## Pair gate rejected; membership seed selected

The non-overlapping pair gate improved aggregate clean SKU-days in 3/3 folds,
but selected-pair direction persisted only 40-46% of the time. Its gain was a
renormalization spillover, so static bakery/SKU eligibility is rejected.

The actual stable mechanism is tier-1 membership restoration. Full promotion
overpredicts the promoted SKU while correcting a larger overprediction on the
other context members. A controlled promotion seed was therefore tested while
keeping mature shares and fallback exactly on baseline.

Selected offline candidate: 5% membership seed.

- clean SKU-day WAPE: 3/3 wins, mean delta -0.000307;
- adjusted-pair clean SKU-day WAPE: 3/3 wins, mean delta -0.000128;
- new-membership context SKU-day WAPE: 3/3 wins, mean delta -0.031466;
- all-holdout SKU-day WAPE: 3/3 wins, mean delta -0.000255;
- exact bakery-hour total preservation.

This is not deployed. Next step is to add the 5% membership-seed candidate to
the local prospective shadow and accumulate independent days before a production
proposal.

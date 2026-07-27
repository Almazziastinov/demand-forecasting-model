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

## Membership seed registered in local shadow

`demand_adjusted_membership_seed_0.05` is now present in the local shadow
manifest under `shadow_candidate_components`; it is not an enabled component.
Its historical evidence ends on 2026-07-19 and the separate candidate counter
is 0/21. A candidate day is recorded only after `evaluated_through` advances
beyond 2026-07-19, with one record per distinct evaluation date. Repeated daily
runs cannot count the same backtest again.

Added `--skip-refresh` to the shadow runner so manifest/registry checks can use
existing local artifacts without rerunning ClickHouse research or changing its
reports. Verified locally with production writes disabled.

## Stockout deficit versus surplus decomposition

Implemented the proposed physical-balance test over all 1,296 clear stockout
cases. The reconstructed 8,305.8-unit deficit was aggregated to 461 positive
bakery-days and compared with closing surplus on other SKUs. The strict donor
definition requires balance consistency, hourly/daily sales agreement, a
one-day product, exclusion of recipient SKUs, and a one-unit reserve.

At that baseline, 5,754.9 units (69.3%) are coverable by same-bakery/day surplus
and 2,550.8 units (30.7%) remain a bakery-volume gap. A temporal sensitivity,
where a donor must still record sales no earlier than the latest stockout hour,
retains 63.5% coverage. Reserve sensitivity yields 82.0% / 69.3% / 57.2%
allocation components for 0 / 1 / 2 reserved units.

The baseline day split is 281 allocation-plus-excess, 26 approximately
balanced, 133 mixed, and 21 volume-shortage days. Closing surplus is only
moderately higher on stockout days than normal days (median 24 versus 21), so
it constrains the decomposition but does not prove direct SKU-to-SKU causality.

Decision: use the decomposition as an offline regime label. Covered deficit
belongs to the dynamic-allocation problem; uncovered deficit belongs to demand
preprocessing and bakery-volume uplift; mixed days use both. Residual surplus
above deficit is tracked separately as overproduction. Production was not
touched. Full details: `docs/stockout_surplus_coverage_20260722.md`.

## Simplified stockout-adjusted data layer

The allocation interpretation was subsequently demoted: same-day surplus is
not evidence that one SKU pulled volume from another. The primary path now
restores censored stockout demand without donor or reallocation assumptions;
allocation remains a later prospective research question.

Added `build_stockout_adjusted_demand_dataset.py`. Its explicit target contract
keeps observed sales as a lower bound, stores imputed demand and a capped point
estimate separately, and records target provenance and reconstruction
confidence. The materialized read-only pilot dataset has 114,852 SKU-days, all
1,296 accepted stockouts, and 8,305.8 imputed units (0.815% of observed demand).
Of the stockouts, 868 have high reconstruction evidence, 395 medium, and 33
insufficient; 36 censored rows receive no point adjustment and are marked
ineligible by the suggested starting weights.

The configured cap binds on 661 cases (51.0%). A 0.50/10-unit policy yields
6,204.0 imputed units, while 1.00/20 yields 9,214.3. The next model experiment
must therefore compare lower-bound, weighted-point, and cap variants instead
of treating the current point estimate as ground truth. No production writes
were performed. Details: `docs/stockout_adjusted_demand_dataset_20260722.md`.

## Bakery-target backtest on the simplified demand data

Ran the production-family bakery-day LightGBM at three temporal cutoffs for
observed-sales, confidence-weighted, and conservative 50%/10-unit targets.
Stockout holdouts were scored separately against observed lower bounds,
conservative points, and full 75%/20-unit points; clean days use observed sales.

The conservative target is selected for the next offline stage. On the two
non-overlapping 14-day holdouts it improves clean-day absolute aggregate bias
2/2 (mean -32.4 units) and reconstructed-stockout absolute bias 2/2 (mean
-802.9 units). Full-point stockout underforecast falls by 529.9 units per
window, with 272.9 units of added overforecast. The weighted variant improves
clean-day bias only 1/2 independent windows.

Observed stockout-sales bias is not treated as decisive because those sales
are censored. The clean control is small (19 and 23 bakery-days), so the result
is directional and not production evidence. Next: pass the conservative target
through the SKU/profile layer and check clean-SKU bias plus delivery of uplift
to affected SKUs. Production remained unchanged. Details:
`docs/stockout_adjusted_bakery_target_experiment_20260722.md`.

## Conservative SKU/profile end-to-end backtest

Passed the conservative 50%/10-unit demand reconstruction through the current
normalized SKU-hour profile on two non-overlapping 14-day holdouts. The
experiment decomposes bakery-total-only, profile-only, combined, and
guarded-profile effects. Actual hourly bakery shape is held as an oracle so the
test isolates daily total and SKU allocation.

The original interpretation over-weighted evaluation against observed sales.
The corrected primary target is reconstructed demand over all SKU-days:
observed sales on clean rows and observed plus imputation on stockout rows.
All variants are also aligned to the same union support before scoring.

Under the corrected evaluation, the reconstructed profile with the observed
bakery total improves all-SKU reconstructed-demand MAE in 2/2 windows (mean
delta -0.0196). It is therefore a valid offline distribution candidate.

The conservative bakery total adds 833.1 units per window on average, but the
unchanged profile sends only 46.6 units to stockout SKU-days, closing 3.2% of
their pooled reconstructed gap. The full adjusted profile raises delivery to
20.6% and improves reconstructed-demand aggregate bias 2/2, but SKU-day MAE
improves only 1/2 and is neutral on average. The guarded profile also wins MAE
only 1/2, despite a favourable mean driven by the second window.

The profile-only variant moves about 251 units per window toward stockout SKUs
while holding the total nearly fixed, proving that normalized reconstructed
profiles introduce implicit transfers. Full end-to-end delivery is also
unstable: 9.1% of the gap in the first independent window and 34.9% in the
second. Most uplift is dispersed over mature clean SKU shares.

Decision: retain the reconstructed normalized profile offline, but do not
promote the full bakery uplift. Next sweep 0/25/50/75/100% of the conservative
bakery correction with the reconstructed profile, using all-SKU reconstructed
demand on equal support as the primary target. Production remained unchanged.
See `docs/stockout_adjusted_sku_profile_experiment_20260727.md`.

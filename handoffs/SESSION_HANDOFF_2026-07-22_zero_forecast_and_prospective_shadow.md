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

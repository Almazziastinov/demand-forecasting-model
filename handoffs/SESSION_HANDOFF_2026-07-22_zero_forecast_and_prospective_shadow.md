# Session handoff — zero forecast causes and prospective shadow

## Completed

- Classified all 47 clear-stockout rows with missing/zero forecasts.
- Confirmed 46 historical assortment exclusions and one forecast-grid drop.
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

1. Trace product 4944 at bakery 257 through forecast-grid construction.
2. Validate that the production refresh publishes effective assortment before
   forecast generation, without changing production during investigation.
3. Run the shadow once on each new Moscow calendar day until 21 observations.
4. Review gates and only then prepare a separate production proposal.

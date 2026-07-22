# Stockout-adjusted bakery target experiment

Date: 2026-07-22

Status: offline rolling backtest; production unchanged

## Design

The production-family bakery-day LightGBM was retrained on three target
variants at cutoffs 2026-06-21, 2026-06-28, and 2026-07-05. Each holdout is the
following 14 days and only adjustments dated on or before the cutoff enter
training.

Variants:

- `observed_sales_target`: unchanged observed bakery sales;
- `weighted_reconstructed_target`: the current reconstruction multiplied by
  the suggested 0.8 high-confidence / 0.5 medium-confidence weights;
- `conservative_reconstructed_target`: full reconstruction under the stricter
  50% and 10-unit per-SKU-day cap.

True latent demand is not observable on stockout days. Predictions are
therefore scored against three diagnostics rather than one claimed truth:
observed sales as a lower bound, the conservative reconstructed point, and the
current 75% / 20-unit point. Clean days are scored against observed sales.

The middle rolling window overlaps the other two. The primary stability check
uses the two non-overlapping holdouts after cutoffs 2026-06-21 and 2026-07-05.

## Result

The conservative target is the stronger candidate.

Across the two non-overlapping windows:

- clean-day absolute aggregate bias improves in 2/2 windows, by 32.4 units per
  window on average;
- absolute bias against the conservative stockout reconstruction improves in
  2/2, by 802.9 units per window;
- absolute bias against the full stockout reconstruction also improves in
  2/2, by 802.9 units per window;
- stockout underforecast against the full point falls by 529.9 units per
  window, while overforecast grows by 272.9 units;
- against observed stockout sales alone, absolute bias improves only 1/2 and
  worsens by 38.9 units on average. This is not a rejection because observed
  sales are censored lower bounds in this group.

The weighted target is weaker: clean-day absolute bias improves only 1/2
non-overlapping windows, while reconstructed-stockout bias improves 2/2.

The three-window rolling result is consistent with the main conclusion. The
conservative target improves clean-day absolute bias in 3/3 rolling windows
and reconstructed-stockout bias in 3/3.

## Limitations and decision

Only 19 and 23 clean pilot bakery-days are available in the two independent
holdouts because clear stockouts occur on most bakery-days. The evidence is
therefore directional, not sufficient for production promotion.

The conservative target is selected for the next offline stage. It should now
be passed through the SKU/profile layer to determine whether the added bakery
volume reaches the affected SKU without introducing systematic clean-SKU
overforecast. The weighted variant remains diagnostic only.

No allocation-transfer assumption, donor subtraction, forecast publication,
or ClickHouse write was used.

## Artifacts

- Experiment: `scripts/experiment_stockout_adjusted_bakery_targets.py`
- Tests: `tests/test_experiment_stockout_adjusted_bakery_targets.py`
- Reports: `reports/stockout_adjusted_bakery_target_experiment/`

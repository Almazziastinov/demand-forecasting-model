# Direct demand target versus selective uplift — 2026-08-25

## Design

Three causal 14-day folds compare the same bakery LightGBM baseline with two
ways of using conservative reconstructed lost demand:

1. direct demand target: train the bakery model on `sales + lost demand`;
2. selective uplift: retain the sales-target forecast, predict stockout
   probability and conditional lost quantity, and cap the addition at 8% of
   the baseline forecast.

Cutoffs are 2026-06-21, 2026-06-28 and 2026-07-05. Only information on or
before each cutoff is used for training. The evidence is limited to 11 pilot
bakeries for which reconstructed demand is available.

## Mean result across folds

| Variant | Forecast | WAPE | Bias | Recognized lost | True overforecast | Overforecast rows |
|---|---:|---:|---:|---:|---:|---:|
| Sales target | 173,518 | 6.024% | -1.946% | 37.41% | 3,619 | 58.3 |
| Direct demand target | 174,296 | 5.878% | -1.507% | 41.90% | 3,881 | 65.3 |
| Selective uplift | 174,925 | **5.848%** | **-1.151%** | **43.46%** | 4,167 | 68.7 |

Direct demand training improves WAPE on two of three folds and raises
recognized lost demand on every fold. Selective uplift improves the baseline
WAPE on all three folds, beats direct demand on two of three, and produces the
highest recognized-lost coverage on every fold.

Both challengers increase true overforecast. Relative to the sales-target
baseline, the mean increase is approximately 261 units and seven rows for
direct demand, and 548 units and 10.3 rows for selective uplift. This is a
real tradeoff, not a passed production gate.

## Interpretation

The experiment supports continuing both approaches. Direct demand training is
the simpler and more conservative candidate. Selective uplift extracts more
of the lost-demand signal and has the best mean WAPE, but its current
probability-times-quantity rule is too permissive. The next selective variant
should calibrate its probability threshold and cap using an inner training
fold, with no tuning on the outer test dates.

Neither result can be extrapolated to the whole network yet because the lost
demand labels cover only 11 bakeries and a short May-July period. The next
required gate is a wider reconstructed-demand dataset followed by the same
frozen comparison and predictive SKU allocation.

Artifacts:

- `scripts/compare_direct_demand_and_selective_uplift.py`
- `reports/direct_demand_vs_selective_uplift_20260825/summary.csv`
- `reports/direct_demand_vs_selective_uplift_20260825/metrics.csv`
- `reports/direct_demand_vs_selective_uplift_20260825/predictions.parquet`

Production writes: none.

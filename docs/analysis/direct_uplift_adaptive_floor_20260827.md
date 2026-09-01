# Direct predictive uplift and adaptive floor (2026-08-27)

## Compared candidates

1. `Direct + P50`: direct predicted SKU demand normalized to the P50 bakery-day volume.
2. `Direct + predictive uplift + P50`: a forecast-time expected lost-demand correction redistributes exactly the same P50 volume.
3. `Direct + predictive uplift + adaptive floor`: a causally selected SKU floor may add volume above P50 for repeatedly constrained SKU.

No candidate uses incumbent SKU/category shares, hourly profiles, the previous
Predictive allocation, or the previous universal floor.

## Predictive uplift

The uplift is a two-stage expected-loss model:

`P(clear stockout) * E(imputed lost demand | clear stockout)`.

Only forecast-time causal Direct features are used. Current-day availability and
last-sale time are used to construct historical labels but are excluded from the
features. The expected lost units are added to Direct raw SKU demand and then
normalized back to exactly the Direct P50 bakery-day volume.

On all 20 rolling dates, WAPE changes from 41.83% to 41.63% at identical volume;
surplus falls by 3,591 and underbake also falls by 3,591. The improvement is small
but directionally consistent on both frozen folds.

## Adaptive floor selection

Parameters were selected only on the earlier blocked fold. The rule minimizes
surplus among candidates that beat actual-state underbake. The selected causal
gate is:

- at least 8 matching weekday observations;
- historical clear-stockout rate at least 75%;
- historical mean imputed loss at least 4 units;
- floor target `0.8 * historical demand P67`;
- increment capped by both +5 units and +10%.

This is applied unchanged to the later current fold.

### Independent current fold

| Candidate | WAPE | Surplus | Underbake | Recognized imputed loss |
| --- | ---: | ---: | ---: | ---: |
| Direct + P50 | 41.65% | 191,477 | 438,117 | 88,764 |
| Direct + uplift + P50 | 41.45% | **189,975** | 436,615 | 96,218 |
| Direct + uplift + adaptive floor | **41.25%** | 194,659 | **428,976** | **101,861** |

The selected floor reduces underbake by 7,639 for 4,684 additional surplus
units versus uplift-only and reduces total imbalance. Maximum top-SKU share is
17.52% and no current-fold bakery-day reaches 20%.

## Kazan two-day FIFO economics

| Candidate | Gross profit | Delta vs actual | Service level | Lost demand |
| --- | ---: | ---: | ---: | ---: |
| Actual state | 102.187m | 0 | 69.10% | 740,187 |
| Current | 93.632m | -8.555m | 65.25% | 832,182 |
| Direct + P50 | 106.374m | +4.186m | 71.80% | 675,362 |
| Direct + uplift + P50 | 108.112m | +5.925m | 71.95% | 671,926 |
| Direct + uplift + adaptive floor | **108.869m** | **+6.682m** | **72.58%** | **656,622** |

Uplift-only adds 1.739m gross profit versus Direct P50 at the same target-volume
policy. The adaptive floor adds another 0.757m, 17,218 production units and
reduces simulated lost demand by 15,304 versus uplift-only.

## Status and limitations

The architecture is validated as a research candidate, not production-ready.
The uplift and floor still depend on calibrated reconstructed-demand labels,
whose level may be aggressive. Probability calibration, rolling stability over
more folds, category/SKU economic tails, capacity, batch rounding, and a fully
causal assortment universe remain open requirements. Production and dev state
were not changed.

Artifacts:

- `scripts/build_direct_uplift_floor_candidates.py`
- `scripts/select_direct_adaptive_floor.py`
- `scripts/evaluate_direct_uplift_floor_economics.py`
- `reports/direct_uplift_floor_20260827/summary.json`
- `reports/direct_uplift_floor_20260827/selected_floor_summary.json`
- `reports/direct_uplift_floor_20260827/economics/summary.csv`

# Volume weighting and soft normalization (2026-08-27)

## Tested changes

Two proposed mechanisms were tested on the same expanding walk-forward contour:

1. Train Direct with `sqrt(1 + historical mean SKU volume)` sample weights,
   normalized to mean one and capped at ten.
2. Replace hard uplift normalization with soft normalization. Alpha is the share
   of expected-loss uplift that increases bakery volume; the remaining share is
   redistributed. `alpha=0` is full normalization and `alpha=1` is none.

Core SKUs are identified causally as the products covering the first 70% of each
bakery's trailing 56-day volume. After normalization, a Core forecast cannot
fall below its own Direct P50 baseline. The validated strict adaptive floor is
then applied.

## Volume-weighted training result

Volume-weighted Direct is rejected. On the calibrated evaluation folds, WAPE
worsens from 41.08% unweighted Direct P50 to 41.33%, underbake increases, and the
Bakery 29 / SKU 1071 forecast on August 23 rises from 246 to 275 before soft
uplift. Larger training weights do not solve the reallocation problem and make
the original incident less safe.

All recommended soft-normalization results therefore use the original unweighted
Direct model.

## Soft-normalization result

### Calibrated-demand operational metrics

| Candidate | WAPE | Surplus | Underbake | SKU 1071 WAPE/bias |
| --- | ---: | ---: | ---: | ---: |
| Actual state | - | 333,672 | 880,968 | - |
| Direct P50 | 41.08% | 333,682 | 782,440 | 25.38% / -6.61% |
| Previous final | 40.64% | 340,594 | 763,672 | 26.37% / -15.47% |
| Original Direct alpha=.25 + floor | **40.29%** | 407,649 | 686,849 | 25.32% / -6.24% |
| Original Direct alpha=.50 + floor | 40.50% | 481,967 | **618,233** | **25.33% / -2.92%** |
| Original Direct alpha=1 + floor | 42.36% | 650,682 | 500,169 | 27.54% / +8.24% |

Complete removal of normalization is rejected: it over-expands volume, worsens
WAPE and overcorrects SKU 1071. The useful region is alpha .25-.50.

Alpha .25 minimizes total imbalance and WAPE. Alpha .50 follows the stated
operational preference for lower underbake: versus actual state it reduces
underbake by 262,734 units (29.8%) at 148,295 additional surplus units (44.4%).

For Bakery 29 / SKU 1071 / August 23, alpha .25 remains exactly at Direct P50
246 against actual 161; alpha .50 is 252. Both remain far below incumbent 441,
and volume weighting is not used.

## Kazan FIFO economics across three evaluation folds

| Scenario | Previous final | Alpha .25 | Alpha .50 |
| --- | ---: | ---: | ---: |
| Conservative 50% | +3.02% | +3.76% | **+4.56%** |
| Calibrated 100% | +6.57% | +9.06% | **+11.43%** |
| Upper 150% | +8.35% | +12.80% | **+16.88%** |

In the calibrated scenario, alpha .50 produces 1.497m units versus 1.372m for
the previous final, reduces FIFO lost demand from 504,502 to 406,145, raises
strategy expiry from 19,804 to 25,239 and terminal carry from 52,555 to 73,496.
Its extra profit is therefore accompanied by a material capacity/carry increase.

## Tail and decision

Soft normalization still has a local concentration tail: bakery 244 / SKU 11018
on July 27 receives 124 against demand 32 and reaches a 23.3% bakery share.
Therefore neither alpha candidate is ready for shadow publication without an
explicit causal tail cap and capacity/rounding validation.

Recommended operating candidates:

- alpha .25 for balanced WAPE/imbalance and lower operational risk;
- alpha .50 when the explicit objective is minimum underbake and capacity can
  absorb the additional volume.

Given the user's stated priority of reducing underbake, alpha .50 is the leading
research candidate, with alpha .25 retained as the safe comparator. Production
and dev state were not changed.

Artifacts:

- `scripts/test_weighted_direct_soft_normalization.py`
- `scripts/evaluate_weighted_soft_normalization_economics.py`
- `reports/weighted_direct_soft_normalization_20260827/metrics.csv`
- `reports/weighted_direct_soft_normalization_20260827/economics/summary.csv`

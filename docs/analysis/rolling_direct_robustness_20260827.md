# Rolling robustness of Direct uplift and adaptive floor (2026-08-27)

## Scope and causality

Only four real historical forecast blocks are available in the snapshot archive:
July 22-26, July 27-August 2, August 11-13, and August 17/18/21-23. Missing
production snapshots were not synthesized. The first block starts floor
calibration; the following three are independent evaluation folds.

Direct and expected-loss uplift are retrained with expanding history before each
fold. Starting with fold two, floor parameters are selected only on prior folds.
Three reconstructed-demand scenarios are tested: 50%, 100%, and 150% of the
calibrated imputed loss.

## Aggregate evaluation across three independent folds

| Demand scenario | Candidate | WAPE | Surplus | Underbake | Recognized loss |
| --- | --- | ---: | ---: | ---: | ---: |
| Conservative 50% | Direct P50 | 33.37% | 360,848 | 413,468 | 123,284 |
| Conservative 50% | + uplift | 33.12% | 357,989 | 410,610 | 129,327 |
| Conservative 50% | + adaptive floor | **33.10%** | 361,837 | **406,214** | **132,183** |
| Calibrated 100% | Direct P50 | 41.08% | 333,682 | 782,440 | 150,450 |
| Calibrated 100% | + uplift | 40.86% | **330,731** | 779,489 | 164,159 |
| Calibrated 100% | + adaptive floor | **40.64%** | 340,594 | **763,672** | **175,815** |
| Upper 150% | Direct P50 | 48.07% | 325,832 | 1,170,728 | 158,299 |
| Upper 150% | + uplift | 47.90% | **323,080** | 1,167,976 | 180,790 |
| Upper 150% | + adaptive floor | **47.56%** | 334,835 | **1,145,800** | **197,847** |

Uplift improves WAPE, surplus and underbake at unchanged volume in every demand
scenario. Adaptive floor improves WAPE and underbake in every one of the nine
fold/scenario evaluations. The expanding selector chooses the same strict rule
for all later folds and scenarios: n>=8, stockout rate>=75%, mean imputed loss>=4,
0.8*P67, cap min(+5,+10%).

## Economic robustness

Kazan two-day FIFO gross-profit deltas versus actual state across the three
evaluation folds are:

| Scenario | Direct P50 | + uplift | + adaptive floor |
| --- | ---: | ---: | ---: |
| Conservative 50% | +1.589m (+2.04%) | +2.229m (+2.86%) | **+2.350m (+3.02%)** |
| Calibrated 100% | +3.234m (+4.14%) | +4.565m (+5.84%) | **+5.138m (+6.57%)** |
| Upper 150% | +3.724m (+4.76%) | +5.619m (+7.18%) | **+6.539m (+8.35%)** |

Every candidate is positive versus actual in every individual evaluation fold,
including the conservative scenario. The economic ranking is stable.

## Tail audit and blocker

Against Direct P50 in the calibrated scenario, the final candidate improves
absolute error for 71.8% of 188 bakeries, 68.0% of 25 categories, and 67.2% of
528 products. The aggregate improvement is therefore broad but not universal.

SKU 1071 is the largest negative product tail. Across evaluation folds:

- Direct P50: WAPE 25.38%, bias -6.61%, forecast 258,641 vs demand 276,935;
- uplift: WAPE 26.42%, bias -15.71%, forecast 233,432;
- final floor: WAPE 26.37%, bias -15.47%, forecast 234,103.

Expected-loss uplift reallocates 25,209 units away from SKU 1071; floor restores
only 670. This fixes earlier concentration incidents on some dates but creates
aggregate underforecast on others. Products 10346, 11573, 57 and 1076 are also
material negative tails.

## Decision

The architecture passes aggregate rolling and economic sensitivity checks but
is not ready for shadow deployment. The next experiment must add a causal
SKU-level uplift gate or shrinkage rule selected on prior folds, specifically
preventing expected-loss reallocation when it has historically worsened a
high-volume SKU. The gate must be tested without changing Direct itself or
reintroducing incumbent/category shares.

Production and dev state were not changed.

Artifacts:

- `scripts/rolling_validate_direct_uplift_floor.py`
- `scripts/evaluate_rolling_direct_robustness.py`
- `reports/rolling_direct_uplift_floor_20260827/aggregate_metrics.csv`
- `reports/rolling_direct_uplift_floor_20260827/floor_selections.csv`
- `reports/rolling_direct_uplift_floor_20260827/economic_summary.csv`
- `reports/rolling_direct_uplift_floor_20260827/tail_by_bakery.csv`
- `reports/rolling_direct_uplift_floor_20260827/tail_by_category.csv`
- `reports/rolling_direct_uplift_floor_20260827/tail_by_product.csv`

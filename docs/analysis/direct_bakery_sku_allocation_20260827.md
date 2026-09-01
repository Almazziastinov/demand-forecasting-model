# Direct bakery-day to SKU allocation (2026-08-27)

## Decision

The first clean daily allocation candidate removes the incumbent category layer:

`bakery-day volume -> predicted demand for every SKU -> bakery-day normalization`.

The incumbent forecast is used only for two inputs that are outside allocation:

- the bakery-day volume already produced by the bakery-level model;
- the assortment universe present in the historical forecast snapshot.

Incumbent SKU shares, incumbent category totals/shares, hourly profiles and old
uplift outputs are not model features and are not constraints. Category is only a
categorical feature. Category totals are outputs of the SKU forecasts.

## Model and causal design

- LightGBM Poisson regression predicts positive daily sales for each SKU.
- Features contain bakery, SKU, category and weekday identifiers, bakery-day
  volume context, causal 7/14/56-day quantities and shares, four same-weekday
  observations, recent presence, trend and historical category share.
- Every historical feature ends on the day before the prediction date.
- Frozen folds are identical to the earlier Predictive comparison:
  `2026-07-22..2026-08-02` trained through July 21, and eight selected current
  dates trained through August 10.
- Predicted SKU demands are normalized only within bakery-day. The maximum
  conservation error is `4.55e-13` units.

## Frozen-fold results

### Current fold: 1,406 bakery-days

| Candidate | SKU WAPE | MAE | Category WAPE | Bias |
| --- | ---: | ---: | ---: | ---: |
| Current allocation | 56.62% | 2.888 | 28.68% | -0.47% |
| Previous Predictive | 44.46% | 2.267 | 28.68% | -0.47% |
| Direct bakery -> SKU | **33.17%** | **1.692** | **13.20%** | -0.47% |

Direct allocation wins on 1,326/1,406 bakery-days against current allocation
and on 1,297/1,406 against the previous Predictive model.

### Earlier blocked fold: 2,154 bakery-days

| Candidate | SKU WAPE | Category WAPE | Bias |
| --- | ---: | ---: | ---: |
| Current allocation | 40.39% | 21.33% | +13.20% |
| Previous Predictive | 39.54% | 21.33% | +13.20% |
| Direct bakery -> SKU | **38.00%** | **19.82%** | +13.20% |

The improvement is smaller but remains positive on the temporally earlier fold.
All candidates retain the same bakery-day bias because the daily total is fixed.

## Forecast-shape checks

On the current fold, the maximum single-SKU bakery-day share falls from 49.1%
in the current system and 39.0% in previous Predictive to 16.1%. There are no
bakery-days above 20%, and no near-zero direct forecasts. Previous Predictive
places near-zero forecasts on 2,911 rows with positive actual sales; direct
allocation places none.

SKU 1071 current-fold WAPE changes from 60.87% current and 22.78% previous
Predictive to 15.82% direct. Its bias changes from +54.35% and +13.63% to -5.81%.
It is still the largest predicted SKU on 1,158/1,406 bakery-days, so network-wide
SKU leadership must be investigated separately from excessive concentration.

## Bakery 29, 2026-08-23

- SKU 1071: actual 161, current 440.98, previous Predictive 300.56, direct 221.43.
- Savory bakery category: actual 767, inherited old total 1,101.56, direct
  emergent total 856.31.

This confirms the earlier diagnosis: preserving the incumbent category total was
the main remaining source of the incident. Direct allocation materially corrects
both the category and SKU without a hard category layer, although SKU 1071 is
still 60 units above observed sales in this bakery-day.

## Limitations and next test

This is not a production candidate yet:

1. The target is observed sales, not reconstructed demand, so stockout-driven
   lost demand is not learned directly.
2. The assortment universe still comes from the historical forecast snapshot;
   a fully autonomous system needs a separate causal assortment policy.
3. The large improvement on the current fold must be repeated on more rolling
   folds and evaluated against operational underproduction/residual metrics.

The next controlled experiment should keep this exact architecture and compare
sales target versus conservatively reconstructed demand target. Bakery-day
volume policy and SKU allocation should remain separate so target changes do not
reintroduce old category or hourly allocations.

Artifacts:

- `scripts/backtest_direct_bakery_sku_allocation.py`
- `reports/direct_bakery_sku_allocation_20260827/predictions.parquet`
- `reports/direct_bakery_sku_allocation_20260827/summary.json`

No production or development database state was changed.

## Kazan two-day FIFO economics

The candidate was also evaluated in the same Kazan-only economic contour used
for the earlier allocation candidates: SKU prices and costs from the markup
workbook, 30% discount for yesterday's product, two-day FIFO inventory, opening
stock and transfers.

| Scenario | Gross profit | Delta vs actual | Service level | Lost demand |
| --- | ---: | ---: | ---: | ---: |
| Actual state | 102.187m | 0 | 69.10% | 740,187 |
| Current allocation | 93.632m | -8.555m | 65.25% | 832,182 |
| Previous Predictive P50 | 101.601m | -0.586m | 71.19% | 690,107 |
| Direct, same bakery volume | 100.627m | -1.560m | 66.76% | 796,066 |
| Direct +2% | 101.632m | -0.555m | 67.63% | 775,393 |
| Direct with P50 bakery volume | **106.374m** | **+4.186m** | **71.80%** | **675,362** |
| Previous Predictive floor | 111.641m | +9.454m | 78.43% | 516,532 |

At equal incumbent bakery volume, direct allocation improves gross profit by
6.995m relative to current allocation. With the same P50 volume policy, direct
allocation improves gross profit by 4.772m relative to previous Predictive P50,
serves 14,745 more reconstructed-demand units, and uses 3,022 fewer production
units.

The previous floor still has the highest simulated gross profit, but it also
produces 224,025 units more than direct P50 and depends most strongly on the
reconstructed-demand target that earlier validation found to be aggressive.
Therefore this ranking is not enough to select the floor for production. The
clean conclusion is narrower: after controlling for bakery volume, direct SKU
allocation materially improves the economic result.

Additional artifacts:

- `scripts/evaluate_direct_kazan_two_day_economics.py`
- `reports/direct_bakery_sku_allocation_20260827/economics/summary.csv`
- `reports/direct_bakery_sku_allocation_20260827/economics/by_category.csv`

# Joint demand-volume and SKU-allocation diagnostic — 2026-08-25

## Objective and definitions

This diagnostic combines a higher bakery forecast with predictive-choice SKU
allocation. It uses the eight completed current production dates and the
corrected observable universe: 1,406 bakery-days across 176 bakeries.

- Demand is `sold + conservative lost demand` (`strict_demand`).
- Recognized lost demand is
  `min(lost demand, max(forecast - sold, 0))`.
- True overforecast is `max(forecast - strict_demand, 0)`.

The current dates are used to inspect the volume frontier. Therefore the
uplift percentage selected below is diagnostic, not a frozen validation or a
production parameter.

## Bakery-level result

| Variant | Forecast | WAPE | Bias | Recognized lost | True overforecast |
|---|---:|---:|---:|---:|---:|
| Raw base | 1,329,668 | 7.50% | -4.26% | 13,343 (33.3%) | 22,487 |
| Current base_recent | 1,342,510 | 7.59% | -3.34% | 15,027 (37.5%) | 29,524 |
| +2% volume | 1,369,361 | **7.25%** | -1.41% | **18,827 (47.0%)** | 40,571 |
| +4% volume | 1,396,211 | 7.26% | +0.53% | 22,625 (56.5%) | 54,066 |

The strict-demand total is 1,388,881 units. A uniform +2% uplift closes more
than half of the current negative bias and gives the best bakery WAPE on the
tested frontier. A +4% uplift crosses into positive aggregate bias.

However, a uniform uplift necessarily increases bakery-level overforecast:
at +2%, true-overforecast bakery-days rise from 524 to 635. A deployable
volume correction therefore must be selective by bakery/day rather than a
single network multiplier.

## SKU-level joint result

| Variant | WAPE | Recognized lost | True overforecast qty | Overforecast rows |
|---|---:|---:|---:|---:|
| Current base_recent + incumbent allocation | 56.13% | 12,121 (30.3%) | 366,588 | 136,107 |
| Current volume + predictive allocation | **44.13%** | 13,375 (33.4%) | 283,272 | 132,089 |
| +2% volume + predictive allocation | 44.54% | **14,158 (35.4%)** | **299,556** | **133,945** |
| +4% volume + predictive allocation | 45.03% | 14,940 (37.3%) | 316,360 | 135,718 |

Relative to the incumbent, the +2% joint candidate:

- raises forecast volume by 26,850 units;
- recognizes 2,037 additional lost-demand units at SKU level;
- reduces true SKU overforecast by 67,032 units;
- removes 2,162 true-overforecast SKU rows;
- improves strict-demand SKU WAPE by 11.59 percentage points.

Thus the joint objective is achievable at SKU level: volume and recognized
lost demand rise while true overforecast falls because predictive allocation
removes much larger mix errors. It is not yet achieved simultaneously at the
bakery-day level because the tested volume uplift is uniform.

## Decision

Use +2% only as a diagnostic center point for the next blocked experiment.
An initial frozen split used 2026-08-11, 12, 13 and 17 for calibration and
2026-08-18, 21, 22 and 23 for testing. A smoothed bakery-specific residual
correction raised volume by 1.07% and recognized lost demand, but did not pass
the bakery overforecast gate: overforecast bakery-days increased from 251 to
274 and true-overforecast quantity from 15,098 to 18,566. The uniform +2%
candidate was better on bakery WAPE (7.96% versus 8.38%) but increased the
same risks further. Simple residual calibration is therefore rejected.

The next volume model must predict a bounded, bakery/day-specific uplift from
causal history and must be accepted only when it:

1. raises network volume and recognized lost demand;
2. improves strict-demand WAPE and bias;
3. does not increase bakery-level true-overforecast events;
4. retains the SKU-level gains after predictive allocation.

Artifacts:

- `scripts/backtest_joint_demand_allocation.py`
- `reports/joint_demand_allocation_20260825/metrics.csv`
- `reports/joint_demand_allocation_20260825/blocked_metrics.csv`
- `reports/joint_demand_allocation_20260825/predictions.parquet`
- `reports/joint_demand_allocation_20260825/summary.json`

Production writes: none.

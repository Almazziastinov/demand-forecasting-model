# Alpha 0.25 stateful operational simulation — 2026-08-31

> Superseded for model selection: the original run below applied an unsupported
> 1,200-unit daily capacity assumption. The bakery has no measured production
> limit in the available data. The primary result is now the unconstrained run
> in `reports/alpha25_stateful_operations_no_capacity_20260831/`. Capacity is an
> output to be justified economically, not an exogenous constraint.

## Corrected unconstrained result

| Candidate | Production | Surplus | Underbake | Imbalance | Service level |
| --- | ---: | ---: | ---: | ---: | ---: |
| Actual state | 1,954,926 | 408,112 | 839,984 | 1,248,095 | 69.08% |
| Current | 2,157,519 | 646,129 | 731,312 | 1,377,441 | 73.08% |
| Direct P50 | 2,311,246 | 669,667 | 570,926 | **1,240,593** | 78.99% |
| Previous final | 2,329,183 | 679,063 | 558,201 | **1,237,265** | 79.45% |
| Alpha 0.25 + tail cap | 2,412,525 | 762,056 | **499,476** | 1,261,532 | **81.62%** |

Corrected Kazan FIFO gross-profit delta versus actual for Current / Previous
final / Alpha 0.25 is `-4.76/+3.23/+3.26%` under conservative demand,
`+0.83/+11.29/+12.64%` under calibrated demand, and
`+3.50/+15.48/+18.46%` under upper demand.

The unconstrained Alpha 0.25 candidate therefore remains profitable even in
the conservative scenario. It defines a recommended production level; it is
not clipped by an unsupported operational ceiling.

## Final economic alpha selection

The unconstrained stateful simulation was expanded once to the frozen grid
`alpha = 0, 0.25, 0.50, 0.75, 1.00`. This is the final model-selection test
before integration.

| Alpha | Conservative profit vs actual | Calibrated profit vs actual | Calibrated underbake | Calibrated surplus |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | +3.23% | +11.04% | 555,306 | 675,944 |
| 0.25 | **+3.27%** | +12.65% | 499,396 | 762,207 |
| 0.50 | +3.08% | +13.49% | 449,101 | 853,589 |
| 0.75 | +2.75% | **+13.94%** | 402,302 | 949,735 |
| 1.00 | +2.32% | +13.89% | **360,960** | 1,055,282 |

Alpha 0.75 maximizes calibrated profit, but Alpha 0.25 maximizes conservative
profit and materially reduces underbake versus the incumbent. Alpha 0.25 is
therefore frozen as the robust integration candidate. The causal tail cap costs
only about 0.003 percentage points of calibrated profit and remains enabled as
protection against SKU-share outliers.

Frozen candidate:

`Direct bakery-day-to-SKU -> predictive expected-loss uplift -> Core-SKU
protection -> alpha=0.25 soft volume expansion -> adaptive floor -> causal tail
cap`.

## Purpose

This simulation removes the main limitation of the one-day operational replay:
each candidate now carries its own fresh remainder into the next day. The next
day's production need is computed from that simulated stock, not from the
historically observed stock.

No production or dev state was changed.

## Execution model

For every contiguous evaluation segment, bakery and SKU:

1. initialize carry from the causal opening stock on the first segment day;
2. compute net production need from forecast target, carry and transfers;
3. round bakeable production to the effective SKU multiple;
4. enforce the shared daily core-production screen of 1,200 units per bakery;
5. sell yesterday's stock first at a 30% discount, then fresh stock;
6. expire unsold yesterday stock and carry fresh remainder into the next day.

The operational metadata was loaded from the read-only local snapshot dated
2026-08-27 because ClickHouse TLS connection timed out. The output records this
source explicitly.

## Calibrated demand result

| Candidate | Production | Surplus | Underbake | Imbalance | Service level |
| --- | ---: | ---: | ---: | ---: | ---: |
| Actual state | 1,954,926 | 408,112 | 839,984 | 1,248,095 | 69.08% |
| Current | 2,114,831 | 633,227 | 763,849 | 1,397,075 | 71.88% |
| Direct P50 | 2,249,840 | 649,452 | 619,205 | 1,268,657 | 77.21% |
| Previous final | 2,269,156 | **657,293** | 605,263 | **1,262,556** | 77.72% |
| Alpha 0.25 + tail cap | 2,334,066 | 732,395 | **561,824** | 1,294,219 | **79.32%** |

Relative to Current, Alpha 0.25 removes 202,024 underbake units and improves
service level by 7.44 percentage points. Relative to Previous final, it removes
43,439 underbake units at the cost of 75,102 additional surplus units and
31,663 additional total imbalance.

Alpha 0.25 reduces underbake versus Previous final in every evaluation fold.
It does not minimize symmetric imbalance; this is a deliberate consequence of
the accepted business priority to minimize underbake first.

## Kazan FIFO economics

| Scenario | Current vs actual | Previous final vs actual | Alpha 0.25 vs actual |
| --- | ---: | ---: | ---: |
| Conservative | -6.82% | +0.04% | -0.40% |
| Calibrated | -1.59% | +7.38% | **+7.66%** |
| Upper | +0.93% | +11.26% | **+12.31%** |

For calibrated demand, Alpha 0.25 is positive in all three independent folds:
`+9.50%`, `+8.03%`, and `+5.05%` versus actual. It wins aggregate calibrated
profit over Previous final by about 220 thousand currency units, but the
conservative scenario exposes a small downside versus actual and Previous
final.

## Validation and decision

Validation passed:

- no negative production or carry;
- served quantity never exceeds demand;
- inventory conservation maximum numerical error is below `6e-14`;
- maximum modeled core production is exactly the 1,200-unit cap.

The capacity-constrained conclusion in this section is obsolete. The corrected
unconstrained candidate is suitable for a non-writing shadow run. Production
activation still requires current-horizon validation, but not an invented
capacity cap.

## Artifacts

- `scripts/simulate_alpha25_stateful_operations.py`
- `reports/alpha25_stateful_operations_20260831/daily_rows.parquet`
- `reports/alpha25_stateful_operations_20260831/summary.csv`
- `reports/alpha25_stateful_operations_20260831/economic_rows.parquet`
- `reports/alpha25_stateful_operations_20260831/economic_summary.csv`

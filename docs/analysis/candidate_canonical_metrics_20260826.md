# Canonical metrics and recognized lost demand (2026-08-26)

Current, P50 + Predictive, and P50 + Predictive + simple floor were evaluated
on the same 20 dates and 282,842 controlled SKU-days. The target is calibrated
demand: observed sales plus reconstructed post-last-sale loss.

## SKU-day metrics

| Variant | WAPE | MAE | RMSE | Bias | sMAPE | R2 |
|---|---:|---:|---:|---:|---:|---:|
| Current | 48.92% | 6.182 | 13.734 | -23.47% | 98.49% | 0.645 |
| P50 + Predictive | **44.81%** | **5.663** | 12.132 | -14.85% | 89.26% | 0.723 |
| P50 + Predictive + floor | 45.29% | 5.723 | **11.801** | **+0.06%** | **87.42%** | **0.738** |

P50 + Predictive is best on absolute SKU error (WAPE/MAE). Floor is almost
unbiased and improves large-error-sensitive RMSE and R2, but slightly worsens
WAPE because some added SKU volume lands on the wrong rows.

## Recognized reconstructed loss at SKU level

Recognized loss is `min(max(plan - observed sales, 0), reconstructed loss)`
per SKU-day.

| Variant | Recognized units | Coverage | Covered loss rows |
|---|---:|---:|---:|
| Current | 163,060 | 15.73% | 38.83% |
| P50 + Predictive | 230,685 | 22.25% | 48.80% |
| P50 + Predictive + floor | **387,117** | **37.34%** | **67.85%** |

Total reconstructed loss on scope is 1,036,832 units across 99,240 SKU-days.

## Bakery-day metrics

| Variant | WAPE | MAE | RMSE | Bias | sMAPE | R2 | Recognized loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current | 24.73% | 248.75 | 323.81 | -23.47% | 27.15% | 0.506 | 235,552 (22.72%) |
| P50 + Predictive | 17.64% | 177.36 | 241.18 | -14.85% | 18.69% | 0.726 | 467,550 (45.09%) |
| P50 + Predictive + floor | **12.51%** | **125.85** | **170.83** | **+0.06%** | **13.24%** | **0.862** | **814,892 (78.59%)** |

Bakery aggregation allows overforecast on one SKU to compensate underforecast
on another. Therefore bakery metrics describe total-volume accuracy, while
SKU metrics are authoritative for allocation quality and operational
recognized loss.

Artifacts:

- `scripts/evaluate_candidate_canonical_metrics.py`
- `reports/candidate_canonical_metrics_20260826/`

Production was unchanged.

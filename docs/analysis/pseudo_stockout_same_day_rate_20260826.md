# Same-day-rate pseudo-stockout validation (2026-08-26)

The relaxed lost-demand estimator was tested by taking SKU-days with sales
through at least 21:00, hiding all sales after synthetic cutoffs, and comparing
the capped same-day-rate estimate with the actual hidden tail. The production
rule was reproduced: opening 07:00, close 23:00, and cap min(10 units, 50% of
observed sales).

| Synthetic cutoff | Cases | True hidden | Predicted | Recovery | Bias |
|---|---:|---:|---:|---:|---:|
| 15:00 | 3,286 | 47,839 | 22,329 | 46.7% | -53.3% |
| 16:00 | 3,423 | 41,130 | 23,526 | 57.2% | -42.8% |
| 17:00 | 3,577 | 32,919 | 24,879 | 75.6% | -24.4% |
| 18:00 | 3,676 | 24,509 | 25,118 | 102.5% | +2.5% |

The estimator is approximately calibrated only near 18:00. It materially
underestimates earlier stockouts, primarily because one fixed cap suppresses
the longer missing tail. Therefore the current 191,866 recognized lost units
are a conservative lower bound, not a validated demand point estimate.

The selective floor still improves the chosen label, but its apparent win over
"observed underbake" cannot yet be treated as a real operational win. The next
label experiment should calibrate cap/rate by last-sale hour and validate on a
separate holdout before further floor optimization.

Cold-start is not the dominant remaining issue: rows with fewer than six
same-weekday observations contribute 31,898 of the selective candidate's
170,270 underbake, while rows with 8+ observations contribute 122,921.

Artifacts are in `reports/pseudo_stockout_same_day_rate_20260826/`.
Production was not changed.

# Selective SKU floor on calibrated demand (2026-08-26)

A causal same-weekday P67 SKU-demand floor was searched on top of calibrated
bakery quantiles and frozen Predictive allocation. Each reference uses only
the preceding 56 days. The eight operational dates were split chronologically
into four calibration and four untouched test dates.

The grid covered P50-P95, optional +2% bakery uplift, 3-8 required history
observations, floor scales 0.60-1.30, and per-row additive caps 2-100 units.

Calibration-selected candidates:

| Candidate | Total surplus | Total underbake | Test surplus | Test underbake |
|---|---:|---:|---:|---:|
| Balanced: P50, n>=7, scale .75, cap 10 | 263,550 | 388,110 | 115,820 | 208,397 |
| Center: P50, n>=6, scale .95, cap 8 | 324,338 | 336,729 | 145,633 | 181,535 |
| Under-first extreme: P95+2%, n>=3, scale 1.30, cap 100 | 798,235 | 156,419 | 377,668 | 83,309 |

On the test half, actual-state surplus/underbake is 90,712/229,282 and current
forecast underbake is 315,042. The balanced candidate reduces test underbake
by 20,885 versus actual and 106,645 versus current. The center candidate
reduces it by 47,747 versus actual and 133,507 versus current, at 54,920 more
surplus than actual. The extreme candidate confirms that the floor can drive
underbake much lower, but its surplus cost is operationally excessive.

Even the most aggressive tested candidate leaves 156,419 underbake across all
dates. A same-weekday floor cannot cover rows without sufficient history and
cannot perfectly anticipate day-specific SKU mix. Reaching zero would require
either indiscriminate volume or additional category/SKU priors, and should not
be treated as attainable from this mechanism alone.

The center candidate is the useful next research point because it honors the
underbake-first objective while avoiding the extreme candidate's 798k
surplus. No rollout is authorized.

Artifacts:

- `scripts/search_calibrated_selective_sku_floor.py`
- `reports/calibrated_selective_sku_floor_20260826/grid.csv`
- `reports/calibrated_selective_sku_floor_20260826/feasible_top100.csv`

Production was unchanged.

# Selective causal SKU floor (2026-08-26)

The aggregate P67 x0.70 floor was decomposed before promotion. Of its 95,349
added units, 61,642 (64.6%) reduced underbake and 33,707 became additional
surplus. Rows with 8+ same-weekday observations were 73.3% useful, while rows
with only 3-4 observations were only 35.6% useful.

The eight dates were split chronologically into four calibration and four
test dates. The minimum aggregate-passing x0.70 rule won only the first half
and failed all four later dates. It is rejected as unstable.

A selective rule was then calibrated to create a 20% underbake safety margin
on the first half:

- base plan: P85 + Predictive +2%;
- floor: same-weekday P67 demand;
- minimum observations: 6;
- scale: 0.83;
- maximum additive uplift: 15 units per SKU-day.

| Scope | Surplus | Underbake | Observed underbake | Pass |
|---|---:|---:|---:|---:|
| Calibration dates | 171,680 | 80,632 | 101,084 | yes |
| Test dates | 143,931 | 89,638 | 90,782 | yes |
| All eight dates | 315,612 | 170,270 | 191,866 | yes |

Full candidate volume is 1,373,874 and total imbalance is 485,882. It reduces
underbake by 239,082 units (58.4%) versus current and beats observed underbake
by 21,596 units. Relative to observed state, surplus is 144,230 units higher.

This is a frozen-split research result, not rollout evidence. More dates and
pseudo-stockout validation are still required. Production was not changed.

Artifacts are stored in `reports/sku_floor_decomposition_20260826/`.

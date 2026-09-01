# Calibrated selective floor decomposition (2026-08-26)

The selected center candidate (P50 + Predictive allocation + same-weekday P67
floor, n>=6, scale .95, cap 8) adds 265,893 units across 61,355 of 110,090
SKU-days. Of the uplift, 150,118 units reduce underbake and 115,775 become
additional surplus, for 56.46% marginal efficiency.

The result is stable across the frozen split: calibration efficiency is
54.21% and untouched test efficiency is 58.44%. Test uplift is 141,384 units,
of which 82,626 reduce underbake and 58,758 become surplus.

History depth matters. Rows with 8+ same-weekday observations have 58.76%
efficiency; rows with 6-7 observations have only 46.33%. The latter consume
49,242 uplift units but produce 26,426 surplus units. This is the clearest
guardrail opportunity.

Selected products:

| Product | Added | Useful | Surplus added | Efficiency | Remaining underbake |
|---|---:|---:|---:|---:|---:|
| 10340 Kystyby P | 5,265 | 3,229 | 2,036 | 61.32% | 21,247 |
| 1071 | 2,632 | 1,389 | 1,242 | 52.79% | 16,863 |
| 11474 Makovka | 8,263 | 4,261 | 4,002 | 51.57% | 4,441 |

Kystyby P is not a poor floor target: its uplift efficiency is above the
network average. Its large remaining underbake is not fixed by an 8-unit cap,
so the position needs a dedicated cap/scale test rather than exclusion.

The next candidate should keep the center rule for 8+ observations, tighten
or disable it for 6-7 observations, and test a larger product-specific cap
only for historically efficient high-underbake SKUs. Selection must use the
calibration half and report the second half untouched.

Artifacts are under
`reports/calibrated_selective_floor_decomposition_20260826/`. Production was
unchanged.

# Two-level selective SKU floor (2026-08-26)

The default center floor was tightened to require eight same-weekday
observations. Product-specific expanded caps were then selected using only the
first four calibration dates. A product had to receive at least 20 uplift
units, pass a calibration efficiency threshold, and retain enough underbake.
The last four dates remained untouched.

| Candidate | Total surplus | Total underbake | Test surplus | Test underbake |
|---|---:|---:|---:|---:|
| Standard n>=8, scale .95, cap 8 | 297,912 | 359,545 | 132,031 | 193,657 |
| Balanced two-level | 299,618 | 357,324 | 133,128 | 192,221 |
| Under-center two-level | 347,105 | 323,167 | 156,760 | 174,098 |
| Under-first two-level | 390,521 | 300,611 | 178,379 | 161,646 |

The under-center rule selects 49 products on calibration using efficiency
>=50% and remaining underbake >=200, then applies same-weekday P67 scale 1.05
with cap 15. On test it reduces underbake by 19,560 versus the standard n>=8
rule for 24,729 additional surplus. Compared with the earlier n>=6 center, it
reduces test underbake by 7,437 for 11,128 additional surplus; break-even is
about 1.50 units of surplus cost per unit of underbake cost.

Kystyby P (10340) is selected: calibration efficiency 57.03% and remaining
underbake 9,012. SKU 1071 (39.32%) and Makovka 11474 (48.50%) are excluded by
the frozen 50% efficiency threshold. This is evidence that product-specific
guardrails remove some waste instead of expanding every high-underbake SKU.

The under-center candidate matches the stated preference when underbake costs
more than roughly 1.5 times surplus. The under-first candidate buys another
22,556 reduction in underbake at 43,416 additional surplus and is less
efficient. No rollout is authorized; a longer rolling validation is required.

Artifacts are under `reports/two_level_selective_floor_20260826/`.
Production was unchanged.

# High-quantile and +2% operational grid (2026-08-26)

P75/P80/P85/P90/P95 were evaluated with frozen Predictive allocation, both
as-is and with a further 2% bakery-volume uplift. The common scope is eight
dates, 175 bakeries and 267 production SKUs. Candidate selection is
lexicographic: underbake first, then surplus.

| Candidate | Volume | Surplus | Underbake | Imbalance |
|---|---:|---:|---:|---:|
| Actual state | 1,178,537 | 171,382 | 191,866 | 363,248 |
| Current | 1,046,259 | 227,078 | 409,352 | 636,430 |
| Predictive | 1,051,099 | 155,382 | 332,816 | 488,199 |
| Predictive +2% | 1,072,121 | 165,874 | 322,286 | 488,160 |
| P75 | 1,188,173 | 231,722 | 272,082 | 503,804 |
| P75 +2% | 1,211,936 | 246,351 | 262,948 | 509,299 |
| P80 | 1,203,051 | 240,814 | 266,296 | 507,110 |
| P80 +2% | 1,227,112 | 255,886 | 257,307 | 513,193 |
| P85 | 1,218,994 | 250,847 | 260,386 | 511,234 |
| P85 +2% | 1,243,374 | 266,431 | 251,590 | 518,020 |
| P90 | 1,248,848 | 270,193 | 249,878 | 520,070 |
| P90 +2% | 1,273,825 | 286,730 | 241,438 | 528,168 |
| P95 | 1,296,098 | 302,795 | 235,230 | 538,025 |
| P95 +2% | 1,322,020 | 320,724 | **227,237** | 547,961 |

P95 +2% minimizes underbake among tested plans, reducing it by 182,115 units
(44.5%) versus current. It still does not reach the actual 191,866 level and
adds substantial surplus. Raising bakery volume has diminishing returns
because Predictive shares continue to place volume on the wrong SKUs. A
causal SKU-level protective floor is therefore the next required experiment.

Production was not changed. Metrics are stored in
`reports/network_quantile_high_grid_20260826/metrics.csv`.

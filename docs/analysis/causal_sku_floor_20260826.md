# Causal SKU protective floor (2026-08-26)

A causal SKU floor was evaluated on top of `P85 + Predictive +2%`. For each
forecast date and bakery-SKU, the floor uses only positive-demand observations
from the preceding 56 days on the same weekday. At least three observations
are required. The final plan is `max(base plan, scaled historical quantile)`;
floor volume is additive and is not taken from other SKUs.

The P85 bakery-day totals have an oracle lower bound of 54,772 underbake units,
showing that some bakery-days remain volume-constrained even with perfect SKU
placement. The causal floor can add volume beyond those totals.

The first stable candidate that beats observed underbake is the same-weekday
P67 floor at scale 0.70:

| State/plan | Volume | Surplus | Underbake | Imbalance |
|---|---:|---:|---:|---:|
| Actual state | 1,178,537 | 171,382 | 191,866 | 363,248 |
| Current forecast | 1,046,259 | 227,078 | 409,352 | 636,430 |
| P85 + Predictive +2% | 1,243,374 | 266,431 | 251,590 | 518,020 |
| P85 +2% + P67 floor x0.70 | 1,338,723 | 300,138 | **189,948** | 490,086 |

The candidate reduces underbake by 219,404 units (53.6%) versus current and
beats observed underbake by 1,918 units. It adds 95,349 units versus the P85
base and has 128,756 more surplus than the observed state. A scale of 0.69
only beats observed underbake by seven units and is too fragile.

Aggressive floors cannot reach zero underbake: rows with fewer than three
same-weekday observations leave an asymptotic residual of roughly 19 thousand
units even while volume and surplus become economically unacceptable. Those
rows require a separate cold-start/category-prior floor.

This is research-only. Production was not changed. Metrics are in
`reports/sku_floor_grid_20260826/metrics.csv` and `refined_metrics.csv`.

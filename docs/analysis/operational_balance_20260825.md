# Operational balance gate — 2026-08-25

## Contract and scope

The current eight completed dates were restricted to 267 products present in
the production-release source, 176 observable bakeries and 110,674 SKU rows.

Because opening inventory is unavailable, observed surplus is an operational
proxy: `max(same-day production - sales, 0)`. Observed underbake is
conservative lost demand. Candidate-plan surplus and underbake are evaluated
against strict demand. This is a useful gate but not a full inventory-flow
simulation; opening and ending stock must be added when available.

## Result

| State or plan | Volume | Surplus | Underbake | Total imbalance |
|---|---:|---:|---:|---:|
| Observed production state | 1,074,186 | **142,200** | **40,035** | **182,235** |
| Incumbent forecast plan | 1,049,963 | 253,316 | 284,335 | 537,651 |
| Predictive allocation plan | 1,053,605 | 182,764 | 210,141 | 392,906 |
| Predictive +2% plan | 1,074,677 | 195,400 | 201,704 | 397,104 |

Predictive allocation removes 144,746 units of imbalance versus the incumbent
forecast plan, but remains 210,671 units worse than the observed production
state. Adding 2% reduces projected underbake but increases surplus by more,
so total imbalance worsens by 4,198 units versus predictive allocation alone.

## Asymmetric business priority

Surplus and underbake must not be treated as equally costly. Predictive
allocation dominates the incumbent forecast plan: it reduces both projected
surplus by 70,552 units and underbake by 74,194 units.

Relative to predictive allocation alone, the +2% plan trades 12,635 additional
surplus units for 8,437 fewer underbake units. It is preferable whenever one
unit of underbake costs more than `1.50` units of surplus. At a 2:1 business
weight, +2% is therefore better than the unchanged-volume predictive plan.

The direct-demand and selective bakery experiments show the same tradeoff.
Versus the sales-target model, direct demand reduces mean underforecast by 516
units for 261 additional overforecast units (break-even weight `0.51`), while
selective reduces it by 859 units for 548 additional overforecast units
(break-even `0.64`). If underbake is more expensive than surplus, both are
economically preferable in those limited folds.

## Decision

The primary gate is now underbake, with surplus reported as the cost of the
improvement. On the available proxy, no forecast candidate yet beats observed
underbake (`40,035`): predictive has `210,141` and predictive +2% has
`201,704`. Thus the forecast candidates still do not beat the observed state,
although predictive +2% is preferable to predictive alone when the underbake
weight is above 1.50.

The next data requirement is reliable opening and ending SKU inventory. Until
then, this proxy must be reported explicitly and cannot be described as actual
ending leftovers.

Artifacts:

- `scripts/evaluate_operational_balance.py`
- `reports/operational_balance_20260825/metrics.csv`
- `reports/operational_balance_20260825/rows.parquet`
- `reports/operational_balance_20260825/summary.json`

Production writes: none.

# Causal economic floor gate (2026-08-27)

Research-only walk-forward test on the clean Kazan two-day FIFO scope.

## Rule

The base plan is P50+Predictive. For each future uninterrupted block, floor is
enabled by product only from economic results observed in earlier blocks.
SKU incremental profit per extra floor unit is shrunk toward its category with
a 50-unit prior; the category rate is shrunk toward the network rate with a
200-unit prior. Floor is enabled when the resulting expected incremental
profit is positive. These constants and the zero-profit threshold were fixed
before the evaluation run.

The first block is warm-up and uses P50. Blocks 2-4 are the honest evaluation
window. Prices/costs are product-specific and yesterday sales are discounted
30%. Opening-stock expiry and terminal carry remain separately identified.

## Walk-forward result, blocks 2-4

| Variant | Production | Served | Lost | Strategy expiry | Terminal carry | Gross profit | Delta vs actual |
|---|---:|---:|---:|---:|---:|---:|---:|
| Actual state | 722,939 | 694,223 | 338,537 | 7,423 | 43,928 | 43.408m | - |
| Current | 651,574 | 614,323 | 418,437 | 14,061 | 45,300 | 35.370m | -8.039m |
| P50+Predictive | 767,662 | 719,291 | 313,469 | 16,622 | 53,863 | 41.389m | -2.020m |
| Universal floor | 869,844 | 797,489 | 235,271 | 22,241 | 72,230 | 46.958m | +3.549m |
| Causal economic gate | 869,208 | 797,099 | 235,661 | 22,162 | 72,062 | 46.949m | +3.541m |

The gate blocks 19, 7, and 7 observed floor-increment products in the three
evaluation blocks. It reduces production by 636 units, strategy expiry by 79,
and terminal carry by 167 versus universal floor, but also serves 390 fewer
units and loses 8.2k gross profit. It trails universal floor in each evaluation
block by 5.6k, 2.2k, and 0.5k respectively.

## Interpretation

The tested gate does not add predictive economic value. With only one prior
block at the first decision and strong positive aggregate floor economics,
shrinkage enables floor for almost every product. Products that were
unprofitable in the warm-up often became profitable later, so a static
product-level past-profit sign is not stable enough. Universal floor remains
the better tested candidate on this limited horizon, but this is not yet enough
for automation: the evaluation contains only three short future blocks and
terminal carry is not liquidated beyond each block.

Production state was not changed. Outputs are in
`reports/causal_economic_floor_gate_20260827/`.

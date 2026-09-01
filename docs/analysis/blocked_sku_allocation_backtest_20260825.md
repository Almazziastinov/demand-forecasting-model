# Blocked SKU allocation validation — 2026-08-25

## Design

The causal-trend formula and blend coefficients selected on the August
`base_norm_recent` dates were frozen and evaluated without retuning on the
earlier 2026-07-17..2026-08-02 production period.

- 17 completed dates;
- 184 observable bakeries;
- 3,072 evaluable bakery-days;
- 556,924 SKU rows;
- 464 forecast-only bakery-days excluded as DQ, not synthetic zero demand;
- every method preserves the incumbent bakery/category total.

This period uses the historical `base_bakery_raw_uplift_sku` production
scenario, so it validates allocation robustness rather than the current
bakery-total model.

## Observed-sales results

| Method | SKU WAPE | Delta vs incumbent | Better bakery-days | Better dates |
|---|---:|---:|---:|---:|
| Incumbent | 40.4570% | — | — | — |
| Same weekday | 42.2699% | +1.8129 pp | 468 / 3,072 | 0 / 17 |
| Full causal trend | 40.9225% | +0.4656 pp | 1,125 / 3,072 | 1 / 17 |
| Blend 25% | **40.2778%** | **-0.1792 pp** | **2,007 / 3,072** | **17 / 17** |
| Blend 50% | 40.2926% | -0.1644 pp | 1,716 / 3,072 | 16 / 17 |
| Blend 75% | 40.5075% | +0.0505 pp | 1,427 / 3,072 | 8 / 17 |

The large August gain from full causal allocation does not generalize. The
25% blend does generalize, but its aggregate gain is deliberately modest.

All methods retain the same +15.0550% inherited total bias. Maximum
category-total conservation error is below `2.3e-13` units.

## Concentration

The blocked period has no incumbent bakery-day at or above 20% top-SKU share.
Blend 25% preserves this behavior: p95 top share changes from 13.5222% to
13.4743%, with zero cases at or above 20%.

This confirms that the extreme concentration defect is scenario/time
dependent. The blend does not introduce it in the clean blocked period.

## Decision

- Reject full causal-trend replacement as the rollout candidate.
- Retain the 25% blend as the conservative shadow candidate.
- Do not claim the 0.1792 pp blocked gain as sufficient for canary.
- Rebuild forecast-conditioned predictive choice with explicit production
  runs, non-oracle totals and the same observability rules before final model
  selection.
- Run the 25% blend prospectively and require both SKU-WAPE improvement and
  concentration reduction before canary.

Artifacts:

- `scripts/backtest_blocked_current_sku_allocation.py`
- `reports/blocked_sku_allocation_backtest_20260825/summary.json`
- `reports/blocked_sku_allocation_backtest_20260825/predictions.parquet`

Production writes: none.

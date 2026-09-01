# Current-scenario SKU allocation backtest — 2026-08-25

## Scope and information boundary

- Production dates: 2026-08-11, 12, 13, 17, 18, 21, 22 and 23.
- One explicit `base_norm_recent` production run per date.
- 176 observable bakeries, 1,406 bakery-days and 264,481 forecast SKU rows.
- 304 forecast-only bakery-days and 158,267 forecast units are excluded from
  quality metrics and retained as a separate DQ population.
- Historical allocation inputs use sales strictly before each forecast date.
- Every challenger preserves the incumbent `date × bakery × category` total.

## Methods

- `incumbent`: current production SKU allocation.
- `same_weekday`: shares from the previous four matching weekdays.
- `causal_trend`: 56-day causal fallback adjusted by the clipped ratio of the
  latest seven-day SKU share to the preceding seven-day share.
- `blend_25/50/75`: incumbent blended with causal-trend allocation.

The causal-trend method is a reproducible regime-aware proxy. It is not the
older forecast-conditioned LightGBM choice model, whose current-date feature
builder is not present in this working tree.

## Strict-demand results

| Method | SKU WAPE | Allocation WAPE | Better bakery-days |
|---|---:|---:|---:|
| Incumbent | 56.1277% | 48.1153% | — |
| Same weekday | 45.3244% | 34.2975% | 1,075 / 1,406 |
| Causal trend | **43.7352%** | **31.6819%** | 1,233 / 1,406 |
| Blend 25% | 51.3018% | 42.1652% | **1,333 / 1,406** |
| Blend 50% | 47.3575% | 36.9630% | 1,306 / 1,406 |
| Blend 75% | 44.7114% | 33.0905% | 1,270 / 1,406 |

Causal trend improves aggregate SKU WAPE by 12.3924 pp and wins on every
evaluated date. Blend 25% is the most conservative option by bakery-day
stability, but captures less of the aggregate gain.

All methods have identical aggregate bias because category totals are fixed.
Maximum category-total conservation error is below `2.3e-13` units.

## Concentration

| Method | p95 top SKU share | >=20% | >=30% | >=40% |
|---|---:|---:|---:|---:|
| Incumbent | 30.36% | 438 | 73 | 10 |
| Same weekday | 18.59% | 55 | 15 | 2 |
| Causal trend | **18.52%** | **55** | **15** | **1** |
| Blend 25% | 27.00% | 301 | 39 | 4 |
| Blend 50% | 24.03% | 171 | 25 | 2 |
| Blend 75% | 20.75% | 87 | 17 | 1 |

This confirms that the current production concentration incident exists on
observable bakery-days. Daily causal allocation substantially reduces it.

## SKU 1071

| Method | WAPE | Bias | Forecast | Strict demand |
|---|---:|---:|---:|---:|
| Incumbent | 60.27% | +53.38% | 196,520 | 128,124 |
| Same weekday | **22.08%** | +10.95% | 142,148 | 128,124 |
| Causal trend | 22.53% | +13.48% | 145,397 | 128,124 |

The current allocation materially overstates SKU 1071. Both daily variants
remove most of that excess without changing category totals.

## Bakery 29 on 2026-08-23

- Incumbent SKU WAPE: 89.47%.
- Same-weekday SKU WAPE: 65.12%.
- Causal-trend SKU WAPE: 64.93%.
- Bakery total is unchanged at 1,779.65 units for every method.

The incident improves materially but is not fully solved by the simple daily
challengers.

## Decision

1. Do not replace production. The subsequent blocked fold rejected full
   causal-trend replacement.
2. Carry only the 25% blend into prospective shadow: it improved blocked WAPE
   from 40.4570% to 40.2778%, won all 17 dates and improved 2,007 of 3,072
   bakery-days.
3. Reconstruct the older forecast-conditioned predictive-choice model on the
   same current inputs before selecting the final challenger.
4. Apply hourly scheduling only after SKU-day totals are fixed, with exact
   daily conservation.
5. Add observability and concentration gates before any canary.

Artifacts:

- `scripts/backtest_current_sku_allocation.py`
- `reports/current_sku_allocation_backtest_20260825/summary.json`
- `reports/current_sku_allocation_backtest_20260825/metrics_by_date.csv`
- `reports/current_sku_allocation_backtest_20260825/predictions.parquet`

Production writes: none.

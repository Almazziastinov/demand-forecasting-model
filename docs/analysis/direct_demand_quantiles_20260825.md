# Direct-demand quantile comparison — 2026-08-25

## Design

Direct-demand LightGBM models were trained on `sales + conservative lost
demand` at P50, P55, P60, P67 and P75. The same three causal 14-day folds and
11 labeled pilot bakeries were used as in the direct-demand/selective test.
All quantile variants use a fixed 180-tree configuration.

## Mean results

| Variant | Forecast | Demand bias | WAPE | Recognized lost | Underbake | Overforecast | Break-even weight |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sales target | 173,518 | -1.95% | **6.02%** | 37.41% | 7,041 | **3,619** | — |
| P50 | 175,665 | -0.73% | 6.15% | 45.81% | 6,083 | 4,808 | 1.24 |
| P55 | 177,580 | +0.35% | 6.30% | 52.07% | 5,252 | 5,893 | 1.27 |
| P60 | 179,580 | +1.48% | 6.61% | 55.77% | 4,533 | 7,173 | 1.42 |
| P67 | 182,216 | +2.97% | 7.02% | 63.31% | 3,576 | 8,852 | 1.51 |
| P75 | 185,438 | +4.79% | 7.75% | 70.92% | **2,616** | 11,114 | 1.69 |

Break-even weight is the underbake-to-surplus cost ratio above which the
variant is preferable to the sales-target baseline. Every quantile reduces
underbake on all three folds. WAPE improves on only one of three folds for
P50-P67 and none for P75, confirming that WAPE and the operational objective
select different forecasts.

Relative to observed sales, aggregate bias is +0.29% for P50, +1.38% for P55,
+2.53% for P60, +4.03% for P67 and +5.87% for P75. P50 therefore already
passes the aggregate `forecast >= sales` requirement. P55 is slightly above
the desired 0..+1% sales-bias band but is close.

## Business-weight selection

- Equal underbake/surplus weight: sales-target baseline remains best.
- Underbake weight 1.5: P55 has the lowest weighted operational loss.
- Underbake weight 2.0: P67 is best.
- Underbake weight 3.0: P75 is best.

The earlier P60 hypothesis is rejected for a 1.5 weight. P55 is the empirical
center candidate, while P50 is the conservative candidate. Choosing between
them must use a business-approved weight and a wider label universe; it must
not be tuned again on these outer folds.

Artifacts:

- `scripts/compare_direct_demand_quantiles.py`
- `reports/direct_demand_quantiles_20260825/summary.csv`
- `reports/direct_demand_quantiles_20260825/metrics.csv`
- `reports/direct_demand_quantiles_20260825/predictions.parquet`

Production writes: none.

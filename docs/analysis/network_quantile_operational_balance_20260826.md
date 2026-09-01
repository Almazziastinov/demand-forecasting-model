# Network quantile operational balance (2026-08-26)

The bakery-day dataset was extended continuously through 2026-08-23 from
deduplicated `fct_check_lines`. Relaxed stockout labels were rebuilt for the
network using the agreed availability and same-day-rate rule. Quantile models
were trained with information through 2026-08-10 and evaluated on the same
eight completed dates used by the operational-balance study.

The common scope contains eight dates, 175 bakeries, 267 production SKUs and
110,090 SKU-days. Bakery 277 was excluded because it has no training history
and therefore no quantile prediction on six evaluation dates.

All P50-P75 bakery totals were allocated with the frozen forecast-conditioned
Predictive allocation shares. The incumbent and Predictive +2% rows retain
their original definitions.

| State/plan | Volume | Surplus | Underbake | Total imbalance |
|---|---:|---:|---:|---:|
| Actual available-to-sell state | 1,178,537 | 171,382 | 191,866 | 363,248 |
| Current forecast | 1,046,259 | 227,078 | 409,352 | 636,430 |
| Predictive allocation | 1,051,099 | 155,382 | 332,816 | 488,199 |
| Predictive +2% | 1,072,121 | 165,874 | 322,286 | **488,160** |
| P50 + Predictive | 1,114,341 | 188,786 | 302,977 | 491,763 |
| P55 + Predictive | 1,129,245 | 196,908 | 296,196 | 493,103 |
| P60 + Predictive | 1,140,822 | 203,565 | 291,276 | 494,841 |
| P67 + Predictive | 1,162,823 | 216,446 | 282,156 | 498,602 |
| P75 + Predictive | 1,188,173 | 231,722 | **272,082** | 503,804 |

P75 is only 9,636 units above actual available volume, but higher volume does
not solve the SKU-placement error: its underbake is still 80,216 units above
the observed state. With equal unit costs, Predictive +2% has the lowest
forecast imbalance, narrowly ahead of Predictive alone. With underbake cost
1.5 times surplus, P67 is best; at weight 2.0, P75 is best.

The relaxed detector estimates much more lost demand than the previous
conservative label. Its absolute level still requires pseudo-stockout
validation. This comparison is research-only; production was not changed.

Artifacts:

- `reports/network_quantiles_20260826/predictions.parquet`
- `reports/network_quantile_operational_balance_20260826/metrics.csv`
- `reports/network_quantile_operational_balance_20260826/rows.parquet`

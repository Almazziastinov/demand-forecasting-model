# Available-to-sell operational balance (2026-08-26)

The observed-state row was rebuilt on the same eight dates, 176 bakeries and
267 products as the prior forecast-plan comparison.

## Contract

- opening stock is the positive closing balance from the previous day;
- total available to sell = production + opening stock + received - sent;
- observed surplus = max(total available to sell - observed sales, 0);
- underbake = conservative recognized lost demand;
- forecast volume is evaluated as-is against strict demand.

All fact components were read from deduplicated `fct_production_release`,
`fct_moves`, and `fct_check_lines`. The incomplete mart view was rejected: it
covered only 91 of the 267 products and understated sales by 67,072 units on
the intersecting rows.

## Result

| State/plan | Volume | Surplus | Underbake | Total imbalance |
|---|---:|---:|---:|---:|
| Actual available-to-sell state | 1,183,823 | **172,466** | **40,035** | **212,501** |
| Current forecast | 1,049,963 | 253,316 | 284,335 | 537,651 |
| Predictive allocation | 1,053,605 | 182,764 | 210,141 | 392,906 |
| Predictive +2% | 1,074,677 | 195,400 | 201,704 | 397,104 |

The forecast rows are unchanged from the previous comparison. The corrected
actual row is higher because it now includes prior-day stock and net incoming
movements. Predictive allocation still improves substantially on the current
forecast, but neither predictive candidate matches the actual underbake.

Artifacts:

- `scripts/evaluate_available_to_sell_balance.py`
- `reports/available_to_sell_balance_20260826/metrics.csv`
- `reports/available_to_sell_balance_20260826/rows.parquet`
- `reports/available_to_sell_balance_20260826/summary.json`

No production state was changed.

# Calibrated post-last-sale target: operational comparison (2026-08-26)

The network post-last-sale label was rebuilt with cutoff-hour coefficients
fitted on August 1-10 and validated on August 11-23. Total reconstructed loss
in the historical label increased from 2.774 million under the fixed cap to
6.212 million units.

Bakery quantile models were trained through August 10 and evaluated on the
same eight dates, 175 bakeries and 267 production SKUs as the earlier
operational comparison. Existing Predictive SKU allocation was kept frozen.

| Variant | Volume | Surplus | Underbake | Imbalance |
|---|---:|---:|---:|---:|
| Actual state | 1,178,537 | 171,371 | 468,732 | 640,103 |
| Current | 1,046,259 | 197,038 | 626,677 | 823,714 |
| Predictive | 1,051,099 | 140,446 | 565,245 | 705,691 |
| Predictive +2% | 1,072,121 | 149,074 | 552,851 | 701,925 |
| P50 | 1,197,614 | 208,563 | 486,847 | 695,411 |
| P67 | 1,260,464 | 241,614 | 457,048 | 698,661 |
| P85 +2% | 1,379,405 | 310,911 | 407,404 | 718,315 |
| P95 +2% | 1,462,792 | 364,416 | 377,522 | 741,938 |

P50 has the lowest equal-cost imbalance. P67 is the first tested quantile to
beat actual-state underbake, by 11,684 units, while adding 70,243 surplus.
P95 +2% reduces underbake by 249,155 versus current and by 91,210 versus the
actual state, but adds 193,045 surplus versus actual. Bakery volume alone
still cannot eliminate underbake because the residual is SKU-placement error.

This comparison is research-only. The coefficients are based on ten
calibration dates and require a longer temporal validation before being used
as production labels.

Artifacts:

- `scripts/apply_post_last_sale_calibration.py`
- `scripts/run_calibrated_network_quantiles.py`
- `scripts/evaluate_calibrated_quantile_balance.py`
- `reports/calibrated_stockout_network_20260826/sku_day_demand.csv`
- `reports/calibrated_network_quantiles_20260826/predictions.parquet`
- `reports/calibrated_quantile_operational_balance_20260826/metrics.csv`

Production was unchanged.

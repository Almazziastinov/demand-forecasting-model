# Post-last-sale demand calibration (2026-08-26)

The same-day-rate estimator was calibrated by synthetic stockouts on network
SKU-days whose actual last sale was at or after 21:00. Sales after fixed
cutoffs from 07:00 through 18:00 were hidden. August 1-10 was used only to fit
one multiplier per cutoff hour; August 11-23 was a frozen holdout.

The current `min(raw rate, 10 units, 50% observed sales)` cap is strongly
downward biased for early stockouts. On holdout it recovered only 4.5% at
07:00, 14.7% at 10:00, 42.1% at 15:00, and 69.6% at 17:00. The calibrated
rate recovered 83.9%, 90.1%, 96.8%, and 99.3% respectively. At 18:00 it
recovered 103.6%.

The fitted raw-rate multipliers range from 0.139 at 07:00 to about 0.69 at
15:00-16:00 and 0.601 at 18:00. This removes most aggregate bias, but
case-level WAPE remains 46-64%; it is a label reconstruction, not an accurate
SKU-day forecast. Very early cutoffs retain 8-15% aggregate under-recovery on
holdout, so they should remain conservative until validated on more dates.

Artifacts:

- `scripts/calibrate_post_last_sale_demand.py`
- `reports/post_last_sale_calibration_20260826/calibration_coefficients.csv`
- `reports/post_last_sale_calibration_20260826/holdout_metrics.csv`
- `reports/post_last_sale_calibration_20260826/cases.parquet`

Production was not changed. The next research step is to rebuild the
post-last-sale demand label with frozen hour coefficients and rerun the
bakery-volume and SKU-allocation operational comparison.

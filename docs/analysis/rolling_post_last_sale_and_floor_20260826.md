# Rolling validation of calibrated loss and simple SKU floor (2026-08-26)

Hourly network sales were extracted read-only for June 1-August 23: 6.289m
hourly SKU rows, 175 bakeries and 669 products. Nine weekly pseudo-stockout
folds used the preceding 21 days to fit cutoff-hour coefficients and the next
seven days as test.

Aggregate weekly recovery ranged from 94.04% to 102.19%. Mean recovery by
cutoff is 97.69%-99.77%; 07:00 is least stable, with a minimum fold recovery
of 82.38%. This supports hour calibration as an aggregate label mechanism,
while confirming that individual SKU-day reconstruction remains noisy.

The model comparison used four causal folds and 20 saved Predictive allocation
forecast dates from July 22-August 23. Models were trained only through each
fold cutoff. This is a rolling one-day-ahead style evaluation: later dates in
each fold use observed intervening lag history, not a fixed 14-day recursive
horizon.

| Variant | Volume | Surplus | Underbake | Imbalance |
|---|---:|---:|---:|---:|
| Current | 3,451,467 | 714,389 | 1,510,091 | 2,224,480 |
| Predictive, same volume | 3,451,467 | 638,293 | 1,433,995 | **2,072,288** |
| P50 + Predictive | 3,828,563 | 831,826 | 1,250,431 | 2,082,257 |
| P50 + Predictive + simple floor | 4,392,380 | 1,123,840 | **978,629** | 2,102,469 |

Predictive allocation improves both surplus and underbake at unchanged volume
on every aggregate comparison and remains the allocation baseline. P50 trades
193,532 additional surplus for 183,564 less underbake versus same-volume
Predictive; break-even underbake weight is about 1.05. The simple floor trades
292,014 additional surplus for 271,802 less underbake versus P50; break-even is
about 1.07. It reduces underbake on every fold but worsens equal-cost imbalance
on the first two folds and improves it on the last two.

Thus floor is justified under the stated underbake-first objective, but not
under equal unit costs. The product-specific two-level rule is not included
because the available saved forecasts do not provide a clean pre-fold
calibration window for selecting product lists in every fold. It must be
validated prospectively or after reconstructing additional historical
Predictive forecasts.

Artifacts:

- `scripts/extract_hourly_sales_for_rolling_backtest.py`
- `scripts/rolling_validate_post_last_sale_calibration.py`
- `scripts/rolling_backtest_floor_vs_no_floor.py`
- `reports/rolling_post_last_sale_calibration_20260826/`
- `reports/rolling_floor_vs_no_floor_20260826/`

Production was unchanged.

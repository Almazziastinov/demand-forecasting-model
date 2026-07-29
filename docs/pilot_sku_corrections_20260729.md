# Pilot SKU corrections deployed on 2026-07-29

## Scope

The daily Bitrix24 baking-plan publisher applies SKU corrections only to the
10-pilot-bakery set:

`20, 21, 22, 28, 80, 89, 107, 221, 222, 257`.

Bakery 16 (Kulagina 4) is not part of this rollout. The correction changes the
SKU mix inside each bakery/category, but preserves the original
`date x bakery x category` forecast total. It does not write corrected values
back to production forecast snapshot tables.

Processing order:

1. new-SKU forecast cold start;
2. mature-SKU systematic correction;
3. previous-day stock subtraction;
4. kratnost rounding;
5. calculation of total stock available for sale.

## Forecast cold start

Products `11573` and `11574` have sales history from 2026-05-30, but forecast
coverage appeared much later. They therefore use their own bakery/product
sales rather than a product analogue.

The floor is an exponentially weighted moving average of sales:

- alpha: `0.90`;
- minimum history: 3 sales days;
- information boundary: strictly before the forecast date;
- corrected forecast: `max(base forecast, sales EWMA floor)`;
- missed-demand estimates are not used in this layer;
- cold start ends after 13 positive-forecast days.

Starting with the 14th positive-forecast day, the pair can enter the mature
correction layer if it meets the mature eligibility gates. The two mechanisms
cannot apply to the same pair on the same date.

## Mature-SKU correction

The demand target is sales plus a conservative missed-demand estimate when the
produced quantity sold out before closing. Registry eligibility requires:

- 49-day history;
- at least 28 observed days;
- at least 14 positive-forecast days;
- age of at least 28 days;
- at least 150 demand units;
- absolute aggregate bias of at least 15%;
- error directionality of at least 40%;
- same-direction seven-day bias of at least 10%.

The full multiplier is `demand / forecast`. It has no hard coefficient bounds.
An adaptive smoothing strength in `[0.10, 0.30]` uses directionality, recent
bias, history length, volume, and repeated missed-demand evidence. The applied
multiplier is geometrically smoothed:

`applied_multiplier = full_multiplier ** smoothing`.

The minimum-positive-forecast-days gate was added after a real false positive:
product 4944 at bakery 257 had long sales history but only one forecast day.
Without the gate, historical zero forecasts created a false `-97.5%` bias and
an applied coefficient of `3.03`.

## Backtest

The combined walk-forward test uses only information available before each
forecast date and covers 28 days through 2026-07-28.

| Metric | Baseline | Corrected | Delta |
| --- | ---: | ---: | ---: |
| Total WAPE | 25.7551% | 25.0720% | -0.6831 pp |
| Underforecast quantity | 19,545.46 | 18,973.43 | -572.04 |
| Overforecast quantity | 23,589.05 | 23,017.01 | -572.04 |
| New-SKU WAPE | 95.0597% | 57.4101% | -37.6496 pp |

Category-total preservation keeps total forecast quantity and aggregate bias
unchanged.

## Runtime and deployment

Runtime files:

- `/opt/scripts/publish_pilot_forecast.py`;
- `/opt/src/experiments_v2/sku_cold_start.py`;
- `/opt/src/experiments_v2/sku_systematic_correction.py`.

Server:
`82bb03a8-c356-4225-97a4-a1540cdc29e6`.

Timer:

- unit: `pilot-forecast-publish.timer`;
- schedule: `03:00 UTC` / `06:00 MSK`;
- state after deployment: enabled and active.

Remote dry-run for 2026-07-30:

- 18 bakery/SKU cold-start floors;
- 426 changed rows after both correction layers;
- 535 final rows across 10 bakeries;
- generated workbook size: 28,739 bytes;
- Bitrix24 send skipped.

Rollback publisher:

`/opt/scripts/publish_pilot_forecast.py.backup_20260729_sku_corrections`.

The old publisher does not import the added modules, so restoring that file and
removing or leaving the unused modules both disable the correction.

## Verification

```bash
python -m pytest \
  tests/test_sku_cold_start.py \
  tests/test_sku_systematic_correction.py \
  tests/test_publish_pilot_forecast.py -v

python scripts/backtest_combined_sku_corrections.py
```

Generated local artifacts:

- `reports/combined_sku_correction_backtest/summary.json`;
- `reports/combined_sku_correction_backtest/backtest_rows.csv`;
- `reports/combined_sku_correction_backtest/cold_registry_history.csv`;
- `reports/combined_sku_correction_backtest/mature_registry_history.csv`.

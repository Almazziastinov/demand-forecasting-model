# Pilot forecast bias by stockout status

Date: 2026-07-20

## Question

Compare forecast displacement from observed sales in two auditable groups:

1. clear stockout SKU-days;
2. confirmed non-stockout SKU-days.

WMAPE is intentionally not used. On a stockout day, observed sales are
censored by produced quantity and do not represent full demand. Positive
forecast bias in that group is therefore interpreted as headroom over realized
sales, not automatically as forecast error.

## Data contract

- Grain: date, bakery, product.
- Forecast: latest available lead-1 row by `generated_at` from
  `sku_forecast_day_snapshots`.
- Forecast dates: 2026-06-01 through 2026-07-19 (49 days).
- Clear stockout: reliable inventory stockout, at least three normal reference
  days, SKU ends at least two hours early, and the bakery sells at least 50
  units after the SKU disappears.
- Confirmed non-stockout: consistent inventory balance, hourly/daily sales
  agreement, and no inventory stockout.
- Ambiguous rows are excluded.
- Dates without any lead-1 forecast run are excluded. A missing SKU inside a
  covered forecast date is retained as a zero forecast because the model did
  not publish that SKU.

## Aggregate result

| Group | SKU-days | Forecast coverage | Sales | Forecast | Aggregate bias | Mean bias per SKU-day | Forecast below observed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Clear stockout | 1,296 | 96.37% | 19,614 | 21,828.65 | +11.29% | +1.71 | 37.4% |
| Confirmed non-stockout | 6,419 | 91.93% | 84,760 | 102,797.15 | +21.28% | +2.81 | 20.9% |

When restricted to published forecast pairs, aggregate bias is +13.12% for
clear stockouts and +22.91% for confirmed non-stockouts. Missing forecasts are
therefore not the reason for the gap.

The model has about ten percentage points more headroom on non-stockout days
than on clear stockout days. This is the opposite of the desired allocation:
the extra forecast is concentrated more strongly where supply was already
sufficient.

In 485 of 1,296 clear stockout SKU-days, forecast quantity was below already
observed sales. These are guaranteed underforecast cases even before estimating
lost demand.

## Stability by period

| Period | Group | SKU-days | Aggregate bias |
| --- | --- | ---: | ---: |
| 2026-06 | Clear stockout | 776 | +12.16% |
| 2026-06 | Confirmed non-stockout | 3,911 | +23.03% |
| 2026-07-01..15 | Clear stockout | 401 | +7.33% |
| 2026-07-01..15 | Confirmed non-stockout | 1,955 | +16.99% |
| 2026-07-16..19 | Clear stockout | 119 | +16.54% |
| 2026-07-16..19 | Confirmed non-stockout | 553 | +24.89% |

The absolute bias changes with forecast versions and dates, so this period
split is descriptive rather than causal. The relative gap persists in every
period: clear-stockout bias is 8 to 11 percentage points lower.

## Bakery pattern

All 11 bakeries have positive aggregate bias in both groups. Clear-stockout
bias is lower than non-stockout bias in 10 of 11 bakeries. Bakery 28 is the only
near-tie (+15.95% stockout versus +16.60% non-stockout). The largest overall
positive biases are at bakery 257 (+36.82% stockout and +57.36% non-stockout),
but the same allocation gap remains.

## Manual review

The generated report directory contains:

- `manual_cases.csv`: largest positive and negative bias cases in each group;
- `sku_day_comparison.csv`: all 7,715 eligible SKU-days;
- `by_bakery.csv`, `by_category.csv`, and `by_date.csv`: statistical slices.

Priority manual cases are clear stockouts where forecast is below observed
sales. Examples include:

- 2026-07-19, bakery 89, `Элеш с курицей`: sold 30, forecast 22.31;
- 2026-07-16, bakery 89, `Беккен капуста`: sold 50, forecast 42.28;
- 2026-07-15, bakery 222, `Киш курица`: sold 20, forecast 11.70;
- 2026-06-26, bakery 28, `Капуста и курица`: sold 8, no published forecast.

For each selected case, inspect the hourly sales curve, last sale hour, normal
last sale hour, produced quantity, closing stock, and bakery activity after the
SKU disappeared.

## Interpretation and next step

The first statistical pass supports the stockout hypothesis: current forecasts
are generally high versus sales, but the relative uplift is weaker precisely on
clear stockout SKU-days. A global forecast reduction would therefore worsen
the allocation problem.

The next step is manual review of a balanced sample:

1. 20 clear stockouts where forecast is below observed sales;
2. 20 clear stockouts with the largest positive headroom;
3. 20 confirmed non-stockouts with the largest positive bias.

The review should decide whether the issue is primarily insufficient SKU-day
quantity, incorrect hourly timing, or assortment omission. No production
change should be made from the aggregate bias table alone.

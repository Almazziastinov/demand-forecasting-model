# Matched stockout forecast-bias analysis

Date: 2026-07-20

## Goal

Test whether the lower forecast headroom on stockout days remains after
controlling for the main selection effects. Each clear stockout SKU-day is
matched to up to three confirmed non-stockout days with:

- the same bakery;
- the same product;
- the same weekday;
- date distance no greater than 28 days;
- produced quantity within 25% of the stockout day's production.

Observed sales on stockout days remain censored, so the analysis compares
forecast headroom rather than treating sales as true demand.

## Main result

- Clear stockout cases available: 1,296.
- Matched cases: 733 (56.56% coverage).
- Median forecast / sales on stockout days: **1.025**.
- Median forecast / sales on matched non-stockout days: **1.508**.
- Median within-case difference: **-0.416**.
- Stockout ratio is below its matched control in **79.40%** of cases.
- Forecast is below already observed sales in 326 matched cases (44.47%).
- Forecast is not above produced quantity in 349 cases (47.61%).
- Median forecast minus produced quantity is only +0.18 units.

The matched comparison therefore confirms that the aggregate difference is not
explained only by comparing different products, weekdays, bakeries, or release
sizes. For the same SKU under similar production, the forecast has materially
less headroom on days when the SKU actually runs out.

## Sensitivity

| Date window | Production tolerance | Matched cases | Stockout ratio below control | Median stockout ratio | Median control ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 14 days | 25% | 542 | 79.15% | 1.015 | 1.473 |
| 28 days | 15% | 635 | 81.10% | 1.023 | 1.540 |
| 28 days | 25% | 733 | 79.40% | 1.025 | 1.508 |
| 28 days | 50% | 945 | 72.59% | 1.047 | 1.479 |

The direction is stable across all tested matching parameters. Tighter
production matching strengthens the result while reducing coverage.

## Interpretation

Selection still matters: unusually strong demand makes stockout more likely.
However, the matched result shows a practical allocation problem beyond the
raw aggregate averages. Current forecasts preserve large buffers on comparable
non-stockout days but are close to realized sales and production on stockout
days.

This is consistent with one or more of the following:

- historical sales for frequently stocked-out products are censored;
- SKU rolling means and caps inherit the censored level;
- bakery-level volume is allocated toward products that already have excess
  supply;
- some stockout-prone products are absent from the forecast assortment;
- the hourly stockout correction does not create enough SKU-day quantity.

The matched comparison does not by itself identify which mechanism dominates.

## Manual review sample

`reports/pilot_stockout_matched_bias/manual_review_cases.csv` contains three
balanced buckets:

1. 20 stockout cases with forecast furthest below observed sales;
2. 20 cases with the largest forecast/sales gap versus matched controls;
3. 20 cases with the largest positive stockout headroom.

High-priority guaranteed-underforecast examples include:

- 2026-06-12, bakery 221, `Треугольник курица безд`: sold 140, forecast 53.17;
- 2026-07-11, bakery 107, `Треугольник курица безд`: sold 119, forecast 87.78;
- 2026-06-01, bakery 22, `Элеш с курицей`: sold 61, forecast 35.30;
- 2026-07-16, bakery 16, `Пирожок яблоко`: sold 50, forecast 37.40;
- bakery 257 has several clear stockouts with no published SKU forecast.

## Decision

The stockout/non-stockout bias gap is real enough to proceed to manual case
review. Do not globally increase or decrease bakery forecasts from this result.
The next decision must be made at SKU-day/hour level after identifying whether
each failure comes from daily quantity, hourly timing, or assortment coverage.

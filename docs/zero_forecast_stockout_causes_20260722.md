# Zero-forecast stockout causes — 2026-07-22

## Result

The 47 clear-stockout SKU-days without a forecast are not primarily an
allocation-share failure.  They split as follows:

- 46 cases were outside `assortment_city_products` on the event date;
- 1 case was in the effective assortment, was bakeable, and is present in the
  current SKU profile, but was absent from every stored forecast snapshot for
  that bakery/SKU/date.

The 46 assortment exclusions cover 8 bakeries and 11 SKUs.  The largest group
is `Капуста и курица`: 18 cases in 7 bakeries and 79 observed sold units.

The remaining grid-drop case is:

- 2026-07-19, bakery 257 (`Ярмарочная 12 Чебоксары`);
- product 4944 (`Пирожок капуста и курица`);
- 15 observed sold units;
- selected run `prod_base_bakery_raw_uplift_sku_20260719_h14`.

A direct query found no snapshot for bakery 257 / product 4944 / 2026-07-19
in any stored run.  The failure is therefore upstream of SKU-share allocation
inside the completed forecast grid.

## Interpretation

The assortment refresh remains the highest-value fix.  A SKU that sells but is
absent from the effective assortment never reaches allocation, regardless of
the quality of its learned share.  The one residual case needs a trace through
forecast-grid construction after assortment and bakeability filters.

`sku_hour_share_profile_smoothed_embedded` is not versioned.  Its presence is
reported only as current-state diagnostic evidence and is not used to claim
what the profile contained on the historical event date.

Artifacts:

- `reports/zero_forecast_stockout_causes/cases.csv`
- `reports/zero_forecast_stockout_causes/summary.json`

All ClickHouse work was read-only.  Production was not changed.

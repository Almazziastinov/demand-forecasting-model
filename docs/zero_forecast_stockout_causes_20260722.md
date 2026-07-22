# Zero-forecast stockout causes — 2026-07-22

## Result

The 47 clear-stockout SKU-days without a forecast are not an allocation-share
failure. All 47 were outside the latest `assortment_city_products` batch that
was actually available before the corresponding historical forecast run.

The 46 assortment exclusions cover 8 bakeries and 11 SKUs.  The largest group
is `Капуста и курица`: 18 cases in 7 bakeries and 79 observed sold units.

An earlier version of this report classified one row as a forecast-grid drop.
That was lookahead: the relevant assortment row had `valid_from=2026-07-19`
but was loaded only on 2026-07-20, after the historical run. The corrected
analysis uses both effective dates and `loaded_at <= run_generated_at`, and
selects only the latest city batch exactly as production allocation does.

## Interpretation

The root cause was the stale allocation allowlist documented in
`docs/allocation_refresh_fix_20260720.md`: daily refresh updated
`bakeable_products`, while forecast allocation still read an old
`assortment_city_products` batch. The failed weekly profile refresh on 19 July
was a concurrent freshness problem, but it is not needed to explain these 47
zero-forecast cases because the assortment filter rejects them first.

The repair deployed on 20 July refreshed allocation assortment daily and added
a two-day freshness guard. A read-only verification against
`prod_base_bakery_raw_uplift_sku_20260722_h14` found all 18 affected
bakery/SKU pairs present on all 14 forecast days (18/18 pairs, zero missing).

`sku_hour_share_profile_smoothed_embedded` is not versioned.  Its presence is
reported only as current-state diagnostic evidence and is not used to claim
what the profile contained on the historical event date.

Artifacts:

- `reports/zero_forecast_stockout_causes/cases.csv`
- `reports/zero_forecast_stockout_causes/summary.json`

All ClickHouse work in this investigation was read-only. Production was not
changed in this session.

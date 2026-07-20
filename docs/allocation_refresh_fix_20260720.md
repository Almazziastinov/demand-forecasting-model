# Allocation refresh fix — 2026-07-20

## Root cause

The daily production refresh successfully rebuilt `bakeable_products`, while
forecast allocation continued to read `assortment_city_products`. The former
was current through 2026-07-19; the latter had not received a new batch since
2026-06-30. The stale allocation allowlist removed established products even
though their profile rows and sales history were present.

The separate weekly SKU-profile refresh also failed on 2026-07-19 during the
June export window after a transient ClickHouse connection refusal. The
profile table therefore remained at its 2026-07-14 build.

## Implementation

- The daily dataset refresh now writes a full, all-category city/SKU batch to
  `assortment_city_products` from the same trailing seven-day sales extract.
- Cities absent from the current `mart_sales_60d` window retain their latest
  known assortment in an explicitly marked carry-forward batch. This currently
  applies to Bugulma and Novokuznetsk; it prevents a city-wide forecast outage
  without pretending that fresh sales were available.
- The bakeable city/bakery layers continue to be written separately to
  `bakeable_products`; this table remains the baking-plan allowlist and is not
  substituted for the full forecast assortment.
- Allocation now reads only the latest assortment batch effective on the
  forecast date instead of unioning every historical active batch.
- Production inference rejects an assortment batch older than two days.
- Production inference checks `weekly_profile_refresh_last_run.json` and
  rejects a SKU profile whose data endpoint is more than eight days old.
- Monthly ClickHouse export windows retry transient query failures three times
  without discarding already exported earlier windows in the same run.

## Validation

The proposed fresh full assortment contains 1,972 city/SKU rows across nine
cities and 641 SKUs. Applied to the current active forecast rows as a read-only
screen:

| Allowlist | Forecast rows retained | Forecast quantity removed |
| --- | ---: | ---: |
| Latest old batch | 431,764 | 19,563.4 |
| Fresh seven-day batch | 436,748 | 6,582.5 |

All 12 control pairs previously removed by the stale allowlist are retained by
the fresh batch. This includes the seven established products at bakery 257
and `Капуста и курица` in five Kazan pilot bakeries.

A direct switch to `bakeable_products` was explicitly rejected during offline
validation: it would also remove bread, drinks, cakes, and other legitimate
forecast categories. Keeping separate full-allocation and baking-plan
assortments avoids that regression.

## Deployment boundary

Deployed to the production writer VM on 2026-07-20. The daily refresh inserted
2,190 allocation-assortment rows effective 2026-07-19. The weekly profile was
rebuilt through 2026-07-19 and loaded 3,537,105 profile rows.

The weekly uplift loader was also corrected to replace only the requested
`profile_version`, rather than truncating every version. Production version
`pilots_evening_20260716` remained present with 27,155 rows; new version
`weekly_20260720` was added with 27,154 rows. A `--skip-export` recovery mode
allows a completed raw export to be reused after a safe interruption.

Final production inference activated
`prod_base_bakery_raw_uplift_sku_20260720_h14` at 2026-07-20 19:03 MSK with
489,130 SKU-day and 5,499,898 SKU-hour snapshot rows. `verify_prod_deploy`
ended with `VERIFY OK`. All 12 control bakery/SKU pairs have non-zero forecasts
on all 14 horizon days. Both `forecast-production.timer` and
`weekly-profile-refresh.timer` are enabled and active.

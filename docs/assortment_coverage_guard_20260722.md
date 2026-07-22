# Assortment coverage guard — 2026-07-22

## Purpose

Prevent a repeat of the stale allocation-assortment failure that produced 47
clear-stockout SKU-days without forecasts. The guard runs after selection and
freshness validation of the assortment batch, but before SKU profile streaming,
allocation, snapshot loading, or run activation.

## Rule

For each bakery in the forecast run, read the previous seven calendar days of
positive sales. A bakery/SKU pair is considered established enough to block the
run when it sold:

- on at least two distinct days; and
- at least two units in total.

Every such pair must be covered by either its city assortment or its explicit
bakery-scoped assortment. Missing pairs are written to
`assortment_coverage_guard.csv`; the summary is written to
`assortment_coverage_guard.json`; then allocation raises `RuntimeError` before
anything is published.

One-day/new-product observations remain diagnostic and do not block the run,
preserving the existing cold-start policy.

## Read-only validation

Validation against `prod_base_bakery_raw_uplift_sku_20260722_h14`:

- bakeries: 211;
- recent bakery/SKU pairs inspected: 29,578;
- active city/SKU assortment pairs: 2,583;
- blocking missing pairs: 0.

The current dataset therefore passes the proposed guard. Production was not
changed or redeployed.

An emergency CLI bypass exists as `--disable-assortment-coverage-guard`, but
normal automated runs leave the guard enabled.

## Publication-boundary test

An isolated orchestration test intentionally raises the same error as a
corrupted-assortment guard. It confirms that `load_forecast_run` is never
called. A paired positive test confirms that a passing allocation reaches
`load_forecast_run` while `activate_run` remains disabled. Together with the
pair-level tests, this verifies both detection and the publication boundary
without writing to ClickHouse.

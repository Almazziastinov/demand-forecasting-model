# Automatic assortment policy

Date: 2026-08-26

## Approved operating model

The primary assortment source is automatic: positive bakery/SKU sales in the
seven complete days before the forecast date. Human-maintained assortment is
not part of the normal flow.

An effective-dated `force_include` or `force_exclude` record is permitted only
as a temporary, audited emergency override. Every override requires an end
date, reason and author; after expiry the result returns to the automatic
seven-day policy.

## Local implementation

- `src/experiments_v2/effective_assortment.py` builds the automatic result,
  applies temporary overrides and diagnoses missing `baking_sku_meta` rows.
- `assortment_emergency_overrides` is an append-only ClickHouse control table.
  `scripts/manage_assortment_override.py` is dry-run by default and requires
  the explicit `--apply` flag before it writes an exception.
- `src/experiments_v2/sku_cold_start.py` no longer limits discovery to product
  ids 11573 and 11574. It can also seed a zero-forecast row for a new SKU that
  is already present in the effective assortment before applying a
  category-neutral floor.
- The production runner's local default allocation table is now
  `bakery_product_assortment_embedded`. The refresh expands the automatic
  city+bakery layers, applies active temporary overrides, and writes the same
  effective pairs consumed by SKU allocation and the baking plan. An explicit
  environment value still takes precedence.
- The pilot publisher now discovers cold-start candidates across products,
  loads missing catalogue and baking metadata, and can add an absent forecast
  row before applying a category-neutral floor.

## Bakery 270 regression contract

For forecast date 2026-08-20, a sale on 2026-08-10 is outside the seven-day
window and does not retain product 11573. Sales on 2026-08-19 include products
11575 and 11615 automatically. A missing `baking_sku_meta` record is reported
as `missing_baking_sku_meta` rather than silently losing the SKU.

## Deployment boundary

This is a local code and contract change only. No production table, active
run, timer or publisher was changed. Before rollout, run a full network dry-run,
verify the VM runtime is compatible, back up the affected tables, deploy as a
single unit and retain the previous active forecast until verification passes.

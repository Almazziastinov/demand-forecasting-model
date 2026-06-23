# Session Handoff - 2026-06-23 - Dynamic Assortment from Forecast

## Summary

Replaced the static OCR-based assortment source with a fully dynamic pipeline
that derives `assortment_city_products` and `bakeable_products` directly from
the active ClickHouse forecast run.

Previously the assortment was built from OCR director files and manual partner
markups. Now it reflects exactly the SKU positions forecasted with `forecast_qty > 0`
in the current active run — meaning the recent-sales filter
(`runner_city_prior_soft_weekpart`) automatically handles exclusions: any SKU
with no recent sales gets zeroed in the forecast and therefore never appears
in the assortment or baking plan.

## New / Changed Files

### `scripts/build_city_assortment_from_forecast.py` (NEW)

Replaces the OCR-based assortment builder. Queries ClickHouse for the active
run and returns DISTINCT `(city, product_id, product_name, category_name)` from
`sku_forecast_day_embedded` WHERE `forecast_qty > 0`, joined to
`bakery_forecast_day_embedded` to get city.

Key decisions:
- No 80%-bakeries threshold — niche single-location SKUs are valid and must not
  be excluded at this stage.
- Inactive products filtered via `NOT startsWith(product_name, 'я_не_исп')` and
  `NOT startsWith(product_name, 'я не исп')` directly in SQL.
- `product_id` cast to `String` (`toString(s.product_id)`) to match the
  `assortment_city_products` schema.
- `product_name` and `category_name` filled with `""` after query because
  `sku_forecast_day_embedded` has them as `Nullable(String)` but
  `assortment_city_products` requires non-nullable `String`.

Output: same CSV schema as the old script — fully compatible with
`load_city_assortment_to_clickhouse.py`.

### `scripts/build_bakeable_products_table.py` (UPDATED)

Added category-filter mode as the new default. A product is bakeable if its
`category_name` contains any of the default patterns (case-insensitive
substring match):

```python
DEFAULT_BAKEABLE_CATEGORY_PATTERNS = ["пирог", "выпечка", "фастфуд"]
```

This covers: Пироги сытные, Пироги сладкие, Выпечка сытная, Выпечка сладкая,
Фастфуд. Хлеб and Пирожные are explicitly excluded (not in patterns).

Legacy `--markup-xlsx` mode is kept for backward compatibility.

Result on active run: 624 bakeable rows out of 2012 total assortment positions.

## Pipeline (run in order after each new active run)

```powershell
# Step 1 — build full assortment CSV from forecast
.venv\Scripts\python.exe scripts\build_city_assortment_from_forecast.py

# Step 2 — filter bakeable categories
.venv\Scripts\python.exe scripts\build_bakeable_products_table.py

# Step 3 — load assortment to ClickHouse
.venv\Scripts\python.exe scripts\load_city_assortment_to_clickhouse.py --replace-current

# Step 4 — load bakeable allowlist to ClickHouse
.venv\Scripts\python.exe scripts\load_bakeable_products_to_clickhouse.py --replace-current
```

Steps 3 and 4 use `--replace-current` which deletes rows matching the same
`source` + `valid_from` before inserting, so re-runs are idempotent.

## Production State After This Session

Active run: `prod_weatherfix2_uplifted_bakery_norm_uplift_sku_20260623_h14`

ClickHouse tables updated (prod):
- `assortment_city_products`: 2012 rows inserted (10 cities, 773 unique products)
- `assortment_source_audit`: 2012 rows inserted
- `bakeable_products`: 624 rows inserted (5 categories: Выпечка сытная,
  Выпечка сладкая, Пироги сытные, Пироги сладкие, Фастфуд)

## How the Excel Baking Plan Uses This

`apps/forecast_embedded/app/services/bakery.py::get_bakeable_products(city, date)`
queries `bakeable_products` filtered by city and `max(valid_from) <= effective_date`.

The result is passed to `baking_plan.build_baking_plan_workbook()` as
`assortment_rows`, which acts as an allowlist — only products present in this
list appear in the Excel. Products from the template not in the allowlist are
skipped; products in the allowlist not in the template are appended as extra rows.

Because `bakeable_products` is derived from the forecast (which already has
the recent filter applied), the Excel plan automatically reflects only positions
with recent sales activity in bakeable categories.

## Notes

- Pipeline is currently manual — must be re-run after each `activate_run`.
- `load_bakeable_products_to_clickhouse.py` already existed; no changes needed.
- The `fillna("")` fix for `product_name`/`category_name` in
  `build_assortment_table()` was critical — without it ClickHouse rejected the
  insert with `DataError: Invalid None value in non-Nullable column`.

# Baking Plan Implementation

Business rules and algorithm design for `apps/baking_plan/`. Reverted from
the MILP allocator (built 2026-07-09..11, redesigned 2026-07-13) back to a
template-driven approach on 2026-07-14 for the pilot launch — see
`docs/ops/DECISIONS.md` for the decision and its rationale, and
`docs/ops/CURRENT_STATE.md` for deploy details. This file replaces the
MILP-era `docs/baking_plan_implementation.md`.

## Definition

План выпекания = an Excel file listing each bakery's currently-assorted
SKUs and the baking-window quantities assigned to them. Window
*assignment* — which of the day's baking windows a given SKU bakes in — is
not computed by any algorithm; it comes directly from the reference Excel
template, whose C:L cells a technologist pre-filled for each SKU row.
Quantities are recomputed every request from the **live** hourly forecast
(`sku_forecast_hour_embedded`), not copied statically from the template —
only the window *assignment* is static, the *amount* is dynamic.

## Demand input

`apps/baking_plan/service.py` orchestrates a single request:

1. `demand.load_revenue_bucket_input(bakery_id)` — latest known monthly
   revenue → `templates.revenue_bucket()` → one of
   `до 1,5 млн` / `до 2,5 млн` / `от 2,5 млн` / `от 3млн`.
2. `templates.template_path_for_bakery(bakery_id)` — an individual
   override (`assets/individual/{bakery_id}_*.xlsx`) takes priority over
   the base template entirely; otherwise the base template's sheet is
   selected by the revenue bucket (`templates.select_sheet_name`, falls
   back to `DEFAULT_BUCKET` or the first sheet if the bucket doesn't match
   any sheet name — individual templates typically have one custom sheet
   with a bakery-specific label, not one of the four standard names, and
   rely on this fallback).
3. `templates.parse_comments_sheet` — SKU → `{dough_group, kratnost,
   station, is_two_day, is_on_demand}` from the "комментарии" sheet
   (parsed before other sheets are dropped).
4. `templates.parse_windows` — window boundaries from row 5 of the
   selected sheet (`4:00-7:00`, `7:00-8:00`, ...).
5. `assortment.get_bakeable_products(city, forecast_date, bakery_id)` —
   the current bakeable assortment (city + bakery scope, 80% threshold —
   unchanged since the MILP era, see `docs/ops/DATA_CONTRACTS.md`).
6. `demand.load_hourly` — hourly forecast for every assortment product_id,
   for both `forecast_date` and `forecast_date + 1` (the latter only used
   for SKUs whose template row has a defrost cell).

Gap-handling:

- Template row whose SKU doesn't resolve to a current assortment product
  (`allocation.resolve_assortment_product`, fuzzy name match via
  `allocation.sku_match_keys`) → row skipped entirely.
- Template row with no non-empty C:L cells → skipped (nothing to
  schedule).
- Assortment product with no matching template row → still included, at
  the end of its category, with all window cells empty and `Итого` =
  the SKU's raw full-day forecast sum (not kratnost-rounded). This is the
  pilot's intentional simplification for new/rarely-sold SKUs the
  reference template was never updated for — see "Known limitations"
  below.

## SKU metadata ("комментарии" sheet, parsed per request)

- **Кратность** (`kratnost`) — production multiple. Used as
  `allocate_template_row`'s `round_to` when the SKU is found in
  "комментарии"; otherwise `allocation.schedule_round_to` derives it from
  the GCD of the template's own pre-filled window quantities for that row.
- **Тесто-группа** (`dough_group`), **`is_two_day`**, **`is_on_demand`**,
  **`station`** ("Стол") — parsed but not currently consumed by the
  allocation logic itself (дефрост/двухдневка placement is read from the
  template's own pre-filled cells, not derived from these flags — see
  below). Kept for future use and because `station` is copied into the
  rendered plan's "Стол" column via row-snapshot restore.

## Windows

Window boundaries come from row 5 of the selected sheet
(`templates.WINDOWS_HEADER_ROW`); each revenue-tier sheet can have a
different window set (fewer windows for lower-revenue bakeries). Data rows
start at row 6 (`templates.PLAN_START_ROW`).

## Дефрост

A template cell is classified as defrost prep by its **text content**
(`allocation.is_defrost_cell` — contains "дефр" or "ночн"), not by the SKU
name or any comментарии flag — SKU names frequently contain "ночного
брожжения" while still having plain integer bake cells elsewhere in the
same row. Defrost quantity = tomorrow's forecast summed over hours 6–11
(`allocation.DEFROST_EARLY_CUTOFF = 12`), rounded to the row's kratnost,
falling back to today's early hours if no next-day forecast exists. No
PDF-derived cap (unlike the MILP era's `NIGHT_STORAGE_DIRECT_UNITS_BY_SKU`)
— this is a deliberate pilot simplification, see `docs/ops/DECISIONS.md`.
Defrost columns never contribute to the same row's regular-window coverage.

## Regular-window allocation (`allocation.allocate_template_row`)

For each non-defrost column in the row's schedule (read via
`allocation.read_row_schedule`, in window-start order): sum the live
hourly forecast over the window's `coverage_hours` (the sales hours it's
responsible for — from the previous window's end hour, or
`FIRST_SALES_HOUR` for the first window, through the hour before the next
window's end, or `LAST_SALES_HOUR` for the last window), subtract any
carried-over surplus from the previous window, round up to the row's
`round_to`, and carry the new surplus forward. A SKU scheduled in only one
window still absorbs the whole day's demand (coverage spans the full sales
day), not just the hours nominally "near" that window.

## Rendering (`apps/baking_plan/rendering.py`)

Unlike the MILP era (which built a fresh `Workbook()` every call), rendering
**mutates the loaded template sheet in place**: each matched row's original
cell styles are snapshotted (`rendering.snapshot_row`) before being
overwritten with computed values, and unmatched (leftover-assortment) rows
reuse the first matched row's style as a plain, unstyled prototype. This
preserves the reference template's actual visual formatting rather than
re-deriving it. Rows are grouped by category (`rendering.GROUP_SORT_ORDER`:
Выпечка сытная → Выпечка сладкая → Пироги сытные → Пироги сладкие →
Фастфуд), sorted within category by original template row order for
matched rows and by name for leftover rows. `Итого` = the sum of what was
actually scheduled (kratnost-rounded) for matched rows, or the raw forecast
total for leftover rows.

## Known limitations (all deliberate 2026-07-14 pilot simplifications)

- **No capacity/mощность checking at all** — matches the original
  pre-MILP system's documented limitation. A plan can schedule more than a
  bakery's ovens/bakers can physically produce in a window; nothing flags
  it.
- **Leftover (not-in-template) assortment SKUs get no window breakdown,
  only a raw total.** Acceptable for the pilot because it targets
  top-selling SKUs, which are expected to already have template rows;
  new/low-volume SKUs are the ones that fall into this bucket.
- **`bakery_month_revenue_embedded` is stale** (one-time manual backfill,
  May 2026 data, never refreshed) — revenue-tier sheet selection uses it
  as-is; a bakery whose revenue has since crossed a tier boundary will get
  the wrong sheet until this table is refreshed.
- **No PDF-derived night-storage caps** on defrost quantity (removed from
  the MILP era's `NIGHT_STORAGE_DIRECT_UNITS_BY_SKU`/
  `NIGHT_PREP_LABOR_MINUTES_BY_SKU`).
- **SKU name matching is fuzzy** (`allocation.sku_match_keys`,
  `SKU_ALIAS_TO_CANONICAL`) between the template (a static reference file)
  and live ClickHouse product names — a product renamed in the catalogue
  without a matching alias entry will silently fall through to the
  leftover-assortment bucket.

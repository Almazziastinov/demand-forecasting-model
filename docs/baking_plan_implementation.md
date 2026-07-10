# Baking Plan Implementation

Business rules and algorithm design for `apps/baking_plan/` (rebuilt from
scratch 2026-07-09, MILP wired end-to-end and deployed 2026-07-11 — see
`docs/ops/DECISIONS.md` for the architecture decision and
`docs/ops/CURRENT_STATE.md` for deploy details). This file replaces the
pre-teardown `docs/baking_plan_implementation.md`, which described the old
peak-detection/clustering algorithm.

## Definition

План выпекания = an Excel file listing each bakery's currently-assorted
SKUs and the baking-window quantities assigned to them. Previously window
assignment was expert opinion (a technologist's manual call); this rebuild
replaces that with a data-driven allocation: real hourly demand forecast +
real production physics (oven/baker capacity, tray capacity, prep time,
shelf life) fed into a constrained optimizer. The technological constraints
themselves are not removed, only the decision of how to satisfy them.

## Demand input

The hourly-SKU-volume curve (`sku_forecast_hour_embedded` in ClickHouse) is
the single input the allocator consumes — it's already bakery volume ×
SKU-share × hourly-profile, computed upstream by the forecast pipeline.
`apps/baking_plan/demand.py` joins this to the bakery's assortment
(`bakeable_products`, city+bakery scope, filtered to 5 bakeable categories:
Пироги сытные, Пироги сладкие, Выпечка сытная, Выпечка сладкая, Фастфуд)
and to SKU metadata (`baking_sku_meta` — see below).

Gap-handling:

- Assortment SKU with no `baking_sku_meta` row → skipped from the plan,
  logged as a data-quality gap (not defaulted or hard-failed).
- SKU with no forecast for the date → kept in the output with all window
  cells empty and `Итого = 0` (bakers should see the full eligible
  assortment even at zero forecast demand, not have rows vanish).

## SKU metadata (`baking_sku_meta`, ClickHouse, keyed by `product_id`)

- **Кратность** (`kratnost`) — tray capacity: how many units fit in one
  bake per tray/plate. A physical constant, not a rounding convenience.
- **Тесто-группа** (`dough_group`) — auto-derived from
  `dim_recipes.material_name` (materials starting with "Тесто", mapped via
  `baking_dough_group_mapping`) so new products flow in automatically from
  recipe data; manually overridden for edge cases with no dough material
  (frozen semi-finished goods, in-store-baked items, bought-in bases).
  SKUs in the same dough group should cluster in nearby windows rather
  than being scheduled independently.
- **`is_two_day`** — двухдневка flag (see below). Manual, no recipe-driven
  source.
- **`station`** ("Стол") and **`is_on_demand`** — manual/operational,
  no auto-derivation.

Seeded once from the old reference Excel via
`scripts/seed_baking_plan_tables.py` (one-time bridge, not a recurring
pipeline step); 71/80 SKUs matched by name, 9 unmatched (mostly one-off
project items) intentionally left out.

## Production capacity (`baking_capacity_config` /
`baking_category_molding_minutes`, ClickHouse; bakery-override-ready,
currently one global default row)

- Лепка/formation time: **4 min/unit** for Пироги сытные and Пироги
  сладкие; **1 min/unit** for every other category.
- Bake time: **30 min/batch**, regardless of category.
- Oven holds **6 trays** per batch.
- Default staffing: **2 bakers, 2 ovens** per bakery (placeholder, ready
  for per-bakery override later).

Capacity is modeled in **two independent dimensions per window** —
baker-minutes (`bakers_count * window_duration_minutes`) and tray-slots
(`ovens_count * (window_duration_minutes // bake_minutes) *
trays_per_oven_batch`). Both must be respected; a window can be tray-slack
but baker-minute-bound (or vice versa) — this is correct multi-resource
packing behavior, not a bug, and shows up as apparently "unused" tray
capacity when baker-minutes is actually the binding constraint.

## Windows

Window boundaries (`4:00-7:00`, `7:00-8:00`, ...) come from row 3 of the
reference template's "План выпекания" sheet
(`apps/baking_plan/templates.py:parse_windows`) — not a ClickHouse table,
since only one window set exists today (no per-revenue-bucket variants are
wired up yet).

Each window's demand share is computed from its own nominal
`[start_hour, end_hour)` slice of the hourly curve, then **all windows are
rescaled proportionally so their total equals the SKU's full-day demand**
— this spreads the whole day's volume (including hours after the last
window's nominal end, e.g. evening sales after 16:00) across the existing
windows without needing real per-bakery closing-hour data. A SKU with zero
demand in every window's nominal hours falls back to an even split.

## Дефрост vs двухдневка — two independent mechanics, not one

These were originally conflated during implementation and then corrected;
they must stay separate:

- **Дефрост**: a SKU gets its **normal** same-day window production
  *plus* an extra overnight batch on top, sized from **tomorrow's**
  early-morning forecast hours (06:00–12:00). Triggered by SKU identity
  (`DEFROST_SKU_NAMES` in `apps/baking_plan/constants.py` — a hardcoded
  5-name placeholder list; **no ClickHouse source of truth exists yet**,
  open question). Rendered as `"N (ночная дефр)"` with a light-coral fill
  in the last window only.
- **Двухдневка** (`is_two_day`): **zero** regular-window production
  today (today's stock was already baked in yesterday's last window) —
  the last window instead bakes the SKU's entire **next day's** forecast
  in one batch. Rendered as a plain number with a light-purple fill, no
  text suffix.

Both mechanics are business-mandatory: they must always be produced in
full regardless of their own sales priority, and only the capacity left
over after satisfying them goes to regular SKUs in that window. In the
MILP objective, their shortfall is weighted at a fixed
`MANDATORY_SHORTFALL_WEIGHT = 10_000.0` — high enough to dominate any
realistic sales-priority weight without numerical-conditioning risk.

## Slot-priority (regular SKUs)

When regular SKUs compete for capacity, higher `avg_daily_sales` wins.
Dough-group cohesion is a secondary constraint — same-group SKUs should
cluster in nearby windows rather than being scattered by pure priority
rank, though this isn't a hard constraint in the MILP formulation (it
still comes out tighter than greedy's explicit whole-group-or-nothing
rule in practice).

## Algorithm: MILP chosen over greedy

Both a greedy allocator (`algorithms/greedy.py`, signed-stock rationing,
kept as reference/fallback) and a MILP allocator
(`algorithms/milp.py`, `scipy.optimize.milp`/HiGHS) were built and
compared on real data (`scripts/compare_baking_algorithms.py`). MILP won
on both weighted shortfall and dough-group window cohesion and was chosen
for production (`apps/baking_plan/service.py` wires `allocate_milp_detailed`
in). Regular, дефрост, and двухдневка demand are three independent
variable/constraint families in the MILP so дефрост/двухдневка can never
be produced early or shorted in favor of regular SKUs.

Solver options: `time_limit=15s`, `mip_rel_gap=0.005` — needed because
full-day proportional demand redistribution makes capacity genuinely
scarce across (nearly) every window simultaneously once SKU count exceeds
~40, and HiGHS's branch-and-bound can otherwise run long. A time-limited
run returns its best feasible solution rather than searching indefinitely;
failure is detected via `result.x is None`, not `result.success`.

## Rendering (`apps/baking_plan/rendering.py`)

The "План выпекания" sheet is built from scratch every call (openpyxl),
not by mutating the reference template — the SKU list size varies per
bakery/date, so the template's row structure doesn't generalize. Layout:

- Rows grouped by category in a fixed order (Выпечка сытная, Выпечка
  сладкая, Пироги сытные, Пироги сладкие, Фастфуд), sorted by
  `avg_daily_sales` descending within each category.
- `MANDATORY_ASSORTMENT` — a hardcoded 10-SKU set that management tracks
  and which must always appear on the plan — highlighted yellow on the
  Стол+Наименование cells. No ClickHouse source of truth yet; if this
  needs to vary per bakery/city later it should move to a table mirroring
  `baking_sku_meta`.
- `Итого` = the sum of what's actually scheduled across the window cells
  (kratnost-rounded production, including дефрост/двухдневка quantities),
  **not** the raw model forecast — e.g. forecast `41.21` with `20+20`
  scheduled shows `Итого = 40`.

## Known open items

- `DEFROST_SKU_NAMES` and `MANDATORY_ASSORTMENT` are hardcoded constants
  with no ClickHouse source of truth.
- Individual per-bakery template overrides (`assets/individual/`) are
  unused; only `scope='base'`/city-scope assortment rows are seeded today.
- Multi-revenue-bucket window sets were never built — one global window
  set is used for every bakery.
- No anti-waste pressure in the MILP objective — shortfall is penalized
  but overproduction isn't, so two-day/defrost allocations can exceed
  exact demand.
- Solve time varies 2–19s depending on branch-and-bound exploration order
  near the time cap; acceptable for a synchronous download button today
  but worth revisiting (lower time limit or background-job pattern) if it
  becomes a UX problem.

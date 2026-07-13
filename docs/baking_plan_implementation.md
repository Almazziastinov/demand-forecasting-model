# Baking Plan Implementation

Business rules and algorithm design for `apps/baking_plan/` (rebuilt from
scratch 2026-07-09, MILP wired end-to-end 2026-07-11, merged-demand MILP
redesign 2026-07-13 — see `docs/ops/DECISIONS.md` for both architecture
decisions and `docs/ops/CURRENT_STATE.md` for deploy details). This file
replaces the pre-teardown `docs/baking_plan_implementation.md`, which
described the old peak-detection/clustering algorithm.

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
- SKU whose `dough_group` marks it as "Замороженные полуфабрикаты" →
  skipped from the production plan. These products have no baking process;
  they are placed on the showcase rather than occupying a baking window,
  oven slot, baker minute, or helper minute.
- SKU with no forecast for the date → kept in the output with all window
  cells empty and `Итого = 0` (bakers should see the full eligible
  assortment even at zero forecast demand, not have rows vanish).

**Yesterday's defrost credit (2026-07-13):** for `DEFROST_SKU_NAMES`
members only, today's early-hour (`DEFROST_HOURS`, 06:00–11:00) demand is
reduced by whatever `sku_forecast_hour_snapshots` (`lead_days = 1`) says
was forecast for those hours *as of yesterday* — that overnight batch was
already baked, so today's own plan shouldn't ask for it again
(`apps/baking_plan/demand.py:_load_yesterday_defrost_offset` /
`_apply_defrost_offset`). Verified this lead-1 snapshot table tracks
closely (within a few percent) with directly querying the prior day's own
forecast run, so it's not a guess at a run-naming convention. Clamped at
0 — an overnight batch that turned out larger than today's (possibly
revised) forecast just means surplus shelf stock, not negative demand.
On real bakery-21 data this closed roughly a third of the day's total
regular shortfall.

## SKU metadata (`baking_sku_meta`, ClickHouse, keyed by `product_id`)

- **Кратность** (`kratnost`) — production multiple / tray capacity from the
  reference data. For non-pie SKUs it is used directly; for pie categories
  the allocator overrides oven fit to 4 pies per tray-slot from the updated
  operational input.
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

- Лепка/formation time: **1 min/unit** for regular baked goods,
  **4 min/unit** for pies whose `dough_group` contains "песоч", and
  **2 min/unit** for other pies.
- Bake time: **30 min/batch**, regardless of category.
- Oven holds **6 trays** per batch.
- Pies use an effective oven kratnost of **4 pies per tray-slot**. The
  historical `baking_sku_meta.kratnost` still controls production multiples
  for non-pie SKUs, but pie oven fit is overridden in code from the updated
  operational input.
- Default staffing: **2 bakers, 2 baker assistants, 2 ovens** per bakery
  (placeholder, ready for per-bakery override later).

Capacity is modeled in **three independent dimensions per window**:
core baker-minutes (`bakers_count * window_duration_minutes`) for
Выпечка сытная/сладкая and Пироги сытные/сладкие, helper-minutes
(`helpers_count * window_duration_minutes`) for products outside those core
categories, and shared tray-slots (`ovens_count *
(window_duration_minutes // bake_minutes) * trays_per_oven_batch`). All
three must be respected; a window can be tray-slack but labor-bound (or vice
versa), and helper-owned products no longer consume core baker minutes.

There is also a **daily core-unit throughput cap** independent of per-SKU
minutes: one baker can produce roughly **600 total units** across pies +
small baked goods in normal conditions and up to **800 total units** at peak
pace over the 12-hour window grid. The first solve uses `600 *
bakers_count`; if that leaves shortfall, the accelerated retry uses `800 *
bakers_count`. Helper-owned categories do not consume this core-unit cap.

**Molding-pace floor and automatic retry (2026-07-13, revised
2026-07-13):** if the plan can't be fully covered at the normal pace,
`service.build_baking_plan_workbook` retries once at a fixed floor pace for
non-pie categories — 54 sec/unit. Pie timing is not taken from the old
category floor anymore; it is resolved by SKU/dough group as described
above.
If shortfall remains even at the floor, `service.py` scans every window's
actual resource usage and adds a note to the rendered plan naming which
physical resource(s) are maxed out — "требуется дополнительно: пекарь",
"помощник пекаря", and/or "печь" — rather than leaving the shortfall
unexplained.

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

## Дефрост vs двухдневка — two independent mechanics, one shared pool

These are still business-distinct and must stay conceptually separate,
but as of the 2026-07-13 redesign they are **not** separate variable
families in the solver — see "Algorithm" below for why that changed.

- **Дефрост / прямой ночной запас**: a SKU gets its **normal** same-day
  window production *plus* an extra overnight batch on top, sized from
  **tomorrow's** early-morning forecast hours (06:00–12:00) and capped by
  the recommended SKU quantity from the freezer/refrigerator night-storage
  PDFs dated 15.05.2026 (`NIGHT_STORAGE_DIRECT_UNITS_BY_SKU` in
  `apps/baking_plan/constants.py`; `DEFROST_SKU_NAMES` remains a
  backwards-compatible alias). The next day credits that same capped
  amount back out of early-hour regular demand via the lead-1 snapshot, so
  the plan does not ask the day shift to remake stock already left
  overnight. Rendered as
  `"N (ночная дефр)"` with a light-coral
  fill — now in **whichever window(s) the post-processing step
  consolidates it into**, preferring the latest window with spare
  capacity rather than being hardwired to the last window (see below).
  The label text is a deliberate cosmetic choice made 2026-07-13: the
  post-processing pass usually does land defrost in the last window
  (that's what it's tuned to prefer), so "ночная" reads correctly in the
  common case even though it's no longer a hard guarantee.
- **Двухдневка** (`is_two_day`): **zero** regular-window production
  today (today's stock was already baked in yesterday's last window) —
  the entire **next day's** forecast bakes in one batch, again placed by
  post-processing to prefer the latest feasible window(s) rather than
  always the last one. Rendered as a plain number with a light-purple
  fill, no text suffix.
- **Ночные заготовки**: prep-only rows from the refrigerator/freezer PDFs
  (formed blanks / dough balls for жар киши and сметанники) are not
  treated as finished stock and do not reduce SKU demand. Instead they
  reduce daytime labor minutes for the mapped SKUs through
  `NIGHT_PREP_LABOR_MINUTES_BY_SKU`.

Both mechanics are business-mandatory: they must always be produced in
full regardless of their own sales priority, and only the capacity left
over after satisfying them goes to regular SKUs in that window.

## Slot-priority (regular SKUs)

When regular SKUs compete for capacity, higher `avg_daily_sales` wins.
Dough-group cohesion is a secondary constraint — same-group SKUs should
cluster in nearby windows rather than being scattered by pure priority
rank, though this isn't a hard constraint in the MILP formulation (it
still comes out tighter than greedy's explicit whole-group-or-nothing
rule in practice).

Production is not forced to open a full extra batch for a tiny tail above a
multiple. For example, a regular SKU with forecast 61 and `kratnost = 20`
may be planned as 60 rather than 80. The tolerance is intentionally small
(up to 50% of `kratnost`) and does **not** drop the first partial batch to
zero: forecast 5 with `kratnost = 20` still produces 20 unless
capacity/priority prevents it. With `kratnost = 20`, forecast 70 can stay
at 60, while 71 requires the next batch and plans 80.

## Algorithm: MILP chosen over greedy

Both a greedy allocator (`algorithms/greedy.py`, signed-stock rationing,
kept as reference/fallback) and a MILP allocator
(`algorithms/milp.py`, `scipy.optimize.milp`/HiGHS) were built and
compared on real data (`scripts/compare_baking_algorithms.py`). MILP won
on both weighted shortfall and dough-group window cohesion and was chosen
for production (`apps/baking_plan/service.py` wires `allocate_milp_detailed`
in).

Solver options: `time_limit=15s`, `mip_rel_gap=0.005` — needed because
full-day proportional demand redistribution makes capacity genuinely
scarce across (nearly) every window simultaneously once SKU count exceeds
~40, and HiGHS's branch-and-bound can otherwise run long. A time-limited
run returns its best feasible solution rather than searching indefinitely;
failure is detected via `result.x is None`, not `result.success`.

**Merged-demand model (2026-07-13 redesign — see `docs/ops/DECISIONS.md`
for the full rationale and rejected alternatives).** Regular, дефрост, and
двухдневка are **not** independent variable families. Each SKU has a
single shared `trays[sku][w]` pool across all windows, sized against the
SKU's **combined** target (regular demand + defrost top-up + two-day
next-day volume). Two constraint tiers price shortfall differently:

1. `shortfall_regular[sku][w]` — per-window cumulative shortfall against
   regular demand only, weighted by the SKU's own sales-priority weight.
2. `mandatory_shortfall[sku]` — a **single checkpoint at the last window**,
   measured against the combined target
   (`cum_regular_demand[last] + defrost_demand + two_day_demand`),
   weighted at a fixed `MANDATORY_SHORTFALL_WEIGHT = 10_000.0` — high
   enough to dominate any realistic sales-priority weight without
   numerical-conditioning risk.

The SKU's total shortfall cost is
`max(shortfall_regular[last], mandatory_shortfall[sku])`, not their sum —
summing would double-count the same units. One subtlety that caused a
real bug: for SKUs with zero defrost/two-day component
(`mandatory_extra <= 0`), the mandatory checkpoint's lower bound must be
relaxed to `-inf` rather than left at its default — otherwise it silently
duplicates the regular constraint and applies the 10,000× weight to
*ordinary* shortfall too, breaking priority ordering for every regular
SKU sharing a window with a mandatory one.

**Why window placement is a separate post-processing pass, not a solver
objective term.** The MILP is correctness-only: it doesn't care *which*
window a mandatory SKU's production lands in, only that the combined
total is met by the last window. Left alone, HiGHS scatters a SKU's
mandatory tail across whichever windows happen to have slack in its first
feasible solution — sometimes 6–7 different windows for one SKU. A
scalar `LATE_WINDOW_PREFERENCE_WEIGHT` objective term was tried and
rejected: weighted below `ANTI_WASTE_WEIGHT` it was too small to reliably
beat `MIP_REL_GAP`'s tolerance (placement became non-deterministic
between otherwise-identical runs); weighted large enough to matter
(~1e-3–1e-1) it started trading real production for placement —
confirmed by a real overproduction case (120 units baked against ~75.7
units of actual combined demand) that reproduced only when the weight was
active and disappeared entirely with the weight removed.

The fix is a two-step post-solve, correctness-preserving pipeline in
`algorithms/milp.py`:

1. **`_split_tail(produced_by_window, windows, mandatory_amount)`** —
   pure bookkeeping. Given the SKU's raw per-window solver output, claims
   `mandatory_amount` units working backward from the last window
   (splitting a single window's quantity if the boundary falls inside
   it), separating "regular" from "mandatory" without changing any
   quantity. `mandatory_amount` itself must be computed from the
   **kratnost-rounded** regular-only target
   (`ceil(cum_demand[last]/kratnost - eps) * kratnost`), not the raw
   continuous solver value — using the raw value produced fractional
   `Итого` figures (e.g. `40.0265...`) in an earlier iteration.
2. **`_shift_to_later_windows(movable, fixed, skus, windows, capacity,
   molding_minutes_map)`** — destination-first bin-packing across *all*
   SKUs' pooled mandatory totals simultaneously: for each window from
   last to first, offers remaining capacity to SKUs sorted by
   `avg_daily_sales` descending. `fixed` (the already-separated regular
   production) only contributes to used-capacity accounting; it is never
   itself moved, so regular production's intra-day placement — driven by
   the actual hourly sales curve — is never disturbed. (An earlier
   version shifted the SKU's *whole* merged total including regular
   production and was caught before shipping — it risked corrupting
   regular placement to make room for mandatory consolidation.)

Only after both passes does `_split_tail`'s defrost/two-day portion get
divided by `product_name in DEFROST_SKU_NAMES` for rendering.
`ANTI_WASTE_WEIGHT` (tiny per-tray objective weight preventing
gratuitous overproduction ties) was raised from `1e-5` to `1e-2` during
this work purely for solve-speed — a weight sweep from `1e-5` to `0.3`
produced identical production/shortfall at every tested value; the
"waste" investigated at the time turned out to be kratnost-rounding, not
a real bug.

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
  (kratnost-rounded production), **not** the raw model forecast — e.g.
  forecast `41.21` with `20+20` scheduled shows `Итого = 40`. **Excludes
  дефрост** (2026-07-13 fix): the defrost top-up bakes tonight for
  *tomorrow's* early hours, so counting it in today's Итого overstated
  today's production. **Includes двухдневка in full** — it fully replaces
  today's regular production for that SKU, so it belongs in today's total.
- **Shortfall highlighting (2026-07-13).** Per-SKU shortfall (forecast
  minus what capacity allowed, computed net of any defrost shortfall so
  the defrost-vs-regular distinction doesn't leak into the regular
  shortfall color) drives the Итого cell's fill: red
  (`SHORTFALL_FULL_FILL`) with the integer forecast shown when nothing
  was produced at all, orange (`SHORTFALL_PARTIAL_FILL`) with a
  `"{produced}/{forecast}"` string when partially covered. No highlight,
  and no format change, when produced matches forecast within rounding.
- **Capacity note row (2026-07-13).** When `service.py` can't fully cover
  demand even at the molding-pace floor, a bold note row is inserted at
  row 3 (`CAPACITY_NOTE_FILL`, shifting the header down to row 4) naming
  which physical resource(s) are the bottleneck — "требуется
  дополнительно: пекарь", "помощник пекаря", and/or "печь". If the floor
  pace *did* resolve the shortfall, a softer informational note about the
  accelerated pace is shown instead.

## Known open items

- `NIGHT_STORAGE_DIRECT_UNITS_BY_SKU`,
  `NIGHT_PREP_LABOR_MINUTES_BY_SKU`, and `MANDATORY_ASSORTMENT` are
  hardcoded constants with no ClickHouse source of truth.
- `MOLDING_MINUTES_FLOOR` (54 sec/unit for non-pie categories),
  `UTILIZATION_THRESHOLD = 0.99`, `helpers_count = 2`,
  `PIE_EFFECTIVE_KRATNOST = 4`, and the sand-dough marker `"песоч"` are
  hardcoded with no ClickHouse-backed source of truth yet — same category
  of open item as the two above.
- `NORMAL_DAILY_CORE_UNITS_PER_BAKER = 600` and
  `PEAK_DAILY_CORE_UNITS_PER_BAKER = 800` are hardcoded until moved to a
  ClickHouse-backed capacity table.
- The yesterday-defrost credit (`_load_yesterday_defrost_offset`) only
  applies to direct night-storage members. `is_two_day` SKUs have no
  analogous "yesterday" top-up concept — they bake their *entire* next
  day in one batch rather than a partial early-hour amount, so there's
  no partial credit to reconcile.
- Individual per-bakery template overrides (`assets/individual/`) are
  unused; only `scope='base'`/city-scope assortment rows are seeded today.
- Multi-revenue-bucket window sets were never built — one global window
  set is used for every bakery.
- ~~No anti-waste pressure in the MILP objective~~ — resolved by
  `ANTI_WASTE_WEIGHT` (see Algorithm section); a weight sweep confirmed
  the originally-observed "waste" was kratnost-rounding, not genuine
  overproduction, and the current weight is tuned for solve-speed only.
- Solve time varies 2–19s depending on branch-and-bound exploration order
  near the time cap; acceptable for a synchronous download button today
  but worth revisiting (lower time limit or background-job pattern) if it
  becomes a UX problem.

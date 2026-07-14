# Baking Plan

Standalone logical unit that generates the per-bakery baking-window Excel
plan. Runs in-process inside `forecast_embedded` (mounted as a router), but
is a separate Python package with its own module boundary.

Rebuilt twice: torn down and rebuilt as a MILP allocator 2026-07-09..11,
then reverted to a template-driven approach 2026-07-14 for the pilot launch
— window *assignment* now comes straight from the Excel template (which
C:L cells a technologist pre-filled), not computed by any algorithm. See
`docs/baking_plan_implementation.md` for the full spec and
`docs/ops/DECISIONS.md` for the 2026-07-14 decision and its rationale.

## Boundary

- This package may import ClickHouse plumbing from `app.db` /
  `app.settings` / `app.table_names` (shared infra, not business logic) —
  wrapped in `_clickhouse.py`.
- `forecast_embedded` (the `app` package) must not import from
  `baking_plan.*` except to mount `baking_plan.router.router` in
  `app/main.py`. No other file under `apps/forecast_embedded/app` should
  reach into this package.
- All forecast/assortment/revenue reads happen inside this package
  (`demand.py`, `assortment.py`), keyed only by
  `run_id`/`forecast_date`/`bakery_id`/`city` — `router.py` resolves those
  from the request and does no data fetching of its own beyond that.

## Layout

```
apps/baking_plan/
  service.py      -- public entrypoint: build_baking_plan_workbook(...);
                      orchestrates template selection, window/comments
                      parsing, per-row allocation, and rendering
  router.py         -- FastAPI router: GET /bakery/{bakery_id}/baking-plan.xlsx
  templates.py        -- xlsx template selection (individual override or
                          base + revenue-tier sheet), window-header parsing
                          (row 5), "комментарии" sheet parsing
  allocation.py          -- pure functions: read a template row's pre-filled
                             window schedule, fuzzy-match SKU names against
                             live assortment, size quantities from the
                             live hourly forecast into those windows
  demand.py                 -- ClickHouse reads: hourly forecast (today +
                                next day, by product_id), bakery revenue
                                (for revenue-tier sheet selection)
  assortment.py                -- bakeable-products allowlist (city + bakery
                                   scope, unchanged since the MILP era)
  rendering.py                    -- writes computed plan rows back into the
                                     loaded template sheet in place (row
                                     snapshot/restore, category grouping,
                                     Итого column) — preserves the
                                     template's own visual styling
  assets/
    template.xlsx                   -- base template: one sheet per revenue
                                        tier (до 1,5/2,5 млн, от 2,5 млн,
                                        от 3млн) + "комментарии"
    individual/{id}_*.xlsx            -- per-bakery override templates,
                                          take priority over revenue tier
```

## What this package deliberately does not do (2026-07-14)

Reverted for the pilot, per `docs/ops/DECISIONS.md`:

- **No algorithmic window placement.** Neither the pre-MILP peak-detection
  distribution nor the MILP solver decide which window a SKU bakes in —
  `allocation.read_row_schedule` always reads it from the template's own
  pre-filled cells.
- **No capacity/mощность checking.** No baker-minutes, tray-slots, daily
  caps, shortfall highlighting, or "требуется дополнительно" notes — a
  documented known limitation, not a bug, matching the pre-MILP system.
- **No PDF-derived night-storage caps.** Defrost quantity is simply
  tomorrow's early-hour (6–11) forecast, uncapped.

## Deploy note

Blackhole deploys have historically uploaded only
`apps/forecast_embedded/app/*` (see `docs/ops/CURRENT_STATE.md`,
2026-07-07 entry). Any future Blackhole deploy that touches this package
must also upload `apps/baking_plan/*`, or add it to whatever deploy
tooling replaces the manual upload.

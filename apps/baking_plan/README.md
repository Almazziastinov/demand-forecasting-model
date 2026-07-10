# Baking Plan

Standalone logical unit that generates the per-bakery baking-window Excel plan.
Runs in-process inside `forecast_embedded` (mounted as a router), but is a
separate Python package with its own module boundary — the old implementation
was torn down 2026-07-09 and rebuilt from scratch here, data-driven instead
of expert-opinion-driven. See project memory `baking_plan_business_rules` /
`baking_plan_package_split` for the full design history.

## Boundary

- This package may import ClickHouse plumbing from `app.db` /
  `app.settings` / `app.table_names` (shared infra, not business logic) —
  wrapped in `_clickhouse.py`.
- `forecast_embedded` (the `app` package) must not import from
  `baking_plan.*` except to mount `baking_plan.router.router` in
  `app/main.py`. No other file under `apps/forecast_embedded/app` should
  reach into this package.
- All forecast/assortment/metadata reads happen inside this package
  (`demand.py`), keyed only by `run_id`/`forecast_date`/`bakery_id`/`city` —
  `router.py` resolves those from the request and does no data fetching of
  its own beyond that.

## Layout

```
apps/baking_plan/
  service.py           -- public entrypoint: build_baking_plan_workbook(...)
  router.py             -- FastAPI router: GET /bakery/{bakery_id}/baking-plan.xlsx
  demand.py              -- SkuDemand assembly: assortment + baking_sku_meta + forecast + sales
  assortment.py            -- bakeable-products allowlist (city + bakery scope)
  capacity.py               -- oven/baker capacity + category molding-time lookups
  templates.py               -- window-layout parsing ("План выпекания" header row) +
                                 "комментарии" sheet parsing (used only by the one-time seed script)
  rendering.py                -- builds the "План выпекания" sheet from scratch each call
  algorithms/
    common.py                   -- shared window-demand / defrost-demand math
    greedy.py                     -- window-by-window greedy allocator (kept as reference/fallback)
    milp.py                        -- scipy.optimize.milp allocator — chosen algorithm (2026-07-10)
  assets/
    template.xlsx                   -- reference file: window-layout source + comments-sheet seed source
    individual/{id}_*.xlsx            -- per-bakery override templates (not yet restored)
```

## Algorithm

`allocate_milp` (whole-day MILP, `scipy.optimize.milp`/HiGHS) is the chosen
allocator — it beat the greedy allocator on both weighted shortfall and
dough-group window cohesion in a real-data comparison (see
`baking_plan_package_split` memory, 2026-07-10 decision). `greedy.py` is
kept in the codebase as a reference/fallback, not currently wired into
`service.py`. `scripts/compare_baking_algorithms.py` can re-run the
comparison against any (bakery, date).

## Deploy note

Blackhole deploys have historically uploaded only
`apps/forecast_embedded/app/*` (see `docs/ops/CURRENT_STATE.md`,
2026-07-07 entry). Any future Blackhole deploy that touches this package
must also upload `apps/baking_plan/*`, or add it to whatever deploy
tooling replaces the manual upload.

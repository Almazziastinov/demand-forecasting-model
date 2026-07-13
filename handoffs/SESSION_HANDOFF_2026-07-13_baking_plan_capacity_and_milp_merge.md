# Session Handoff — 2026-07-13 — Baking Plan Capacity Guardrails and Merged-Demand MILP Redesign

## Scope

- Studied `docs/` and updated `CLAUDE.md` to describe the two-part repo
  (legacy local ML pipeline vs the live VM/ClickHouse/Blackhole
  production system) and point future sessions at `docs/ops/` as the
  live source of truth — this gap had caused confusion in earlier
  sessions about which docs to trust.
- Deep-dove `apps/baking_plan/` (built in the 2026-07-09/11 sessions) and
  evaluated generated plan `.xlsx` files against real bakery-21 data,
  which surfaced several real gaps: no visibility into why some SKUs got
  zero production, дефрост always landing in the first window instead of
  being consolidated late, and `Итого` incorrectly folding дефрост into
  today's total.
- Implemented capacity guardrails: a molding-pace floor with automatic
  retry, per-window shortage recommendations ("нужен пекарь"/"нужна
  печь"), and red/orange highlighting for unfulfilled SKUs.
- Implemented yesterday's-defrost-batch crediting against today's
  early-hour demand.
- Redesigned the MILP to merge regular + дефрост + двухдневка demand into
  one shared solve per SKU (previously three independent variable
  families), with window placement moved to a correctness-preserving
  post-processing pass — this was an explicit architecture correction
  from the user after an earlier same-session attempt got window
  placement wrong.
- Deployed the finished code to the Blackhole embedded app
  (`bakery-forecast-embedded`) via the VibeCode `/exec` API, verified
  live with an HTTP smoke test.
- Backfilled `docs/baking_plan_implementation.md` and
  `docs/ops/DECISIONS.md` to describe the shipped architecture, and wrote
  this handoff.

## Why the window-scattering problem mattered

The reference Excel plan bakers work from is meant to read as "bake this
consolidated batch late in the day," not "bake a little bit of this SKU
in six different windows." The original MILP (regular / дефрост /
двухдневка as three independent variable families, each guaranteed via
its own weighted shortfall) satisfied the *quantity* guarantee correctly
but left window *placement* entirely up to whichever feasible solution
HiGHS found first — which in practice scattered a single SKU's mandatory
production across up to 6-7 windows. This is disruptive on a bakery floor
and was the direct trigger for the redesign.

## Capacity guardrails (molding-pace floor, shortage note, highlighting)

`apps/baking_plan/capacity.py` gained `MOLDING_MINUTES_FLOOR` — a floor
pace (54 sec/unit default, 3:30/unit for Пироги сытные/сладкие) that
`service.build_baking_plan_workbook` retries at automatically if the
normal pace (1-4 min/unit) can't cover demand. If shortfall remains even
at the floor, `service._capacity_recommendation` scans every window's
actual baker-minutes and tray-slot usage and names which physical
resource is maxed out, surfaced as a bold note row in the rendered
workbook. `rendering.py` highlights the `Итого` cell red (zero produced)
or orange (`{produced}/{forecast}`) per SKU so a shortfall is visible at
a glance instead of silently showing a lower-than-expected number.

## Yesterday's defrost credit

`apps/baking_plan/demand.py:_load_yesterday_defrost_offset` queries
`sku_forecast_hour_snapshots` (`lead_days = 1`) for what was forecast for
today's early hours (`DEFROST_HOURS`, 06:00-11:00) *as of yesterday* —
that overnight batch was already baked, so today's plan shouldn't ask for
it again. Verified this snapshot table tracks closely with directly
querying the prior day's own forecast run. On real bakery-21 data this
closed roughly a third of the day's total regular shortfall.

## Merged-demand MILP redesign

Full technical detail lives in `docs/baking_plan_implementation.md`
("Algorithm" section) and `docs/ops/DECISIONS.md` (2026-07-13 entry) —
summary here:

- Each SKU now has a single shared `trays[sku][w]` pool sized against its
  combined target (regular + defrost + two-day), with the "always fully
  produced" guarantee preserved via a `mandatory_shortfall[sku]`
  checkpoint at the last window, weighted `10_000.0`.
- Window placement of the mandatory portion is decided by a **post-solve**
  pass — `_split_tail` (separates regular from mandatory using
  kratnost-rounded boundaries) then `_shift_to_later_windows`
  (destination-first bin-packing, latest window first, across all SKUs) —
  not by the solver's objective.
- A scalar `LATE_WINDOW_PREFERENCE_WEIGHT` objective term was tried first
  and rejected: too small a weight was swamped by HiGHS's
  `mip_rel_gap=0.005` tolerance (non-deterministic placement between
  runs); too large a weight caused real overproduction (a reproducible
  120-vs-75.7-unit case). A pre-solve greedy backward-fill allocator was
  also considered and rejected by the user — it would lose joint
  optimization with regular SKUs, which was the whole point of the MILP
  rebuild.
- Two real bugs were found and fixed during this work: (1) the mandatory
  checkpoint's lower bound needed `-inf` for SKUs with no defrost/two-day
  component, otherwise it silently duplicated the regular constraint and
  broke priority ordering (caught by
  `test_two_day_always_wins_capacity_over_higher_priority_regular_sku`
  failing backwards); (2) `_split_tail`'s mandatory boundary needed the
  kratnost-rounded regular target, not the raw continuous solver value,
  to avoid fractional `Итого` figures like `40.0265...`.

### Result on real bakery-21 data

RED (fully unfulfilled) shortfall rows: 0. ORANGE (partial): 1. All 5
дефрост SKUs consolidated into (mostly) one late window each (down from
scattered placement). Двухдневка SKUs consolidated into 2-3 windows each
(down from up to 7). All `Итого` values are clean integers.

## Code Changes

All bundled in one commit (`3b18eac`) since the features were designed
and tested together as one iterative session:

- `apps/baking_plan/algorithms/milp.py` — merged-pool model,
  `_split_tail`, `_shift_to_later_windows`, `ANTI_WASTE_WEIGHT` raised
  `1e-5` -> `1e-2` (solve-speed only, confirmed via weight sweep).
- `apps/baking_plan/capacity.py` — `MOLDING_MINUTES_FLOOR`,
  `resolve_molding_minutes_floor`.
- `apps/baking_plan/demand.py` — yesterday-defrost offset loading/apply.
- `apps/baking_plan/rendering.py` — red/orange highlighting,
  `capacity_note` row, `Итого` excludes дефрост, defrost suffix text
  changed to `"(доп. партия на завтра)"` (no longer claims "night only").
- `apps/baking_plan/service.py` — two-stage pace retry,
  `_capacity_recommendation`.
- `tests/test_baking_plan_milp.py` (10 tests, incl. the critical
  `test_two_day_always_wins_capacity_over_higher_priority_regular_sku`),
  `tests/test_baking_plan_shortfall.py` (new, ~9 tests),
  `tests/test_baking_plan_demand_offset.py` (new, 4 tests).

## Deploy Status

| Artifact | Status |
|---|---|
| Code (`3b18eac`) | ✅ pushed to `origin/master`, pulled to Blackhole VM |
| `/opt/app/app` (mirrors `apps/forecast_embedded/app`) | ✅ replaced, backup at `/opt/app/app_backup_20260713_072134` |
| `/opt/baking_plan` (mirrors `apps/baking_plan`) | ✅ replaced, backup at `/opt/baking_plan_backup_20260713_072134` |
| `bakery-forecast-embedded` service | ✅ restarted, active, `/health` OK |
| Smoke test | ✅ `GET /bakery/21/baking-plan.xlsx?date=2026-07-10&run_id=prod_base_bakery_no_sku_uplift_20260710_h14` → HTTP 200, 8338-byte valid xlsx |
| `docs/ops/CURRENT_STATE.md` | ✅ updated with deploy entry (`50a49f2`) |
| `docs/baking_plan_implementation.md`, `docs/ops/DECISIONS.md` | ✅ this session (see Commits) |

Deploy method note: staged the new code into `/tmp/deploy_stage` and ran
dependency-free plain-Python checks (no `pytest` — installing it into the
live prod venv was correctly blocked by the safety classifier) against
the 7 most critical invariants *before* touching `/opt/app` or
`/opt/baking_plan`, per the user's explicit "backup+preflight, then
replace" choice.

## Pending Issues

- `DEFROST_SKU_NAMES`, `MANDATORY_ASSORTMENT`, `MOLDING_MINUTES_FLOOR`,
  and `UTILIZATION_THRESHOLD` remain hardcoded constants with no
  ClickHouse source of truth — flagged as open items in
  `docs/baking_plan_implementation.md`, not new to this session but worth
  a reminder since two of them (`MOLDING_MINUTES_FLOOR`,
  `UTILIZATION_THRESHOLD`) were added this session.
- Yesterday's-defrost credit only applies to `DEFROST_SKU_NAMES` members;
  `is_two_day` SKUs have no analogous partial "yesterday" concept since
  they bake their entire next day in one batch.
- An earlier, unrelated user message this session ("можешь гллчнуть еще
  за одно") was never clarified (likely a typo for "глянуть" / "take a
  look at") — no action taken, follow up if it resurfaces.

## Commits

| Hash | Message |
|---|---|
| `3b18eac` | feat: merge defrost/двухдневка into baking-plan MILP, clean up window placement |
| `50a49f2` | docs: record baking plan MILP redesign deploy to Blackhole (2026-07-13) |

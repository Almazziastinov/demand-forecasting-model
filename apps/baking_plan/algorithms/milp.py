"""Baking-window allocator, solved whole-day-at-once via `scipy.optimize.milp`
(HiGHS backend, bundled with scipy — no extra dependency).

Model per (bakery, date):

- `trays[sku][w]` — integer >= 0, number of tray-batches of `sku` baked in
  window `w`. One shared pool per SKU across regular, дефрост, and
  двухдневка demand — there is no separate defrost/two_day variable family
  (2026-07-12: merged per explicit user request, so the whole day's need —
  today's forecast + дефрост + двухдневка — enters the solver as one
  combined volume per SKU, rather than three competing tray pools).
- `shortfall_regular[sku][w]` — continuous >= 0, the running stock deficit
  against *regular* demand only, at window `w`'s checkpoint:
  `kratnost[sku] * cumsum(trays)[w] + shortfall_regular[sku][w] >=
  cum_regular_demand[sku][w]`. Standard lost-sales/backlog LP formulation,
  weighted by `avg_daily_sales[sku]` in the objective.
- `mandatory_shortfall[sku]` — continuous >= 0, a *second*, independent
  checkpoint at the last window only, against the combined regular + extra
  (дефрост + двухдневка) total: `kratnost[sku] * cumsum(trays)[last] +
  mandatory_shortfall[sku] >= cum_regular_demand[sku][last] +
  defrost_demand[sku] + two_day_demand[sku]`. Weighted at
  `MANDATORY_SHORTFALL_WEIGHT` (far above any realistic `avg_daily_sales`)
  so дефрост/двухдневка are still effectively always produced in full even
  at the expense of a higher-priority regular SKU sharing the same
  capacity — this is the one piece of the old three-pool model that had to
  survive the merge (confirmed with user 2026-07-12; see
  `test_two_day_always_wins_capacity_over_higher_priority_regular_sku`).
  Since this checkpoint's target is always >= the regular-only target,
  `shortfall_by_sku[sku] = max(shortfall_regular[sku][last],
  mandatory_shortfall[sku])` is the correct *total* unmet demand (summing
  them would double-count the overlapping gap).
- Capacity per window: baker-minutes and oven-tray-slots
  (`capacity.window_capacity`), shared by the one `trays` pool.
- Objective: minimize `avg_daily_sales[sku] * shortfall_regular` +
  `MANDATORY_SHORTFALL_WEIGHT * mandatory_shortfall` + `ANTI_WASTE_WEIGHT *
  trays` (see ANTI_WASTE_WEIGHT for why the last term exists).

The merged model has no opinion about *which* window a SKU's дефрост/
двухдневка increment lands in — trays cost the same everywhere, so among
equally-optimal solutions the solver is indifferent, and empirically (both
in small hand-worked examples and on the real ~66-SKU bakery-21 problem)
that increment can land anywhere, including the first window of the day.
Two post-solve passes fix this without touching correctness — in this
order:

1. `_split_tail()` — per SKU, claims the last `defrost_demand +
   two_day_demand` units off the solver's raw per-window output, working
   backward from the last window, and labels them "mandatory" (regular
   production keeps whatever's left). This must run *before* any shifting:
   regular's own per-window split reflects its real intra-day sales curve
   (`shortfall_regular`'s per-window checkpoints), so it must never be
   moved — only the split-off mandatory portion is free to relocate.
2. `_shift_to_later_windows()` — moves the (now-separated) mandatory
   portion toward the last window wherever spare capacity allows, treating
   regular production as a fixed background it must route around rather
   than something it can also push aside.

Dough-group cohesion is **not** a hard constraint here (see project plan /
memory for the rationale) — it's evaluated as a comparison metric instead,
in `scripts/compare_baking_algorithms.py`.
"""

from __future__ import annotations

# ruff: noqa: E501
import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from ..capacity import CapacityConfig, resolve_molding_minutes, window_capacity
from ..demand import SkuDemand
from ..templates import Window
from .common import DEFROST_SKU_NAMES, defrost_demand, two_day_demand, window_demand

# Safety net, not a tuning knob: at realistic scale (~40+ SKUs) with demand
# spread across the full day (see algorithms/common.py's proportional
# rescaling), capacity is genuinely scarce in most windows simultaneously —
# a real bin-packing-hard MILP, not a bug — and HiGHS's branch-and-bound can
# take exponentially longer as SKU count grows. Capping runtime trades a
# certified-optimal solution for a bounded one — HiGHS returns its best
# feasible solution found within the budget (`mip_rel_gap` also lets it
# stop early once within the gap of proven optimal).
SOLVE_TIME_LIMIT_SECONDS = 15

# A relative gap applied to an objective dominated by MANDATORY_SHORTFALL_WEIGHT
# can let the solver settle for a "good enough" solution that still leaves a
# small amount of *mandatory* shortfall unresolved even when spare capacity
# exists to fix it. A tighter gap closes that slop; kept well above 0 to
# still cap worst-case solve time.
MIP_REL_GAP = 0.005

# Дефрост and двухдневка are business-mandatory — always baked in full,
# never traded away for regular production regardless of either side's
# avg_daily_sales. 10_000 comfortably dominates any realistic priority
# value (avg_daily_sales is typically single/double digits) without the
# numerical-conditioning risk of a much larger coefficient spread.
MANDATORY_SHORTFALL_WEIGHT = 10_000.0

# Tray production itself costs nothing in the objective — only shortfall
# does — so once shortfall is fully minimized, *any* amount of further
# production is an equally "optimal" solution, including gratuitous
# overproduction with no offsetting benefit. This tiny per-tray weight
# breaks that tie in favor of the minimal-production solution without
# distorting real priority: even a few thousand gratuitous trays cost less
# than a single unit of the lowest realistic shortfall weight.
#
# 1e-2 (2026-07-12): swept 1e-5..0.3 on the real ~66-SKU bakery-21 problem
# and production/shortfall were byte-for-byte identical at every value
# tested — any "waste" was already the kratnost-rounding floor, not
# gratuitous slop. What *did* change: solve time dropped sharply once the
# weight left the 1e-5 range (clearer gradient for branch-and-bound), which
# matters for the synchronous xlsx-download endpoint and the
# SOLVE_TIME_LIMIT_SECONDS safety margin.
ANTI_WASTE_WEIGHT = 1e-2


def _shift_to_later_windows(
    movable: dict[tuple[str, str], float],
    fixed: dict[tuple[str, str], float],
    skus: list[SkuDemand],
    windows: list[Window],
    capacity: CapacityConfig,
    molding_minutes_map: dict[str, float],
) -> dict[tuple[str, str], float]:
    """Re-pack `movable` quantities into the latest windows with spare
    capacity, without changing any SKU's total quantity. `fixed` (regular
    production) contributes to capacity usage but is never itself moved —
    only the already-split-off mandatory portion (see `_split_tail`) is
    passed as `movable`, so regular's own per-window sales-curve placement
    is never disturbed.

    Pools each SKU's movable total (ignoring which window it started in —
    the merged MILP has no preference among equally-optimal solutions, see
    module docstring) and fills windows *destination-first*, latest to
    earliest: for each window, offer it to every SKU with remaining
    unplaced demand (highest `avg_daily_sales` first, matching the
    tie-break used everywhere else demand competes for capacity) before
    moving to the next-earliest window. This matters when several
    mandatory SKUs compete for the same scarce late capacity — an
    earlier, per-entry "move this one entry as late as it'll go" pass
    processed entries in whatever order they happened to start in and
    could let one SKU's small early-window entry claim a late slot before
    a different SKU's larger pooled total got a chance, leaving that SKU
    scattered even when a smarter global packing would have consolidated
    it (found 2026-07-12 on the real bakery-21 problem: one SKU still
    landed in the very first window despite ample later slack elsewhere).
    Destination-first packing doesn't need to be globally optimal to fix
    that — it only needs to not play favorites by processing order.
    """
    if not windows:
        return movable

    n_win = len(windows)
    kratnost_by_id = {sku.product_id: sku.kratnost for sku in skus}
    category_by_id = {sku.product_id: sku.category_name for sku in skus}
    priority_by_id = {sku.product_id: sku.avg_daily_sales for sku in skus}
    window_caps = [window_capacity(w, capacity) for w in windows]
    label_idx = {w.label: idx for idx, w in enumerate(windows)}

    used_baker_min = [0.0] * n_win
    used_trays = [0.0] * n_win
    for (pid, label), qty in fixed.items():
        idx = label_idx[label]
        molding_minutes = resolve_molding_minutes(category_by_id[pid], molding_minutes_map)
        used_baker_min[idx] += qty * molding_minutes
        used_trays[idx] += qty / kratnost_by_id[pid]

    remaining_by_sku: dict[str, float] = {}
    for (pid, _label), qty in movable.items():
        remaining_by_sku[pid] = remaining_by_sku.get(pid, 0.0) + qty
    sku_order = sorted(remaining_by_sku, key=lambda pid: priority_by_id[pid], reverse=True)

    result: dict[tuple[str, str], float] = {}
    for dst_idx in range(n_win - 1, -1, -1):
        for pid in sku_order:
            remaining_qty = remaining_by_sku.get(pid, 0.0)
            if remaining_qty <= 1e-9:
                continue
            kratnost = kratnost_by_id[pid]
            molding_minutes = resolve_molding_minutes(category_by_id[pid], molding_minutes_map)
            remaining_baker_min = window_caps[dst_idx].baker_minutes - used_baker_min[dst_idx]
            remaining_trays = window_caps[dst_idx].tray_slots - used_trays[dst_idx]
            max_by_baker_min = remaining_baker_min / molding_minutes if molding_minutes > 0 else float("inf")
            movable_trays = max(0.0, min(remaining_qty / kratnost, remaining_trays, max_by_baker_min))
            movable_trays = float(np.floor(movable_trays + 1e-9))
            if movable_trays <= 0:
                continue
            move_qty = movable_trays * kratnost
            key = (pid, windows[dst_idx].label)
            result[key] = result.get(key, 0.0) + move_qty
            used_baker_min[dst_idx] += move_qty * molding_minutes
            used_trays[dst_idx] += movable_trays
            remaining_by_sku[pid] -= move_qty
    return result


def _split_tail(
    produced_by_window: dict[str, float], windows: list[Window], mandatory_amount: float
) -> tuple[dict[str, float], dict[str, float]]:
    """Split one SKU's per-window production into (regular_part,
    mandatory_part) by claiming the last `mandatory_amount` units, working
    backward from the last window — splitting a single window's quantity if
    the boundary falls inside it. Pure bookkeeping for rendering; the
    numbers themselves were already decided by the solver (and the shift
    pass above), so this never changes what gets produced, only how it's
    labeled.
    """
    regular_part = dict(produced_by_window)
    mandatory_part: dict[str, float] = {}
    remaining = mandatory_amount
    for window in reversed(windows):
        if remaining <= 1e-9:
            break
        label = window.label
        qty = regular_part.get(label, 0.0)
        if qty <= 0:
            continue
        take = min(qty, remaining)
        regular_part[label] = qty - take
        if regular_part[label] <= 1e-9:
            del regular_part[label]
        mandatory_part[label] = mandatory_part.get(label, 0.0) + take
        remaining -= take
    return regular_part, mandatory_part


def allocate_milp_detailed(
    skus: list[SkuDemand],
    windows: list[Window],
    capacity: CapacityConfig,
    molding_minutes_map: dict[str, float],
) -> tuple[
    dict[tuple[str, str], float],
    dict[tuple[str, str], float],
    dict[tuple[str, str], float],
    dict[str, float],
    dict[str, float],
]:
    """Solve and return (regular_allocation, defrost_allocation, two_day_allocation,
    shortfall_by_sku, defrost_shortfall_by_sku). See module docstring for the model.

    `defrost_shortfall_by_sku` is just the defrost component on its own —
    `rendering.py` subtracts it back out of `shortfall_by_sku` when
    computing the Итого column, which represents *today's own* demand
    (regular + двухдневка) and deliberately excludes the extra defrost
    batch baked for tomorrow.
    """
    if not windows or not skus:
        return {}, {}, {}, {}, {}

    n_sku = len(skus)
    n_win = len(windows)
    last_w = n_win - 1

    raw_demand = np.array([window_demand(sku, windows) for sku in skus])  # (n_sku, n_win)
    cum_demand = np.cumsum(raw_demand, axis=1)
    mandatory_extra = np.array(
        [defrost_demand(sku) + two_day_demand(sku) for sku in skus]
    )  # (n_sku,) — at most one of the two terms is ever nonzero per SKU

    kratnost = np.array([sku.kratnost for sku in skus], dtype=float)
    molding_min = np.array(
        [resolve_molding_minutes(sku.category_name, molding_minutes_map) for sku in skus],
        dtype=float,
    )
    priority = np.array([sku.avg_daily_sales for sku in skus], dtype=float)

    n_regular = n_sku * n_win
    # trays, shortfall_regular: each n_sku*n_win. mandatory_shortfall: n_sku
    # (one day-total checkpoint, not per-window — see module docstring).
    n_vars = 2 * n_regular + n_sku

    def trays_idx(i: int, w: int) -> int:
        return i * n_win + w

    def shortfall_idx(i: int, w: int) -> int:
        return n_regular + i * n_win + w

    def mandatory_shortfall_idx(i: int) -> int:
        return 2 * n_regular + i

    c = np.zeros(n_vars)
    for i in range(n_sku):
        c[[shortfall_idx(i, w) for w in range(n_win)]] = priority[i]
        c[mandatory_shortfall_idx(i)] = MANDATORY_SHORTFALL_WEIGHT
        c[[trays_idx(i, w) for w in range(n_win)]] = ANTI_WASTE_WEIGHT

    # Regular cumulative coverage: kratnost[i] * sum_{w'<=w} trays[i,w'] + shortfall_regular[i,w] >= cum_demand[i,w]
    n_cov_rows = n_sku * n_win
    a_cov = np.zeros((n_cov_rows, n_vars))
    lb_cov = np.zeros(n_cov_rows)
    row = 0
    for i in range(n_sku):
        for w in range(n_win):
            for w2 in range(w + 1):
                a_cov[row, trays_idx(i, w2)] = kratnost[i]
            a_cov[row, shortfall_idx(i, w)] = 1.0
            lb_cov[row] = cum_demand[i, w]
            row += 1
    coverage_constraint = LinearConstraint(a_cov, lb_cov, np.full(n_cov_rows, np.inf))

    # Combined (regular + дефрост/двухдневка) coverage, last checkpoint only
    # — and *only* for SKUs that actually have a mandatory extra. For a
    # regular-only SKU this row would otherwise exactly duplicate its own
    # regular-checkpoint target, forcing mandatory_shortfall to absorb the
    # same gap as shortfall_regular and so silently taxing every regular
    # SKU's own shortfall at MANDATORY_SHORTFALL_WEIGHT too — which defeats
    # priority-based rationing among regular SKUs entirely (caught via
    # test_two_day_always_wins_capacity_over_higher_priority_regular_sku).
    # -inf makes the row a no-op: mandatory_shortfall then has no reason to
    # be anything but its own lower bound, 0.
    a_mand = np.zeros((n_sku, n_vars))
    lb_mand = np.full(n_sku, -np.inf)
    for i in range(n_sku):
        for w2 in range(n_win):
            a_mand[i, trays_idx(i, w2)] = kratnost[i]
        a_mand[i, mandatory_shortfall_idx(i)] = 1.0
        if mandatory_extra[i] > 1e-9:
            lb_mand[i] = cum_demand[i, last_w] + mandatory_extra[i]
    mandatory_constraint = LinearConstraint(a_mand, lb_mand, np.full(n_sku, np.inf))

    # Per-window capacity: baker-minutes and tray-slots, one shared trays pool.
    window_caps = [window_capacity(w, capacity) for w in windows]
    a_cap = np.zeros((2 * n_win, n_vars))
    ub_cap = np.zeros(2 * n_win)
    for w in range(n_win):
        for i in range(n_sku):
            a_cap[2 * w, trays_idx(i, w)] = kratnost[i] * molding_min[i]
            a_cap[2 * w + 1, trays_idx(i, w)] = 1.0
        ub_cap[2 * w] = window_caps[w].baker_minutes
        ub_cap[2 * w + 1] = window_caps[w].tray_slots
    capacity_constraint = LinearConstraint(a_cap, np.full(2 * n_win, -np.inf), ub_cap)

    integrality = np.zeros(n_vars)
    integrality[:n_regular] = 1  # trays integer; both shortfall families stay continuous

    result = milp(
        c,
        constraints=[coverage_constraint, mandatory_constraint, capacity_constraint],
        integrality=integrality,
        bounds=Bounds(lb=0, ub=np.inf),
        options={"time_limit": SOLVE_TIME_LIMIT_SECONDS, "mip_rel_gap": MIP_REL_GAP},
    )
    if result.x is None:
        raise RuntimeError(f"MILP solve failed: {result.message}")

    # Raw per-(sku, window) production straight from the solver — split
    # BEFORE any shifting, so regular's own per-window placement (its real
    # sales-curve checkpoints) is captured before the mandatory portion is
    # allowed to move (see module docstring).
    regular: dict[tuple[str, str], float] = {}
    mandatory: dict[tuple[str, str], float] = {}  # defrost + two_day, not yet split by type
    shortfall_by_sku: dict[str, float] = {}
    defrost_shortfall_by_sku: dict[str, float] = {}

    for i, sku in enumerate(skus):
        produced_by_window: dict[str, float] = {}
        for w in range(n_win):
            trays = round(result.x[trays_idx(i, w)])
            if trays > 0:
                produced_by_window[windows[w].label] = trays * sku.kratnost

        if mandatory_extra[i] > 1e-9:
            # Claim (total actually produced) - (regular's own rounded-up
            # minimum), not the raw continuous mandatory_extra[i] demand or
            # the raw continuous cum_demand[i, last_w] — neither is a
            # kratnost multiple (e.g. 19.97 for kratnost=20), so subtracting
            # them directly would split off a fractional "leftover" instead
            # of a clean whole-tray amount, which then can't be shifted (a
            # fractional remainder floors to 0 movable trays) and shows up
            # as fragments scattered across whichever windows the raw solve
            # happened to touch. Rounding regular's own target up to the
            # nearest kratnost multiple first gives the true tray boundary:
            # anything the solver produced beyond that minimum exists only
            # because of the mandatory addition.
            total_produced = sum(produced_by_window.values())
            regular_only_min = (
                math.ceil(cum_demand[i, last_w] / kratnost[i] - 1e-9) * kratnost[i] if kratnost[i] > 0 else 0.0
            )
            mandatory_to_claim = max(0.0, total_produced - regular_only_min)
            regular_part, mandatory_part = _split_tail(produced_by_window, windows, mandatory_to_claim)
            for label, qty in regular_part.items():
                regular[(sku.product_id, label)] = qty
            for label, qty in mandatory_part.items():
                mandatory[(sku.product_id, label)] = qty
        else:
            for label, qty in produced_by_window.items():
                regular[(sku.product_id, label)] = qty

        regular_shortfall_last = max(0.0, result.x[shortfall_idx(i, last_w)])
        mandatory_shortfall = max(0.0, result.x[mandatory_shortfall_idx(i)])
        shortfall_by_sku[sku.product_id] = max(regular_shortfall_last, mandatory_shortfall)
        defrost_shortfall_by_sku[sku.product_id] = (
            mandatory_shortfall if sku.product_name in DEFROST_SKU_NAMES else 0.0
        )

    mandatory = _shift_to_later_windows(mandatory, regular, skus, windows, capacity, molding_minutes_map)

    sku_by_id = {sku.product_id: sku for sku in skus}
    defrost_out: dict[tuple[str, str], float] = {}
    two_day_out: dict[tuple[str, str], float] = {}
    for (pid, label), qty in mandatory.items():
        target = defrost_out if sku_by_id[pid].product_name in DEFROST_SKU_NAMES else two_day_out
        target[(pid, label)] = qty

    return regular, defrost_out, two_day_out, shortfall_by_sku, defrost_shortfall_by_sku


def allocate_milp(
    skus: list[SkuDemand],
    windows: list[Window],
    capacity: CapacityConfig,
    molding_minutes_map: dict[str, float],
) -> dict[tuple[str, str], float]:
    """Return {(product_id, window.label): qty} for every SKU actually scheduled.

    Merges the regular, defrost and двухдневка allocations into one dict —
    use `allocate_milp_detailed` when they need to stay distinguishable.
    """
    regular, defrost_out, two_day_out, _shortfall_by_sku, _defrost_shortfall_by_sku = allocate_milp_detailed(
        skus, windows, capacity, molding_minutes_map
    )
    merged = dict(regular)
    for key, qty in defrost_out.items():
        merged[key] = merged.get(key, 0) + qty
    for key, qty in two_day_out.items():
        merged[key] = merged.get(key, 0) + qty
    return merged

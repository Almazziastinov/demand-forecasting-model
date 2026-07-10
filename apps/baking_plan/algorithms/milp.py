"""MILP baking-window allocator, solved whole-day-at-once via `scipy.optimize.milp`
(HiGHS backend, bundled with scipy — no extra dependency).

Model per (bakery, date):

- `trays[sku][w]` — integer >= 0, number of tray-batches of `sku` baked in
  window `w`, covering that window's own regular demand slice.
  `qty[sku][w] = trays[sku][w] * kratnost[sku]`. Zero for `is_two_day` SKUs
  in every window (see `algorithms/common.py` — their regular-window demand
  is always zero).
- `shortfall[sku][w]` — continuous >= 0, the running stock deficit at
  window `w`'s checkpoint: `kratnost[sku] * cumsum(trays)[w] + shortfall[sku][w]
  >= cum_demand[sku][w]`. Standard lost-sales/backlog LP formulation.
- `defrost_trays[sku]` / `defrost_shortfall[sku]` — a **separate** pair for
  `DEFROST_SKU_NAMES` members' overnight batch (next-day early hours), *on
  top of* their normal same-day production.
- `two_day_trays[sku]` / `two_day_shortfall[sku]` — a **separate** pair for
  `is_two_day` SKUs' last-window batch, sized to the *entire* next day's
  demand (not just early hours) — replaces all of tomorrow's regular
  production, since these SKUs have zero regular-window demand of their
  own.

  Both of these must be modeled as separate variables, not folded into the
  last window's regular demand target: if they were just added to the
  target, the cumulative-coverage constraint has no way to *forbid*
  producing that quantity in an earlier window too (cumulative production
  only needs to clear the bar by the last checkpoint, so the solver is free
  to move it earlier) — which is wrong, both only make sense at the last
  window. These variables are deliberately excluded from every window's
  regular coverage chain and only appear in the last window's capacity
  constraint.
- Capacity per window: baker-minutes and oven-tray-slots
  (`capacity.window_capacity`). The last window's capacity constraint
  additionally includes `defrost_trays` and `two_day_trays`.
- Objective: minimize the weighted sum of all three shortfall families
  (`avg_daily_sales[sku] * shortfall`) — higher-selling SKUs are costlier to
  leave short, the MILP's analogue of the greedy allocator's priority
  ranking.

Dough-group cohesion is **not** a hard constraint here (see project plan /
memory for the rationale) — it's evaluated as a comparison metric instead,
in `scripts/compare_baking_algorithms.py`.
"""

from __future__ import annotations

# ruff: noqa: E501
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from ..capacity import CapacityConfig, resolve_molding_minutes, window_capacity
from ..demand import SkuDemand
from ..templates import Window
from .common import defrost_demand, two_day_demand, window_demand

# Safety net, not a tuning knob: at realistic scale (~40+ SKUs) with demand
# spread across the full day (see algorithms/common.py's proportional
# rescaling), capacity is genuinely scarce in most windows simultaneously —
# a real bin-packing-hard MILP, not a bug — and HiGHS's branch-and-bound can
# take exponentially longer as SKU count grows (measured 2026-07-10: 35
# SKUs ~0.1s, 39 SKUs ~5s, 40 SKUs did not return in 45s). Capping runtime
# trades a certified-optimal solution for a bounded one — HiGHS returns its
# best feasible solution found within the budget, which is usually close to
# optimal (`mip_rel_gap` also lets it stop early once within 2% of proven
# optimal, rather than always spending the full budget).
SOLVE_TIME_LIMIT_SECONDS = 15

# Дефрост and двухдневка are business-mandatory — always baked in full in
# the last window, never traded away for regular production regardless of
# either side's avg_daily_sales. 10_000 comfortably dominates any realistic
# priority value (avg_daily_sales is typically single/double digits) without
# the numerical-conditioning risk of a much larger coefficient spread.
MANDATORY_SHORTFALL_WEIGHT = 10_000.0

# A relative gap applied to an objective dominated by MANDATORY_SHORTFALL_WEIGHT
# can let the solver settle for a "good enough" solution that still leaves a
# small amount of *mandatory* shortfall unresolved even when spare capacity
# exists to fix it (observed 2026-07-10: 2.1 units of двухдневка shortfall
# left on the table with 10 spare tray-slots available). A tighter gap
# closes that slop; kept well above 0 to still cap worst-case solve time.
MIP_REL_GAP = 0.005


def allocate_milp_detailed(
    skus: list[SkuDemand],
    windows: list[Window],
    capacity: CapacityConfig,
    molding_minutes_map: dict[str, int],
) -> tuple[dict[tuple[str, str], float], dict[str, float], dict[str, float]]:
    """Solve and return (regular_allocation, defrost_allocation, two_day_allocation).

    Kept apart so callers that render the plan can distinguish a plain
    same-day quantity, a "(ночная дефр)" overnight top-up, and a двухдневка
    SKU's full-next-day batch. `allocate_milp()` below merges all three
    into the last window's cell for callers that only care about total
    quantity per (SKU, window).
    """
    if not windows or not skus:
        return {}, {}, {}

    n_sku = len(skus)
    n_win = len(windows)
    last_w = n_win - 1

    raw_demand = np.array([window_demand(sku, windows) for sku in skus])  # (n_sku, n_win)
    cum_demand = np.cumsum(raw_demand, axis=1)
    defrost = np.array([defrost_demand(sku) for sku in skus])  # (n_sku,)
    two_day = np.array([two_day_demand(sku) for sku in skus])  # (n_sku,)

    kratnost = np.array([sku.kratnost for sku in skus], dtype=float)
    molding_min = np.array(
        [resolve_molding_minutes(sku.category_name, molding_minutes_map) for sku in skus],
        dtype=float,
    )
    priority = np.array([sku.avg_daily_sales for sku in skus], dtype=float)

    n_regular = n_sku * n_win
    n_vars = 2 * n_regular + 4 * n_sku

    def trays_idx(i: int, w: int) -> int:
        return i * n_win + w

    def shortfall_idx(i: int, w: int) -> int:
        return n_regular + i * n_win + w

    def defrost_trays_idx(i: int) -> int:
        return 2 * n_regular + i

    def defrost_shortfall_idx(i: int) -> int:
        return 2 * n_regular + n_sku + i

    def two_day_trays_idx(i: int) -> int:
        return 2 * n_regular + 2 * n_sku + i

    def two_day_shortfall_idx(i: int) -> int:
        return 2 * n_regular + 3 * n_sku + i

    c = np.zeros(n_vars)
    for i in range(n_sku):
        c[[shortfall_idx(i, w) for w in range(n_win)]] = priority[i]
        # Дефрост/двухдневка must always be baked in full — a weight far
        # above any realistic avg_daily_sales makes the solver treat their
        # shortfall as effectively forbidden (never traded away for regular
        # production) while still using an LP penalty rather than a hard
        # equality, so a genuinely infeasible edge case degrades gracefully
        # instead of erroring the whole solve. Whatever last-window capacity
        # remains after satisfying them is then optimized for regular
        # production, same objective as everyone else.
        c[defrost_shortfall_idx(i)] = MANDATORY_SHORTFALL_WEIGHT
        c[two_day_shortfall_idx(i)] = MANDATORY_SHORTFALL_WEIGHT

    # Regular cumulative coverage: kratnost[i] * sum_{w'<=w} trays[i,w'] + shortfall[i,w] >= cum_demand[i,w]
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

    # Defrost coverage: a separate, single checkpoint per SKU (last window only).
    a_defrost = np.zeros((n_sku, n_vars))
    for i in range(n_sku):
        a_defrost[i, defrost_trays_idx(i)] = kratnost[i]
        a_defrost[i, defrost_shortfall_idx(i)] = 1.0
    defrost_constraint = LinearConstraint(a_defrost, defrost, np.full(n_sku, np.inf))

    # Двухдневка coverage: a separate, single checkpoint per SKU (last window only).
    a_two_day = np.zeros((n_sku, n_vars))
    for i in range(n_sku):
        a_two_day[i, two_day_trays_idx(i)] = kratnost[i]
        a_two_day[i, two_day_shortfall_idx(i)] = 1.0
    two_day_constraint = LinearConstraint(a_two_day, two_day, np.full(n_sku, np.inf))

    # Per-window capacity: baker-minutes and tray-slots. Last window also
    # carries the defrost and двухдневка batches' resource usage.
    window_caps = [window_capacity(w, capacity) for w in windows]
    a_cap = np.zeros((2 * n_win, n_vars))
    ub_cap = np.zeros(2 * n_win)
    for w in range(n_win):
        for i in range(n_sku):
            a_cap[2 * w, trays_idx(i, w)] = kratnost[i] * molding_min[i]
            a_cap[2 * w + 1, trays_idx(i, w)] = 1.0
        ub_cap[2 * w] = window_caps[w].baker_minutes
        ub_cap[2 * w + 1] = window_caps[w].tray_slots
    for i in range(n_sku):
        a_cap[2 * last_w, defrost_trays_idx(i)] = kratnost[i] * molding_min[i]
        a_cap[2 * last_w + 1, defrost_trays_idx(i)] = 1.0
        a_cap[2 * last_w, two_day_trays_idx(i)] = kratnost[i] * molding_min[i]
        a_cap[2 * last_w + 1, two_day_trays_idx(i)] = 1.0
    capacity_constraint = LinearConstraint(a_cap, np.full(2 * n_win, -np.inf), ub_cap)

    integrality = np.zeros(n_vars)
    integrality[:n_regular] = 1  # regular trays integer
    integrality[2 * n_regular : 2 * n_regular + n_sku] = 1  # defrost trays integer
    integrality[2 * n_regular + 2 * n_sku : 2 * n_regular + 3 * n_sku] = 1  # two_day trays integer

    result = milp(
        c,
        constraints=[coverage_constraint, defrost_constraint, two_day_constraint, capacity_constraint],
        integrality=integrality,
        bounds=Bounds(lb=0, ub=np.inf),
        options={"time_limit": SOLVE_TIME_LIMIT_SECONDS, "mip_rel_gap": MIP_REL_GAP},
    )
    if result.x is None:
        raise RuntimeError(f"MILP solve failed: {result.message}")

    regular: dict[tuple[str, str], float] = {}
    defrost_out: dict[str, float] = {}
    two_day_out: dict[str, float] = {}
    for i, sku in enumerate(skus):
        for w in range(n_win):
            trays = round(result.x[trays_idx(i, w)])
            if trays > 0:
                key = (sku.product_id, windows[w].label)
                regular[key] = regular.get(key, 0) + trays * sku.kratnost
        defrost_trays = round(result.x[defrost_trays_idx(i)])
        if defrost_trays > 0:
            defrost_out[sku.product_id] = defrost_trays * sku.kratnost
        two_day_trays = round(result.x[two_day_trays_idx(i)])
        if two_day_trays > 0:
            two_day_out[sku.product_id] = two_day_trays * sku.kratnost
    return regular, defrost_out, two_day_out


def allocate_milp(
    skus: list[SkuDemand],
    windows: list[Window],
    capacity: CapacityConfig,
    molding_minutes_map: dict[str, int],
) -> dict[tuple[str, str], float]:
    """Return {(product_id, window.label): qty} for every SKU actually scheduled.

    Merges the defrost and двухдневка batches into the last window's cell —
    use `allocate_milp_detailed` when they need to stay distinguishable.
    """
    regular, defrost_out, two_day_out = allocate_milp_detailed(
        skus, windows, capacity, molding_minutes_map
    )
    if not windows:
        return regular
    last_label = windows[-1].label
    merged = dict(regular)
    for pid, qty in defrost_out.items():
        key = (pid, last_label)
        merged[key] = merged.get(key, 0) + qty
    for pid, qty in two_day_out.items():
        key = (pid, last_label)
        merged[key] = merged.get(key, 0) + qty
    return merged

"""Greedy, window-by-window baking-window allocator.

Processes windows in chronological order, tracking a signed stock balance
per SKU: `stock += qty_baked - demand_this_window`. Rounding a small need up
to a full kratnost batch often overproduces relative to that window's own
demand — the excess must carry forward as a credit that offsets later
windows' needs, or the same SKU would get a fresh full batch every window
that has any nonzero residual demand. When stock goes negative (a window's
demand outstrips available stock), the deficit becomes this window's
production target.

When capacity is short to cover every target, ration by dough-group: groups
are ranked by their highest-priority member's average sales, and each group
is scheduled *whole* or not at all — a group is never split across a
capacity shortfall, it either fully fits in this window or its entire
shortfall stays negative stock (rolls into next window's target).

The last window gets extra components added on top of regular demand (see
`algorithms/common.py` for the дефрост vs двухдневка distinction):
`defrost_demand` (only for `DEFROST_SKU_NAMES` members) and `two_day_demand`
(only for `is_two_day` SKUs — whose regular-window demand is already zero
every other window, per `window_demand`).
"""

from __future__ import annotations

# ruff: noqa: E501
import math

from ..capacity import (
    CapacityConfig,
    effective_kratnost,
    is_core_baking_category,
    resolve_molding_minutes_for_sku,
    window_capacity,
)
from ..demand_milp import SkuDemand
from ..templates import Window
from .common import defrost_demand, two_day_demand, window_demand


def _group_and_rank(candidates: list[SkuDemand]) -> list[list[SkuDemand]]:
    groups: dict[str, list[SkuDemand]] = {}
    for sku in candidates:
        groups.setdefault(sku.dough_group, []).append(sku)
    return sorted(
        groups.values(),
        key=lambda members: max(s.avg_daily_sales for s in members),
        reverse=True,
    )


def _ration(
    candidates: list[SkuDemand],
    trays_needed: dict[str, int],
    core_labor_min_needed: dict[str, float],
    helper_labor_min_needed: dict[str, float],
    tray_slots: int,
    baker_minutes: float,
    helper_minutes: float,
    kratnost_by_id: dict[str, int],
    remaining_core_units: float,
) -> dict[str, float]:
    scheduled: dict[str, float] = {}
    remaining_trays = tray_slots
    remaining_baker_min = baker_minutes
    remaining_helper_min = helper_minutes
    for group_members in _group_and_rank(candidates):
        group_trays = sum(trays_needed[s.product_id] for s in group_members)
        group_baker_min = sum(core_labor_min_needed[s.product_id] for s in group_members)
        group_helper_min = sum(helper_labor_min_needed[s.product_id] for s in group_members)
        group_core_units = sum(
            trays_needed[s.product_id] * kratnost_by_id[s.product_id]
            for s in group_members
            if is_core_baking_category(s.category_name)
        )
        if (
            group_trays <= remaining_trays
            and group_baker_min <= remaining_baker_min
            and group_helper_min <= remaining_helper_min
            and group_core_units <= remaining_core_units
        ):
            for sku in group_members:
                scheduled[sku.product_id] = trays_needed[sku.product_id] * kratnost_by_id[sku.product_id]
            remaining_trays -= group_trays
            remaining_baker_min -= group_baker_min
            remaining_helper_min -= group_helper_min
            remaining_core_units -= group_core_units
        # else: whole group skipped this window; its target rolls into next window's need
    return scheduled


def allocate_greedy(
    skus: list[SkuDemand],
    windows: list[Window],
    capacity: CapacityConfig,
    molding_minutes_map: dict[str, int],
    core_unit_cap: int | None = None,
) -> dict[tuple[str, str], float]:
    """Return {(product_id, window.label): qty} for every SKU actually scheduled."""
    if not windows:
        return {}

    base_demand = {sku.product_id: window_demand(sku, windows) for sku in skus}
    stock = dict.fromkeys((sku.product_id for sku in skus), 0.0)
    result: dict[tuple[str, str], float] = {}
    remaining_core_units = float("inf") if core_unit_cap is None else float(core_unit_cap)

    last_index = len(windows) - 1
    for w, window in enumerate(windows):
        raw_demand: dict[str, float] = {}
        for sku in skus:
            pid = sku.product_id
            amount = base_demand[pid][w]
            if w == last_index:
                amount += defrost_demand(sku) + two_day_demand(sku)
            raw_demand[pid] = amount

        # Production target: only what's needed to clear a stock deficit —
        # existing positive stock (overproduction carried from an earlier
        # window) offsets this window's demand first.
        target = {
            sku.product_id: max(0.0, raw_demand[sku.product_id] - stock[sku.product_id])
            for sku in skus
        }
        candidates = [sku for sku in skus if target[sku.product_id] > 1e-9]

        scheduled: dict[str, float] = {}
        if candidates:
            trays_needed: dict[str, int] = {}
            core_labor_min_needed: dict[str, float] = {}
            helper_labor_min_needed: dict[str, float] = {}
            kratnost_by_id: dict[str, int] = {}
            for sku in candidates:
                kratnost = effective_kratnost(sku)
                kratnost_by_id[sku.product_id] = kratnost
                trays = math.ceil(target[sku.product_id] / kratnost)
                trays_needed[sku.product_id] = trays
                molding_min = resolve_molding_minutes_for_sku(sku, molding_minutes_map)
                labor_min = trays * kratnost * molding_min
                if is_core_baking_category(sku.category_name):
                    core_labor_min_needed[sku.product_id] = labor_min
                    helper_labor_min_needed[sku.product_id] = 0.0
                else:
                    core_labor_min_needed[sku.product_id] = 0.0
                    helper_labor_min_needed[sku.product_id] = labor_min

            cap = window_capacity(window, capacity)
            total_trays = sum(trays_needed.values())
            total_baker_min = sum(core_labor_min_needed.values())
            total_helper_min = sum(helper_labor_min_needed.values())

            if (
                total_trays <= cap.tray_slots
                and total_baker_min <= cap.baker_minutes
                and total_helper_min <= cap.helper_minutes
                and sum(
                    trays_needed[sku.product_id] * kratnost_by_id[sku.product_id]
                    for sku in candidates
                    if is_core_baking_category(sku.category_name)
                )
                <= remaining_core_units
            ):
                scheduled = {
                    sku.product_id: trays_needed[sku.product_id] * kratnost_by_id[sku.product_id]
                    for sku in candidates
                }
            else:
                scheduled = _ration(
                    candidates,
                    trays_needed,
                    core_labor_min_needed,
                    helper_labor_min_needed,
                    cap.tray_slots,
                    cap.baker_minutes,
                    cap.helper_minutes,
                    kratnost_by_id,
                    remaining_core_units,
                )

        for sku in skus:
            pid = sku.product_id
            qty = scheduled.get(pid, 0.0)
            if qty > 0:
                result[(pid, window.label)] = qty
                if is_core_baking_category(sku.category_name):
                    remaining_core_units -= qty
            stock[pid] = stock[pid] - raw_demand[pid] + qty

    return result

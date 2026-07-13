"""Public entrypoint for the baking-plan package.

`router.py` (and nothing else outside this package) calls into this module.
"""

from __future__ import annotations

# ruff: noqa: E501
from io import BytesIO

import openpyxl

from . import capacity, demand
from .algorithms.milp import allocate_milp_detailed
from .demand import SkuDemand
from .rendering import render_workbook
from .templates import BASE_TEMPLATE_PATH, Window, parse_windows

# Below this, a shortfall is treated as solver/rounding noise, not a real gap.
SHORTFALL_TOLERANCE = 1e-6

# A window resource at or above this utilization is considered "maxed out"
# for the purpose of the capacity-shortage recommendation below.
UTILIZATION_THRESHOLD = 0.99


def _has_shortfall(shortfall_by_sku: dict[str, float]) -> bool:
    return any(value > SHORTFALL_TOLERANCE for value in shortfall_by_sku.values())


def _capacity_recommendation(
    skus: list[SkuDemand],
    windows: list[Window],
    capacity_config: capacity.CapacityConfig,
    molding_map: dict[str, float],
    regular_alloc: dict[tuple[str, str], float],
    defrost_alloc: dict[tuple[str, str], float],
    two_day_alloc: dict[tuple[str, str], float],
) -> list[str]:
    """Scan every window's actual resource usage (at the pace that was just
    solved) and report which physical resource(s) are pinned at capacity
    somewhere in the day — the reason a shortfall survived even at the floor
    molding pace. Core baker-minutes, helper-minutes, and tray-slots are
    independent physical resources (see capacity.window_capacity), so any
    combination can be binding; only report what's actually maxed. Regular,
    defrost, and two_day allocations can each land in any window now (see
    algorithms/milp.py), so all three are checked in every window, not just
    the last.
    """
    needs_baker = False
    needs_helper = False
    needs_oven = False
    for window in windows:
        wc = capacity.window_capacity(window, capacity_config)
        used_baker_minutes = 0.0
        used_helper_minutes = 0.0
        used_trays = 0.0
        for sku in skus:
            key = (sku.product_id, window.label)
            qty = regular_alloc.get(key, 0.0) + defrost_alloc.get(key, 0.0) + two_day_alloc.get(key, 0.0)
            if not qty:
                continue
            molding_minutes = capacity.resolve_molding_minutes_for_sku(sku, molding_map)
            if capacity.is_core_baking_category(sku.category_name):
                used_baker_minutes += qty * molding_minutes
            else:
                used_helper_minutes += qty * molding_minutes
            used_trays += qty / capacity.effective_kratnost(sku)
        if wc.baker_minutes > 0 and used_baker_minutes / wc.baker_minutes >= UTILIZATION_THRESHOLD:
            needs_baker = True
        if wc.helper_minutes > 0 and used_helper_minutes / wc.helper_minutes >= UTILIZATION_THRESHOLD:
            needs_helper = True
        if wc.tray_slots > 0 and used_trays / wc.tray_slots >= UTILIZATION_THRESHOLD:
            needs_oven = True

    recommendation = []
    if needs_baker:
        recommendation.append("пекарь")
    if needs_helper:
        recommendation.append("помощник пекаря")
    if needs_oven:
        recommendation.append("печь")
    return recommendation


def build_baking_plan_workbook(
    *,
    run_id: str,
    forecast_date: str,
    bakery_id: int,
    bakery_name: str,
    city: str,
) -> bytes:
    """Build the baking-plan .xlsx file content for one bakery/date.

    Returns the raw bytes of the generated workbook.
    """
    skus, _skipped = demand.build_sku_demand(
        run_id=run_id, forecast_date=forecast_date, bakery_id=bakery_id, city=city
    )
    if not skus:
        raise RuntimeError(f"No bakeable SKUs with metadata found for bakery_id={bakery_id}")

    windows_workbook = openpyxl.load_workbook(BASE_TEMPLATE_PATH, data_only=True)
    windows = parse_windows(windows_workbook)

    capacity_config = capacity.get_capacity_config(bakery_id)
    molding_map = capacity.get_molding_minutes_map()

    normal_core_cap = capacity.daily_core_unit_cap(capacity_config, peak=False)
    peak_core_cap = capacity.daily_core_unit_cap(capacity_config, peak=True)
    regular_alloc, defrost_alloc, two_day_alloc, shortfall_by_sku, defrost_shortfall_by_sku = (
        allocate_milp_detailed(skus, windows, capacity_config, molding_map, core_unit_cap=normal_core_cap)
    )

    # Normal pace can't cover today's demand: before concluding this bakery
    # genuinely lacks capacity, retry at the floor pace (fastest realistic
    # molding speed a baker can sustain, confirmed by the user 2026-07-11 —
    # see capacity.MOLDING_MINUTES_FLOOR). Most days this branch never runs.
    capacity_note: str | None = None
    if _has_shortfall(shortfall_by_sku):
        regular_alloc, defrost_alloc, two_day_alloc, shortfall_by_sku, defrost_shortfall_by_sku = (
            allocate_milp_detailed(
                skus,
                windows,
                capacity_config,
                capacity.MOLDING_MINUTES_FLOOR,
                core_unit_cap=peak_core_cap,
            )
        )
        if _has_shortfall(shortfall_by_sku):
            missing = _capacity_recommendation(
                skus,
                windows,
                capacity_config,
                capacity.MOLDING_MINUTES_FLOOR,
                regular_alloc,
                defrost_alloc,
                two_day_alloc,
            )
            if missing:
                capacity_note = (
                    "Даже при ускоренном темпе мелкоштучки (54 сек/шт; пироги считаются по тесто-группе) "
                    f"план не выполняется полностью — требуется дополнительно: {', '.join(missing)}."
                )
        if capacity_note is None:
            capacity_note = (
                "Для выполнения плана сегодня требуется ускоренный темп лепки: "
                "54 сек/шт для мелкоштучки; пироги считаются по тесто-группе."
            )

    workbook = render_workbook(
        bakery_name=bakery_name,
        forecast_date=forecast_date,
        windows=windows,
        skus=skus,
        regular_alloc=regular_alloc,
        defrost_alloc=defrost_alloc,
        two_day_alloc=two_day_alloc,
        shortfall_by_sku=shortfall_by_sku,
        defrost_shortfall_by_sku=defrost_shortfall_by_sku,
        capacity_note=capacity_note,
    )
    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()

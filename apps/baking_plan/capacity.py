"""Oven/baker capacity and category molding-time lookups.

Reads `baking_capacity_config` (bakery override, falls back to the global
`bakery_id IS NULL` default row) and `baking_category_molding_minutes`
(category -> minutes/unit, `''` is the default-fallback category).

The optimizer treats core bakery categories and helper-owned categories as
separate labor pools: bakers work on pies/baked goods, while baker assistants
can take products outside those core categories. Oven tray capacity remains
shared.

Also defines the floor (fastest realistic) molding pace for non-pie
categories — used by the pace-search in `service.py`: if the normal pace
can't fit demand into capacity, the solver is retried at this floor before
falling back to a capacity-shortage recommendation. Pie categories are now
resolved by SKU/dough group instead: sand-dough pies use 4 min/unit, other
pies use 2 min/unit.
"""

from __future__ import annotations

# ruff: noqa: E501
from dataclasses import dataclass

from .constants import NIGHT_PREP_LABOR_MINUTES_BY_SKU
from ._clickhouse import get_client, records, table_name
from .demand_milp import SkuDemand

from .templates import Window

CAPACITY_TABLE = table_name("baking_capacity_config")
MOLDING_MINUTES_TABLE = table_name("baking_category_molding_minutes")


@dataclass(frozen=True)
class CapacityConfig:
    bakers_count: int
    ovens_count: int
    trays_per_oven_batch: int
    bake_minutes: int
    helpers_count: int = 2


@dataclass(frozen=True)
class WindowCapacity:
    baker_minutes: float
    helper_minutes: float
    tray_slots: int


CORE_BAKING_CATEGORIES = {
    "Выпечка сытная",
    "Выпечка сладкая",
    "Пироги сытные",
    "Пироги сладкие",
}
PIE_CATEGORIES = {
    "Пироги сытные",
    "Пироги сладкие",
}
SAND_DOUGH_MARKER = "песоч"
PIE_EFFECTIVE_KRATNOST = 4
NORMAL_DAILY_CORE_UNITS_PER_BAKER = 600
PEAK_DAILY_CORE_UNITS_PER_BAKER = 800


def get_capacity_config(bakery_id: int) -> CapacityConfig:
    client = get_client()
    query = f"""
        select bakers_count, ovens_count, trays_per_oven_batch, bake_minutes
        from {CAPACITY_TABLE} final
        where is_active = 1
          and (bakery_id = %(bakery_id)s or bakery_id is null)
        order by (bakery_id is null) asc, valid_from desc
        limit 1
        """
    df = client.query_df(query, parameters={"bakery_id": bakery_id})
    rows = records(df)
    if not rows:
        raise RuntimeError(f"No baking_capacity_config row found for bakery_id={bakery_id}")
    row = rows[0]
    return CapacityConfig(
        bakers_count=int(row["bakers_count"]),
        ovens_count=int(row["ovens_count"]),
        trays_per_oven_batch=int(row["trays_per_oven_batch"]),
        bake_minutes=int(row["bake_minutes"]),
    )


def get_molding_minutes_map() -> dict[str, int]:
    client = get_client()
    df = client.query_df(
        f"select category_name, minutes_per_unit from {MOLDING_MINUTES_TABLE} final where is_active = 1"
    )
    return {row["category_name"]: int(row["minutes_per_unit"]) for row in records(df)}


def resolve_molding_minutes(category_name: str, minutes_map: dict[str, int]) -> int:
    if category_name in minutes_map:
        return minutes_map[category_name]
    return minutes_map.get("", 1)


def is_core_baking_category(category_name: str) -> bool:
    return category_name in CORE_BAKING_CATEGORIES


def is_pie_category(category_name: str) -> bool:
    return category_name in PIE_CATEGORIES


def effective_kratnost(sku: SkuDemand) -> int:
    """Return oven tray capacity for the SKU.

    The historical `baking_sku_meta.kratnost` came from production multiples,
    not always from physical oven fit. For pies, the current business input is
    explicit: one oven tray can hold 4 pies.
    """
    if is_pie_category(sku.category_name):
        return PIE_EFFECTIVE_KRATNOST
    return sku.kratnost


def resolve_molding_minutes_for_sku(
    sku: SkuDemand,
    minutes_map: dict[str, float],
) -> float:
    """Return labor minutes/unit for allocation.

    Pies depend on dough group: sand dough stays at 4 min/unit, other pie
    doughs use 2 min/unit. Non-pie baked goods use the operational 1 min/unit
    default unless a table override is present.
    """
    prep_minutes = NIGHT_PREP_LABOR_MINUTES_BY_SKU.get(sku.product_name)
    if prep_minutes is not None:
        return prep_minutes
    if is_pie_category(sku.category_name):
        dough_group = (sku.dough_group or "").lower()
        return 4.0 if SAND_DOUGH_MARKER in dough_group else 2.0
    return float(resolve_molding_minutes(sku.category_name, minutes_map))


def daily_core_unit_cap(config: CapacityConfig, *, peak: bool = False) -> int:
    units_per_baker = PEAK_DAILY_CORE_UNITS_PER_BAKER if peak else NORMAL_DAILY_CORE_UNITS_PER_BAKER
    return config.bakers_count * units_per_baker


# Floor pace (minutes/unit) for non-pie categories — see module docstring.
# Keyed the same way as `baking_category_molding_minutes` (`''` = default
# fallback).
MOLDING_MINUTES_FLOOR: dict[str, float] = {
    "": 54 / 60,
}


def resolve_molding_minutes_floor(
    category_name: str, floor_map: dict[str, float] = MOLDING_MINUTES_FLOOR
) -> float:
    if category_name in floor_map:
        return floor_map[category_name]
    return floor_map.get("", 54 / 60)


def window_capacity(window: Window, config: CapacityConfig) -> WindowCapacity:
    duration_minutes = (window.end_hour - window.start_hour) * 60
    baker_minutes = config.bakers_count * duration_minutes
    helper_minutes = config.helpers_count * duration_minutes
    bake_cycles = duration_minutes // config.bake_minutes
    tray_slots = config.ovens_count * bake_cycles * config.trays_per_oven_batch
    return WindowCapacity(baker_minutes=baker_minutes, helper_minutes=helper_minutes, tray_slots=tray_slots)

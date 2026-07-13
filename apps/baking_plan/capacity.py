"""Oven/baker capacity and category molding-time lookups.

Reads `baking_capacity_config` (bakery override, falls back to the global
`bakery_id IS NULL` default row) and `baking_category_molding_minutes`
(category -> minutes/unit, `''` is the default-fallback category).

Also defines the floor (fastest realistic) molding pace per category —
used by the pace-search in `service.py`: if the normal pace can't fit
demand into capacity, the solver is retried at this floor before falling
back to a capacity-shortage recommendation. Hardcoded like
`DEFROST_SKU_NAMES`/`MANDATORY_ASSORTMENT` elsewhere in this package — no
ClickHouse source of truth exists yet for a per-category minimum pace,
confirmed directly by the user (2026-07-11): 54s/unit for the default
(1 min normal) categories, 3:30/unit for Пироги сытные/сладкие (4 min
normal).
"""

from __future__ import annotations

# ruff: noqa: E501
from dataclasses import dataclass

from ._clickhouse import get_client, records, table_name

from .templates import Window

CAPACITY_TABLE = table_name("baking_capacity_config")
MOLDING_MINUTES_TABLE = table_name("baking_category_molding_minutes")


@dataclass(frozen=True)
class CapacityConfig:
    bakers_count: int
    ovens_count: int
    trays_per_oven_batch: int
    bake_minutes: int


@dataclass(frozen=True)
class WindowCapacity:
    baker_minutes: float
    tray_slots: int


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


# Floor pace (minutes/unit) — see module docstring. Keyed the same way as
# `baking_category_molding_minutes` (`''` = default fallback).
MOLDING_MINUTES_FLOOR: dict[str, float] = {
    "": 54 / 60,
    "Пироги сытные": 210 / 60,
    "Пироги сладкие": 210 / 60,
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
    bake_cycles = duration_minutes // config.bake_minutes
    tray_slots = config.ovens_count * bake_cycles * config.trays_per_oven_batch
    return WindowCapacity(baker_minutes=baker_minutes, tray_slots=tray_slots)

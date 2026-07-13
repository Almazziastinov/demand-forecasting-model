from __future__ import annotations

# ruff: noqa: E501
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps"))
sys.path.insert(0, str(ROOT / "apps" / "forecast_embedded"))

from baking_plan.algorithms.common import DEFROST_SKU_NAMES  # noqa: E402
from baking_plan.algorithms.milp import (  # noqa: E402
    _shift_to_later_windows,
    _split_tail,
    _swap_mandatory_labels_later,
    allocate_milp,
)
from baking_plan.capacity import (  # noqa: E402
    CapacityConfig,
    effective_kratnost,
    is_core_baking_category,
    resolve_molding_minutes_for_sku,
    window_capacity,
)
from baking_plan.demand import SkuDemand  # noqa: E402
from baking_plan.templates import Window  # noqa: E402

WINDOWS = [Window("0-4", 0, 4), Window("4-8", 4, 8)]
CAPACITY = CapacityConfig(bakers_count=1, ovens_count=1, trays_per_oven_batch=2, bake_minutes=60)
MOLDING = {"": 1, "Пироги сытные": 4}


def make_sku(
    product_id: str,
    *,
    name: str | None = None,
    dough_group: str = "G",
    kratnost: int = 1,
    category: str = "Фастфуд",
    is_two_day: bool = False,
    avg_daily_sales: float = 1.0,
    hourly_qty: dict | None = None,
    next_day_hourly_qty: dict | None = None,
) -> SkuDemand:
    return SkuDemand(
        product_id=product_id,
        product_name=name or product_id,
        category_name=category,
        dough_group=dough_group,
        kratnost=kratnost,
        station="Пекарь",
        is_two_day=is_two_day,
        is_on_demand=False,
        hourly_qty=hourly_qty or {},
        next_day_hourly_qty=next_day_hourly_qty or {},
        avg_daily_sales=avg_daily_sales,
    )


def _assert_capacity_respected(result, skus, windows, capacity, molding_map):
    sku_by_id = {s.product_id: s for s in skus}
    for window in windows:
        trays = sum(
            math.ceil(qty / effective_kratnost(sku_by_id[pid]))
            for (pid, label), qty in result.items()
            if label == window.label
        )
        baker_min = sum(
            qty * resolve_molding_minutes_for_sku(sku_by_id[pid], molding_map)
            for (pid, label), qty in result.items()
            if label == window.label and is_core_baking_category(sku_by_id[pid].category_name)
        )
        helper_min = sum(
            qty * resolve_molding_minutes_for_sku(sku_by_id[pid], molding_map)
            for (pid, label), qty in result.items()
            if label == window.label and not is_core_baking_category(sku_by_id[pid].category_name)
        )
        cap = window_capacity(window, capacity)
        assert trays <= cap.tray_slots
        assert baker_min <= cap.baker_minutes
        assert helper_min <= cap.helper_minutes


def test_prioritizes_higher_avg_sales_when_capacity_binds():
    # Single tight window: both SKUs want 5, only 8 tray-slots total, but
    # each alone needs the whole window's baker-minutes budget is generous
    # (default molding = 1 min/unit) so tray_slots (8) is the binding
    # constraint. Minimizing weighted shortfall means giving the
    # higher-priority SKU everything it needs first.
    windows = [WINDOWS[0]]
    high = make_sku("HIGH", dough_group="G1", avg_daily_sales=10, hourly_qty={0: 5})
    low = make_sku("LOW", dough_group="G2", avg_daily_sales=1, hourly_qty={0: 5})
    result = allocate_milp([high, low], windows, CAPACITY, MOLDING)

    assert result.get(("HIGH", "0-4"), 0) == 5
    assert result.get(("LOW", "0-4"), 0) == 3  # only 3 tray-slots left after HIGH
    _assert_capacity_respected(result, [high, low], windows, CAPACITY, MOLDING)


def test_night_prep_sku_uses_reduced_daytime_molding_minutes():
    prep = make_sku(
        "PREP",
        name="Жар Киш курица",
        category="Пироги сытные",
        dough_group="песочка",
    )
    regular_sand_pie = make_sku(
        "PIE",
        name="Губадия",
        category="Пироги сладкие",
        dough_group="песочка",
    )

    assert resolve_molding_minutes_for_sku(prep, MOLDING) == 1.0
    assert resolve_molding_minutes_for_sku(regular_sand_pie, MOLDING) == 4.0


def test_two_day_always_wins_capacity_over_higher_priority_regular_sku():
    # Дефрост/двухдневка are business-mandatory: always baked in full,
    # never traded away for regular production regardless of relative
    # avg_daily_sales. LOW_PRIORITY_TWO_DAY needs 5, HIGH_PRIORITY_REGULAR
    # wants 10, only 8 tray-slots total — a pure sales-priority allocator
    # would give the high-priority regular SKU everything and leave the
    # двухдневка short, but that must not happen here.
    windows = [WINDOWS[0]]
    two_day = make_sku(
        "LOW_PRIORITY_TWO_DAY",
        dough_group="G1",
        is_two_day=True,
        avg_daily_sales=1,  # far lower priority than the regular SKU
        hourly_qty={},
        next_day_hourly_qty={0: 5},
    )
    regular = make_sku(
        "HIGH_PRIORITY_REGULAR",
        dough_group="G2",
        avg_daily_sales=100,  # far higher priority than the двухдневка SKU
        hourly_qty={0: 10},
    )
    result = allocate_milp([two_day, regular], windows, CAPACITY, MOLDING)

    assert result.get(("LOW_PRIORITY_TWO_DAY", "0-4"), 0) == 5  # fully baked regardless of priority
    assert result.get(("HIGH_PRIORITY_REGULAR", "0-4"), 0) == 3  # gets whatever capacity remains
    _assert_capacity_respected(result, [two_day, regular], windows, CAPACITY, MOLDING)


def test_two_day_has_zero_regular_windows_and_full_next_day_in_last_window():
    two_day = make_sku(
        "D",
        name="Сочень",
        is_two_day=True,
        avg_daily_sales=1,
        hourly_qty={0: 99},  # must be ignored — двухдневка has zero regular demand
        next_day_hourly_qty={0: 1, 6: 3, 7: 2, 15: 1},  # full-day total = 7 (fits window capacity)
    )
    result = allocate_milp([two_day], WINDOWS, CAPACITY, MOLDING)

    assert ("D", "0-4") not in result
    # Exactly 7: ANTI_WASTE_WEIGHT keeps total production at the exact
    # demand, and _shift_to_later_windows consolidates it into the last
    # window regardless of which window the raw solve happened to use.
    assert result[("D", "4-8")] == 7


def test_defrost_is_extra_on_top_of_regular_production_not_tied_to_two_day():
    defrost_name = next(iter(DEFROST_SKU_NAMES))
    sku = make_sku(
        "F",
        name=defrost_name,
        is_two_day=False,
        avg_daily_sales=1,
        hourly_qty={0: 4},
        next_day_hourly_qty={6: 3, 7: 2, 20: 100},  # only hours 6-11 count: 3+2=5
    )
    other = make_sku(
        "G",
        name="Не в списке дефроста",
        is_two_day=False,
        avg_daily_sales=1,
        hourly_qty={},
        next_day_hourly_qty={6: 3, 7: 2},
    )
    result = allocate_milp([sku, other], WINDOWS, CAPACITY, MOLDING)

    assert result[("F", "0-4")] == 4
    assert result[("F", "4-8")] == 5
    assert ("G", "0-4") not in result
    assert ("G", "4-8") not in result


def test_capacity_never_exceeded_under_heavy_demand():
    skus = [
        make_sku("A", dough_group="GA", avg_daily_sales=10, hourly_qty={0: 20}),
        make_sku("B", dough_group="GB", avg_daily_sales=5, hourly_qty={0: 20}),
        make_sku("C", dough_group="GC", avg_daily_sales=1, hourly_qty={0: 20}),
    ]
    result = allocate_milp(skus, WINDOWS, CAPACITY, MOLDING)
    _assert_capacity_respected(result, skus, WINDOWS, CAPACITY, MOLDING)


def test_two_day_can_land_in_an_earlier_window_when_the_last_window_has_no_capacity():
    # Last window has zero duration -> zero baker-minutes and zero
    # tray-slots (window_capacity derives both from duration).
    # _shift_to_later_windows tries to move the mandatory portion there,
    # finds no capacity, and leaves it in the earlier window instead.
    windows = [Window("0-4", 0, 4), Window("4-4", 4, 4)]
    two_day = make_sku(
        "D",
        name="Сочень",
        is_two_day=True,
        avg_daily_sales=1,
        next_day_hourly_qty={0: 5},
    )
    result = allocate_milp([two_day], windows, CAPACITY, MOLDING)
    assert result.get(("D", "0-4"), 0) == 5
    assert ("D", "4-4") not in result


def test_no_gratuitous_overproduction_when_shortfall_already_zero():
    # Trays cost nothing directly in the objective (only shortfall does), so
    # without ANTI_WASTE_WEIGHT the solver could "optimally" produce far more
    # than needed once demand is fully covered. Neither the split nor the
    # shift pass change total quantity, so this stays exact (kratnost=1
    # here, so no rounding overshoot is even possible either).
    two_day = make_sku(
        "D",
        name="Сочень",
        is_two_day=True,
        avg_daily_sales=1,
        next_day_hourly_qty={0: 1, 6: 3, 7: 2, 15: 1},  # total = 7
    )
    result = allocate_milp([two_day], WINDOWS, CAPACITY, MOLDING)
    total_produced = sum(qty for (pid, _label), qty in result.items() if pid == "D")
    assert total_produced == pytest.approx(7.0)


def test_defrost_prefers_the_last_window_when_multiple_are_equally_feasible():
    # The merged MILP is indifferent to which window the defrost increment
    # lands in (see module docstring) — _shift_to_later_windows is what
    # consolidates it into the last window regardless of the solver's raw
    # (non-deterministic) choice.
    defrost_name = next(iter(DEFROST_SKU_NAMES))
    sku = make_sku(
        "F",
        name=defrost_name,
        is_two_day=False,
        avg_daily_sales=1,
        hourly_qty={},
        next_day_hourly_qty={6: 3, 7: 2},  # defrost demand = 5
    )
    result = allocate_milp([sku], WINDOWS, CAPACITY, MOLDING)
    assert result.get(("F", "4-8"), 0) == 5
    assert ("F", "0-4") not in result


def test_split_tail_claims_from_the_end_and_splits_a_window_if_needed():
    windows = [Window("0-4", 0, 4), Window("4-8", 4, 8), Window("8-12", 8, 12)]
    produced_by_window = {"0-4": 10.0, "4-8": 10.0, "8-12": 10.0}
    # Mandatory amount (15) spans the last two windows: all of "8-12" (10)
    # plus 5 of "4-8"'s 10, splitting that window between regular/mandatory.
    regular_part, mandatory_part = _split_tail(produced_by_window, windows, 15.0)
    assert regular_part == {"0-4": 10.0, "4-8": 5.0}
    assert mandatory_part == {"8-12": 10.0, "4-8": 5.0}


def test_shift_to_later_windows_routes_around_fixed_usage_and_splits_across_windows():
    # "8-12" (last) already has 6 of its 8 tray-slots used by fixed
    # (regular) production for a different SKU — the movable mandatory
    # portion (3 trays) can only fit 2 there, so 1 tray spills back to
    # "4-8" rather than being stranded in "0-4".
    windows = [Window("0-4", 0, 4), Window("4-8", 4, 8), Window("8-12", 8, 12)]
    f_sku = make_sku("F", kratnost=10, category="Фастфуд")
    other_sku = make_sku("OTHER", kratnost=10, category="Фастфуд")
    movable = {("F", "0-4"): 30.0}  # 3 trays, all sitting in the first window
    fixed = {("OTHER", "8-12"): 60.0}  # 6 of "8-12"'s 8 tray-slots already used
    shifted = _shift_to_later_windows(movable, fixed, [f_sku, other_sku], windows, CAPACITY, MOLDING)
    assert shifted == {("F", "8-12"): 20.0, ("F", "4-8"): 10.0}
    assert sum(shifted.values()) == 30.0  # total quantity unchanged
    assert fixed == {("OTHER", "8-12"): 60.0}  # fixed background never touched


def test_shift_to_later_windows_converts_labor_minutes_to_trays():
    # Regression: remaining_labor / minutes_per_unit is a unit count, not a
    # tray count. For kratnost=10 and 1 min/unit, 10 spare minutes allow only
    # one tray (10 units), not ten trays.
    windows = [Window("0-1", 0, 1), Window("1-2", 1, 2)]
    capacity = CapacityConfig(bakers_count=1, ovens_count=10, trays_per_oven_batch=10, bake_minutes=60)
    sku = make_sku("F", kratnost=10, category="Выпечка сытная")
    fixed = {("F", "1-2"): 50.0}  # 50 of 60 baker-minutes already used in the last window
    movable = {("F", "0-1"): 30.0}

    shifted = _shift_to_later_windows(movable, fixed, [sku], windows, capacity, MOLDING)

    assert shifted == {("F", "1-2"): 10.0, ("F", "0-1"): 20.0}
    assert sum(shifted.values()) == 30.0


def test_swap_mandatory_labels_later_exchanges_same_sku_regular_batch():
    sku = make_sku("KISH", category="Выпечка сытная", kratnost=10)
    regular = {("KISH", "11-12"): 10.0}
    mandatory = {("KISH", "10-11"): 10.0}
    windows = [Window("10-11", 10, 11), Window("11-12", 11, 12)]

    regular_out, mandatory_out = _swap_mandatory_labels_later(regular, mandatory, [sku], windows)

    assert regular_out == {("KISH", "10-11"): 10.0}
    assert mandatory_out == {("KISH", "11-12"): 10.0}
    assert regular == {("KISH", "11-12"): 10.0}
    assert mandatory == {("KISH", "10-11"): 10.0}


def test_helper_pool_keeps_fastfood_from_consuming_core_baker_minutes():
    windows = [Window("0-1", 0, 1)]
    capacity = CapacityConfig(
        bakers_count=1,
        ovens_count=20,
        trays_per_oven_batch=10,
        bake_minutes=60,
        helpers_count=1,
    )
    core = make_sku("CORE", category="Выпечка сытная", hourly_qty={0: 60}, avg_daily_sales=10)
    helper_owned = make_sku("HELPER", category="Фастфуд", hourly_qty={0: 60}, avg_daily_sales=10)

    result = allocate_milp([core, helper_owned], windows, capacity, MOLDING)

    assert result.get(("CORE", "0-1"), 0) == 60
    assert result.get(("HELPER", "0-1"), 0) == 60
    _assert_capacity_respected(result, [core, helper_owned], windows, capacity, MOLDING)


def test_pie_effective_kratnost_is_four_units_per_oven_slot():
    windows = [Window("0-1", 0, 1)]
    capacity = CapacityConfig(bakers_count=10, ovens_count=1, trays_per_oven_batch=1, bake_minutes=60)
    pie = make_sku(
        "PIE",
        category="Пироги сытные",
        dough_group="дрожжевое",
        kratnost=1,
        hourly_qty={0: 4},
    )

    result = allocate_milp([pie], windows, capacity, MOLDING)

    assert result.get(("PIE", "0-1"), 0) == 4
    _assert_capacity_respected(result, [pie], windows, capacity, MOLDING)


def test_non_sand_pies_use_two_minutes_and_sand_pies_use_four_minutes():
    windows = [Window("0-1", 0, 1)]
    capacity = CapacityConfig(
        bakers_count=1,
        ovens_count=10,
        trays_per_oven_batch=10,
        bake_minutes=60,
    )
    non_sand = make_sku(
        "NON_SAND",
        category="Пироги сытные",
        dough_group="дрожжевое",
        hourly_qty={0: 28},
        avg_daily_sales=10,
    )
    sand = make_sku(
        "SAND",
        category="Пироги сладкие",
        dough_group="песочка",
        hourly_qty={0: 30},
        avg_daily_sales=1,
    )

    result = allocate_milp([non_sand, sand], windows, capacity, MOLDING)

    assert result.get(("NON_SAND", "0-1"), 0) == 28
    assert ("SAND", "0-1") not in result
    _assert_capacity_respected(result, [non_sand, sand], windows, capacity, MOLDING)


def test_core_daily_unit_cap_limits_baked_goods_but_not_helper_owned_skus():
    windows = [Window("0-1", 0, 1), Window("1-2", 1, 2)]
    capacity = CapacityConfig(
        bakers_count=10,
        ovens_count=20,
        trays_per_oven_batch=10,
        bake_minutes=60,
        helpers_count=10,
    )
    core = make_sku("CORE", category="Выпечка сытная", hourly_qty={0: 80}, avg_daily_sales=10)
    helper_owned = make_sku("HELPER", category="Фастфуд", hourly_qty={0: 80}, avg_daily_sales=10)

    result = allocate_milp([core, helper_owned], windows, capacity, MOLDING, core_unit_cap=50)

    assert sum(qty for (pid, _), qty in result.items() if pid == "CORE") == 50
    assert sum(qty for (pid, _), qty in result.items() if pid == "HELPER") == 80
    _assert_capacity_respected(result, [core, helper_owned], windows, capacity, MOLDING)


def test_tiny_tail_above_kratnost_does_not_force_extra_batch():
    windows = [Window("0-1", 0, 1)]
    capacity = CapacityConfig(bakers_count=10, ovens_count=10, trays_per_oven_batch=10, bake_minutes=60)
    sku = make_sku(
        "TRIANGLE",
        category="Выпечка сытная",
        kratnost=20,
        hourly_qty={0: 61},
    )

    result = allocate_milp([sku], windows, capacity, MOLDING)

    assert result.get(("TRIANGLE", "0-1"), 0) == 60


def test_tail_above_half_kratnost_forces_extra_batch():
    windows = [Window("0-1", 0, 1)]
    capacity = CapacityConfig(bakers_count=10, ovens_count=10, trays_per_oven_batch=10, bake_minutes=60)
    sku = make_sku(
        "TRIANGLE",
        category="Выпечка сытная",
        kratnost=20,
        hourly_qty={0: 71},
    )

    result = allocate_milp([sku], windows, capacity, MOLDING)

    assert result.get(("TRIANGLE", "0-1"), 0) == 80


def test_first_partial_batch_is_not_rounded_down_to_zero():
    windows = [Window("0-1", 0, 1)]
    capacity = CapacityConfig(bakers_count=10, ovens_count=10, trays_per_oven_batch=10, bake_minutes=60)
    sku = make_sku(
        "SMALL",
        category="Выпечка сытная",
        kratnost=20,
        hourly_qty={0: 5},
    )

    result = allocate_milp([sku], windows, capacity, MOLDING)

    assert result.get(("SMALL", "0-1"), 0) == 20

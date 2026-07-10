from __future__ import annotations

# ruff: noqa: E501
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps"))
sys.path.insert(0, str(ROOT / "apps" / "forecast_embedded"))

from baking_plan.algorithms.common import DEFROST_SKU_NAMES  # noqa: E402
from baking_plan.algorithms.milp import allocate_milp  # noqa: E402
from baking_plan.capacity import CapacityConfig, resolve_molding_minutes, window_capacity  # noqa: E402
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
            math.ceil(qty / sku_by_id[pid].kratnost)
            for (pid, label), qty in result.items()
            if label == window.label
        )
        baker_min = sum(
            qty * resolve_molding_minutes(sku_by_id[pid].category_name, molding_map)
            for (pid, label), qty in result.items()
            if label == window.label
        )
        cap = window_capacity(window, capacity)
        assert trays <= cap.tray_slots
        assert baker_min <= cap.baker_minutes


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
    # >= 7 rather than == : the objective only penalizes shortfall, not
    # overproduction, so any qty >= demand is an equally-optimal solution —
    # HiGHS isn't guaranteed to return the tightest one.
    assert result[("D", "4-8")] >= 7


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

from __future__ import annotations

# ruff: noqa: E501
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps"))
sys.path.insert(0, str(ROOT / "apps" / "forecast_embedded"))

from baking_plan.algorithms.common import DEFROST_SKU_NAMES, window_demand  # noqa: E402
from baking_plan.algorithms.greedy import allocate_greedy  # noqa: E402
from baking_plan.capacity import CapacityConfig, resolve_molding_minutes, window_capacity  # noqa: E402
from baking_plan.demand import SkuDemand  # noqa: E402
from baking_plan.templates import Window  # noqa: E402

WINDOWS = [Window("0-4", 0, 4), Window("4-8", 4, 8)]
# duration=4h=240min; bake_cycles=240//60=4; tray_slots=1*4*2=8; baker_minutes=1*240=240
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


def test_window_demand_rescales_to_full_day_total():
    # WINDOWS here are "0-4" and "4-8", but real sales can fall outside
    # [0, 8) entirely (e.g. evening sales after the last window's nominal
    # end) — that demand must not be silently invisible to the whole
    # allocation (found 2026-07-10: a bakery's real sales ran until 21:00
    # against a window list stopping at 16:00). Rather than dumping it all
    # onto the last window (which blew its capacity 17x over in practice),
    # the whole day's total is spread proportionally to each window's
    # share of the windowed-hours-only demand curve.
    sku = make_sku(
        "H",
        hourly_qty={0: 5, 3: 1, 4: 2, 7: 1, 20: 9, 23: 4},
    )
    # windowed-only shares: window "0-4" = 5+1=6, window "4-8" = 2+1=3 (2:1 ratio)
    # full-day total = 5+1+2+1+9+4 = 22
    demands = window_demand(sku, WINDOWS)
    assert sum(demands) == pytest.approx(22)  # nothing lost — full day covered
    assert demands[0] == pytest.approx(2 * demands[1])  # 2:1 ratio preserved
    assert demands[0] == pytest.approx(22 * 6 / 9)
    assert demands[1] == pytest.approx(22 * 3 / 9)


def test_window_demand_falls_back_to_even_split_when_no_windowed_hours_have_demand():
    # All demand falls outside every window's own hours — no curve shape to
    # distribute by, so split evenly rather than dropping it.
    sku = make_sku("I", hourly_qty={20: 10, 22: 6})  # nothing in [0,8)
    demands = window_demand(sku, WINDOWS)
    assert demands == [8.0, 8.0]


def test_priority_resolves_overflow_and_respects_capacity():
    skus = [
        make_sku("A", dough_group="GA", avg_daily_sales=10, hourly_qty={0: 5}),
        make_sku("B", dough_group="GB", avg_daily_sales=5, hourly_qty={0: 5}),
        make_sku("C", dough_group="GC", avg_daily_sales=1, hourly_qty={0: 5}),
    ]
    result = allocate_greedy(skus, WINDOWS, CAPACITY, MOLDING)

    # Window 0-4 can only hold 8 units combined; A (highest priority) wins its
    # full 5, leaving only 3 spare — not enough for B or C's 5 each.
    assert result[("A", "0-4")] == 5
    assert ("B", "0-4") not in result
    assert ("C", "0-4") not in result

    # B's unmet 5 carries to window 4-8 and gets scheduled there (nothing else
    # competes as urgently); C's carries too but the window is full after B.
    assert result[("B", "4-8")] == 5
    assert ("C", "4-8") not in result

    _assert_capacity_respected(result, skus, WINDOWS, CAPACITY, MOLDING)


def test_two_day_has_zero_regular_windows_and_full_next_day_in_last_window():
    # Двухдневка: no production on regular windows (today's stock was baked
    # yesterday); the last window bakes tomorrow's ENTIRE day, not just
    # early hours. hourly_qty (today) is ignored entirely for scheduling.
    two_day = make_sku(
        "D",
        name="Сочень",
        is_two_day=True,
        avg_daily_sales=1,
        hourly_qty={0: 99},  # must be ignored — двухдневка has zero regular demand
        next_day_hourly_qty={0: 1, 6: 3, 7: 2, 15: 1},  # full-day total = 7 (fits window capacity)
    )
    result = allocate_greedy([two_day], WINDOWS, CAPACITY, MOLDING)

    assert ("D", "0-4") not in result
    assert result[("D", "4-8")] == 7  # full next-day total, not just hours 6-11


def test_defrost_is_extra_on_top_of_regular_production_not_tied_to_two_day():
    # Дефрост: triggered by product_name membership in DEFROST_SKU_NAMES,
    # independent of is_two_day. Regular same-day windows still apply; the
    # last window gets extra next-day early-hours (6-11) demand added on top.
    defrost_name = next(iter(DEFROST_SKU_NAMES))
    sku = make_sku(
        "F",
        name=defrost_name,
        is_two_day=False,
        avg_daily_sales=1,
        hourly_qty={0: 4},  # normal same-day demand, covered by window 0-4
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
    result = allocate_greedy([sku, other], WINDOWS, CAPACITY, MOLDING)

    assert result[("F", "0-4")] == 4  # regular same-day demand
    assert result[("F", "4-8")] == 5  # extra overnight batch, hours 6-11 only
    assert ("G", "0-4") not in result
    assert ("G", "4-8") not in result  # not in DEFROST_SKU_NAMES: ignored


def test_group_cohesion_defers_whole_group_when_it_does_not_fit():
    # Second window is longer (8h vs 4h) so it has double capacity — enough
    # to absorb the carried-over group once deferred, without changing the
    # story in the first (tight) window.
    windows = [Window("0-4", 0, 4), Window("4-12", 4, 12)]
    a1 = make_sku("A1", dough_group="GA", avg_daily_sales=5, hourly_qty={0: 3})
    a2 = make_sku("A2", dough_group="GA", avg_daily_sales=5, hourly_qty={0: 7})
    result = allocate_greedy([a1, a2], windows, CAPACITY, MOLDING)

    # Combined group need is 10 > the window's 8-tray capacity. Even though
    # A1 alone (3) would fit, the whole group is deferred together rather
    # than scheduling A1 and dropping A2.
    assert ("A1", "0-4") not in result
    assert ("A2", "0-4") not in result
    # Deferred demand carries forward and both get scheduled once there's
    # room (second window has double the capacity: 16 tray-slots).
    assert result[("A1", "4-12")] == 3
    assert result[("A2", "4-12")] == 7

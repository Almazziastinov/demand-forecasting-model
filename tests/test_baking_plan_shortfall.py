from __future__ import annotations

# ruff: noqa: E501
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps"))
sys.path.insert(0, str(ROOT / "apps" / "forecast_embedded"))

from baking_plan import service  # noqa: E402
from baking_plan.algorithms.milp import allocate_milp_detailed  # noqa: E402
from baking_plan.capacity import (  # noqa: E402
    CapacityConfig,
    MOLDING_MINUTES_FLOOR,
    resolve_molding_minutes_floor,
)
from baking_plan.demand import SkuDemand  # noqa: E402
from baking_plan.rendering import (  # noqa: E402
    SHORTFALL_FULL_FILL,
    SHORTFALL_PARTIAL_FILL,
    render_workbook,
)
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


def test_resolve_molding_minutes_floor_matches_known_categories():
    assert resolve_molding_minutes_floor("Пироги сладкие") == pytest.approx(210 / 60)
    assert resolve_molding_minutes_floor("Пироги сытные") == pytest.approx(210 / 60)
    # Unknown category falls back to the '' default floor (54s), same
    # fallback pattern as resolve_molding_minutes().
    assert resolve_molding_minutes_floor("Выпечка сытная") == pytest.approx(54 / 60)
    assert MOLDING_MINUTES_FLOOR[""] == pytest.approx(54 / 60)


def test_allocate_milp_detailed_reports_shortfall_for_undersupplied_sku():
    # Same setup as test_baking_plan_milp.test_prioritizes_higher_avg_sales_when_capacity_binds:
    # single tight window, 8 tray-slots total, HIGH wins its full 5, LOW only
    # gets 3 of its 5 — LOW's shortfall should show up as exactly 2.
    windows = [WINDOWS[0]]
    high = make_sku("HIGH", dough_group="G1", avg_daily_sales=10, hourly_qty={0: 5})
    low = make_sku("LOW", dough_group="G2", avg_daily_sales=1, hourly_qty={0: 5})
    _regular, _defrost, _two_day, shortfall, _defrost_shortfall = allocate_milp_detailed(
        [high, low], windows, CAPACITY, MOLDING
    )
    assert shortfall["HIGH"] == pytest.approx(0.0, abs=1e-6)
    assert shortfall["LOW"] == pytest.approx(2.0, abs=1e-6)


def test_allocate_milp_detailed_zero_shortfall_when_capacity_is_sufficient():
    windows = [WINDOWS[0]]
    sku = make_sku("A", avg_daily_sales=1, hourly_qty={0: 3})
    _regular, _defrost, _two_day, shortfall, _defrost_shortfall = allocate_milp_detailed(
        [sku], windows, CAPACITY, MOLDING
    )
    assert shortfall["A"] == pytest.approx(0.0, abs=1e-6)


def test_has_shortfall():
    assert not service._has_shortfall({"A": 0.0, "B": 1e-9})
    assert service._has_shortfall({"A": 0.0, "B": 0.5})
    assert not service._has_shortfall({})


def test_capacity_recommendation_flags_baker_when_only_baker_minutes_maxed():
    windows = [Window("W", 0, 1)]  # 60 minutes
    cap = CapacityConfig(bakers_count=1, ovens_count=10, trays_per_oven_batch=10, bake_minutes=60)
    # baker_minutes = 1*60 = 60 (tight); tray_slots = 10*(60//60)*10 = 100 (roomy)
    molding = {"": 1}
    sku = make_sku("A", kratnost=1, category="Фастфуд")
    regular_alloc = {("A", "W"): 60.0}  # 60 baker-min used of 60 (100%); 60 trays of 100 (60%)
    rec = service._capacity_recommendation([sku], windows, cap, molding, regular_alloc, {}, {})
    assert rec == ["пекарь"]


def test_capacity_recommendation_flags_oven_when_only_trays_maxed():
    windows = [Window("W", 0, 1)]  # 60 minutes
    cap = CapacityConfig(bakers_count=100, ovens_count=1, trays_per_oven_batch=1, bake_minutes=60)
    # baker_minutes = 100*60 = 6000 (roomy); tray_slots = 1*(60//60)*1 = 1 (tight)
    molding = {"": 1}
    sku = make_sku("A", kratnost=1, category="Фастфуд")
    regular_alloc = {("A", "W"): 1.0}  # 1 baker-min of 6000 (~0%); 1 tray of 1 (100%)
    rec = service._capacity_recommendation([sku], windows, cap, molding, regular_alloc, {}, {})
    assert rec == ["печь"]


def test_capacity_recommendation_empty_when_nothing_maxed():
    windows = [Window("W", 0, 1)]
    cap = CapacityConfig(bakers_count=10, ovens_count=10, trays_per_oven_batch=10, bake_minutes=60)
    molding = {"": 1}
    sku = make_sku("A", kratnost=1, category="Фастфуд")
    regular_alloc = {("A", "W"): 1.0}
    rec = service._capacity_recommendation([sku], windows, cap, molding, regular_alloc, {}, {})
    assert rec == []


def _cell_by_name(sheet, name: str):
    for row in sheet.iter_rows(min_row=1):
        for cell in row:
            if cell.value == name:
                return sheet.cell(row=cell.row, column=sheet.max_column)
    raise AssertionError(f"row for {name!r} not found")


def test_render_workbook_marks_full_and_partial_shortfall_distinctly():
    windows = [WINDOWS[0]]
    zero_sku = make_sku("ZERO", name="Zero Product", category="Фастфуд", avg_daily_sales=1)
    partial_sku = make_sku("PARTIAL", name="Partial Product", category="Фастфуд", avg_daily_sales=1)
    ok_sku = make_sku("OK", name="OK Product", category="Фастфуд", avg_daily_sales=1)

    regular_alloc = {("PARTIAL", "0-4"): 3.0, ("OK", "0-4"): 4.0}
    shortfall_by_sku = {"ZERO": 5.0, "PARTIAL": 2.0, "OK": 0.0}

    workbook = render_workbook(
        bakery_name="Test",
        forecast_date="2026-07-10",
        windows=windows,
        skus=[zero_sku, partial_sku, ok_sku],
        regular_alloc=regular_alloc,
        defrost_alloc={},
        two_day_alloc={},
        shortfall_by_sku=shortfall_by_sku,
    )
    sheet = workbook.active

    zero_total = _cell_by_name(sheet, "Zero Product")
    assert zero_total.value == 5
    assert zero_total.fill.fgColor.rgb == SHORTFALL_FULL_FILL.fgColor.rgb

    partial_total = _cell_by_name(sheet, "Partial Product")
    assert partial_total.value == "3/5"
    assert partial_total.fill.fgColor.rgb == SHORTFALL_PARTIAL_FILL.fgColor.rgb

    ok_total = _cell_by_name(sheet, "OK Product")
    assert ok_total.value == 4.0
    assert ok_total.fill.fgColor.rgb != SHORTFALL_FULL_FILL.fgColor.rgb
    assert ok_total.fill.fgColor.rgb != SHORTFALL_PARTIAL_FILL.fgColor.rgb


def test_render_workbook_итого_excludes_defrost_component():
    # Дефрост is an extra batch for *tomorrow*, not today's own demand —
    # Итого should reflect only the regular (today's) production even when
    # a defrost top-up shares the same window cell, and defrost's own
    # shortfall must not leak into the Итого shortfall/forecast math either.
    windows = [WINDOWS[0]]
    fully_covered = make_sku("F", name="Full Product", category="Фастфуд", avg_daily_sales=1)
    defrost_short = make_sku("G", name="Defrost-Short Product", category="Фастфуд", avg_daily_sales=1)

    regular_alloc = {("F", "0-4"): 4.0, ("G", "0-4"): 4.0}
    defrost_alloc = {("F", "0-4"): 20.0}
    # F: fully covered, no shortfall anywhere.
    # G: regular fully covered (0 regular shortfall), but its defrost batch
    # is short by 3 — that 3 belongs to "tomorrow's" tally, not today's Итого.
    shortfall_by_sku = {"F": 0.0, "G": 3.0}
    defrost_shortfall_by_sku = {"F": 0.0, "G": 3.0}

    workbook = render_workbook(
        bakery_name="Test",
        forecast_date="2026-07-10",
        windows=windows,
        skus=[fully_covered, defrost_short],
        regular_alloc=regular_alloc,
        defrost_alloc=defrost_alloc,
        two_day_alloc={},
        shortfall_by_sku=shortfall_by_sku,
        defrost_shortfall_by_sku=defrost_shortfall_by_sku,
    )
    sheet = workbook.active

    full_total = _cell_by_name(sheet, "Full Product")
    assert full_total.value == 4.0  # not 24 — defrost's 20 excluded from Итого
    assert full_total.fill.fgColor.rgb != SHORTFALL_FULL_FILL.fgColor.rgb
    assert full_total.fill.fgColor.rgb != SHORTFALL_PARTIAL_FILL.fgColor.rgb

    defrost_short_total = _cell_by_name(sheet, "Defrost-Short Product")
    assert defrost_short_total.value == 4.0  # not flagged short — the shortfall was all defrost's
    assert defrost_short_total.fill.fgColor.rgb != SHORTFALL_FULL_FILL.fgColor.rgb
    assert defrost_short_total.fill.fgColor.rgb != SHORTFALL_PARTIAL_FILL.fgColor.rgb


def test_render_workbook_capacity_note_shifts_header_and_shows_note():
    windows = [WINDOWS[0]]
    sku = make_sku("A", name="A Product", category="Фастфуд")
    note = "Требуется дополнительно: пекарь."
    workbook = render_workbook(
        bakery_name="Test",
        forecast_date="2026-07-10",
        windows=windows,
        skus=[sku],
        regular_alloc={},
        defrost_alloc={},
        two_day_alloc={},
        capacity_note=note,
    )
    sheet = workbook.active
    assert sheet.cell(row=3, column=1).value == note
    assert sheet.cell(row=4, column=1).value == "Стол"

from __future__ import annotations

# ruff: noqa: E501
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps"))
sys.path.insert(0, str(ROOT / "apps" / "forecast_embedded"))

from baking_plan.demand import (  # noqa: E402
    _apply_defrost_offset,
    _cap_defrost_offset,
    _is_no_bake_meta,
    _load_yesterday_defrost_offset,
)


def test_apply_defrost_offset_subtracts_per_hour():
    hourly = {
        "A": {6: 5.0, 7: 3.0, 20: 9.0},  # hour 20 untouched — outside DEFROST_HOURS
        "B": {6: 2.0},
    }
    offset = {
        "A": {6: 2.0, 7: 1.0},
    }
    _apply_defrost_offset(hourly, offset)
    assert hourly["A"] == {6: 3.0, 7: 2.0, 20: 9.0}
    assert hourly["B"] == {6: 2.0}  # no offset for B, untouched


def test_apply_defrost_offset_clamps_at_zero():
    hourly = {"A": {6: 1.0}}
    offset = {"A": {6: 5.0}}  # yesterday over-baked relative to today's revised forecast
    _apply_defrost_offset(hourly, offset)
    assert hourly["A"] == {6: 0.0}


def test_apply_defrost_offset_creates_missing_product_entry():
    # A product present in the offset but absent from today's hourly (e.g.
    # today's forecast has no rows at all for it) still ends up as an
    # explicit zeroed entry rather than crashing.
    hourly: dict = {}
    offset = {"A": {6: 3.0}}
    _apply_defrost_offset(hourly, offset)
    assert hourly["A"] == {6: 0.0}


def test_load_yesterday_defrost_offset_empty_product_ids_skips_query():
    # Mirrors _load_hourly's early-return pattern — must not hit ClickHouse
    # (and therefore not require a live client) when there's nothing to ask for.
    assert _load_yesterday_defrost_offset(21, "2026-07-10", []) == {}


def test_cap_defrost_offset_uses_pdf_quantity_limit():
    offset = {
        "A": {6: 25.0, 7: 10.0},
        "B": {6: 5.0},
        "C": {6: 99.0},
    }
    capped = _cap_defrost_offset(
        offset,
        {
            "A": "Сосиска в тесте",  # refrigerator PDF limit: 30
            "B": "Пицца с колбасой",  # limit: 10, raw value below limit
            "C": "Не дефрост",
        },
    )
    assert capped == {
        "A": {6: 25.0, 7: 5.0},
        "B": {6: 5.0},
    }


def test_no_bake_meta_detects_frozen_semi_finished_goods():
    assert _is_no_bake_meta({"dough_group": "Замороженные полуфабрикаты (ничего с ними не делаем)"})
    assert not _is_no_bake_meta({"dough_group": "Тесто сдобное"})

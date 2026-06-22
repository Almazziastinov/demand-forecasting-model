from pathlib import Path
import sys
from io import BytesIO

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = ROOT / "apps" / "forecast_embedded"
sys.path.insert(0, str(APP_ROOT))

from app.services.baking_plan import BakingWindow  # noqa: E402
from app.services.baking_plan import ScheduledColumn  # noqa: E402
from app.services.baking_plan import allocate_template_row  # noqa: E402
from app.services.baking_plan import build_product_hour_lookup  # noqa: E402
from app.services.baking_plan import build_baking_plan_workbook  # noqa: E402
from app.services.baking_plan import build_assortment_lookup  # noqa: E402
from app.services.baking_plan import coverage_hours  # noqa: E402
from app.services.baking_plan import is_defrost_cell  # noqa: E402
from app.services.baking_plan import normalize_sku_name  # noqa: E402
from app.services.baking_plan import revenue_bucket  # noqa: E402
from app.services.baking_plan import schedule_round_to  # noqa: E402
from app.services.baking_plan import sku_match_keys  # noqa: E402
from app.services.baking_plan import template_path_for_bakery  # noqa: E402


# Window labels mirror the real template row 5.
WINDOWS = {
    3: BakingWindow(3, "4:00-7:00", 4, 7),
    4: BakingWindow(4, "7:00-8:00", 7, 8),
    5: BakingWindow(5, "8:00-9:00", 8, 9),
    8: BakingWindow(8, "11:00-12:00", 11, 12),
    9: BakingWindow(9, "12:00-13:00", 12, 13),
    12: BakingWindow(12, "15:00-16:00", 15, 16),
}


def _schedule(*columns, defrost=()):
    return [
        ScheduledColumn(
            window=WINDOWS[c],
            is_defrost=c in defrost,
            note="20 (ночная дефр)" if c in defrost else None,
        )
        for c in columns
    ]


def _schedule_with_quantities(items, defrost=()):
    return [
        ScheduledColumn(
            window=WINDOWS[c],
            is_defrost=c in defrost,
            note="20 (РЅРѕС‡РЅР°СЏ РґРµС„СЂ)" if c in defrost else None,
            quantity=q,
        )
        for c, q in items
    ]


def test_coverage_hours_tiles_multi_window_without_gap_or_overlap() -> None:
    windows = [WINDOWS[3], WINDOWS[8], WINDOWS[12]]
    cov = coverage_hours(windows)
    assert cov[3] == [6, 7, 8, 9, 10, 11]
    assert cov[8][0] == 12
    all_hours = [h for hours in cov.values() for h in hours]
    assert len(all_hours) == len(set(all_hours))  # no overlap
    assert set(all_hours) == set(range(6, 24))  # no gap


def test_coverage_hours_single_late_window_absorbs_full_day() -> None:
    # A SKU baked only once midday must still cover the whole day, not drop morning.
    assert coverage_hours([WINDOWS[12]])[12] == list(range(6, 24))
    assert coverage_hours([WINDOWS[5]])[5] == list(range(6, 24))


def test_allocate_only_fills_scheduled_columns() -> None:
    rows = [
        {"product_name": "Треугольник курица", "hour": h, "forecast_qty": q}
        for h, q in {6: 3, 7: 12, 8: 15, 9: 15, 10: 20, 11: 30, 12: 27}.items()
    ]
    lookup = build_product_hour_lookup(rows)

    result = allocate_template_row(
        template_sku_name="Треугольник курица (тесто ночного брожжения)",
        schedule=_schedule(3, 8),
        product_hour_lookup=lookup,
    )

    # Scheduled in cols 3 and 8 only — no other column appears.
    assert set(result.keys()) == {3, 8}
    assert result[3] == 3 + 12 + 15 + 15 + 20 + 30  # hours 6..11
    assert result[8] == 27  # hour 12..end


def test_schedule_round_to_uses_template_batch_gcd() -> None:
    assert schedule_round_to(_schedule_with_quantities([(3, 20), (8, 20)])) == 20
    assert (
        schedule_round_to(_schedule_with_quantities([(3, 20), (8, 10), (12, 20)]))
        == 10
    )


def test_allocate_rounds_up_and_carries_surplus_to_next_bakes() -> None:
    rows = [
        {"product_name": "batch sku", "hour": h, "forecast_qty": q}
        for h, q in {6: 21, 12: 25, 16: 1}.items()
    ]
    lookup = build_product_hour_lookup(rows)

    result = allocate_template_row(
        template_sku_name="batch sku",
        schedule=_schedule_with_quantities([(3, 20), (8, 20), (12, 20)]),
        product_hour_lookup=lookup,
    )

    assert result == {3: 40, 8: 20}


def test_allocate_adds_midday_split_for_single_early_bake() -> None:
    rows = [
        {"product_name": "split sku", "hour": h, "forecast_qty": q}
        for h, q in {7: 20, 12: 15, 18: 25}.items()
    ]
    lookup = build_product_hour_lookup(rows)

    result = allocate_template_row(
        template_sku_name="split sku",
        schedule=_schedule_with_quantities([(3, 20)]),
        product_hour_lookup=lookup,
        available_windows=WINDOWS,
    )

    assert result == {3: 20, 8: 40}


def test_defrost_cell_detection_keys_off_cell_value_not_sku_name() -> None:
    assert is_defrost_cell("20 (ночная дефр)") is True
    assert is_defrost_cell(20) is False
    # SKU names contain these tokens but plain integer cells stay normal bakes.
    assert normalize_sku_name("Треугольник курица (тесто ночного брожжения)") == (
        "треугольник курица"
    )


def test_defrost_sized_from_next_day_early_window_with_annotation() -> None:
    today = build_product_hour_lookup(
        [{"product_name": "Вак-бэлиш", "hour": 7, "forecast_qty": 50}]
    )
    next_day = build_product_hour_lookup(
        [
            {"product_name": "Вак-бэлиш", "hour": 7, "forecast_qty": 30},
            {"product_name": "Вак-бэлиш", "hour": 8, "forecast_qty": 5},
            {"product_name": "Вак-бэлиш", "hour": 15, "forecast_qty": 99},  # late
        ]
    )

    result = allocate_template_row(
        template_sku_name="Вак-бэлиш (ночная дефростация)",
        schedule=_schedule(3, 12, defrost=(12,)),
        product_hour_lookup=today,
        next_day_hour_lookup=next_day,
    )

    # Col 12 is defrost: sized from next-day early window (7+8 = 35), late hour 15
    # excluded; annotation preserved; number replaced.
    assert result[12] == "35 (ночная дефр)"
    # Col 3 (bake) covers today; defrost column never contributes to today's bake.
    assert result[3] == 50


def test_defrost_falls_back_to_today_when_next_day_absent() -> None:
    today = build_product_hour_lookup(
        [{"product_name": "Вак-бэлиш", "hour": 7, "forecast_qty": 40}]
    )

    result = allocate_template_row(
        template_sku_name="Вак-бэлиш (ночная дефростация)",
        schedule=_schedule(3, 12, defrost=(12,)),
        product_hour_lookup=today,
        next_day_hour_lookup=None,
    )

    assert result[12] == "40 (ночная дефр)"


def test_allocate_returns_empty_when_sku_absent_from_forecast() -> None:
    lookup = build_product_hour_lookup(
        [{"product_name": "Что-то ещё", "hour": 7, "forecast_qty": 10}]
    )
    result = allocate_template_row(
        template_sku_name="Несуществующий SKU",
        schedule=_schedule(3, 9),
        product_hour_lookup=lookup,
    )
    assert result == {}


def test_revenue_bucket_matches_template_thresholds() -> None:
    assert revenue_bucket(1_499_999) == "до 1,5 млн"
    assert revenue_bucket(1_500_000) == "до 2,5 млн"
    assert revenue_bucket(2_500_000) == "от 2,5 млн"
    assert revenue_bucket(3_000_000) == "от 3млн"


def test_sku_match_keys_include_known_forecast_aliases() -> None:
    assert "треугольник курица" in sku_match_keys("Треугольник курица безд")
    assert "жар пицца оригинальная" in sku_match_keys("ЖарПицца Оригинальная")
    assert "пирожок булочка с яблоками" in sku_match_keys("Пирожок яблоко")


def test_workbook_uses_active_assortment_and_adds_unscheduled_rows() -> None:
    content = build_baking_plan_workbook(
        bakery={"bakery_id": 22, "bakery_name": "Тест", "city": "Казань"},
        forecast_date="2026-06-20",
        sku_hour_rows=[
            {
                "product_id": "1",
                "product_name": "Треугольник курица безд",
                "category_name": "Выпечка сытная",
                "hour": 7,
                "forecast_qty": 12,
            },
            {
                "product_id": "2",
                "product_name": "Новая позиция",
                "category_name": "Новая группа",
                "hour": 8,
                "forecast_qty": 7,
            },
        ],
        assortment_rows=[
            {
                "product_id": "1",
                "product_name": "Треугольник курица безд",
                "category_name": "Выпечка сытная",
            },
            {
                "product_id": "2",
                "product_name": "Новая позиция",
                "category_name": "Новая группа",
            },
        ],
        bucket="до 1,5 млн",
    )

    sheet = load_workbook(BytesIO(content))["План выпекания"]
    names = {
        sheet.cell(row=row, column=2).value: row
        for row in range(1, sheet.max_row + 1)
        if sheet.cell(row=row, column=2).value
    }

    assert sheet.cell(row=5, column=13).value == "Итого"
    assert sheet.max_column == 13
    assert sheet.column_dimensions["M"].hidden is False
    assert sheet.cell(row=6, column=1).value == "Выпечка сытная"
    assert "Треугольник курица безд" in names
    assert "Новая позиция" in names
    assert "Пирожок капуста курица" not in names

    new_row = names["Новая позиция"]
    assert all(
        sheet.cell(row=new_row, column=column).value is None
        for column in range(3, 13)
    )
    assert sheet.cell(row=new_row, column=13).value == 7
    assert sheet.cell(row=new_row, column=2).fill.fill_type is None
    assert all(
        not sheet.row_dimensions[row].hidden
        for row in range(1, sheet.max_row + 1)
    )


def test_assortment_lookup_prefers_regular_product_over_order_variant() -> None:
    lookup, _ = build_assortment_lookup(
        [
            {
                "product_id": "1",
                "product_name": "Клубника и банан ЗКЗ",
                "category_name": "Заказная продукция",
            },
            {
                "product_id": "2",
                "product_name": "Клубника и банан НОВЫЙ",
                "category_name": "Пироги сладкие",
            },
        ]
    )

    assert lookup["клубника банан"]["product_id"] == "2"


def test_pilot_bakery_uses_individual_template() -> None:
    assert template_path_for_bakery(22).name == "22_sibirskiy_trakt_25.xlsx"
    assert template_path_for_bakery(999999).name == "baking_plan_template.xlsx"

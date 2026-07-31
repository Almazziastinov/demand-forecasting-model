from io import BytesIO

from openpyxl import load_workbook

from scripts.publish_pilot_forecast import (
    PILOT_BAKERY_IDS,
    _build_excel,
    _round_up_kratnost,
)


def test_expanded_pilot_contains_approved_bakeries_without_kulagina() -> None:
    assert PILOT_BAKERY_IDS == [
        1,
        20,
        21,
        22,
        28,
        39,
        41,
        56,
        57,
        66,
        67,
        69,
        80,
        89,
        107,
        125,
        149,
        155,
        160,
        221,
        222,
        257,
    ]
    assert 16 not in PILOT_BAKERY_IDS


def test_round_up_kratnost_after_stock_subtraction() -> None:
    forecast = 23.8
    yesterday_stock = 8.0

    net_need = max(forecast - yesterday_stock, 0.0)

    assert net_need == 15.8
    assert _round_up_kratnost(net_need, 10) == 20


def test_stock_can_cover_full_forecast() -> None:
    assert _round_up_kratnost(max(5.0 - 7.0, 0.0), 10) == 0


def test_excel_contains_stock_and_production_plan_columns() -> None:
    rows = [
        {
            "bakery_id": 16,
            "bakery_name": "Кулагина 4 Казань",
            "category": "Выпечка сладкая",
            "product_name": "Ватрушка",
            "forecast": 23.8,
            "yesterday_stock": 8.0,
            "net_need": 15.8,
            "production_plan": 20,
            "total_for_sale": 28.0,
            "kratnost": 10,
        }
    ]

    workbook = load_workbook(BytesIO(_build_excel(rows, "2026-07-29")), data_only=True)
    sheet = workbook["Прогноз"]

    assert [cell.value for cell in sheet[2]] == [
        "Пекарня",
        "Категория",
        "Номенклатура",
        "Прогноз",
        "Остаток со вчерашнего дня",
        "Чистая потребность",
        "План выпуска",
        "Итого на продажу",
        "Кратность",
    ]
    assert [cell.value for cell in sheet[3]] == [
        "Кулагина 4 Казань",
        "Выпечка сладкая",
        "Ватрушка",
        23.8,
        8.0,
        15.8,
        20,
        28.0,
        10,
    ]

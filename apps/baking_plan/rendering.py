"""Build the "План выпекания" xlsx sheet from scratch.

Rows are generated fresh from live data every call — the reference file
`assets/template.xlsx` is a filled *example*, not a blank template with a
fixed row count, so we can't just overwrite cells in it (the SKU list size
varies by bakery/date/assortment). Only the window-boundary layout is read
from that file (`templates.parse_windows`); everything else (styling,
category grouping, row order) is built here to match its visual style.
"""

from __future__ import annotations

# ruff: noqa: E501
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.worksheet.worksheet import Worksheet

from .demand import SkuDemand
from .templates import Window

SHEET_NAME = "План выпекания"
CATEGORY_ORDER = [
    "Выпечка сытная",
    "Выпечка сладкая",
    "Пироги сытные",
    "Пироги сладкие",
    "Фастфуд",
]
HEADER_FILL = PatternFill("solid", fgColor="FFD8D8D8")
CATEGORY_FILL = PatternFill("solid", fgColor="FFD9EAF7")
DEFROST_FILL = PatternFill("solid", fgColor="FFFCE4D6")
TWO_DAY_FILL = PatternFill("solid", fgColor="FFE6D9F2")
MANDATORY_FILL = PatternFill("solid", fgColor="FFFFFF00")
DEFROST_SUFFIX = " (ночная дефр)"

# SKUs management tracks as "обязательный ассортимент" — must always be on
# the plan. No source table for this yet (only one SKU was marked yellow in
# the reference file's "комментарии" sheet); this list was given directly
# by the user 2026-07-10. Matched by exact product_name.
MANDATORY_ASSORTMENT = {
    "Треугольник курица безд",
    "Треугольник говядина безд",
    "Треугольник острый",
    "Вак-бэлиш",
    "Жар пицца с курицей",
    "Сосиска в тесте",
    "Элеш с курицей",
    "Беккен капуста",
    "Сосиска под шубой",
    "Кыстыбый П",
}


def render_workbook(
    *,
    bakery_name: str,
    forecast_date: str,
    windows: list[Window],
    skus: list[SkuDemand],
    regular_alloc: dict[tuple[str, str], float],
    defrost_alloc: dict[str, float],
    two_day_alloc: dict[str, float],
) -> Workbook:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = SHEET_NAME

    total_col = 3 + len(windows)  # A=Стол, B=Наименование, C..=windows, then Итого
    last_window_label = windows[-1].label if windows else None

    sheet["A1"] = f"План выпекания: {bakery_name} на {forecast_date}."
    sheet["A1"].font = Font(bold=True)
    sheet["A2"] = (
        "Количество по SKU-hour прогнозу активного run, разнесено по окнам "
        "выпекания через MILP-оптимизацию; ночная дефростация — по утреннему "
        "прогнозу следующего дня."
    )
    sheet["A2"].font = Font(bold=True)

    header_row = 3
    sheet.cell(row=header_row, column=1, value="Стол")
    sheet.cell(row=header_row, column=2, value="Наименование")
    for idx, window in enumerate(windows):
        sheet.cell(row=header_row, column=3 + idx, value=window.label)
    sheet.cell(row=header_row, column=total_col, value="Итого")
    for col in range(1, total_col + 1):
        cell = sheet.cell(row=header_row, column=col)
        cell.font = Font(bold=True)
        cell.fill = HEADER_FILL

    row = header_row
    for category in CATEGORY_ORDER:
        members = sorted(
            (sku for sku in skus if sku.category_name == category),
            key=lambda sku: sku.avg_daily_sales,
            reverse=True,
        )
        if not members:
            continue

        row += 1
        category_cell = sheet.cell(row=row, column=1, value=category)
        category_cell.font = Font(bold=True)
        sheet.merge_cells(start_row=row, start_column=1, end_row=row, end_column=total_col)
        for col in range(1, total_col + 1):
            sheet.cell(row=row, column=col).fill = CATEGORY_FILL

        for sku in members:
            row += 1
            station_cell = sheet.cell(row=row, column=1, value=sku.station or "")
            name_cell = sheet.cell(row=row, column=2, value=sku.product_name)
            if sku.product_name in MANDATORY_ASSORTMENT:
                station_cell.fill = MANDATORY_FILL
                name_cell.fill = MANDATORY_FILL
            row_total = 0.0
            for idx, window in enumerate(windows):
                is_last_window = window.label == last_window_label
                two_day_qty = two_day_alloc.get(sku.product_id, 0) if is_last_window else 0
                if two_day_qty:
                    window_cell = sheet.cell(row=row, column=3 + idx, value=two_day_qty)
                    window_cell.fill = TWO_DAY_FILL
                    row_total += two_day_qty
                    continue
                qty = regular_alloc.get((sku.product_id, window.label), 0)
                defrost_qty = defrost_alloc.get(sku.product_id, 0) if is_last_window else 0
                cell_value = _format_cell(qty, defrost_qty)
                if cell_value != "":
                    window_cell = sheet.cell(row=row, column=3 + idx, value=cell_value)
                    if defrost_qty:
                        window_cell.fill = DEFROST_FILL
                    row_total += qty + defrost_qty
            # Итого = sum of what's actually scheduled across the windows
            # (kratnost-rounded production), not the raw model forecast —
            # e.g. forecast 41.21 with 20+20 scheduled shows 40, not 41.21.
            sheet.cell(row=row, column=total_col, value=row_total)

    _set_column_widths(sheet, total_col)
    return workbook


def _format_cell(qty: float, defrost_qty: float) -> object:
    if defrost_qty:
        return f"{qty + defrost_qty:g}{DEFROST_SUFFIX}"
    if qty:
        return qty
    return ""


def _set_column_widths(sheet: Worksheet, total_col: int) -> None:
    sheet.column_dimensions["A"].width = 22
    sheet.column_dimensions["B"].width = 34
    for col in range(3, total_col + 1):
        sheet.column_dimensions[sheet.cell(row=3, column=col).column_letter].width = 14

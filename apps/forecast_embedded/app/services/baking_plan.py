from __future__ import annotations

import math
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from openpyxl import load_workbook
from openpyxl.worksheet.worksheet import Worksheet


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE_PATH = ROOT / "assets" / "baking_plan_template.xlsx"
DEFAULT_BUCKET = "до 2,5 млн"
REVENUE_BUCKETS = (
    (1_500_000, "до 1,5 млн"),
    (2_500_000, "до 2,5 млн"),
    (3_000_000, "от 2,5 млн"),
)
WINDOW_COLUMNS = tuple(range(3, 13))
FIRST_SALES_HOUR = 6
LAST_SALES_HOUR = 23
SKU_ALIAS_TO_CANONICAL = {
    "треугольник курица безд": "треугольник курица",
    "треугольник говядина безд": "треугольник говядина",
    "хуплу": "хуплу чебоксары",
    "элеш с курицей": "элеш",
    "конвертик курица": "конвертик с курицей",
    "ватрушка": "ватрушка в ассортменте",
    "жарпицца пикантная": "жар пицца пикантная",
    "жарпицца оригинальная": "жар пицца оригинальная",
    "кыстыбый п": "кыстыбый",
    "киш курица": "киш с курицей",
    "жар киш курица": "жар киш с курицей",
    "трехслойник новый": "трехслойник",
    "пирог ханский": "ханский",
    "капустный": "пирог капустный",
    "капуста и мясо": "пирог капуста мясо",
    "капуста и курица": "пирог капуста курица",
    "горбуша саго": "пирог горбуша саго",
    "пирожок яблоко": "пирожок булочка с яблоками",
    "клубника и банан новый": "клубника банан",
    "клубника и банан зкз": "клубника банан",
    "печенье детское 250": "печенье детское",
}


@dataclass(frozen=True)
class BakingWindow:
    column: int
    label: str
    start_hour: int
    end_hour: int


def normalize_sku_name(value: object) -> str:
    text = str(value or "").lower().replace("ё", "е")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    stop_words = {
        "тесто",
        "ночного",
        "ночное",
        "брожжения",
        "дефростация",
        "ночная",
        "ночн",
        "дефр",
    }
    tokens = [token for token in text.split() if token not in stop_words]
    return " ".join(tokens)


def sku_match_keys(value: object) -> list[str]:
    key = normalize_sku_name(value)
    if not key:
        return []

    keys = [key]
    alias = SKU_ALIAS_TO_CANONICAL.get(key)
    if alias:
        keys.append(alias)

    if "жарпицца" in key:
        keys.append(key.replace("жарпицца", "жар пицца"))

    result: list[str] = []
    seen: set[str] = set()
    for item in keys:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result


def revenue_bucket(revenue: float | int | None) -> str:
    if revenue is None:
        return DEFAULT_BUCKET
    value = float(revenue)
    for threshold, bucket in REVENUE_BUCKETS:
        if value < threshold:
            return bucket
    return "от 3млн"


def parse_window_label(value: object, column: int) -> BakingWindow | None:
    label = str(value or "").strip()
    match = re.search(r"(\d{1,2})[:.]\d{2}\s*-\s*(\d{1,2})[:.]\d{2}", label)
    if not match:
        return None
    return BakingWindow(
        column=column,
        label=label,
        start_hour=int(match.group(1)),
        end_hour=int(match.group(2)),
    )


def coverage_hours(windows: list[BakingWindow]) -> dict[int, list[int]]:
    if not windows:
        return {}

    ordered = sorted(
        windows,
        key=lambda item: (item.start_hour, item.end_hour, item.column),
    )
    result: dict[int, list[int]] = {}
    for index, window in enumerate(ordered):
        if index == 0 and window.start_hour <= 4:
            start_hour = FIRST_SALES_HOUR
        else:
            start_hour = window.end_hour

        if index + 1 < len(ordered):
            end_hour = ordered[index + 1].start_hour
        else:
            end_hour = LAST_SALES_HOUR

        if end_hour >= start_hour:
            result[window.column] = list(range(start_hour, end_hour + 1))
        else:
            result[window.column] = []
    return result


def build_product_hour_lookup(
    sku_hour_rows: list[dict[str, Any]],
) -> dict[str, dict[int, float]]:
    lookup: dict[str, dict[int, float]] = {}
    for row in sku_hour_rows:
        name_keys = sku_match_keys(row.get("product_name"))
        if not name_keys or row.get("hour") is None:
            continue

        hour = int(row["hour"])
        qty = float(row.get("forecast_qty") or 0.0)
        for name_key in name_keys:
            lookup.setdefault(name_key, {})
            lookup[name_key][hour] = lookup[name_key].get(hour, 0.0) + qty
    return lookup


def _resolve_hourly_forecast(
    template_sku_name: object,
    product_hour_lookup: dict[str, dict[int, float]],
) -> dict[int, float] | None:
    for name_key in sku_match_keys(template_sku_name):
        hourly = product_hour_lookup.get(name_key)
        if hourly:
            return hourly
    return None


def allocate_template_row(
    *,
    template_sku_name: object,
    row_windows: list[BakingWindow],
    product_hour_lookup: dict[str, dict[int, float]],
    round_to: int = 1,
) -> dict[int, int]:
    hourly = _resolve_hourly_forecast(template_sku_name, product_hour_lookup)
    if not hourly:
        return {}

    allocated: dict[int, int] = {}
    for column, hours in coverage_hours(row_windows).items():
        qty = sum(hourly.get(hour, 0.0) for hour in hours)
        if qty <= 0:
            continue
        allocated[column] = int(math.ceil(qty / round_to) * round_to)
    return allocated


def _select_sheet_name(bucket: str | None, sheet_names: list[str]) -> str:
    if bucket and bucket in sheet_names:
        return bucket
    if DEFAULT_BUCKET in sheet_names:
        return DEFAULT_BUCKET
    return sheet_names[0]


def _sheet_windows(sheet: Worksheet) -> dict[int, BakingWindow]:
    windows = {}
    for column in WINDOW_COLUMNS:
        parsed = parse_window_label(sheet.cell(row=5, column=column).value, column)
        if parsed:
            windows[column] = parsed
    return windows


def build_baking_plan_workbook(
    *,
    bakery: dict[str, Any],
    forecast_date: str,
    sku_hour_rows: list[dict[str, Any]],
    bucket: str | None = None,
    template_path: Path = DEFAULT_TEMPLATE_PATH,
) -> bytes:
    workbook = load_workbook(template_path)
    selected_sheet_name = _select_sheet_name(bucket, workbook.sheetnames)
    sheet = workbook[selected_sheet_name]

    for worksheet in list(workbook.worksheets):
        if worksheet.title != selected_sheet_name:
            workbook.remove(worksheet)
    sheet.title = "План выпекания"

    sheet["A1"] = (
        f"План выпекания: {bakery.get('bakery_name') or bakery.get('bakery_id')} "
        f"на {forecast_date}. Шаблон: {selected_sheet_name}"
    )
    sheet["A2"] = (
        "Количество рассчитано по SKU-hour прогнозу активного run; "
        "округление вверх до целых штук."
    )

    sheet_windows = _sheet_windows(sheet)
    product_hour_lookup = build_product_hour_lookup(sku_hour_rows)

    for row_index in range(6, sheet.max_row + 1):
        sku_name = sheet.cell(row=row_index, column=2).value
        if not sku_name:
            continue

        row_windows = [
            sheet_windows[column]
            for column in WINDOW_COLUMNS
            if column in sheet_windows
        ]
        if not row_windows:
            continue

        allocated = allocate_template_row(
            template_sku_name=sku_name,
            row_windows=row_windows,
            product_hour_lookup=product_hour_lookup,
        )
        for column in WINDOW_COLUMNS:
            if column in sheet_windows:
                sheet.cell(row=row_index, column=column).value = allocated.get(column)

    output = BytesIO()
    workbook.save(output)
    return output.getvalue()

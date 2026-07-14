"""Read a template row's pre-filled window schedule and size quantities from
live forecast.

Window *assignment* always comes from the template (which C:L cells the
reference file's author filled in for a SKU row) — this module never decides
*which* window a SKU bakes in, only *how much* to bake in the windows the
template already assigned. See `docs/baking_plan_implementation.md`.
"""

from __future__ import annotations

# ruff: noqa: E501
import math
import re
from dataclasses import dataclass
from typing import Any

from openpyxl.worksheet.worksheet import Worksheet

from .templates import Window

FIRST_SALES_HOUR = 6
LAST_SALES_HOUR = 23
# Night-defrost columns are prep for tomorrow's morning batches; size them
# from tomorrow's forecast over hours [FIRST_SALES_HOUR .. DEFROST_EARLY_CUTOFF - 1].
DEFROST_EARLY_CUTOFF = 12
DEFROST_MARKERS = ("дефр", "ночн")

# Template SKU names occasionally drift from the live product catalogue
# (renames, bez-suffix variants). Matched after normalize_sku_name().
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


def is_defrost_cell(value: object) -> bool:
    text = str(value or "").lower().replace("ё", "е")
    return any(marker in text for marker in DEFROST_MARKERS)


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


def _assortment_match_priority(product: dict[str, Any]) -> tuple[int, str]:
    text = f"{product.get('product_name', '')} {product.get('category_name', '')}".casefold()
    is_order_product = any(marker in text for marker in ("зкз", "заказ"))
    return int(is_order_product), str(product.get("product_id") or "")


def build_assortment_lookup(
    assortment_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Map every normalized name key to its best-matching assortment product."""
    by_name: dict[str, dict[str, Any]] = {}
    for record in assortment_rows:
        for name_key in sku_match_keys(record.get("product_name")):
            current = by_name.get(name_key)
            if current is None or _assortment_match_priority(record) < _assortment_match_priority(current):
                by_name[name_key] = record
    return by_name


def resolve_assortment_product(
    template_sku_name: object,
    assortment_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    for name_key in sku_match_keys(template_sku_name):
        product = assortment_lookup.get(name_key)
        if product:
            return product
    return None


@dataclass(frozen=True)
class ScheduledColumn:
    """A C:L cell that the template pre-filled for a SKU row.

    ``is_defrost`` marks night-defrost prep (sized from tomorrow's early
    forecast, excluded from today's coverage). ``note`` keeps the original
    cell text so the defrost annotation (e.g. ``"20 (ночная дефр)"``) can be
    re-emitted.
    """

    window: Window
    column: int
    is_defrost: bool
    note: str | None
    quantity: int | None = None


def _extract_positive_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        rounded = int(round(value))
        return rounded if rounded > 0 and math.isclose(value, rounded) else None
    match = re.search(r"\d+", str(value))
    if not match:
        return None
    parsed = int(match.group(0))
    return parsed if parsed > 0 else None


def read_row_schedule(
    sheet: Worksheet,
    row_index: int,
    sheet_windows: dict[int, Window],
) -> list[ScheduledColumn]:
    """Read which C:L columns the template pre-filled for this SKU row.

    A non-empty cell marks a scheduled baking window. Cells whose text
    carries a night-defrost marker (e.g. ``"20 (ночная дефр)"``) are
    classified as defrost prep. Classification keys off the *cell value*,
    not the SKU name (many SKU names contain "ночного брожжения" yet have
    plain integer bake cells).
    """
    schedule: list[ScheduledColumn] = []
    for column, window in sheet_windows.items():
        value = sheet.cell(row=row_index, column=column).value
        if value is None or str(value).strip() == "":
            continue
        defrost = is_defrost_cell(value)
        schedule.append(
            ScheduledColumn(
                window=window,
                column=column,
                is_defrost=defrost,
                note=str(value) if defrost else None,
                quantity=_extract_positive_int(value),
            )
        )
    return schedule


def schedule_round_to(schedule: list[ScheduledColumn]) -> int:
    """Kratnost fallback when комментарии has no entry for this SKU: derive
    the production multiple from the GCD of the template's own pre-filled
    quantities."""
    quantities = [
        item.quantity for item in schedule if not item.is_defrost and item.quantity is not None and item.quantity > 0
    ]
    if not quantities:
        quantities = [item.quantity for item in schedule if item.quantity is not None and item.quantity > 0]
    if not quantities:
        return 1
    round_to = quantities[0]
    for quantity in quantities[1:]:
        round_to = math.gcd(round_to, quantity)
    return max(round_to, 1)


def _round_up(qty: float, round_to: int) -> int:
    if round_to <= 1:
        return int(math.ceil(qty))
    return int(math.ceil(qty / round_to) * round_to)


def coverage_hours(
    windows: list[Window],
    *,
    first_sales_hour: int = FIRST_SALES_HOUR,
    last_sales_hour: int = LAST_SALES_HOUR,
) -> dict[str, list[int]]:
    """Map each bake window (by label) to the sales hours it covers.

    A batch becomes available at its window's end hour and covers sales
    until the next batch becomes available (the next window's end hour); the
    last window runs through ``last_sales_hour``. The earliest window always
    starts its coverage at ``first_sales_hour`` so a SKU baked only once
    (e.g. midday) still absorbs the whole day's forecast instead of
    silently dropping the morning.
    """
    if not windows:
        return {}
    ordered = sorted(windows, key=lambda item: (item.start_hour, item.end_hour, item.label))
    result: dict[str, list[int]] = {}
    for index, window in enumerate(ordered):
        start_hour = first_sales_hour if index == 0 else max(window.end_hour, first_sales_hour)
        if index + 1 < len(ordered):
            end_hour = ordered[index + 1].end_hour - 1
        else:
            end_hour = last_sales_hour
        result[window.label] = list(range(start_hour, end_hour + 1)) if end_hour >= start_hour else []
    return result


def _early_window_sum(
    hourly: dict[int, float],
    first_sales_hour: int = FIRST_SALES_HOUR,
    cutoff: int = DEFROST_EARLY_CUTOFF,
) -> float:
    return sum(qty for hour, qty in hourly.items() if first_sales_hour <= hour < cutoff)


def allocate_template_row(
    *,
    schedule: list[ScheduledColumn],
    hourly: dict[int, float],
    next_day_hourly: dict[int, float] | None = None,
    round_to: int | None = None,
    first_sales_hour: int = FIRST_SALES_HOUR,
    last_sales_hour: int = LAST_SALES_HOUR,
) -> dict[int, int | str]:
    """Allocate forecast quantities into a row's template-assigned columns.

    Bake windows get the summed forecast over their coverage hours. Defrost
    columns get tomorrow's early-window volume (fallback: today's
    early-window volume) and keep their original annotation text. Defrost
    columns never contribute to today's coverage.
    """
    if not hourly:
        return {}

    effective_round_to = round_to or schedule_round_to(schedule)
    bake_columns = [item for item in schedule if not item.is_defrost]
    bake_windows = [item.window for item in bake_columns]
    column_by_label = {item.window.label: item.column for item in bake_columns}
    allocated: dict[int, int | str] = {}

    coverage = coverage_hours(bake_windows, first_sales_hour=first_sales_hour, last_sales_hour=last_sales_hour)
    carry = 0.0
    for window in sorted(bake_windows, key=lambda item: (item.start_hour, item.end_hour, item.label)):
        demand = sum(hourly.get(hour, 0.0) for hour in coverage.get(window.label, []))
        net_qty = max(demand - carry, 0.0)
        if net_qty <= 0:
            carry = max(carry - demand, 0.0)
            continue
        baked_qty = _round_up(net_qty, effective_round_to)
        allocated[column_by_label[window.label]] = baked_qty
        carry = carry + baked_qty - demand

    defrost_columns = [item for item in schedule if item.is_defrost]
    if defrost_columns:
        source = next_day_hourly if next_day_hourly else hourly
        defrost_qty = _round_up(_early_window_sum(source, first_sales_hour=first_sales_hour), effective_round_to)
        for item in defrost_columns:
            # Strip the template's leading number, keep the annotation (e.g.
            # "(ночная дефр)") and re-emit it with the forecast-sized quantity.
            annotation = re.sub(r"^\s*\d+\s*", "", item.note or "").strip()
            allocated[item.column] = f"{defrost_qty} {annotation}".strip() if annotation else defrost_qty
    return allocated

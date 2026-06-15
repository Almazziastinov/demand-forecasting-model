from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from openpyxl import load_workbook
from openpyxl.worksheet.worksheet import Worksheet


logger = logging.getLogger(__name__)

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
# Night-defrost columns are prep for tomorrow's morning batches; size them from
# tomorrow's forecast over hours [FIRST_SALES_HOUR .. DEFROST_EARLY_CUTOFF - 1].
DEFROST_EARLY_CUTOFF = 12
DEFROST_MARKERS = ("дефр", "ночн")
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


@dataclass(frozen=True)
class ScheduledColumn:
    """A C:L cell that the template pre-filled for a SKU row.

    ``is_defrost`` marks night-defrost prep (sized from tomorrow's early forecast,
    excluded from today's coverage). ``note`` keeps the original cell text so the
    defrost annotation (e.g. ``"20 (ночная дефр)"``) can be re-emitted.
    """

    window: BakingWindow
    is_defrost: bool
    note: str | None
    quantity: int | None = None
    is_artificial: bool = False


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


def coverage_hours(
    windows: list[BakingWindow],
    *,
    first_sales_hour: int = FIRST_SALES_HOUR,
    last_sales_hour: int = LAST_SALES_HOUR,
) -> dict[int, list[int]]:
    """Map each bake window to the sales hours it covers.

    A batch becomes available at its window's end hour and covers sales until the
    next batch becomes available (the next window's end hour); the last window runs
    through ``last_sales_hour``. The earliest window always starts its coverage at
    ``first_sales_hour`` so a SKU baked only once (e.g. midday) still absorbs the
    whole day's forecast instead of silently dropping the morning.
    """
    if not windows:
        return {}

    ordered = sorted(
        windows,
        key=lambda item: (item.start_hour, item.end_hour, item.column),
    )
    result: dict[int, list[int]] = {}
    for index, window in enumerate(ordered):
        if index == 0:
            start_hour = first_sales_hour
        else:
            start_hour = max(window.end_hour, first_sales_hour)

        if index + 1 < len(ordered):
            end_hour = ordered[index + 1].end_hour - 1
        else:
            end_hour = last_sales_hour

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


def read_row_schedule(
    sheet: Worksheet,
    row_index: int,
    sheet_windows: dict[int, BakingWindow],
) -> list[ScheduledColumn]:
    """Read which C:L columns the template pre-filled for this SKU row.

    A non-empty cell marks a scheduled baking window. Cells whose text carries a
    night-defrost marker (e.g. ``"20 (ночная дефр)"``) are classified as defrost
    prep. Classification keys off the *cell value*, not the SKU name (many SKU
    names contain "ночного брожжения" yet have plain integer bake cells).
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
                is_defrost=defrost,
                note=str(value) if defrost else None,
                quantity=_extract_positive_int(value),
            )
        )
    return schedule


def _round_up(qty: float, round_to: int) -> int:
    if round_to <= 1:
        return int(math.ceil(qty))
    return int(math.ceil(qty / round_to) * round_to)


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


def schedule_round_to(schedule: list[ScheduledColumn]) -> int:
    quantities = [
        item.quantity
        for item in schedule
        if not item.is_defrost and item.quantity is not None and item.quantity > 0
    ]
    if not quantities:
        quantities = [
            item.quantity
            for item in schedule
            if item.quantity is not None and item.quantity > 0
        ]
    if not quantities:
        return 1

    round_to = quantities[0]
    for quantity in quantities[1:]:
        round_to = math.gcd(round_to, quantity)
    return max(round_to, 1)


def _early_window_sum(
    hourly: dict[int, float],
    first_sales_hour: int = FIRST_SALES_HOUR,
    cutoff: int = DEFROST_EARLY_CUTOFF,
) -> float:
    return sum(
        qty for hour, qty in hourly.items() if first_sales_hour <= hour < cutoff
    )


def _with_midday_split_window(
    bake_windows: list[BakingWindow],
    *,
    available_windows: dict[int, BakingWindow] | None,
    hourly: dict[int, float],
    first_sales_hour: int,
    last_sales_hour: int,
) -> list[BakingWindow]:
    if len(bake_windows) != 1 or not available_windows:
        return bake_windows

    early_window = bake_windows[0]
    if early_window.end_hour > DEFROST_EARLY_CUTOFF:
        return bake_windows

    later_demand = sum(
        qty
        for hour, qty in hourly.items()
        if DEFROST_EARLY_CUTOFF <= hour <= last_sales_hour
    )
    if later_demand <= 0:
        return bake_windows

    candidates = [
        window
        for window in available_windows.values()
        if window.column != early_window.column
        and window.end_hour >= DEFROST_EARLY_CUTOFF
        and window.start_hour >= first_sales_hour
    ]
    if not candidates:
        return bake_windows

    split_window = sorted(
        candidates,
        key=lambda item: (
            abs(item.end_hour - DEFROST_EARLY_CUTOFF),
            item.end_hour,
            item.column,
        ),
    )[0]
    return [early_window, split_window]


def allocate_template_row(
    *,
    template_sku_name: object,
    schedule: list[ScheduledColumn],
    product_hour_lookup: dict[str, dict[int, float]],
    next_day_hour_lookup: dict[str, dict[int, float]] | None = None,
    round_to: int | None = None,
    available_windows: dict[int, BakingWindow] | None = None,
    first_sales_hour: int = FIRST_SALES_HOUR,
    last_sales_hour: int = LAST_SALES_HOUR,
) -> dict[int, int | str]:
    """Allocate forecast quantities into a row's scheduled columns.

    Bake windows get the summed forecast over their coverage hours. Defrost columns
    get tomorrow's early-window volume (fallback: today's early-window volume) and
    keep their original annotation text. Defrost columns never contribute to today's
    coverage.
    """
    hourly = _resolve_hourly_forecast(template_sku_name, product_hour_lookup)
    if not hourly:
        return {}

    effective_round_to = round_to or schedule_round_to(schedule)
    bake_windows = [item.window for item in schedule if not item.is_defrost]
    bake_windows = _with_midday_split_window(
        bake_windows,
        available_windows=available_windows,
        hourly=hourly,
        first_sales_hour=first_sales_hour,
        last_sales_hour=last_sales_hour,
    )
    allocated: dict[int, int | str] = {}

    coverage = coverage_hours(
        bake_windows,
        first_sales_hour=first_sales_hour,
        last_sales_hour=last_sales_hour,
    )
    carry = 0.0
    for window in sorted(
        bake_windows,
        key=lambda item: (item.start_hour, item.end_hour, item.column),
    ):
        demand = sum(hourly.get(hour, 0.0) for hour in coverage.get(window.column, []))
        net_qty = max(demand - carry, 0.0)
        if net_qty <= 0:
            carry = max(carry - demand, 0.0)
            continue
        baked_qty = _round_up(net_qty, effective_round_to)
        allocated[window.column] = baked_qty
        carry = carry + baked_qty - demand

    defrost_columns = [item for item in schedule if item.is_defrost]
    if defrost_columns:
        next_hourly = None
        if next_day_hour_lookup is not None:
            next_hourly = _resolve_hourly_forecast(
                template_sku_name, next_day_hour_lookup
            )
        source = next_hourly if next_hourly else hourly
        defrost_qty = _round_up(
            _early_window_sum(source, first_sales_hour=first_sales_hour),
            effective_round_to,
        )
        for item in defrost_columns:
            # Strip the template's leading number, keep the annotation (e.g.
            # "(ночная дефр)") and re-emit it with the forecast-sized quantity.
            annotation = re.sub(r"^\s*\d+\s*", "", item.note or "").strip()
            allocated[item.window.column] = (
                f"{defrost_qty} {annotation}".strip() if annotation else defrost_qty
            )
    return allocated


def _select_sheet_name(bucket: str | None, sheet_names: list[str]) -> str:
    if bucket and bucket in sheet_names:
        return bucket
    if bucket:
        # Per-SKU schedules differ per sheet, so a wrong/unknown bucket silently
        # picks the wrong baking schedule. Surface it instead of failing quietly.
        logger.warning(
            "baking_plan: revenue bucket %r does not match any sheet %r; "
            "falling back to default",
            bucket,
            sheet_names,
        )
    if DEFAULT_BUCKET in sheet_names:
        return DEFAULT_BUCKET
    return sheet_names[0]


def _sales_hour_bounds(
    product_hour_lookup: dict[str, dict[int, float]],
) -> tuple[int, int]:
    hours = [hour for hourly in product_hour_lookup.values() for hour in hourly]
    if not hours:
        return FIRST_SALES_HOUR, LAST_SALES_HOUR
    return min(min(hours), FIRST_SALES_HOUR), max(max(hours), LAST_SALES_HOUR)


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
    next_day_sku_hour_rows: list[dict[str, Any]] | None = None,
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
        "Количество по SKU-hour прогнозу активного run, разнесено по расписанию "
        "выпекания шаблона; ночная дефростация — по утреннему прогнозу след. дня."
    )

    sheet_windows = _sheet_windows(sheet)
    product_hour_lookup = build_product_hour_lookup(sku_hour_rows)
    next_day_hour_lookup = (
        build_product_hour_lookup(next_day_sku_hour_rows)
        if next_day_sku_hour_rows
        else None
    )
    first_sales_hour, last_sales_hour = _sales_hour_bounds(product_hour_lookup)

    for row_index in range(6, sheet.max_row + 1):
        sku_name = sheet.cell(row=row_index, column=2).value
        if not sku_name:
            continue

        # The pre-filled C:L cells ARE the per-SKU baking schedule. Rows without
        # any scheduled cell are section headers / sub-items — skip, don't fill.
        schedule = read_row_schedule(sheet, row_index, sheet_windows)
        if not schedule:
            continue

        allocated = allocate_template_row(
            template_sku_name=sku_name,
            schedule=schedule,
            product_hour_lookup=product_hour_lookup,
            next_day_hour_lookup=next_day_hour_lookup,
            first_sales_hour=first_sales_hour,
            last_sales_hour=last_sales_hour,
            available_windows=sheet_windows,
        )
        # Rewrite scheduled cells and any added split window. Other template cells
        # stay untouched.
        for item in schedule:
            column = item.window.column
            sheet.cell(row=row_index, column=column).value = allocated.get(column)
        scheduled_columns = {item.window.column for item in schedule}
        for column, value in allocated.items():
            if column not in scheduled_columns:
                sheet.cell(row=row_index, column=column).value = value

    output = BytesIO()
    workbook.save(output)
    return output.getvalue()

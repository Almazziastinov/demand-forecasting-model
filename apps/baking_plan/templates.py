"""Excel template selection and "комментарии" sheet parsing.

Templates live in `apps/baking_plan/assets/`:
- `template.xlsx` — base template, one sheet per bakery-revenue tier
  (`до 1,5 млн`, `до 2,5 млн`, `от 2,5 млн`, `от 3млн`) plus "комментарии"
- `individual/{bakery_id}_*.xlsx` — per-bakery override, takes priority over
  the revenue-tier sheet selection entirely

Each revenue-tier sheet's data rows (from `PLAN_START_ROW`) have some C:L
window cells pre-filled by a technologist — that is the SKU's baking-window
*assignment*; `allocation.read_row_schedule` reads it directly, this module
never computes window placement.

The "комментарии" sheet groups SKUs by dough (тесто-группа). A row is a
group header when column C (кратность выпуска) is empty or literally
"кратность выпуска"; every row after it belongs to that group until the next
header row. The sheet also has a staffing-norms reference block at the
bottom ("Нормативы по штату...") that is not per-SKU metadata — parsing
stops there.
"""

from __future__ import annotations

# ruff: noqa: E501
import re
from dataclasses import dataclass
from pathlib import Path

from openpyxl.workbook import Workbook

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
BASE_TEMPLATE_PATH = ASSETS_DIR / "template.xlsx"
INDIVIDUAL_TEMPLATES_DIR = ASSETS_DIR / "individual"

COMMENTS_SHEET_NAME = "комментарии"
GROUP_HEADER_LABEL = "кратность выпуска"
ON_DEMAND_MARKER = "запрос"
TWO_DAY_MARKER = "двухдневка"
STAFFING_SECTION_MARKER = "Нормативы"


@dataclass(frozen=True)
class SkuMeta:
    sku_name: str
    station: str | None
    dough_group: str
    kratnost: int
    kratnost_raw: str
    is_on_demand: bool
    is_two_day: bool


def normalize_name(value: object) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"\s+", " ", text)


def _parse_kratnost(raw: object) -> tuple[int, bool]:
    """Return (multiple, is_on_demand). "1 (в ней 6 кусков)" -> (1, False)."""
    if raw is None:
        return 1, False
    if isinstance(raw, (int, float)):
        return int(raw), False
    text = str(raw).strip()
    if ON_DEMAND_MARKER in text.lower():
        return 1, True
    match = re.match(r"\d+", text)
    if match:
        return int(match.group()), False
    return 1, False


def parse_comments_sheet(workbook: Workbook) -> dict[str, SkuMeta]:
    """Parse the "комментарии" sheet into {normalized_sku_name: SkuMeta}."""
    sheet = workbook[COMMENTS_SHEET_NAME]
    result: dict[str, SkuMeta] = {}
    current_group = ""
    for row in sheet.iter_rows(min_row=2):
        station_cell, name_cell, kratnost_cell, two_day_cell = row[0], row[1], row[2], row[3]

        station_value = station_cell.value
        if isinstance(station_value, str) and STAFFING_SECTION_MARKER in station_value:
            break

        name = name_cell.value
        if name is None:
            continue

        kratnost_raw = kratnost_cell.value
        is_header = kratnost_raw is None or (
            isinstance(kratnost_raw, str) and kratnost_raw.strip().lower() == GROUP_HEADER_LABEL
        )
        if is_header:
            current_group = str(name).strip()
            continue

        kratnost, is_on_demand = _parse_kratnost(kratnost_raw)
        two_day_value = two_day_cell.value
        is_two_day = isinstance(two_day_value, str) and TWO_DAY_MARKER in two_day_value.lower()
        station = (
            str(station_value).strip()
            if isinstance(station_value, str) and station_value.strip()
            else None
        )
        result[normalize_name(name)] = SkuMeta(
            sku_name=str(name).strip(),
            station=station,
            dough_group=current_group,
            kratnost=kratnost,
            kratnost_raw="" if kratnost_raw is None else str(kratnost_raw),
            is_on_demand=is_on_demand,
            is_two_day=is_two_day,
        )
    return result


WINDOWS_HEADER_ROW = 5
WINDOW_LABEL_RE = re.compile(r"^(\d{1,2})[:.]\d{2}\s*-\s*(\d{1,2})[:.]\d{2}$")
PLAN_START_ROW = 6


@dataclass(frozen=True)
class Window:
    label: str
    column: int
    start_hour: int
    end_hour: int


def parse_windows(workbook: Workbook, sheet_name: str) -> list[Window]:
    """Parse the window column headers (row 5, columns from C onward)."""
    sheet = workbook[sheet_name]
    windows: list[Window] = []
    for cell in sheet[WINDOWS_HEADER_ROW][2:]:
        value = cell.value
        if value is None:
            continue
        match = WINDOW_LABEL_RE.match(str(value).strip())
        if not match:
            continue
        start_hour, end_hour = int(match.group(1)), int(match.group(2))
        windows.append(
            Window(label=str(value).strip(), column=cell.column, start_hour=start_hour, end_hour=end_hour)
        )
    return windows


def template_path_for_bakery(bakery_id: int) -> Path:
    for path in sorted(INDIVIDUAL_TEMPLATES_DIR.glob(f"{bakery_id}_*.xlsx")):
        return path
    return BASE_TEMPLATE_PATH


# Revenue-tier sheet selection for the base template — individual templates
# (matched above) bypass this entirely, they have their own single sheet.
DEFAULT_BUCKET = "до 2,5 млн"
REVENUE_BUCKETS = (
    (1_500_000, "до 1,5 млн"),
    (2_500_000, "до 2,5 млн"),
    (3_000_000, "от 2,5 млн"),
)


def revenue_bucket(revenue: float | int | None) -> str:
    if revenue is None:
        return DEFAULT_BUCKET
    value = float(revenue)
    for threshold, bucket in REVENUE_BUCKETS:
        if value < threshold:
            return bucket
    return "от 3млн"


def select_sheet_name(bucket: str | None, sheet_names: list[str]) -> str:
    if bucket and bucket in sheet_names:
        return bucket
    if DEFAULT_BUCKET in sheet_names:
        return DEFAULT_BUCKET
    return sheet_names[0]

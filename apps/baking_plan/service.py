"""Public entrypoint for the baking-plan package.

`router.py` (and nothing else outside this package) calls into this module.

Window *assignment* comes straight from the loaded template (whichever C:L
cells a technologist pre-filled); this module's own job is only to resolve
each template row to a live assortment product, pull its forecast, and size
quantities into the windows the template already assigned. See
`docs/baking_plan_implementation.md`.
"""

from __future__ import annotations

# ruff: noqa: E501
from datetime import date as date_type
from datetime import timedelta
from io import BytesIO

import openpyxl

from . import allocation, assortment, demand, rendering, templates


def build_baking_plan_workbook(
    *,
    run_id: str,
    forecast_date: str,
    bakery_id: int,
    bakery_name: str,
    city: str,
) -> bytes:
    """Build the baking-plan .xlsx file content for one bakery/date."""
    revenue = demand.load_revenue_bucket_input(bakery_id)
    bucket = templates.revenue_bucket(revenue)
    template_path = templates.template_path_for_bakery(bakery_id)

    workbook = openpyxl.load_workbook(template_path)
    sku_meta = templates.parse_comments_sheet(workbook)
    selected_sheet_name = templates.select_sheet_name(bucket, workbook.sheetnames)
    windows = templates.parse_windows(workbook, selected_sheet_name)
    sheet_windows = {window.column: window for window in windows}

    sheet = workbook[selected_sheet_name]
    for worksheet in list(workbook.worksheets):
        if worksheet.title != selected_sheet_name:
            workbook.remove(worksheet)
    sheet.title = rendering.SHEET_TITLE

    assortment_rows = assortment.get_bakeable_products(city, forecast_date, bakery_id=bakery_id)
    assortment_lookup = allocation.build_assortment_lookup(assortment_rows)
    all_product_ids = [row["product_id"] for row in assortment_rows]

    next_day_date = (date_type.fromisoformat(forecast_date) + timedelta(days=1)).isoformat()
    hourly = demand.load_hourly(run_id, forecast_date, bakery_id, all_product_ids)
    next_day_hourly = demand.load_hourly(run_id, next_day_date, bakery_id, all_product_ids)

    total_column = 3 + len(windows)
    plan_rows: list[dict] = []
    present_product_ids: set[str] = set()

    for row_index in range(templates.PLAN_START_ROW, sheet.max_row + 1):
        sheet.row_dimensions[row_index].hidden = False
        sku_name = sheet.cell(row=row_index, column=2).value
        if not sku_name:
            continue

        schedule = allocation.read_row_schedule(sheet, row_index, sheet_windows)
        if not schedule:
            continue

        assortment_product = allocation.resolve_assortment_product(sku_name, assortment_lookup)
        if assortment_product is None:
            continue

        product_id = assortment_product["product_id"]
        meta = sku_meta.get(templates.normalize_name(sku_name))
        round_to = meta.kratnost if meta else None

        allocated = allocation.allocate_template_row(
            schedule=schedule,
            hourly=hourly.get(product_id, {}),
            next_day_hourly=next_day_hourly.get(product_id, {}),
            round_to=round_to,
        )
        present_product_ids.add(product_id)
        plan_rows.append(
            {
                "snapshot": rendering.snapshot_row(sheet, row_index, total_column),
                "product_id": product_id,
                "product_name": assortment_product["product_name"],
                "category_name": assortment_product["category_name"],
                "allocated": allocated,
                "total": float(sum(value for value in allocated.values() if isinstance(value, (int, float)))),
                "source_order": row_index,
            }
        )

    for product in assortment_rows:
        product_id = product["product_id"]
        if product_id in present_product_ids:
            continue
        plan_rows.append(
            {
                "snapshot": None,
                "product_id": product_id,
                "product_name": product["product_name"],
                "category_name": product["category_name"],
                "allocated": {},
                "total": sum(hourly.get(product_id, {}).values()),
                "source_order": 10**9,
            }
        )

    rendering.write_plan(
        sheet=sheet,
        windows=windows,
        plan_rows=plan_rows,
        bakery_name=bakery_name,
        forecast_date=forecast_date,
        selected_sheet_name=selected_sheet_name,
    )

    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()

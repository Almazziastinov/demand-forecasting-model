"""Public entrypoint for the baking-plan package.

`router.py` (and nothing else outside this package) calls into this module.
"""

from __future__ import annotations

# ruff: noqa: E501
from io import BytesIO

import openpyxl

from . import capacity, demand
from .algorithms.milp import allocate_milp_detailed
from .rendering import render_workbook
from .templates import BASE_TEMPLATE_PATH, parse_windows


def build_baking_plan_workbook(
    *,
    run_id: str,
    forecast_date: str,
    bakery_id: int,
    bakery_name: str,
    city: str,
) -> bytes:
    """Build the baking-plan .xlsx file content for one bakery/date.

    Returns the raw bytes of the generated workbook.
    """
    skus, _skipped = demand.build_sku_demand(
        run_id=run_id, forecast_date=forecast_date, bakery_id=bakery_id, city=city
    )
    if not skus:
        raise RuntimeError(f"No bakeable SKUs with metadata found for bakery_id={bakery_id}")

    windows_workbook = openpyxl.load_workbook(BASE_TEMPLATE_PATH, data_only=True)
    windows = parse_windows(windows_workbook)

    capacity_config = capacity.get_capacity_config(bakery_id)
    molding_map = capacity.get_molding_minutes_map()

    regular_alloc, defrost_alloc, two_day_alloc = allocate_milp_detailed(
        skus, windows, capacity_config, molding_map
    )

    workbook = render_workbook(
        bakery_name=bakery_name,
        forecast_date=forecast_date,
        windows=windows,
        skus=skus,
        regular_alloc=regular_alloc,
        defrost_alloc=defrost_alloc,
        two_day_alloc=two_day_alloc,
    )
    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()

"""Public entrypoint for the baking-plan package.

`router.py` (and nothing else outside this package) calls into this module.

Window assignment is determined by the MILP solver using the hourly sales
profile from the active forecast run. The Excel template is still used to
select the correct window time-slot structure (one sheet per revenue tier)
and to establish which time windows exist for this bakery. Rendering is done
from scratch via `rendering_milp.render_workbook` — no template mutation.

See `docs/baking_plan_implementation.md` for the business-rule spec and
`apps/baking_plan/algorithms/milp.py` for the solver model.
"""

from __future__ import annotations

import logging
from io import BytesIO

import openpyxl

from . import demand, templates
from .algorithms.milp import allocate_milp_detailed
from .capacity import CapacityConfig, get_capacity_config, get_molding_minutes_map
from .demand_milp import build_sku_demand
from .rendering_milp import MANDATORY_ASSORTMENT, render_workbook

logger = logging.getLogger(__name__)


def build_baking_plan_workbook(
    *,
    run_id: str,
    forecast_date: str,
    bakery_id: int,
    bakery_name: str,
    city: str,
) -> bytes:
    """Build the baking-plan .xlsx file content for one bakery/date."""
    skus, skipped = build_sku_demand(
        run_id=run_id,
        forecast_date=forecast_date,
        bakery_id=bakery_id,
        city=city,
    )
    if skipped:
        logger.warning(
            "baking_plan: %d SKUs skipped (no baking_sku_meta): %s",
            len(skipped),
            skipped,
        )

    # Force mandatory assortment SKUs into the first baking window.
    mandatory_ids = frozenset(
        sku.product_id for sku in skus if sku.product_name in MANDATORY_ASSORTMENT
    )

    # Read the template only to obtain the window time structure for this
    # bakery's revenue tier. The template cells themselves are not used for
    # quantity allocation.
    revenue = demand.load_revenue_bucket_input(bakery_id)
    bucket = templates.revenue_bucket(revenue)
    template_path = templates.template_path_for_bakery(bakery_id)
    workbook = openpyxl.load_workbook(template_path)
    selected_sheet_name = templates.select_sheet_name(bucket, workbook.sheetnames)
    all_windows = templates.parse_windows(workbook, selected_sheet_name)

    # Some templates have both hourly windows (4:00-7:00, 7:00-8:00, ...) and
    # coarser merged windows (8:00-10:00, ...) with different labels in the
    # same sheet. The MILP receives the deduplicated hourly set (first
    # occurrence of each label) so that each window label maps to exactly one
    # column index in the output workbook.
    seen_labels: set[str] = set()
    milp_windows = []
    for w in all_windows:
        if w.label not in seen_labels:
            milp_windows.append(w)
            seen_labels.add(w.label)

    try:
        capacity = get_capacity_config(bakery_id)
    except RuntimeError:
        logger.warning(
            "baking_plan: no baking_capacity_config for bakery %d, using fallback",
            bakery_id,
        )
        capacity = CapacityConfig(
            bakers_count=2, ovens_count=2, trays_per_oven_batch=6, bake_minutes=30
        )
    molding_map = get_molding_minutes_map()

    regular, defrost_out, two_day_out, shortfall_by_sku, defrost_shortfall_by_sku = (
        allocate_milp_detailed(
            skus=skus,
            windows=milp_windows,
            capacity=capacity,
            molding_minutes_map=molding_map,
            mandatory_first_window_ids=mandatory_ids,
        )
    )

    wb = render_workbook(
        bakery_name=bakery_name,
        forecast_date=forecast_date,
        windows=milp_windows,
        skus=skus,
        regular_alloc=regular,
        defrost_alloc=defrost_out,
        two_day_alloc=two_day_out,
        shortfall_by_sku=shortfall_by_sku,
        defrost_shortfall_by_sku=defrost_shortfall_by_sku,
    )

    buffer = BytesIO()
    wb.save(buffer)
    return buffer.getvalue()

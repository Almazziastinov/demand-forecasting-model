"""HTTP surface for the baking-plan package.

Mounted into the `forecast_embedded` FastAPI app (see `app/main.py`). This is
the only file in this package that `forecast_embedded` reaches into; nothing
in `app/routers` or `app/services` should import from `baking_plan.*`
directly.
"""

from __future__ import annotations

# ruff: noqa: E501
import logging
from io import BytesIO

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from app.auth import get_auth_context
from app.services import bakery as bakery_service
from app.services import runs as run_service

from . import service as baking_plan_service

router = APIRouter(tags=["baking_plan"])
logger = logging.getLogger(__name__)


def _resolve_run(auth, run_id: str | None) -> dict | None:
    return run_service.resolve_run(run_id if auth.is_admin else None)


@router.get("/bakery/{bakery_id}/baking-plan.xlsx")
def download_baking_plan(
    request: Request,
    bakery_id: int,
    date: str = Query(...),
    run_id: str | None = Query(default=None),
) -> StreamingResponse:
    auth = get_auth_context(request)
    active_run = _resolve_run(auth, run_id)
    if not active_run:
        raise HTTPException(status_code=404, detail="Forecast run not found")

    bakery_day = bakery_service.get_bakery_day(active_run["run_id"], date, bakery_id, auth)
    if not bakery_day:
        raise HTTPException(status_code=404, detail="Bakery forecast not found")

    city = str(bakery_day.get("city") or "").strip()
    if not city:
        raise HTTPException(status_code=503, detail="Bakery city is unavailable")

    try:
        content = baking_plan_service.build_baking_plan_workbook(
            run_id=active_run["run_id"],
            forecast_date=date,
            bakery_id=bakery_id,
            bakery_name=str(bakery_day.get("bakery_name") or bakery_id),
            city=city,
        )
    except Exception as exc:
        logger.error("baking_plan: workbook generation failed", exc_info=True)
        raise HTTPException(status_code=503, detail="Baking plan is unavailable") from exc

    filename = f"baking_plan_{bakery_id}_{date}_assortment_v2.xlsx"
    return StreamingResponse(
        BytesIO(content),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )

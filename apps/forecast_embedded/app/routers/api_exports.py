from __future__ import annotations

# ruff: noqa: E501

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response

from app.auth import get_auth_context
from app.services import exports as export_service
from app.services import runs as run_service


router = APIRouter(prefix="/api/v1", tags=["exports"])


@router.get("/export/bakery")
def export_bakery_csv(
    request: Request,
    bakery_id: int = Query(...),
    date: str = Query(...),
    run_id: str | None = None,
) -> Response:
    active_run = run_service.resolve_run(run_id)
    if not active_run:
        raise HTTPException(status_code=404, detail="Forecast run not found")

    auth = get_auth_context(request)
    csv_text = export_service.export_bakery_top_sku_csv(
        active_run["run_id"],
        date,
        bakery_id,
        auth,
    )
    filename = f"bakery_{bakery_id}_{date}.csv"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=csv_text, media_type="text/csv; charset=utf-8", headers=headers)

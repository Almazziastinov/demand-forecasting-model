from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from app.services import exports as export_service
from app.services import runs as run_service


router = APIRouter(prefix="/api/v1", tags=["exports"])


@router.get("/export/bakery")
def export_bakery_csv(
    bakery_id: int = Query(...),
    date: str = Query(...),
) -> Response:
    active_run = run_service.get_active_run()
    if not active_run:
        raise HTTPException(status_code=404, detail="No active forecast run found")

    csv_text = export_service.export_bakery_top_sku_csv(active_run["run_id"], date, bakery_id)
    filename = f"bakery_{bakery_id}_{date}.csv"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=csv_text, media_type="text/csv; charset=utf-8", headers=headers)

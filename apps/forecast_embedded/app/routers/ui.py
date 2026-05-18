from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.services import bakery as bakery_service
from app.services import runs as run_service


router = APIRouter(tags=["ui"])
templates = Jinja2Templates(directory="app/templates")


@router.get("/", response_class=HTMLResponse)
def index(request: Request, date: str | None = Query(default=None)) -> HTMLResponse:
    active_run = run_service.get_active_run()
    if not active_run:
        raise HTTPException(status_code=404, detail="No active forecast run found")

    dates = run_service.get_run_dates(active_run["run_id"])
    selected_date = date or (dates[0] if dates else None)
    bakeries = (
        bakery_service.get_bakery_list(active_run["run_id"], selected_date)
        if selected_date
        else []
    )
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "active_run": active_run,
            "dates": dates,
            "selected_date": selected_date,
            "bakeries": bakeries,
        },
    )


@router.get("/bakery/{bakery_id}", response_class=HTMLResponse)
def bakery_detail(
    request: Request,
    bakery_id: int,
    date: str = Query(...),
) -> HTMLResponse:
    active_run = run_service.get_active_run()
    if not active_run:
        raise HTTPException(status_code=404, detail="No active forecast run found")

    bakery_day = bakery_service.get_bakery_day(active_run["run_id"], date, bakery_id)
    if not bakery_day:
        raise HTTPException(status_code=404, detail="Bakery forecast not found")

    return templates.TemplateResponse(
        request,
        "bakery.html",
        {
            "active_run": active_run,
            "selected_date": date,
            "bakery": bakery_day,
            "hourly_total": bakery_service.get_hourly_total(active_run["run_id"], date, bakery_id),
            "top_sku": bakery_service.get_top_sku(active_run["run_id"], date, bakery_id),
        },
    )

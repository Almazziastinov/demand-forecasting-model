from __future__ import annotations

# ruff: noqa: E501

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.auth import get_auth_context
from app.services import bakery as bakery_service
from app.services import runs as run_service


router = APIRouter(tags=["ui"])
templates = Jinja2Templates(directory="app/templates")


@router.get("/", response_class=HTMLResponse)
def index(
    request: Request,
    date: str | None = Query(default=None),
    run_id: str | None = Query(default=None),
) -> HTMLResponse:
    active_run = run_service.resolve_run(run_id)
    if not active_run:
        raise HTTPException(status_code=404, detail="Forecast run not found")

    dates = run_service.get_run_dates(active_run["run_id"])
    selected_date = date or (dates[0] if dates else None)
    auth = get_auth_context(request)
    bakeries = (
        bakery_service.get_bakery_list(active_run["run_id"], selected_date, auth)
        if selected_date
        else []
    )
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "active_run": active_run,
            "runs": run_service.list_runs(),
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
    run_id: str | None = Query(default=None),
) -> HTMLResponse:
    active_run = run_service.resolve_run(run_id)
    if not active_run:
        raise HTTPException(status_code=404, detail="Forecast run not found")

    auth = get_auth_context(request)
    bakery_day = bakery_service.get_bakery_day(active_run["run_id"], date, bakery_id, auth)
    if not bakery_day:
        raise HTTPException(status_code=404, detail="Bakery forecast not found")

    return templates.TemplateResponse(
        request,
        "bakery.html",
        {
            "active_run": active_run,
            "runs": run_service.list_runs(),
            "selected_date": date,
            "bakery": bakery_day,
            "context": bakery_service.get_day_context(active_run["run_id"], date, bakery_day["city"]),
            "hourly_total": bakery_service.get_hourly_total(active_run["run_id"], date, bakery_id, auth),
            "top_sku": bakery_service.get_top_sku(active_run["run_id"], date, bakery_id, auth),
        },
    )

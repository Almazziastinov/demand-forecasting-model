from __future__ import annotations

# ruff: noqa: E501

from fastapi import APIRouter, HTTPException, Query, Request

from app.auth import get_auth_context
from app.services import bakery as bakery_service
from app.services import runs as run_service


router = APIRouter(prefix="/api/v1", tags=["bakeries"])


def _require_run(run_id: str | None = None) -> dict:
    run = run_service.resolve_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Forecast run not found")
    return run


@router.get("/bakeries")
def get_bakeries(
    request: Request,
    date: str = Query(...),
    run_id: str | None = None,
) -> dict:
    active_run = _require_run(run_id)
    auth = get_auth_context(request)
    return {
        "run_id": active_run["run_id"],
        "items": bakery_service.get_bakery_list(active_run["run_id"], date, auth),
    }


@router.get("/bakeries/{bakery_id}/summary")
def get_bakery_summary(
    request: Request,
    bakery_id: int,
    date: str = Query(...),
    run_id: str | None = None,
) -> dict:
    active_run = _require_run(run_id)
    auth = get_auth_context(request)
    bakery_day = bakery_service.get_bakery_day(active_run["run_id"], date, bakery_id, auth)
    if not bakery_day:
        raise HTTPException(status_code=404, detail="Bakery forecast not found")

    return {
        "run_id": active_run["run_id"],
        "bakery": {
            "bakery_id": bakery_day["bakery_id"],
            "bakery_name": bakery_day["bakery_name"],
            "city": bakery_day["city"],
        },
        "day": {
            "forecast_base": bakery_day["forecast_base"],
            "forecast_final": bakery_day["forecast_final"],
        },
        "context": bakery_service.get_day_context(
            active_run["run_id"],
            date,
            bakery_day["city"],
        ),
        "hourly_total": bakery_service.get_hourly_total(active_run["run_id"], date, bakery_id, auth),
        "top_sku": bakery_service.get_top_sku(active_run["run_id"], date, bakery_id, auth),
        "meta": active_run,
    }


@router.get("/bakeries/{bakery_id}/sku-day")
def get_bakery_sku_day(
    request: Request,
    bakery_id: int,
    date: str = Query(...),
    limit: int = Query(default=100, ge=1, le=1000),
    run_id: str | None = None,
) -> dict:
    active_run = _require_run(run_id)
    auth = get_auth_context(request)
    return {
        "run_id": active_run["run_id"],
        "items": bakery_service.get_top_sku(active_run["run_id"], date, bakery_id, auth, limit=limit),
    }


@router.get("/bakeries/{bakery_id}/sku-hour")
def get_bakery_sku_hour(
    request: Request,
    bakery_id: int,
    product_id: int = Query(...),
    date: str = Query(...),
    run_id: str | None = None,
) -> dict:
    active_run = _require_run(run_id)
    auth = get_auth_context(request)
    items = bakery_service.get_sku_hour(active_run["run_id"], date, bakery_id, product_id, auth)
    if not items:
        raise HTTPException(status_code=404, detail="SKU hourly forecast not found")
    return {
        "run_id": active_run["run_id"],
        "items": items,
    }


@router.get("/bakeries/{bakery_id}/hour-discrepancy")
def get_bakery_hour_discrepancy(
    request: Request,
    bakery_id: int,
    date: str = Query(...),
    hour: int = Query(..., ge=0, le=23),
    limit: int = Query(default=10, ge=1, le=50),
    run_id: str | None = None,
) -> dict:
    active_run = _require_run(run_id)
    auth = get_auth_context(request)
    return {
        "run_id": active_run["run_id"],
        **bakery_service.get_hour_discrepancy_contributors(
            active_run["run_id"],
            date,
            bakery_id,
            hour,
            auth,
            limit=limit,
        ),
    }

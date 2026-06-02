from __future__ import annotations

# ruff: noqa: E501

from fastapi import APIRouter, HTTPException, Query

from app.services import bakery as bakery_service
from app.services import runs as run_service


router = APIRouter(prefix="/api/v1", tags=["bakeries"])


def _require_run(run_id: str | None = None) -> dict:
    run = run_service.resolve_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Forecast run not found")
    return run


@router.get("/bakeries")
def get_bakeries(date: str = Query(...), run_id: str | None = None) -> dict:
    active_run = _require_run(run_id)
    return {
        "run_id": active_run["run_id"],
        "items": bakery_service.get_bakery_list(active_run["run_id"], date),
    }


@router.get("/bakeries/{bakery_id}/summary")
def get_bakery_summary(
    bakery_id: int,
    date: str = Query(...),
    run_id: str | None = None,
) -> dict:
    active_run = _require_run(run_id)
    bakery_day = bakery_service.get_bakery_day(active_run["run_id"], date, bakery_id)
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
        "hourly_total": bakery_service.get_hourly_total(active_run["run_id"], date, bakery_id),
        "top_sku": bakery_service.get_top_sku(active_run["run_id"], date, bakery_id),
        "meta": active_run,
    }


@router.get("/bakeries/{bakery_id}/sku-day")
def get_bakery_sku_day(
    bakery_id: int,
    date: str = Query(...),
    limit: int = Query(default=100, ge=1, le=1000),
    run_id: str | None = None,
) -> dict:
    active_run = _require_run(run_id)
    return {
        "run_id": active_run["run_id"],
        "items": bakery_service.get_top_sku(active_run["run_id"], date, bakery_id, limit=limit),
    }


@router.get("/bakeries/{bakery_id}/sku-hour")
def get_bakery_sku_hour(
    bakery_id: int,
    product_id: int = Query(...),
    date: str = Query(...),
    run_id: str | None = None,
) -> dict:
    active_run = _require_run(run_id)
    items = bakery_service.get_sku_hour(active_run["run_id"], date, bakery_id, product_id)
    if not items:
        raise HTTPException(status_code=404, detail="SKU hourly forecast not found")
    return {
        "run_id": active_run["run_id"],
        "items": items,
    }

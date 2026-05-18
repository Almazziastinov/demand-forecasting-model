from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.services import bakery as bakery_service
from app.services import runs as run_service


router = APIRouter(prefix="/api/v1", tags=["bakeries"])


def _require_active_run() -> dict:
    active_run = run_service.get_active_run()
    if not active_run:
        raise HTTPException(status_code=404, detail="No active forecast run found")
    return active_run


@router.get("/bakeries")
def get_bakeries(date: str = Query(...)) -> dict:
    active_run = _require_active_run()
    return {
        "run_id": active_run["run_id"],
        "items": bakery_service.get_bakery_list(active_run["run_id"], date),
    }


@router.get("/bakeries/{bakery_id}/summary")
def get_bakery_summary(bakery_id: int, date: str = Query(...)) -> dict:
    active_run = _require_active_run()
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
) -> dict:
    active_run = _require_active_run()
    return {
        "run_id": active_run["run_id"],
        "items": bakery_service.get_top_sku(active_run["run_id"], date, bakery_id, limit=limit),
    }


@router.get("/bakeries/{bakery_id}/sku-hour")
def get_bakery_sku_hour(
    bakery_id: int,
    product_id: int = Query(...),
    date: str = Query(...),
) -> dict:
    active_run = _require_active_run()
    items = bakery_service.get_sku_hour(active_run["run_id"], date, bakery_id, product_id)
    if not items:
        raise HTTPException(status_code=404, detail="SKU hourly forecast not found")
    return {
        "run_id": active_run["run_id"],
        "items": items,
    }

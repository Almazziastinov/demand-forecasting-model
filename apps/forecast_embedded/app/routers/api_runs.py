from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas import ActiveRunOut
from app.services import runs as run_service


router = APIRouter(prefix="/api/v1", tags=["runs"])


@router.get("/runs/active", response_model=ActiveRunOut)
def get_active_run() -> dict:
    active_run = run_service.get_active_run()
    if not active_run:
        raise HTTPException(status_code=404, detail="No active forecast run found")
    return active_run


@router.get("/runs")
def list_runs(limit: int = 50) -> dict:
    return {"items": run_service.list_runs(limit=limit)}


@router.get("/dates")
def get_dates(run_id: str | None = None) -> dict:
    active_run = run_service.resolve_run(run_id)
    if not active_run:
        raise HTTPException(status_code=404, detail="Forecast run not found")
    return {
        "run_id": active_run["run_id"],
        "dates": run_service.get_run_dates(active_run["run_id"]),
    }

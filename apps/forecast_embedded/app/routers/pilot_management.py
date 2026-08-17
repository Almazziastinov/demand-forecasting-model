"""Pilot management analytics screen (PM-05).

Admin-only routes at /pilot.  Reads from the CSV report directory set via
the PILOT_REPORT_DIR environment variable (defaults to
<repo_root>/reports/pilot_management_summary).
"""

from __future__ import annotations

# ruff: noqa: E501
import os
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Add repo root to sys.path so src.pilot_management_service is importable
# when the app is launched from apps/forecast_embedded/.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.auth import get_auth_context  # noqa: E402
from src.pilot_management_service import PilotManagementService, _pct  # noqa: E402

router = APIRouter(prefix="/pilot", tags=["pilot"])
templates = Jinja2Templates(directory="app/templates")

_DEFAULT_REPORT_DIR = _REPO_ROOT / "reports" / "pilot_management_summary"


def _get_service() -> PilotManagementService:
    report_dir = Path(
        os.environ.get("PILOT_REPORT_DIR", str(_DEFAULT_REPORT_DIR))
    )
    return PilotManagementService(report_dir)


def _require_pilot_user(request: Request) -> None:
    auth = get_auth_context(request)
    if not auth.is_pilot_user:
        raise HTTPException(status_code=403, detail="Управленческая аналитика доступна только директорам, аналитикам и администраторам")


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
def pilot_summary(request: Request) -> HTMLResponse:
    """Pilot summary: company-level KPIs, bakery table, queues overview."""
    _require_pilot_user(request)
    auth = get_auth_context(request)
    svc = _get_service()
    summary = svc.get_pilot_summary()
    if not summary:
        raise HTTPException(status_code=404, detail="Отчёт не найден. Запустите build_pilot_management_summary.py")
    return templates.TemplateResponse(
        request,
        "pilot_management.html",
        {
            "auth": auth,
            "is_admin": auth.is_admin,
            "summary": summary,
            "bakeries": svc.get_bakery_list(),
            "week_trend": svc.get_week_trend(),
            "model_queue": svc.get_model_queue(tiers=("M1",)),
            "execution_queue": svc.get_execution_queue(
                triage_filter=("likely_execution", "needs_joint_review")
            ),
            "dq": svc.get_dq_summary(),
            "dp_queue": svc.get_data_process_queue(tiers=("D1",)),
            "pct": _pct,
        },
    )


@router.get("/bakery/{bakery_id}", response_class=HTMLResponse)
def pilot_bakery(request: Request, bakery_id: int) -> HTMLResponse:
    """Bakery drill-down: bakery KPIs + per-SKU table."""
    _require_pilot_user(request)
    auth = get_auth_context(request)
    svc = _get_service()
    bakery = svc.get_bakery_detail(bakery_id)
    if not bakery:
        raise HTTPException(status_code=404, detail=f"Пекарня {bakery_id} не найдена")
    summary = svc.get_pilot_summary()
    return templates.TemplateResponse(
        request,
        "pilot_bakery.html",
        {
            "auth": auth,
            "is_admin": auth.is_admin,
            "summary": summary,
            "bakery": bakery,
            "skus": svc.get_sku_list(bakery_id),
            "pct": _pct,
        },
    )


@router.get("/bakery/{bakery_id}/sku/{product_id}", response_class=HTMLResponse)
def pilot_sku(request: Request, bakery_id: int, product_id: int) -> HTMLResponse:
    """SKU timeline: day-by-day forecast vs plan vs actual."""
    _require_pilot_user(request)
    auth = get_auth_context(request)
    svc = _get_service()
    bakery = svc.get_bakery_detail(bakery_id)
    days = svc.get_day_list(bakery_id, product_id)
    if not days:
        raise HTTPException(status_code=404, detail=f"Данные для SKU {product_id} в пекарне {bakery_id} не найдены")
    sku_list = svc.get_sku_list(bakery_id)
    sku_meta = next((s for s in sku_list if s["product_id"] == product_id), None)
    return templates.TemplateResponse(
        request,
        "pilot_sku.html",
        {
            "auth": auth,
            "is_admin": auth.is_admin,
            "bakery": bakery,
            "sku": sku_meta,
            "days": days,
            "bakery_id": bakery_id,
            "product_id": product_id,
            "pct": _pct,
        },
    )

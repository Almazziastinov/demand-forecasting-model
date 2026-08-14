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

import io
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
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


def _require_admin(request: Request) -> None:
    auth = get_auth_context(request)
    if not auth.is_admin:
        raise HTTPException(status_code=403, detail="Управленческая аналитика доступна только администраторам")


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
def pilot_summary(
    request: Request,
    category: str | None = Query(default=None),
) -> HTMLResponse:
    """Pilot summary: company-level KPIs, bakery table, SKU table."""
    _require_admin(request)
    auth = get_auth_context(request)
    svc = _get_service()
    summary = svc.get_pilot_summary(category=category)
    if not summary:
        raise HTTPException(status_code=404, detail="Отчёт не найден. Запустите build_pilot_management_summary.py")
    return templates.TemplateResponse(
        request,
        "pilot_management.html",
        {
            "auth": auth,
            "is_admin": auth.is_admin,
            "summary": summary,
            "bakeries": svc.get_bakery_list(category=category),
            "week_trend": svc.get_week_trend(category=category),
            "sku_summary": svc.get_sku_summary(category=category),
            "categories": svc.get_available_categories(),
            "selected_category": category,
            "dq": svc.get_dq_summary(),
            "pct": _pct,
        },
    )


@router.get("/bakery/{bakery_id}", response_class=HTMLResponse)
def pilot_bakery(
    request: Request,
    bakery_id: int,
    category: str | None = Query(default=None),
) -> HTMLResponse:
    """Bakery drill-down: KPIs, week trend, category filter, SKU table."""
    _require_admin(request)
    auth = get_auth_context(request)
    svc = _get_service()
    bakery = svc.get_bakery_kpi(bakery_id, category=category)
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
            "week_trend": svc.get_bakery_week_trend(bakery_id, category=category),
            "skus": svc.get_sku_list(bakery_id, category=category),
            "selected_category": category,
            "pct": _pct,
        },
    )


@router.get("/bakery/{bakery_id}/week/{week_start}", response_class=HTMLResponse)
def pilot_bakery_week(
    request: Request,
    bakery_id: int,
    week_start: str,
    category: str | None = Query(default=None),
) -> HTMLResponse:
    """Week drill-down: day-by-day table + SKU summary for the week."""
    _require_admin(request)
    auth = get_auth_context(request)
    svc = _get_service()
    bakery = svc.get_bakery_kpi(bakery_id, category=category)
    if not bakery:
        raise HTTPException(status_code=404, detail=f"Пекарня {bakery_id} не найдена")
    days = svc.get_bakery_week_days(bakery_id, week_start, category=category)
    if not days:
        raise HTTPException(status_code=404, detail=f"Данных за неделю {week_start} нет")
    skus = svc.get_bakery_week_sku_summary(bakery_id, week_start, category=category)
    return templates.TemplateResponse(
        request,
        "pilot_bakery_week.html",
        {
            "auth": auth,
            "is_admin": auth.is_admin,
            "bakery": bakery,
            "week_start": week_start,
            "days": days,
            "skus": skus,
            "selected_category": category,
            "pct": _pct,
        },
    )


@router.get("/bakery/{bakery_id}/week/{week_start}/day/{date}/export")
def pilot_day_export(
    request: Request,
    bakery_id: int,
    week_start: str,
    date: str,
    category: str | None = Query(default=None),
) -> StreamingResponse:
    """Export one day's SKU data as Excel."""
    import pandas as pd

    _require_admin(request)
    svc = _get_service()
    bakery = svc.get_bakery_kpi(bakery_id)
    if not bakery:
        raise HTTPException(status_code=404, detail=f"Пекарня {bakery_id} не найдена")
    rows = svc.get_bakery_day_detail(bakery_id, date, category=category)
    if not rows:
        raise HTTPException(status_code=404, detail=f"Нет данных за {date}")

    df = pd.DataFrame(rows)
    _COL_NAMES = {
        "bakery_name": "Пекарня",
        "product_name": "Номенклатура",
        "fact_category_name": "Категория",
        "forecast_qty": "Прогноз",
        "issued_total_for_sale": "План (итого на продажу)",
        "produced_qty": "Произведено",
        "sold_qty": "Продано",
        "demand_qty": "Спрос",
        "price": "Цена",
        "revenue": "Выручка",
        "lost_demand_recognized_qty": "Упущ. признанное",
        "eligible_forecast_summary": "Пригодно для прогноза",
        "eligible_lost_demand": "Пригодно для упущ.",
        "execution_status": "Статус выполнения",
    }
    df = df.rename(columns={k: v for k, v in _COL_NAMES.items() if k in df.columns})

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=date)
    buf.seek(0)

    filename_utf8 = f"pilot_{bakery['bakery_name'].replace(' ', '_')[:30]}_{date}.xlsx"
    filename_ascii = f"pilot_{bakery_id}_{date}.xlsx"
    disposition = f'attachment; filename="{filename_ascii}"; filename*=UTF-8\'\'{quote(filename_utf8)}'
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": disposition},
    )


@router.get("/bakery/{bakery_id}/sku/{product_id}", response_class=HTMLResponse)
def pilot_sku(request: Request, bakery_id: int, product_id: int) -> HTMLResponse:
    """SKU timeline: day-by-day forecast vs plan vs actual."""
    _require_admin(request)
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

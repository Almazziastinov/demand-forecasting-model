"""Pilot scope management: add/exclude bakeries from the pilot.

Routes:
  GET  /pilot/config            — management page
  POST /pilot/config/bakery/{id}/add     — add bakery to pilot
  POST /pilot/config/bakery/{id}/exclude — exclude bakery from pilot
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.auth import get_auth_context  # noqa: E402
from app.db import get_client  # noqa: E402
from src.pilot_config_service import PilotConfigService  # noqa: E402

router = APIRouter(prefix="/pilot/config", tags=["pilot-config"])
templates = Jinja2Templates(directory="app/templates")

DEFAULT_SCOPE = "expanded_pilot_38"


def _get_service() -> PilotConfigService:
    return PilotConfigService(get_client())


def _require_pilot_user(request: Request) -> None:
    auth = get_auth_context(request)
    if not auth.is_pilot_user:
        raise HTTPException(status_code=403, detail="Управление пилотом доступно только директорам, аналитикам и администраторам")


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
def pilot_config(request: Request, scope: str = DEFAULT_SCOPE) -> HTMLResponse:
    _require_pilot_user(request)
    auth = get_auth_context(request)
    svc = _get_service()
    bakeries = svc.get_all_bakeries(scope)
    summary = svc.get_summary(scope)
    history = svc.get_history(scope, limit=50)
    return templates.TemplateResponse(
        request,
        "pilot_config.html",
        {
            "auth": auth,
            "is_admin": auth.is_admin,
            "scope": scope,
            "bakeries": bakeries,
            "summary": summary,
            "history": history,
        },
    )


@router.post("/bakery/{bakery_id}/add")
def add_bakery(
    request: Request,
    bakery_id: int,
    reason: str = Form(default=""),
    scope: str = Form(default=DEFAULT_SCOPE),
) -> RedirectResponse:
    _require_pilot_user(request)
    auth = get_auth_context(request)
    svc = _get_service()
    svc.add_bakery(
        scope_name=scope,
        bakery_id=bakery_id,
        changed_by=auth.display_name or auth.user_id or "unknown",
        reason=reason.strip(),
    )
    return RedirectResponse(url=f"/pilot/config?scope={scope}", status_code=303)


@router.post("/bakery/{bakery_id}/exclude")
def exclude_bakery(
    request: Request,
    bakery_id: int,
    reason: str = Form(default=""),
    scope: str = Form(default=DEFAULT_SCOPE),
) -> RedirectResponse:
    _require_pilot_user(request)
    auth = get_auth_context(request)
    svc = _get_service()
    svc.exclude_bakery(
        scope_name=scope,
        bakery_id=bakery_id,
        changed_by=auth.display_name or auth.user_id or "unknown",
        reason=reason.strip(),
    )
    return RedirectResponse(url=f"/pilot/config?scope={scope}", status_code=303)

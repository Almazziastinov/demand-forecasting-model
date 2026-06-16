from __future__ import annotations

from fastapi import APIRouter

from app.settings import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "ok": True,
        "app_env": settings.app_env,
        "table_suffix": settings.table_suffix,
    }

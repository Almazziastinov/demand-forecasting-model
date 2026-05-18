from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routers import api_bakeries
from app.routers import api_exports
from app.routers import api_runs
from app.routers import health
from app.routers import ui
from app.settings import get_settings


BASE_DIR = Path(__file__).resolve().parent
settings = get_settings()

app = FastAPI(title=settings.app_title)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

app.include_router(health.router)
app.include_router(api_runs.router)
app.include_router(api_bakeries.router)
app.include_router(api_exports.router)
app.include_router(ui.router)

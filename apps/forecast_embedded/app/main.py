from __future__ import annotations

# ruff: noqa: E402
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# apps/baking_plan is a sibling package to this app (apps/forecast_embedded/app),
# not a subpackage of it — add apps/ to sys.path so `import baking_plan` resolves.
_APPS_DIR = Path(__file__).resolve().parents[2]
if str(_APPS_DIR) not in sys.path:
    sys.path.insert(0, str(_APPS_DIR))

from baking_plan.router import router as baking_plan_router

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
app.include_router(baking_plan_router)
app.include_router(ui.router)

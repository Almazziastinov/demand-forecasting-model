from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel


class ActiveRunOut(BaseModel):
    run_id: str
    model_version: str
    profile_version: str
    source_kind: str
    horizon_start: date
    horizon_end: date
    generated_at: datetime
    status: str
    is_bias_adjusted: bool


class BakeryListItemOut(BaseModel):
    bakery_id: int
    bakery_name: str
    city: str | None = None
    forecast_final: float


class HourlyForecastItemOut(BaseModel):
    hour: int
    forecast_qty: float


class SkuDayItemOut(BaseModel):
    product_id: int
    product_name: str | None = None
    category_name: str | None = None
    forecast_qty: float


class SkuHourItemOut(BaseModel):
    hour: int
    product_id: int
    product_name: str | None = None
    forecast_qty: float

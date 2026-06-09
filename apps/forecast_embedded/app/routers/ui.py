from __future__ import annotations

# ruff: noqa: E501

import logging
import re
from datetime import date as date_type
from datetime import datetime, timedelta
from io import BytesIO
from urllib.parse import quote
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates

from app.auth import AuthContext, get_auth_context
from app.services import baking_plan as baking_plan_service
from app.services import bakery as bakery_service
from app.services import runs as run_service


router = APIRouter(tags=["ui"])
templates = Jinja2Templates(directory="app/templates")
logger = logging.getLogger(__name__)

WEEKDAYS_RU = {
    0: "Понедельник",
    1: "Вторник",
    2: "Среда",
    3: "Четверг",
    4: "Пятница",
    5: "Суббота",
    6: "Воскресенье",
}

EVENT_LABELS_RU = {
    "regular": "обычный день",
    "pre_event_1_3": "1-3 дня до события",
    "pre_event_4_7": "4-7 дней до события",
    "event": "событие",
    "post_event_1_3": "1-3 дня после события",
    "post_event_4_7": "4-7 дней после события",
}

FILENAME_SAFE_RE = re.compile(r'[\\/:*?"<>|\r\n]+')


def _parse_date(value: str | None) -> date_type | None:
    if not value:
        return None
    return date_type.fromisoformat(value[:10])


def _format_date(value: date_type) -> str:
    return value.strftime("%d.%m.%Y")


def _format_short_date(value: date_type) -> str:
    return value.strftime("%d.%m")


def _safe_filename_part(value: object) -> str:
    cleaned = FILENAME_SAFE_RE.sub(" ", str(value or "")).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or "пекарня"


def _attachment_header(filename: str, fallback_filename: str) -> str:
    quoted = quote(filename)
    return f'attachment; filename="{fallback_filename}"; filename*=UTF-8\'\'{quoted}'


def _event_label(value: str | None) -> str:
    if not value:
        return "обычный день"
    return EVENT_LABELS_RU.get(value, value.replace("_", " "))


def _resolve_run(auth: AuthContext, run_id: str | None) -> dict | None:
    return run_service.resolve_run(run_id if auth.is_admin else None)


def _default_week_start(dates: list[str]) -> str | None:
    yesterday = datetime.now(ZoneInfo("Europe/Moscow")).date() - timedelta(days=1)
    if yesterday.isoformat() in dates:
        return yesterday.isoformat()
    if not dates:
        return None
    parsed_dates = [_parse_date(value) for value in dates]
    valid_dates = [value for value in parsed_dates if value is not None]
    future_dates = [value for value in valid_dates if value >= yesterday]
    return (future_dates[0] if future_dates else valid_dates[0]).isoformat()


def _week_dates(start_date: str) -> list[date_type]:
    start = _parse_date(start_date)
    if start is None:
        return []
    return [start + timedelta(days=offset) for offset in range(7)]


def _with_day_labels(row: dict, current_date: date_type) -> dict:
    prepared = {
        "actual_qty": None,
        "actual_revenue": None,
        "forecast_base": None,
        "forecast_final": None,
        "temp_mean": None,
        "precipitation": None,
        "rain": None,
        "snowfall": None,
        "windspeed_max": None,
        "is_bad_weather": False,
        "holiday_name": None,
        "is_holiday": False,
        "is_pre_holiday": False,
        "is_post_holiday": False,
        "event_window_type": None,
        **row,
    }
    prepared["forecast_date"] = current_date.isoformat()
    prepared["date_label"] = _format_date(current_date)
    prepared["short_date_label"] = _format_short_date(current_date)
    prepared["weekday_label"] = WEEKDAYS_RU[current_date.weekday()]
    prepared["is_weekend"] = current_date.weekday() >= 5
    prepared["event_label"] = _event_label(prepared.get("event_window_type"))
    return prepared


def _prepare_week_rows(raw_rows: list[dict], week: list[date_type]) -> list[dict]:
    by_date = {str(row["forecast_date"])[:10]: row for row in raw_rows}
    rows = [_with_day_labels(by_date.get(day.isoformat(), {}), day) for day in week]
    max_value = max(
        [
            float(row.get("forecast_final") or 0)
            for row in rows
        ]
        + [
            float(row.get("actual_qty") or 0)
            for row in rows
        ]
        + [1.0]
    )
    for row in rows:
        forecast = float(row.get("forecast_final") or 0)
        actual = float(row.get("actual_qty") or 0)
        row["forecast_height"] = max(4, round(forecast / max_value * 100)) if forecast else 0
        row["actual_height"] = max(4, round(actual / max_value * 100)) if actual else 0
        row["has_forecast"] = row.get("forecast_final") is not None
        row["has_actual"] = row.get("actual_qty") is not None
    return rows


def _prepare_hour_chart(rows: list[dict]) -> dict:
    by_hour = {int(row["hour"]): row for row in rows if row.get("hour") is not None}
    points = []
    max_value = 1.0
    for hour in range(24):
        row = by_hour.get(hour, {})
        forecast = float(row.get("forecast_qty") or 0)
        actual = float(row.get("actual_qty") or 0)
        max_value = max(max_value, forecast, actual)
        points.append({"hour": hour, "forecast": forecast, "actual": actual})

    width = 920
    height = 260
    left = 36
    right = 18
    top = 18
    bottom = 34
    plot_width = width - left - right
    plot_height = height - top - bottom

    for point in points:
        point["x"] = left + (plot_width * point["hour"] / 23)
        point["forecast_y"] = top + plot_height - (plot_height * point["forecast"] / max_value)
        point["actual_y"] = top + plot_height - (plot_height * point["actual"] / max_value)
        point["forecast_label_y"] = max(12, point["forecast_y"] - 8)
        point["actual_label_y"] = min(height - bottom - 8, point["actual_y"] + 16)

    return {
        "points": points,
        "forecast_polyline": " ".join(f'{point["x"]:.1f},{point["forecast_y"]:.1f}' for point in points),
        "actual_polyline": " ".join(f'{point["x"]:.1f},{point["actual_y"]:.1f}' for point in points),
        "max_value": max_value,
        "width": width,
        "height": height,
        "left": left,
        "bottom_y": height - bottom,
    }


def _page_context(
    request: Request,
    auth: AuthContext,
    active_run: dict,
    week_start: str,
    selected_bakery_id: int | None = None,
) -> dict:
    bakeries = bakery_service.get_bakery_list(active_run["run_id"], week_start, auth) if week_start else []
    if selected_bakery_id is None and bakeries:
        selected_bakery_id = int(bakeries[0]["bakery_id"])
    return {
        "request": request,
        "active_run": active_run,
        "runs": run_service.list_runs() if auth.is_admin else [],
        "auth": auth,
        "is_admin": auth.is_admin,
        "week_start": week_start,
        "bakeries": bakeries,
        "selected_bakery_id": selected_bakery_id,
        "run_query": f"&run_id={active_run['run_id']}" if auth.is_admin else "",
    }


@router.head("/")
def index_head() -> Response:
    return Response(status_code=200)


@router.get("/", response_class=HTMLResponse)
@router.post("/", response_class=HTMLResponse)
def index(
    request: Request,
    week_start: str | None = Query(default=None),
    date: str | None = Query(default=None),
    bakery_id: int | None = Query(default=None),
    run_id: str | None = Query(default=None),
) -> HTMLResponse:
    auth = get_auth_context(request)
    active_run = _resolve_run(auth, run_id)
    if not active_run:
        raise HTTPException(status_code=404, detail="Forecast run not found")

    dates = run_service.get_run_dates(active_run["run_id"])
    selected_week_start = week_start or date or _default_week_start(dates)
    if not selected_week_start:
        raise HTTPException(status_code=404, detail="Forecast dates not found")

    base_context = _page_context(request, auth, active_run, selected_week_start, bakery_id)
    selected_bakery_id = base_context["selected_bakery_id"]
    week = _week_dates(selected_week_start)
    raw_week_rows = (
        bakery_service.get_bakery_week(
            active_run["run_id"],
            week[0].isoformat(),
            week[-1].isoformat(),
            selected_bakery_id,
            auth,
        )
        if selected_bakery_id and week
        else []
    )
    week_rows = _prepare_week_rows(raw_week_rows, week)

    logger.warning(
        "embedded index request_id=%s user_id=%s email=%s portal_id=%s role=%s is_admin=%s week_start=%s bakeries=%s selected_bakery_id=%s",
        request.headers.get("x-vibe-request-id"),
        auth.user_id,
        auth.email,
        auth.portal_id,
        auth.role,
        auth.is_admin,
        selected_week_start,
        len(base_context["bakeries"]),
        selected_bakery_id,
    )
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            **base_context,
            "week_rows": week_rows,
            "dates": dates,
            "selected_bakery": next(
                (bakery for bakery in base_context["bakeries"] if int(bakery["bakery_id"]) == selected_bakery_id),
                None,
            ),
        },
    )


@router.get("/bakery/{bakery_id}", response_class=HTMLResponse)
def bakery_detail(
    request: Request,
    bakery_id: int,
    date: str = Query(...),
    week_start: str | None = Query(default=None),
    run_id: str | None = Query(default=None),
    category: str | None = Query(default=None),
) -> HTMLResponse:
    auth = get_auth_context(request)
    active_run = _resolve_run(auth, run_id)
    if not active_run:
        raise HTTPException(status_code=404, detail="Forecast run not found")

    dates = run_service.get_run_dates(active_run["run_id"])
    selected_week_start = week_start or _default_week_start(dates) or date
    bakery_day = bakery_service.get_bakery_day(active_run["run_id"], date, bakery_id, auth)
    if not bakery_day:
        raise HTTPException(status_code=404, detail="Bakery forecast not found")

    categories = bakery_service.get_categories(active_run["run_id"], date, bakery_id, auth)
    selected_category = category if category in categories else None
    hourly_total = bakery_service.get_hourly_total(active_run["run_id"], date, bakery_id, auth)
    top_sku = bakery_service.get_top_sku(
        active_run["run_id"],
        date,
        bakery_id,
        auth,
        limit=100,
        category=selected_category,
    )
    selected_date = _parse_date(date)
    return templates.TemplateResponse(
        request,
        "bakery.html",
        {
            **_page_context(request, auth, active_run, selected_week_start, bakery_id),
            "selected_date": date,
            "selected_date_label": _format_date(selected_date) if selected_date else date,
            "selected_weekday_label": WEEKDAYS_RU[selected_date.weekday()] if selected_date else "",
            "bakery": bakery_day,
            "context": bakery_service.get_day_context(active_run["run_id"], date, bakery_day["city"]),
            "hourly_total": hourly_total,
            "hour_chart": _prepare_hour_chart(hourly_total),
            "top_sku": top_sku,
            "categories": categories,
            "selected_category": selected_category,
        },
    )


@router.get("/bakery/{bakery_id}/baking-plan.xlsx")
def download_baking_plan(
    request: Request,
    bakery_id: int,
    date: str = Query(...),
    run_id: str | None = Query(default=None),
    bucket: str | None = Query(default=None),
) -> StreamingResponse:
    auth = get_auth_context(request)
    active_run = _resolve_run(auth, run_id)
    if not active_run:
        raise HTTPException(status_code=404, detail="Forecast run not found")

    bakery_day = bakery_service.get_bakery_day(active_run["run_id"], date, bakery_id, auth)
    if not bakery_day:
        raise HTTPException(status_code=404, detail="Bakery forecast not found")

    sku_hour = bakery_service.get_sku_hour_forecast(active_run["run_id"], date, bakery_id, auth)

    # Night-defrost batches are prep for the next morning; size them from the next
    # day's forecast. A missing next day (horizon end) must not break the export.
    next_day_sku_hour: list[dict] = []
    selected_date = _parse_date(date)
    if selected_date is not None:
        next_date = (selected_date + timedelta(days=1)).isoformat()
        try:
            next_day_sku_hour = bakery_service.get_sku_hour_forecast(
                active_run["run_id"], next_date, bakery_id, auth
            )
        except Exception:
            logger.warning("baking_plan: next-day forecast fetch failed", exc_info=True)
            next_day_sku_hour = []

    revenue_info = bakery_service.get_month_revenue_bucket(date, bakery_id)
    selected_bucket = bucket or (revenue_info or {}).get("revenue_bucket")
    content = baking_plan_service.build_baking_plan_workbook(
        bakery=bakery_day,
        forecast_date=date,
        sku_hour_rows=sku_hour,
        next_day_sku_hour_rows=next_day_sku_hour,
        bucket=selected_bucket,
    )
    bakery_name = _safe_filename_part(bakery_day.get("bakery_name") or bakery_id)
    filename = f"План выпекания - {bakery_name} - {date}.xlsx"
    fallback_filename = f"baking_plan_{bakery_id}_{date}.xlsx"
    return StreamingResponse(
        BytesIO(content),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": _attachment_header(filename, fallback_filename)},
    )


@router.get("/bakery/{bakery_id}/sku/{product_id}", response_class=HTMLResponse)
def sku_detail(
    request: Request,
    bakery_id: int,
    product_id: int,
    date: str = Query(...),
    week_start: str | None = Query(default=None),
    run_id: str | None = Query(default=None),
    category: str | None = Query(default=None),
) -> HTMLResponse:
    auth = get_auth_context(request)
    active_run = _resolve_run(auth, run_id)
    if not active_run:
        raise HTTPException(status_code=404, detail="Forecast run not found")

    dates = run_service.get_run_dates(active_run["run_id"])
    selected_week_start = week_start or _default_week_start(dates) or date
    bakery_day = bakery_service.get_bakery_day(active_run["run_id"], date, bakery_id, auth)
    sku_day = bakery_service.get_sku_day(active_run["run_id"], date, bakery_id, product_id, auth)
    if not bakery_day or not sku_day:
        raise HTTPException(status_code=404, detail="SKU forecast not found")

    sku_hour = bakery_service.get_sku_hour(active_run["run_id"], date, bakery_id, product_id, auth)
    selected_date = _parse_date(date)
    return templates.TemplateResponse(
        request,
        "sku.html",
        {
            **_page_context(request, auth, active_run, selected_week_start, bakery_id),
            "selected_date": date,
            "selected_date_label": _format_date(selected_date) if selected_date else date,
            "selected_weekday_label": WEEKDAYS_RU[selected_date.weekday()] if selected_date else "",
            "bakery": bakery_day,
            "sku": sku_day,
            "sku_hour": sku_hour,
            "hour_chart": _prepare_hour_chart(sku_hour),
            "selected_category": category,
        },
    )

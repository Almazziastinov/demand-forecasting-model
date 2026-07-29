"""Generate and publish the daily pilot forecast summary to Bitrix24 chat.

For the given date (default: today), queries ClickHouse for per-SKU
daily forecasts and previous-day closing stock for all pilot bakeries.
The stock is subtracted from forecast demand before kratnost rounding.
If VIBECODE_API_KEY is set (via .env or environment), uploads the generated
Excel file to the Bitrix24 pilot chat automatically.

Usage (local / on VM):
    python scripts/publish_pilot_forecast.py --env-file .env
    python scripts/publish_pilot_forecast.py --env-file .env --date 2026-07-24
    python scripts/publish_pilot_forecast.py --env-file .env --dry-run
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import date as date_type
from datetime import timedelta
from io import BytesIO
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "apps"))
sys.path.insert(0, str(ROOT / "apps" / "forecast_embedded"))

from src.experiments_v2.sku_systematic_correction import (  # noqa: E402
    CorrectionConfig,
    apply_category_neutral_corrections,
    build_correction_registry,
)
from src.experiments_v2.sku_cold_start import (  # noqa: E402
    ColdStartConfig,
    apply_category_neutral_cold_start,
    build_cold_start_registry,
)

PILOT_BAKERY_IDS = [20, 21, 22, 28, 80, 89, 107, 221, 222, 257]

WEEKDAY_RU = ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"]

BAKEABLE_CATEGORIES = {
    "Пироги сытные",
    "Пироги сладкие",
    "Выпечка сытная",
    "Выпечка сладкая",
    "Фастфуд",
}


def _load_env(env_file: str) -> None:
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _round_up_kratnost(value: float, kratnost: int) -> int:
    """Always round UP to the nearest kratnost multiple (same as MILP)."""
    if value <= 0 or kratnost <= 0:
        return 0
    return int(math.ceil(value / kratnost - 1e-9) * kratnost)


def _build_report(
    forecast_date: str,
    *,
    sku_correction_registry: str | Path | None = None,
    enable_new_sku_cold_start: bool = True,
) -> list[dict]:
    """Build daily pilot rows with previous-day stock and production plan."""
    from app.db import get_client
    from app.table_names import table_name

    client = get_client()

    run_df = client.query_df(
        f"select run_id from {table_name('forecast_runs_embedded')} where status = 'active' limit 1"
    )
    if run_df.empty:
        raise RuntimeError("No active forecast run")
    run_id = str(run_df.iloc[0]["run_id"])

    # Bakery names for pilot bakeries
    # dim_bakeries uses zero-padded string bakery_id ("000000016"), same as baking_sku_meta
    pilot_bids_str = [f"{b:09d}" for b in PILOT_BAKERY_IDS]
    bakery_df = client.query_df(
        "select bakery_id as bid, any(bakery_name) as name, any(city) as city "
        "from dim_bakeries "
        "where bakery_id in %(bids)s "
        "group by bakery_id",
        parameters={"bids": pilot_bids_str},
    )
    bakery_info: dict[int, dict] = {}
    for row in bakery_df.to_dict("records"):
        try:
            bid = int(row["bid"])
            bakery_info[bid] = {"name": row["name"], "city": row["city"]}
        except (TypeError, ValueError):
            pass

    # Daily SKU forecasts for all pilot bakeries
    forecast_df = client.query_df(
        f"""
        select
            bakery_id,
            product_id,
            any(product_name) as product_name,
            any(category_name) as category_name,
            sum(forecast_qty) as forecast_qty
        from {table_name('sku_forecast_day_embedded')}
        where run_id = %(run_id)s
          and forecast_date = %(forecast_date)s
          and bakery_id in %(bids)s
        group by bakery_id, product_id
        """,
        parameters={
            "run_id": run_id,
            "forecast_date": forecast_date,
            "bids": PILOT_BAKERY_IDS,
        },
    )

    if forecast_df.empty:
        print(f"  WARNING: no forecast rows for {forecast_date}, run {run_id}")
        return []

    previous_date = str(date_type.fromisoformat(forecast_date) - timedelta(days=1))
    stock_df = client.query_df(
        """
        select
            toInt64OrZero(toString(m.bakery_id)) as bakery_id,
            toInt64OrZero(toString(m.product_id)) as product_id,
            sum(m.stock_balance) as stock_balance
        from Svezhar.mart_zero_sales_60d as m
        where m.dt = toDate(%(previous_date)s)
          and toInt64OrZero(toString(m.bakery_id)) in %(bids)s
        group by bakery_id, product_id
        """,
        parameters={
            "previous_date": previous_date,
            "bids": PILOT_BAKERY_IDS,
        },
    )
    yesterday_stock: dict[tuple[int, int], float] = {}
    for row in stock_df.to_dict("records"):
        try:
            key = (int(row["bakery_id"]), int(row["product_id"]))
            yesterday_stock[key] = max(float(row.get("stock_balance") or 0), 0.0)
        except (TypeError, ValueError):
            continue

    # SKU meta (kratnost) — base + bakery overrides
    # baking_sku_meta.product_id is zero-padded string ("000001234")
    # sku_forecast_day_embedded.product_id is int64
    all_pids_int = [int(r) for r in forecast_df["product_id"].dropna().unique()]
    all_pids_str = [f"{p:09d}" for p in all_pids_int]
    meta_df = client.query_df(
        f"""
        select product_id, bakery_id, dough_group, kratnost, scope
        from {table_name('baking_sku_meta')} final
        where is_active = 1 and product_id in %(pids)s
        """,
        parameters={"pids": all_pids_str},
    )

    # Build kratnost lookup keyed by int product_id
    base_kratnost: dict[int, int] = {}
    bakery_kratnost: dict[tuple[int, int], int] = {}
    frozen_pids: set[int] = set()
    for row in meta_df.to_dict("records"):
        try:
            pid_int = int(row["product_id"])
        except (TypeError, ValueError):
            continue
        dg = str(row.get("dough_group") or "").lower()
        if "замороженные полуфабрикаты" in dg:
            frozen_pids.add(pid_int)
            continue
        kr = int(row.get("kratnost") or 1) or 1
        if row["scope"] == "bakery" and row["bakery_id"] is not None:
            try:
                bakery_kratnost[(pid_int, int(row["bakery_id"]))] = kr
            except (TypeError, ValueError):
                pass
        else:
            base_kratnost[pid_int] = kr

    corrected_forecast: dict[tuple[int, int], float] = {}
    if sku_correction_registry or enable_new_sku_cold_start:
        eligible = forecast_df.copy()
        eligible["bakery_id"] = pd.to_numeric(
            eligible["bakery_id"],
            errors="coerce",
        )
        eligible["product_id"] = pd.to_numeric(
            eligible["product_id"],
            errors="coerce",
        )
        eligible = eligible[
            eligible["bakery_id"].isin(PILOT_BAKERY_IDS)
            & eligible["category_name"].isin(BAKEABLE_CATEGORIES)
            & ~eligible["product_id"].isin(frozen_pids)
        ].copy()
        has_meta = eligible.apply(
            lambda row: (
                int(row["product_id"]) in base_kratnost
                or (int(row["product_id"]), int(row["bakery_id"]))
                in bakery_kratnost
            ),
            axis=1,
        )
        eligible = eligible[has_meta].copy()
        eligible["date"] = pd.Timestamp(forecast_date)

    if enable_new_sku_cold_start:
        history_from = str(
            date_type.fromisoformat(forecast_date) - timedelta(days=60)
        )
        sales_history = client.query_df(
            """
            select
                dt as date,
                toInt64(bakery_id) as bakery_id,
                toInt64(product_id) as product_id,
                sum(qty_sold) as sold_qty
            from Svezhar.mart_zero_sales_60d
            where dt >= toDate(%(history_from)s)
              and dt < toDate(%(forecast_date)s)
              and toInt64(bakery_id) in %(bids)s
              and toInt64(product_id) in %(product_ids)s
            group by date, bakery_id, product_id
            """,
            parameters={
                "history_from": history_from,
                "forecast_date": forecast_date,
                "bids": PILOT_BAKERY_IDS,
                "product_ids": list(ColdStartConfig().product_ids),
            },
        )
        forecast_history = client.query_df(
            """
            select
                forecast_date as date,
                toInt64(bakery_id) as bakery_id,
                toInt64(product_id) as product_id,
                argMax(forecast_qty, generated_at) as forecast_qty
            from Svezhar.sku_forecast_day_snapshots
            where forecast_date >= toDate(%(history_from)s)
              and forecast_date < toDate(%(forecast_date)s)
              and lead_days = 1
              and toInt64(bakery_id) in %(bids)s
              and toInt64(product_id) in %(product_ids)s
            group by date, bakery_id, product_id
            """,
            parameters={
                "history_from": history_from,
                "forecast_date": forecast_date,
                "bids": PILOT_BAKERY_IDS,
                "product_ids": list(ColdStartConfig().product_ids),
            },
        )
        cold_history = sales_history.merge(
            forecast_history,
            on=["date", "bakery_id", "product_id"],
            how="left",
            validate="one_to_one",
        )
        cold_history["forecast_qty"] = cold_history["forecast_qty"].fillna(0.0)
        cold_registry = build_cold_start_registry(
            cold_history,
            as_of_date=forecast_date,
        )
        eligible = apply_category_neutral_cold_start(
            eligible,
            cold_registry,
        )
        eligible["forecast_qty"] = eligible["cold_start_forecast_qty"]
        print(
            "  new-SKU cold start: "
            f"{len(cold_registry)} bakery/SKU floors, "
            "bakery/category totals preserved"
        )

    mature_registry = pd.DataFrame()
    if sku_correction_registry:
        registry_path = Path(sku_correction_registry)
        if not registry_path.exists():
            raise FileNotFoundError(
                f"SKU correction registry not found: {registry_path}"
            )
        mature_registry = pd.read_csv(registry_path, encoding="utf-8-sig")
    else:
        mature_history_from = str(
            date_type.fromisoformat(forecast_date)
            - timedelta(days=CorrectionConfig().history_days)
        )
        mature_fact = client.query_df(
            """
            select
                dt as date,
                toInt64(bakery_id) as bakery_id,
                toInt64(product_id) as product_id,
                any(product_name) as product_name,
                any(category_name) as category_name,
                sum(qty_sold) as sold_qty,
                sum(qty_produced) as produced_qty,
                max(last_sale_time) as last_sale_time
            from Svezhar.mart_zero_sales_60d
            where dt >= toDate(%(history_from)s)
              and dt < toDate(%(forecast_date)s)
              and toInt64(bakery_id) in %(bids)s
            group by date, bakery_id, product_id
            """,
            parameters={
                "history_from": mature_history_from,
                "forecast_date": forecast_date,
                "bids": PILOT_BAKERY_IDS,
            },
        )
        mature_forecast = client.query_df(
            """
            select
                forecast_date as date,
                toInt64(bakery_id) as bakery_id,
                toInt64(product_id) as product_id,
                argMax(forecast_qty, generated_at) as forecast_qty
            from Svezhar.sku_forecast_day_snapshots
            where forecast_date >= toDate(%(history_from)s)
              and forecast_date < toDate(%(forecast_date)s)
              and lead_days = 1
              and toInt64(bakery_id) in %(bids)s
            group by date, bakery_id, product_id
            """,
            parameters={
                "history_from": mature_history_from,
                "forecast_date": forecast_date,
                "bids": PILOT_BAKERY_IDS,
            },
        )
        mature_history = mature_fact.merge(
            mature_forecast,
            on=["date", "bakery_id", "product_id"],
            how="left",
            validate="one_to_one",
        )
        mature_history["forecast_qty"] = mature_history[
            "forecast_qty"
        ].fillna(0.0)
        last_sale = pd.to_datetime(
            mature_history["last_sale_time"],
            errors="coerce",
        )
        day_start = last_sale.dt.normalize() + pd.Timedelta(hours=7)
        day_end = last_sale.dt.normalize() + pd.Timedelta(hours=19)
        elapsed_hours = (
            (last_sale - day_start).dt.total_seconds() / 3600.0
        )
        remaining_hours = (
            (day_end - last_sale).dt.total_seconds() / 3600.0
        ).clip(lower=0.0)
        full_realization = (
            mature_history["produced_qty"].gt(0)
            & mature_history["sold_qty"].ge(
                mature_history["produced_qty"] - 0.01
            )
            & elapsed_hours.ge(2.0)
            & last_sale.lt(day_end)
        )
        raw_lost = np.where(
            full_realization,
            mature_history["sold_qty"] / elapsed_hours * remaining_hours,
            0.0,
        )
        conservative_cap = np.maximum(
            mature_history["sold_qty"] * 1.5,
            15.0,
        )
        mature_history["lost_demand_qty"] = np.minimum(
            raw_lost,
            conservative_cap,
        )
        mature_history["demand_qty"] = (
            mature_history["sold_qty"]
            + mature_history["lost_demand_qty"]
        )
        mature_registry = build_correction_registry(
            mature_history,
            as_of_date=forecast_date,
        )

    if not mature_registry.empty:
        corrected = apply_category_neutral_corrections(
            eligible,
            mature_registry,
        )
        corrected_forecast = {
            (int(row["bakery_id"]), int(row["product_id"])): float(
                row["corrected_forecast_qty"]
            )
            for row in corrected.to_dict("records")
        }
        changed = (
            corrected["corrected_forecast_qty"]
            - corrected["forecast_qty"]
        ).abs() > 1e-9
        print(
            "  systematic SKU correction: "
            f"{int(changed.sum())} changed rows, "
            "bakery/category totals preserved"
        )
    elif enable_new_sku_cold_start:
        corrected_forecast = {
            (int(row["bakery_id"]), int(row["product_id"])): float(
                row["forecast_qty"]
            )
            for row in eligible.to_dict("records")
        }

    rows = []
    for rec in forecast_df.to_dict("records"):
        try:
            bid = int(rec["bakery_id"])
        except (TypeError, ValueError):
            continue
        try:
            pid_int = int(rec["product_id"])
        except (TypeError, ValueError):
            continue
        if bid not in PILOT_BAKERY_IDS:
            continue
        category = str(rec.get("category_name") or "")
        if category not in BAKEABLE_CATEGORIES:
            continue
        if pid_int in frozen_pids:
            continue
        if pid_int not in base_kratnost and (pid_int, bid) not in bakery_kratnost:
            continue  # no baking meta → not a bakeable SKU

        kratnost = bakery_kratnost.get((pid_int, bid)) or base_kratnost.get(pid_int) or 1
        forecast_qty = corrected_forecast.get(
            (bid, pid_int),
            float(rec.get("forecast_qty") or 0),
        )
        stock_qty = yesterday_stock.get((bid, pid_int), 0.0)
        net_need = max(forecast_qty - stock_qty, 0.0)
        production_plan = _round_up_kratnost(net_need, kratnost)

        bname = bakery_info.get(bid, {}).get("name") or str(bid)
        rows.append({
            "bakery_id": bid,
            "bakery_name": bname,
            "category": category,
            "product_name": str(rec.get("product_name") or ""),
            "forecast": round(forecast_qty, 1),
            "yesterday_stock": round(stock_qty, 1),
            "net_need": round(net_need, 1),
            "production_plan": production_plan,
            "total_for_sale": round(production_plan + stock_qty, 1),
            "kratnost": kratnost,
        })

    rows.sort(key=lambda r: (r["bakery_id"], r["category"], r["product_name"]))
    return rows


def _build_excel(rows: list[dict], forecast_date: str) -> bytes:
    import openpyxl
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    d = date_type.fromisoformat(forecast_date)
    weekday_name = WEEKDAY_RU[d.weekday()]

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Прогноз"

    # Header row
    header_fill = PatternFill("solid", fgColor="1F4E79")
    header_font = Font(bold=True, color="FFFFFF", size=10)
    thin = Side(style="thin", color="CCCCCC")
    cell_border = Border(left=thin, right=thin, top=thin, bottom=thin)

    title_font = Font(bold=True, size=12)
    ws["A1"] = f"Прогноз выпечки — {d.strftime('%d.%m.%Y')} ({weekday_name})"
    ws["A1"].font = title_font
    ws.row_dimensions[1].height = 20

    headers = [
        "Пекарня",
        "Категория",
        "Номенклатура",
        "Прогноз",
        "Остаток со вчерашнего дня",
        "Чистая потребность",
        "План выпуска",
        "Итого на продажу",
        "Кратность",
    ]
    col_widths = [35, 20, 40, 12, 24, 20, 16, 18, 12]

    for col_idx, (h, w) in enumerate(zip(headers, col_widths), start=1):
        cell = ws.cell(row=2, column=col_idx, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = cell_border
        ws.column_dimensions[get_column_letter(col_idx)].width = w
    ws.row_dimensions[2].height = 30

    # Alternating bakery fill
    bakery_fills = [
        PatternFill("solid", fgColor="EBF3FB"),
        PatternFill("solid", fgColor="FFFFFF"),
    ]
    number_fmt = "#,##0.0"
    int_fmt = "#,##0"

    prev_bid = None
    fill_idx = 0
    for data_row in rows:
        if data_row["bakery_id"] != prev_bid:
            fill_idx = 1 - fill_idx
            prev_bid = data_row["bakery_id"]
        fill = bakery_fills[fill_idx]

        row_num = ws.max_row + 1
        values = [
            data_row["bakery_name"],
            data_row["category"],
            data_row["product_name"],
            data_row["forecast"],
            data_row["yesterday_stock"],
            data_row["net_need"],
            data_row["production_plan"],
            data_row["total_for_sale"],
            data_row["kratnost"],
        ]
        fmts = [None, None, None, number_fmt, number_fmt, number_fmt, int_fmt, number_fmt, int_fmt]
        for col_idx, (val, fmt) in enumerate(zip(values, fmts), start=1):
            cell = ws.cell(row=row_num, column=col_idx, value=val)
            cell.fill = fill
            cell.border = cell_border
            cell.font = Font(size=10)
            if fmt:
                cell.number_format = fmt
                cell.alignment = Alignment(horizontal="right")

    ws.freeze_panes = "A3"
    ws.auto_filter.ref = f"A2:{get_column_letter(len(headers))}{ws.max_row}"

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


VIBECODE_API_BASE = "https://vibecode.bitrix24.tech/v1"
# Chat "Пилот выставления планов выпекания ИИ" — diskFolderId 1473995, chatId 179919
PILOT_CHAT_DIALOG_ID = "chat179919"
PILOT_CHAT_ID = 179919
PILOT_CHAT_DISK_FOLDER_ID = 1473995
# Native Bitrix24 webhook base URL — set via B24_WEBHOOK_URL env var (not hardcoded to avoid token leak)
# Example: https://franshizasvezhar.bitrix24.ru/rest/27979/<token>
B24_WEBHOOK_URL_ENV = "B24_WEBHOOK_URL"


def _send_to_chat(file_bytes: bytes, filename: str, forecast_date: str) -> None:
    """Upload Excel to the chat's Disk folder and send it as a file message.

    Flow:
      1. Native B24 disk.folder.uploadfile → get uploadUrl (supports Cyrillic filenames)
      2. POST file bytes to uploadUrl via multipart → get disk object id
      3. im.disk.file.commit → sends file as a proper attachment in chat
      4. Send short text message via VibeCode chats API
    """
    import email.generator
    import io
    import json
    import time as _time
    import urllib.request
    from email.mime.multipart import MIMEMultipart
    from email.mime.application import MIMEApplication

    api_key = os.environ.get("VIBECODE_API_KEY") or ""
    if not api_key:
        raise RuntimeError("VIBECODE_API_KEY not set")
    b24_webhook_base = os.environ.get(B24_WEBHOOK_URL_ENV, "").rstrip("/")
    if not b24_webhook_base:
        raise RuntimeError(f"{B24_WEBHOOK_URL_ENV} not set in environment")

    d = date_type.fromisoformat(forecast_date)
    weekday_name = WEEKDAY_RU[d.weekday()]
    # Russian filename shown in chat
    ru_filename = f"Прогноз_{d.strftime('%d.%m.%Y')}_{weekday_name}.xlsx"

    # Step 1: get uploadUrl from native B24 REST
    step1_body = json.dumps({
        "id": PILOT_CHAT_DISK_FOLDER_ID,
        "data": {"NAME": ru_filename},
        "generateUniqueName": "Y",
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{b24_webhook_base}/disk.folder.uploadfile",
        data=step1_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        step1 = json.loads(resp.read())
    if "error" in step1:
        raise RuntimeError(f"disk.folder.uploadfile failed: {step1}")
    upload_url = step1["result"]["uploadUrl"]
    print(f"  [b24] uploadUrl obtained")

    # Step 2: POST file bytes to uploadUrl as multipart/form-data
    boundary = f"----FormBoundary{int(_time.time())}"
    body_parts = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{ru_filename}"\r\n'
        f"Content-Type: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet\r\n"
        f"\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
    req = urllib.request.Request(
        upload_url,
        data=body_parts,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        step2 = json.loads(resp.read())
    if "error" in step2:
        raise RuntimeError(f"File upload to uploadUrl failed: {step2}")
    disk_id = step2["result"]["ID"]
    print(f"  [b24] file uploaded, disk_id={disk_id}")

    # Step 3: send short text message first via VibeCode
    msg_text = f"Прогноз — {d.strftime('%d.%m.%Y')} ({weekday_name})"
    msg_body = json.dumps({"message": msg_text}).encode("utf-8")
    req = urllib.request.Request(
        f"{VIBECODE_API_BASE}/chats/{PILOT_CHAT_DIALOG_ID}/messages",
        data=msg_body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        msg_result = json.loads(resp.read())
    if not msg_result.get("success"):
        raise RuntimeError(f"Message send failed: {msg_result}")
    print(f"  [vibecode] text message sent, id={msg_result['data']}")

    # Step 4: commit file to chat — sends it as a proper attachment message
    commit_body = json.dumps({
        "CHAT_ID": PILOT_CHAT_ID,
        "DISK_ID": disk_id,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{b24_webhook_base}/im.disk.file.commit",
        data=commit_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        commit_result = json.loads(resp.read())
    if "error" in commit_result:
        raise RuntimeError(f"im.disk.file.commit failed: {commit_result}")
    file_msg_id = commit_result.get("result", {}).get("MESSAGE_ID")
    print(f"  [b24] file message sent, message_id={file_msg_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--date", default=None, help="Forecast date (YYYY-MM-DD); default: today")
    parser.add_argument("--dry-run", action="store_true", help="Build Excel but do not send to Bitrix24")
    parser.add_argument("--out-dir", default="output/pilot_forecast")
    parser.add_argument(
        "--sku-correction-registry",
        default=None,
        help=(
            "Optional mature-SKU correction registry CSV. "
            "If omitted, PILOT_SKU_CORRECTION_REGISTRY is used."
        ),
    )
    args = parser.parse_args()

    if args.env_file and Path(args.env_file).exists():
        _load_env(args.env_file)

    forecast_date = args.date or str(date_type.today())
    d = date_type.fromisoformat(forecast_date)
    weekday_abbr = WEEKDAY_RU[d.weekday()][:2]

    print(f"Pilot forecast summary | date: {forecast_date} ({weekday_abbr})")

    registry_path = (
        args.sku_correction_registry
        or os.environ.get("PILOT_SKU_CORRECTION_REGISTRY")
        or None
    )
    rows = _build_report(
        forecast_date,
        sku_correction_registry=registry_path,
    )
    if not rows:
        print("No data found — aborting.")
        return

    print(f"  {len(rows)} SKU rows across {len({r['bakery_id'] for r in rows})} bakeries")

    file_bytes = _build_excel(rows, forecast_date)

    filename = f"Прогноз_выпечки_{d.strftime('%d.%m.%Y')}_{weekday_abbr}.xlsx"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    out_path.write_bytes(file_bytes)
    print(f"  saved: {out_path}")

    if args.dry_run:
        print("  --dry-run: skipping Bitrix24 send")
        return

    api_key = (
        os.environ.get("VIBECODE_API_KEY")
        or os.environ.get("VIBECODE_API_KEY".lower())
        or ""
    )
    if not api_key:
        print("  VIBECODE_API_KEY not set — skipping Bitrix24 send")
        return

    print(f"  sending to {PILOT_CHAT_DIALOG_ID}...")
    _send_to_chat(file_bytes, filename, forecast_date)
    print("  done.")


if __name__ == "__main__":
    main()

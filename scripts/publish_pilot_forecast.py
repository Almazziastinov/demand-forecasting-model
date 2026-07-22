"""Generate and publish the daily pilot forecast summary to Bitrix24 chat.

For the given date (default: tomorrow), queries ClickHouse for per-SKU
daily forecasts for all pilot bakeries, applies kratnost rounding, and
writes a single Excel file.  If VIBECODE_API_KEY is set (via .env or
environment), uploads the file to the Bitrix24 pilot chat automatically.

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "apps"))
sys.path.insert(0, str(ROOT / "apps" / "forecast_embedded"))

PILOT_BAKERY_IDS = [16, 20, 21, 22, 28, 80, 89, 107, 221, 222, 257]

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


def _build_report(forecast_date: str) -> list[dict]:
    """Return rows: bakery_name, category, product_name, forecast, forecast_kratnost."""
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
    bakery_df = client.query_df(
        "select bakery_id as bid, any(bakery_name) as name, any(city) as city "
        "from dim_bakeries "
        "where bakery_id in %(bids)s "
        "group by bakery_id",
        parameters={"bids": [str(b) for b in PILOT_BAKERY_IDS]},
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
        forecast_qty = float(rec.get("forecast_qty") or 0)
        forecast_rounded = _round_up_kratnost(forecast_qty, kratnost)

        bname = bakery_info.get(bid, {}).get("name") or str(bid)
        rows.append({
            "bakery_id": bid,
            "bakery_name": bname,
            "category": category,
            "product_name": str(rec.get("product_name") or ""),
            "forecast": round(forecast_qty, 1),
            "forecast_kratnost": forecast_rounded,
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

    headers = ["Пекарня", "Категория", "Номенклатура", "Прогноз", "Прогноз (кратность)", "Кратность"]
    col_widths = [35, 20, 40, 12, 20, 12]

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
            data_row["forecast_kratnost"],
            data_row["kratnost"],
        ]
        fmts = [None, None, None, number_fmt, int_fmt, int_fmt]
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


def _send_via_vibecode(file_bytes: bytes, filename: str, forecast_date: str) -> None:
    """Upload Excel to the chat's Disk folder and send it as a file message.

    Flow:
      1. Upload via VibeCode /v1/files/upload → get disk object id
      2. Call native B24 im.disk.file.commit → sends file as a proper chat attachment
      3. Send a short text header via VibeCode chats API
    """
    import base64
    import json
    import urllib.request

    api_key = os.environ.get("VIBECODE_API_KEY") or ""
    if not api_key:
        raise RuntimeError("VIBECODE_API_KEY not set")

    vibe_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Step 1: upload file to the chat's Disk folder via VibeCode
    import time as _time
    d_upload = date_type.fromisoformat(forecast_date)
    ascii_filename = f"forecast_{d_upload.strftime('%Y-%m-%d')}_{int(_time.time())}.xlsx"
    upload_body = json.dumps({
        "folderId": PILOT_CHAT_DISK_FOLDER_ID,
        "filename": ascii_filename,
        "content": base64.b64encode(file_bytes).decode("ascii"),
    }, ensure_ascii=True).encode("utf-8")
    req = urllib.request.Request(
        f"{VIBECODE_API_BASE}/files/upload",
        data=upload_body,
        headers=vibe_headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            upload_result = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read()
        raise RuntimeError(f"File upload HTTP {exc.code}: {body[:500]}")
    if not upload_result.get("success"):
        raise RuntimeError(f"File upload failed: {upload_result}")
    disk_id = upload_result["data"]["id"]
    print(f"  [vibecode] file uploaded, disk_id={disk_id}")

    # Step 2: commit file to the chat via native B24 REST (im.disk.file.commit)
    # This sends the file as a proper attachment message in the chat.
    b24_webhook_base = os.environ.get(B24_WEBHOOK_URL_ENV, "").rstrip("/")
    if not b24_webhook_base:
        raise RuntimeError(f"{B24_WEBHOOK_URL_ENV} not set in environment")
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

    # Step 3: send a text header message via VibeCode
    d = date_type.fromisoformat(forecast_date)
    weekday_name = WEEKDAY_RU[d.weekday()]
    msg_text = (
        f"[b]Прогноз выпечки — {d.strftime('%d.%m.%Y')} ({weekday_name})[/b]\n"
        f"Все пилотные пекарни · {len(PILOT_BAKERY_IDS)} пекарен\n"
        f"Прогноз + прогноз с учётом кратности по каждой позиции"
    )
    msg_body = json.dumps({"message": msg_text}).encode()
    req = urllib.request.Request(
        f"{VIBECODE_API_BASE}/chats/{PILOT_CHAT_DIALOG_ID}/messages",
        data=msg_body,
        headers=vibe_headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        msg_result = json.loads(resp.read())
    if not msg_result.get("success"):
        raise RuntimeError(f"Message send failed: {msg_result}")
    print(f"  [vibecode] header message sent, id={msg_result['data']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--date", default=None, help="Forecast date (YYYY-MM-DD); default: tomorrow")
    parser.add_argument("--dry-run", action="store_true", help="Build Excel but do not send to Bitrix24")
    parser.add_argument("--out-dir", default="output/pilot_forecast")
    args = parser.parse_args()

    if args.env_file and Path(args.env_file).exists():
        _load_env(args.env_file)

    forecast_date = args.date or str(date_type.today() + timedelta(days=1))
    d = date_type.fromisoformat(forecast_date)
    weekday_abbr = WEEKDAY_RU[d.weekday()][:2]

    print(f"Pilot forecast summary | date: {forecast_date} ({weekday_abbr})")

    rows = _build_report(forecast_date)
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
    _send_via_vibecode(file_bytes, filename, forecast_date)
    print("  done.")


if __name__ == "__main__":
    main()

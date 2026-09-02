"""Build and atomically publish the pilot management report to Blackhole."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER_ID = "82bb03a8-c356-4225-97a4-a1540cdc29e6"
DEFAULT_API_BASE = "https://vibecode.bitrix24.tech/v1"
REPORT_NAME = "pilot_management_summary"


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def validate_report(report_dir: Path, date_from: date, date_to: date) -> dict[str, object]:
    required = ("summary.json", "detail.csv", "week_kpi.csv")
    for name in required:
        path = report_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty report file: {name}")

    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8-sig"))
    if summary.get("date_from") != date_from.isoformat() or summary.get("date_to") != date_to.isoformat():
        raise RuntimeError("summary.json date range does not match the requested range")

    dates: set[str] = set()
    last_day_bakeries: set[str] = set()
    with (report_dir / "detail.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            business_date = str(row.get("business_date", ""))[:10]
            bakery_id = str(row.get("bakery_id", "")).strip()
            if business_date:
                dates.add(business_date)
            if business_date == date_to.isoformat() and bakery_id:
                last_day_bakeries.add(bakery_id)

    expected = {
        (date_from + timedelta(days=offset)).isoformat()
        for offset in range((date_to - date_from).days + 1)
    }
    if dates != expected:
        raise RuntimeError(
            f"Report date coverage mismatch: missing={sorted(expected - dates)}, extra={sorted(dates - expected)}"
        )
    if not last_day_bakeries:
        raise RuntimeError(f"No pilot bakeries found for {date_to.isoformat()}")
    return {
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "days": len(dates),
        "last_day_bakeries": len(last_day_bakeries),
        "forecast_source": summary.get("forecast_source"),
    }


def api_request(url: str, api_key: str, payload: dict[str, object], timeout: int) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not result.get("success"):
        raise RuntimeError(f"VibeCode request failed: {result}")
    return result


def deploy_script(date_from: date, date_to: date) -> str:
    return f'''from __future__ import annotations
import csv, json, os, shutil, tarfile, urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

archive = Path("/opt/.pilot_management_summary_upload.tar.gz")
reports_root = Path("/opt/reports").resolve()
backups_root = Path("/opt/backups").resolve()
target = (reports_root / "pilot_management_summary").resolve()
stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
stage = (reports_root / (".pilot_management_summary_stage_" + stamp)).resolve()
old = (reports_root / (".pilot_management_summary_old_" + stamp)).resolve()
backup = (backups_root / ("pilot_management_summary_before_" + stamp)).resolve()
for path, root in ((target, reports_root), (stage, reports_root), (old, reports_root), (backup, backups_root)):
    if path.parent != root:
        raise RuntimeError(f"Unsafe deployment path: {{path}}")
if stage.exists() or old.exists() or backup.exists():
    raise RuntimeError("Deployment staging path already exists")
stage.mkdir(parents=True)
try:
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = (stage / member.name).resolve()
            if stage not in member_path.parents and member_path != stage:
                raise RuntimeError(f"Unsafe archive member: {{member.name}}")
        tar.extractall(stage)
    summary = json.loads((stage / "summary.json").read_text(encoding="utf-8-sig"))
    if summary.get("date_from") != "{date_from.isoformat()}" or summary.get("date_to") != "{date_to.isoformat()}":
        raise RuntimeError("Remote report date validation failed")
    dates = set()
    with (stage / "detail.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            dates.add(str(row.get("business_date", ""))[:10])
    expected = {{
        (date({date_from.year}, {date_from.month}, {date_from.day}) + timedelta(days=i)).isoformat()
        for i in range({(date_to - date_from).days + 1})
    }}
    if dates != expected:
        raise RuntimeError("Remote detail date coverage validation failed")
    if target.exists():
        shutil.copytree(target, backup)
        os.replace(target, old)
    try:
        os.replace(stage, target)
    except Exception:
        if old.exists() and not target.exists():
            os.replace(old, target)
        raise
    if old.exists():
        shutil.rmtree(old)
    urllib.request.urlopen("http://localhost:3000/health", timeout=15).read()
    print(json.dumps({{"deployed": str(target), "backup": str(backup), "date_to": "{date_to.isoformat()}"}}))
finally:
    if stage.exists():
        shutil.rmtree(stage)
    archive.unlink(missing_ok=True)
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date-from", type=date.fromisoformat, default=date(2026, 7, 23))
    parser.add_argument("--date-to", type=date.fromisoformat)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--server-id", default=DEFAULT_SERVER_ID)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    args = parser.parse_args()

    load_env(args.env_file)
    api_key = os.environ.get("VIBECODE_API_KEY", "")
    if not api_key:
        raise RuntimeError("VIBECODE_API_KEY is not configured")
    date_to = args.date_to or (datetime.now(ZoneInfo("Europe/Moscow")).date() - timedelta(days=1))
    if date_to < args.date_from:
        raise RuntimeError("date-to is earlier than date-from")

    with tempfile.TemporaryDirectory(prefix="pilot_management_report_") as temp:
        temp_dir = Path(temp)
        report_dir = temp_dir / REPORT_NAME
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "build_pilot_management_summary.py"),
                "--date-from",
                args.date_from.isoformat(),
                "--date-to",
                date_to.isoformat(),
                "--output-dir",
                str(report_dir),
                "--env-file",
                str(args.env_file),
                "--scope",
                "pilot_dynamic",
            ],
            cwd=ROOT,
            check=True,
        )
        validation = validate_report(report_dir, args.date_from, date_to)
        archive = temp_dir / f"{REPORT_NAME}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for path in sorted(report_dir.iterdir()):
                if path.is_file():
                    tar.add(path, arcname=path.name)

        exec_url = f"{args.api_base.rstrip('/')}/infra/servers/{args.server_id}/exec?stream=false"
        upload_url = exec_url.replace("/exec?stream=false", "/upload")
        api_request(
            upload_url,
            api_key,
            {
                "path": "/opt/.pilot_management_summary_upload.tar.gz",
                "content": base64.b64encode(archive.read_bytes()).decode("ascii"),
                "mode": "600",
            },
            timeout=600,
        )
        encoded = base64.b64encode(deploy_script(args.date_from, date_to).encode("utf-8")).decode("ascii")
        command = f"echo {encoded} | base64 -d | python3"
        result = api_request(exec_url, api_key, {"command": command}, timeout=600)
        remote = result.get("data")
        if not isinstance(remote, dict):
            raise RuntimeError(f"VibeCode exec returned no result data: {result}")
        exit_code = remote.get("exitCode")
        if exit_code != 0:
            detail = remote.get("stderr") or remote.get("stdout")
            raise RuntimeError(
                f"Blackhole deployment failed with exit code {exit_code}: {detail}"
            )
        print(
            json.dumps(
                {"validation": validation, "remote": remote.get("stdout")},
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()

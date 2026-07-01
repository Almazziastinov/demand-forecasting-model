"""Weekly refresh of SKU hour share profile + uplift multipliers in ClickHouse.

Pipeline:
  1. Export raw check lines from ClickHouse -> data/raw/sales_hrs_all_clickhouse.csv
  2. Build SKU hour share profile from CSV
  3. Smooth the profile
  4. Load smoothed profile to ClickHouse (sku_hour_share_profile_smoothed_embedded)
  5. Load raw uplift multipliers to ClickHouse (sku_hour_uplift_multiplier_embedded)

Recent correction (runner_city_prior_soft_weekpart) reads mart_sales_60d live
at allocation time and does NOT need a separate refresh step.

Usage (on VM):
    cd /opt/demand-forecasting-model
    .venv/bin/python scripts/weekly_profile_refresh.py --env-file .env

Optional:
    --date-from 2025-09-01   # default: 12 months back from today
    --date-to   2026-06-30   # default: yesterday
    --profile-version weekly_YYYYMMDD  # default: auto-generated
    --dry-run                # print plan, skip ClickHouse writes
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "weekly_profile_refresh.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

PYTHON = sys.executable

DEFAULT_RAW_OUTPUT = ROOT / "data" / "raw" / "sales_hrs_all_clickhouse.csv"
DEFAULT_SQL_TEMPLATE = ROOT / "scripts" / "clickhouse_export_template.sql"
DEFAULT_PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_SCHEMA_PATH = ROOT / "apps" / "forecast_embedded" / "sql" / "schema.sql"


def run(cmd: list[str], step: str, dry_run: bool = False) -> None:
    log.info("=== STEP: %s ===", step)
    log.info("cmd: %s", " ".join(str(c) for c in cmd))
    if dry_run:
        log.info("DRY RUN — skipping")
        return
    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = round(time.monotonic() - t0, 1)
    if result.returncode != 0:
        log.error("FAILED %s (exit=%d, elapsed=%.1fs)", step, result.returncode, elapsed)
        sys.exit(result.returncode)
    log.info("OK %s (elapsed=%.1fs)", step, elapsed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--date-from", default=None, help="default: 12 months ago")
    parser.add_argument("--date-to", default=None, help="default: yesterday")
    parser.add_argument("--profile-version", default=None, help="default: weekly_YYYYMMDD")
    parser.add_argument("--raw-output", default=str(DEFAULT_RAW_OUTPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_PROCESSED_DIR))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    today = date.today()
    date_to = args.date_to or str(today - timedelta(days=1))
    date_from = args.date_from or str(date(today.year - 1, today.month, today.day))
    profile_version = args.profile_version or f"weekly_{today.strftime('%Y%m%d')}"

    output_dir = Path(args.output_dir)
    raw_output = Path(args.raw_output)
    profile_smoothed = output_dir / "sku_hour_share_profile_smoothed.csv"
    profile_applied = output_dir / "sku_hour_share_profile_daily.csv"
    profile_applied_smoothed = output_dir / "sku_hour_share_profile_daily_smoothed.csv"

    log.info("weekly_profile_refresh start")
    log.info("date_from=%s date_to=%s profile_version=%s", date_from, date_to, profile_version)

    # 1. Export raw checks from ClickHouse
    run([
        PYTHON, "scripts/export_clickhouse_checks.py",
        "--env-file", args.env_file,
        "--sql-template", str(DEFAULT_SQL_TEMPLATE),
        "--date-from", date_from,
        "--date-to", date_to,
        "--output", str(raw_output),
    ], step="1/5 export_checks", dry_run=args.dry_run)

    # 2. Build SKU hour share profile
    run([
        PYTHON, "-m", "src.experiments_v2.build_sku_hour_share_profile",
        "--source-path", str(raw_output),
        "--output-dir", str(output_dir),
    ], step="2/5 build_share_profile", dry_run=args.dry_run)

    # 3. Smooth the profile
    run([
        PYTHON, "-m", "src.experiments_v2.smooth_sku_hour_share_profile",
        "--profile-path", str(output_dir / "sku_hour_share_profile.csv"),
        "--applied-path", str(profile_applied),
        "--output-dir", str(output_dir),
    ], step="3/5 smooth_share_profile", dry_run=args.dry_run)

    # 4. Load smoothed profile to ClickHouse
    run([
        PYTHON, "-m", "pipelines.forecast_publish.sku_hour_profile_store",
        "--mode", "load",
        "--env-file", args.env_file,
        "--schema-path", str(DEFAULT_SCHEMA_PATH),
        "--profile-path", str(profile_smoothed),
        "--truncate",
    ], step="4/5 load_profile_to_clickhouse", dry_run=args.dry_run)

    # 5. Load uplift multipliers to ClickHouse
    run([
        PYTHON, "-m", "pipelines.forecast_publish.sku_hour_profile_store",
        "--mode", "load-uplift-multipliers",
        "--env-file", args.env_file,
        "--schema-path", str(DEFAULT_SCHEMA_PATH),
        "--applied-path", str(profile_applied_smoothed),
        "--profile-version", profile_version,
        "--truncate",
    ], step="5/5 load_uplift_multipliers_to_clickhouse", dry_run=args.dry_run)

    log.info("weekly_profile_refresh DONE profile_version=%s", profile_version)

    summary = {
        "profile_version": profile_version,
        "date_from": date_from,
        "date_to": date_to,
        "finished_at": str(date.today()),
    }
    summary_path = ROOT / "reports" / "weekly_profile_refresh_last_run.json"
    summary_path.parent.mkdir(exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("summary: %s", summary_path)


if __name__ == "__main__":
    main()

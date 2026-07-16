"""
Export raw bakery check data from ClickHouse into the normalized English schema
used by the bakery-driven pipeline.

The script does not hardcode your ClickHouse column names. Instead it reads a
SQL template and expects that template to return a canonical set of columns.

Required columns:
  - check_datetime
  - check_date
  - cash_event_type
  - quantity
  - bakery_id
  - bakery_name
  - city
  - product_id
  - product_name
  - category_name

Optional but useful columns:
  - freshness
  - price
  - line_amount

The SQL template may use the placeholders:
  - {date_from}
  - {date_to}
  - {limit_clause}

Typical usage:
  python scripts/export_clickhouse_checks.py ^
    --sql-template scripts/clickhouse_export_template.sql ^
    --date-from 2025-01-01 ^
    --date-to 2026-05-01 ^
    --output data/raw/sales_hrs_all_clickhouse.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterator

import pandas as pd

try:
    import clickhouse_connect
except ImportError as exc:  # pragma: no cover - runtime environment concern
    clickhouse_connect = None
    CLICKHOUSE_IMPORT_ERROR = exc
else:
    CLICKHOUSE_IMPORT_ERROR = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = ROOT / ".env"
DEFAULT_SQL_TEMPLATE = ROOT / "scripts" / "clickhouse_export_template.sql"
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "sales_hrs_all_clickhouse.csv"

REQUIRED_COLUMNS = [
    "check_datetime",
    "check_date",
    "cash_event_type",
    "quantity",
    "bakery_id",
    "bakery_name",
    "city",
    "product_id",
    "product_name",
    "category_name",
]

OPTIONAL_COLUMNS = [
    "freshness",
    "price",
    "line_amount",
]


def load_env_file(path: str | Path = DEFAULT_ENV_PATH) -> dict[str, str]:
    env: dict[str, str] = {}
    file_path = Path(path)
    if not file_path.exists():
        return env

    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def first_setting(env: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        value = env.get(key) or os.getenv(key)
        if value:
            return value
    return None


def get_connection_settings(env_path: str | Path) -> dict[str, str]:
    env = load_env_file(env_path)
    settings = {
        "host": first_setting(env, "CLICKHOUSE_HOST", "HOST"),
        "port": first_setting(env, "CLICKHOUSE_PORT", "PORT"),
        "username": first_setting(env, "CLICKHOUSE_USER", "USER"),
        "password": first_setting(env, "CLICKHOUSE_PASSWORD", "PASSWORD"),
        "database": first_setting(env, "CLICKHOUSE_DATABASE", "DATABASE"),
        "secure": first_setting(env, "CLICKHOUSE_SECURE", "SECURE"),
        "verify": first_setting(env, "CLICKHOUSE_VERIFY", "VERIFY"),
    }

    missing = [
        key
        for key in ["host", "port", "username", "password", "database"]
        if not settings.get(key)
    ]
    if missing:
        raise ValueError(
            "Missing ClickHouse connection settings: "
            + ", ".join(missing)
            + ". Fill them in .env or environment variables."
        )
    return settings


def create_client(env_path: str | Path):
    if clickhouse_connect is None:  # pragma: no cover - runtime environment concern
        raise ImportError(
            "clickhouse-connect is not installed. Add it to the environment first."
        ) from CLICKHOUSE_IMPORT_ERROR

    settings = get_connection_settings(env_path)
    return clickhouse_connect.get_client(
        host=settings["host"],
        port=int(settings["port"]),
        username=settings["username"],
        password=settings["password"],
        database=settings["database"],
        secure=str(settings.get("secure", "true")).strip().lower()
        in {"1", "true", "yes", "on"},
        verify=str(settings.get("verify", "false")).strip().lower()
        in {"1", "true", "yes", "on"},
    )


def month_windows(date_from: str, date_to: str) -> Iterator[tuple[str, str]]:
    start = pd.Timestamp(date_from).normalize()
    end = pd.Timestamp(date_to).normalize()
    if start > end:
        raise ValueError("date_from must be <= date_to")

    cursor = start
    while cursor <= end:
        month_end = (cursor + pd.offsets.MonthEnd(0)).normalize()
        window_end = min(month_end, end)
        yield cursor.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d")
        cursor = window_end + pd.Timedelta(days=1)


def week_windows(date_from: str, date_to: str) -> Iterator[tuple[str, str]]:
    start = pd.Timestamp(date_from).normalize()
    end = pd.Timestamp(date_to).normalize()
    if start > end:
        raise ValueError("date_from must be <= date_to")

    cursor = start
    while cursor <= end:
        window_end = min(cursor + pd.Timedelta(days=6), end)
        yield cursor.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d")
        cursor = window_end + pd.Timedelta(days=1)


def render_sql(
    template_text: str,
    *,
    date_from: str,
    date_to: str,
    limit: int | None,
) -> str:
    limit_clause = "" if limit is None else f"LIMIT {limit}"
    return template_text.format(
        date_from=date_from,
        date_to=date_to,
        limit_clause=limit_clause,
    )


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "The SQL query did not return the required canonical columns: "
            + ", ".join(missing)
        )


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    ordered = [col for col in REQUIRED_COLUMNS + OPTIONAL_COLUMNS if col in df.columns]
    tail = [col for col in df.columns if col not in ordered]
    return df[ordered + tail]


def export_windows(
    *,
    client,
    sql_template_text: str,
    output_path: Path,
    date_from: str,
    date_to: str,
    batch_mode: str,
    limit: int | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    total_rows = 0
    wrote_header = False

    if batch_mode == "single":
        windows = [(date_from, date_to)]
    elif batch_mode == "monthly":
        windows = list(month_windows(date_from, date_to))
    elif batch_mode == "weekly":
        windows = list(week_windows(date_from, date_to))
    else:
        raise ValueError("batch_mode must be 'single', 'monthly', or 'weekly'")

    for index, (window_from, window_to) in enumerate(windows, start=1):
        sql = render_sql(
            sql_template_text,
            date_from=window_from,
            date_to=window_to,
            limit=limit,
        )
        print(
            f"[{index}/{len(windows)}] Querying {window_from} .. {window_to}",
            flush=True,
        )
        df = client.query_df(sql)
        # clickhouse-connect returns columnless DataFrame for empty result sets
        if df.empty or len(df.columns) == 0:
            print(f"    rows: 0 (skipped)", flush=True)
            continue
        validate_columns(df)
        df = reorder_columns(df)
        rows = len(df)
        total_rows += rows
        print(f"    rows: {rows:,}", flush=True)

        if rows == 0:
            continue

        df.to_csv(
            output_path,
            mode="a",
            index=False,
            encoding="utf-8-sig",
            header=not wrote_header,
        )
        wrote_header = True

    print("=" * 72)
    print("CLICKHOUSE EXPORT COMPLETE")
    print("=" * 72)
    print(f"output: {output_path}")
    print(f"rows:   {total_rows:,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export raw clickhouse checks into the repository raw CSV schema."
    )
    parser.add_argument(
        "--sql-template",
        default=str(DEFAULT_SQL_TEMPLATE),
        help="Path to SQL template that returns canonical raw columns.",
    )
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_PATH),
        help="Path to .env with ClickHouse credentials.",
    )
    parser.add_argument(
        "--date-from",
        required=True,
        help="Inclusive start date in YYYY-MM-DD.",
    )
    parser.add_argument(
        "--date-to",
        required=True,
        help="Inclusive end date in YYYY-MM-DD.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--batch-mode",
        choices=["single", "monthly", "weekly"],
        default="monthly",
        help="How to split the export query across the date range.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional LIMIT for debugging small extracts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sql_template_path = Path(args.sql_template)
    if not sql_template_path.exists():
        raise FileNotFoundError(f"SQL template not found: {sql_template_path}")

    sql_template_text = sql_template_path.read_text(encoding="utf-8")
    client = create_client(args.env_file)
    export_windows(
        client=client,
        sql_template_text=sql_template_text,
        output_path=Path(args.output),
        date_from=args.date_from,
        date_to=args.date_to,
        batch_mode=args.batch_mode,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

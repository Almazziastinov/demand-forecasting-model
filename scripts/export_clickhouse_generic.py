"""
Generic ClickHouse CSV exporter for repository raw extracts.

Unlike export_clickhouse_checks.py, this script does not validate a fixed sales
schema. It simply runs a SQL template with optional date placeholders and saves
the result to CSV, optionally in monthly batches.

Supported placeholders in SQL templates:
  - {date_from}
  - {date_to}
  - {limit_clause}
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.export_clickhouse_checks import DEFAULT_ENV_PATH
from scripts.export_clickhouse_checks import create_client
from scripts.export_clickhouse_checks import month_windows
from scripts.export_clickhouse_checks import render_sql


ROOT = Path(__file__).resolve().parents[1]


def export_query(
    *,
    client,
    sql_template_text: str,
    output_path: Path,
    date_from: str | None,
    date_to: str | None,
    batch_mode: str,
    limit: int | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    if batch_mode == "single":
        if not date_from or not date_to:
            raise ValueError("single batch mode with placeholders requires both --date-from and --date-to")
        windows = [(date_from, date_to)]
    elif batch_mode == "monthly":
        if not date_from or not date_to:
            raise ValueError("monthly batch mode requires both --date-from and --date-to")
        windows = list(month_windows(date_from, date_to))
    elif batch_mode == "none":
        windows = [(date_from or "", date_to or "")]
    else:
        raise ValueError("batch_mode must be one of: none, single, monthly")

    wrote_header = False
    total_rows = 0

    for index, (window_from, window_to) in enumerate(windows, start=1):
        sql = render_sql(
            sql_template_text,
            date_from=window_from,
            date_to=window_to,
            limit=limit,
        )
        if batch_mode == "none":
            print("[1/1] Querying template without date batching", flush=True)
        else:
            print(f"[{index}/{len(windows)}] Querying {window_from} .. {window_to}", flush=True)
        df = client.query_df(sql)
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
    print("CLICKHOUSE GENERIC EXPORT COMPLETE")
    print("=" * 72)
    print(f"output: {output_path}")
    print(f"rows:   {total_rows:,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic ClickHouse SQL-to-CSV exporter")
    parser.add_argument("--sql-template", required=True, help="Path to SQL template file")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH), help="Path to .env with ClickHouse credentials")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--date-from", default=None, help="Inclusive start date in YYYY-MM-DD")
    parser.add_argument("--date-to", default=None, help="Inclusive end date in YYYY-MM-DD")
    parser.add_argument("--batch-mode", choices=["none", "single", "monthly"], default="none")
    parser.add_argument("--limit", type=int, default=None, help="Optional LIMIT for debugging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sql_template_path = Path(args.sql_template)
    if not sql_template_path.exists():
        raise FileNotFoundError(f"SQL template not found: {sql_template_path}")

    sql_template_text = sql_template_path.read_text(encoding="utf-8")
    client = create_client(args.env_file)
    export_query(
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

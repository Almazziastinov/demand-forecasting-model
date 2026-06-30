"""Export bakery-day aggregates from ClickHouse for production refresh."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.export_clickhouse_checks import create_client
from scripts.export_clickhouse_checks import month_windows
from scripts.export_clickhouse_checks import render_sql


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = ROOT / ".env"
DEFAULT_SQL_TEMPLATE = ROOT / "scripts" / "clickhouse_bakery_daily_template.sql"
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "bakery_daily_sales_clickhouse.csv"

REQUIRED_COLUMNS = [
    "date",
    "bakery_id",
    "bakery_name",
    "city",
    "bakery_sales",
    "line_amount_sum",
    "priced_quantity",
    "price_x_qty_sum",
]


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "The SQL query did not return the required bakery-day columns: "
            + ", ".join(missing)
        )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize ClickHouse result columns to the required bakery-day schema.

    Some ClickHouse/proxy combinations return expression names instead of SELECT
    aliases for aggregate queries. The SQL template controls column order, so a
    positional fallback is safer than failing the production refresh.
    """
    if all(col in df.columns for col in REQUIRED_COLUMNS):
        return df
    if len(df.columns) < len(REQUIRED_COLUMNS):
        return df

    work = df.copy()
    rename_map = {
        current: expected
        for current, expected in zip(
            work.columns[: len(REQUIRED_COLUMNS)],
            REQUIRED_COLUMNS,
        )
    }
    return work.rename(columns=rename_map)


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    ordered = [col for col in REQUIRED_COLUMNS if col in df.columns]
    tail = [col for col in df.columns if col not in ordered]
    return df[ordered + tail]


def export_daily_windows(
    *,
    client,
    sql_template_text: str,
    output_path: Path,
    date_from: str,
    date_to: str,
    batch_mode: str,
    limit: int | None,
) -> dict[str, int | str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    if batch_mode == "single":
        windows = [(date_from, date_to)]
    elif batch_mode == "monthly":
        windows = list(month_windows(date_from, date_to))
    else:
        raise ValueError("batch_mode must be 'single' or 'monthly'")

    total_rows = 0
    wrote_header = False
    for index, (window_from, window_to) in enumerate(windows, start=1):
        sql = render_sql(
            sql_template_text,
            date_from=window_from,
            date_to=window_to,
            limit=limit,
        )
        print(
            f"[{index}/{len(windows)}] Querying bakery-day "
            f"{window_from} .. {window_to}",
            flush=True,
        )
        df = client.query_df(sql)
        df = normalize_columns(df)
        if df.empty and len(df.columns) == 0:
            df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        validate_columns(df)
        df = reorder_columns(df)
        rows = len(df)
        total_rows += rows
        print(f"    bakery-day rows: {rows:,}", flush=True)
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
    print("CLICKHOUSE BAKERY-DAY EXPORT COMPLETE")
    print("=" * 72)
    print(f"output: {output_path}")
    print(f"rows:   {total_rows:,}")
    return {
        "output_path": str(output_path),
        "rows": total_rows,
        "windows": len(windows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export bakery-day ClickHouse aggregates for production refresh."
    )
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH))
    parser.add_argument("--sql-template", default=str(DEFAULT_SQL_TEMPLATE))
    parser.add_argument("--date-from", required=True)
    parser.add_argument("--date-to", required=True)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--batch-mode",
        choices=["single", "monthly"],
        default="monthly",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sql_template_text = Path(args.sql_template).read_text(encoding="utf-8")
    export_daily_windows(
        client=create_client(args.env_file),
        sql_template_text=sql_template_text,
        output_path=Path(args.output),
        date_from=args.date_from,
        date_to=args.date_to,
        batch_mode=args.batch_mode,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

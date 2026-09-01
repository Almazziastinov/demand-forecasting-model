"""Read-only ClickHouse extraction of network hourly SKU sales for backtests."""

from __future__ import annotations

import argparse
from pathlib import Path

import clickhouse_connect
import pandas as pd
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
SALE_HEX = "D09FD180D0BED0B4D0B0D0B6D0B0"


def client_from_env(path: Path):
    values = dotenv_values(path)
    prefix = "CLICKHOUSE_" if "CLICKHOUSE_HOST" in values else ""
    return clickhouse_connect.get_client(
        host=values[f"{prefix}HOST"],
        port=int(values[f"{prefix}PORT"]),
        username=values[f"{prefix}USER"],
        password=values[f"{prefix}PASSWORD"],
        database=values[f"{prefix}DATABASE"],
        secure=True,
        verify=False,
        connect_timeout=30,
        send_receive_timeout=300,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--date-from", default="2026-06-01")
    parser.add_argument("--date-to", default="2026-08-23")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".codex_tmp/rolling_hourly_sales_20260601_20260823.parquet",
    )
    args = parser.parse_args()

    operational = pd.read_parquet(
        ROOT / "reports/calibrated_quantile_operational_balance_20260826/rows.parquet",
        columns=["bakery_id"],
    )
    bakery_ids = tuple(sorted(operational["bakery_id"].astype(int).unique()))
    query = f"""
        select check_date date,
               toInt64OrZero(toString(bakery_id)) bakery_id,
               toInt64OrZero(toString(product_id)) product_id,
               toHour(check_datetime) hour,
               sum(toFloat64(quantity)) sold
        from (
            select distinct check_datetime, check_date, bakery_id, product_id,
                   quantity, price, line_amount, cash_event_type
            from Svezhar.fct_check_lines
            where hex(cash_event_type) = '{SALE_HEX}'
              and check_date between toDate(%(date_from)s) and toDate(%(date_to)s)
              and toInt64OrZero(toString(bakery_id)) in %(bakery_ids)s
              and quantity > 0
        )
        group by date, bakery_id, product_id, hour
    """
    frame = client_from_env(args.env_file).query_df(
        query,
        parameters={
            "date_from": args.date_from,
            "date_to": args.date_to,
            "bakery_ids": bakery_ids,
        },
    )
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print(
        f"rows={len(frame)} dates={frame['date'].min().date()}..{frame['date'].max().date()} "
        f"bakeries={frame['bakery_id'].nunique()} products={frame['product_id'].nunique()} "
        f"sold={frame['sold'].sum():.3f}"
    )


if __name__ == "__main__":
    main()

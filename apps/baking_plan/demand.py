"""Live-data reads this package needs beyond the assortment allowlist:
hourly SKU forecast (today + next day, by product_id) and the bakery's
revenue tier (for base-template sheet selection).
"""

from __future__ import annotations

# ruff: noqa: E501
from ._clickhouse import get_client, records, table_name

SKU_FORECAST_HOUR_TABLE = table_name("sku_forecast_hour_embedded")
REVENUE_TABLE = table_name("bakery_month_revenue_embedded")


def load_hourly(
    run_id: str, forecast_date: str, bakery_id: int, product_ids: list[str]
) -> dict[str, dict[int, float]]:
    """Per-(product_id, hour) forecast_qty for one bakery/date."""
    if not product_ids:
        return {}
    client = get_client()
    pid_by_int = {int(pid): pid for pid in product_ids}
    df = client.query_df(
        f"""
        select product_id, hour, sum(forecast_qty) as qty
        from {SKU_FORECAST_HOUR_TABLE}
        where run_id = %(run_id)s
          and forecast_date = %(forecast_date)s
          and bakery_id = %(bakery_id)s
          and product_id in %(pids)s
        group by product_id, hour
        """,
        parameters={
            "run_id": run_id,
            "forecast_date": forecast_date,
            "bakery_id": bakery_id,
            "pids": list(pid_by_int.keys()),
        },
    )
    result: dict[str, dict[int, float]] = {}
    for row in records(df):
        pid = pid_by_int.get(int(row["product_id"]))
        if pid is None:
            continue
        result.setdefault(pid, {})[int(row["hour"])] = float(row["qty"])
    return result


def load_revenue_bucket_input(bakery_id: int) -> float | None:
    """Latest known monthly revenue for this bakery, or None if unknown.

    Feeds `templates.revenue_bucket()`; a stale/missing value falls back to
    `templates.DEFAULT_BUCKET` there, not an error — this data is a one-time
    manual backfill (see `docs/ops/DECISIONS.md`), not a live feed.
    """
    client = get_client()
    df = client.query_df(
        f"""
        select revenue
        from {REVENUE_TABLE} final
        where bakery_id = %(bakery_id)s
        order by month_start desc
        limit 1
        """,
        parameters={"bakery_id": bakery_id},
    )
    rows = records(df)
    return float(rows[0]["revenue"]) if rows else None

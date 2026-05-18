from __future__ import annotations

from app.db import get_client


BAKERY_DAY_TABLE = "bakery_forecast_day_embedded"
SKU_DAY_TABLE = "sku_forecast_day_embedded"
SKU_HOUR_TABLE = "sku_forecast_hour_embedded"


def _records(df):
    rows = []
    for record in df.to_dict("records"):
        normalized = {}
        for key, value in record.items():
            if hasattr(value, "isoformat"):
                normalized[key] = value.isoformat()
            else:
                normalized[key] = value
        rows.append(normalized)
    return rows


def get_bakery_list(run_id: str, forecast_date: str) -> list[dict]:
    client = get_client()
    query = """
        select bakery_id, bakery_name, city, forecast_final
        from {table}
        where run_id = %(run_id)s
          and forecast_date = %(forecast_date)s
        order by forecast_final desc, bakery_name asc
        """.format(table=BAKERY_DAY_TABLE)
    df = client.query_df(query, parameters={"run_id": run_id, "forecast_date": forecast_date})
    return _records(df)


def get_bakery_day(run_id: str, forecast_date: str, bakery_id: int) -> dict | None:
    client = get_client()
    query = """
        select bakery_id, bakery_name, city, forecast_base, forecast_final
        from {table}
        where run_id = %(run_id)s
          and forecast_date = %(forecast_date)s
          and bakery_id = %(bakery_id)s
        limit 1
        """.format(table=BAKERY_DAY_TABLE)
    df = client.query_df(
        query,
        parameters={
            "run_id": run_id,
            "forecast_date": forecast_date,
            "bakery_id": bakery_id,
        },
    )
    if df.empty:
        return None
    return _records(df)[0]


def get_top_sku(run_id: str, forecast_date: str, bakery_id: int, limit: int = 20) -> list[dict]:
    client = get_client()
    query = """
        select product_id, product_name, category_name, forecast_qty
        from {table}
        where run_id = %(run_id)s
          and forecast_date = %(forecast_date)s
          and bakery_id = %(bakery_id)s
        order by forecast_qty desc, product_name asc
        limit %(limit)s
        """.format(table=SKU_DAY_TABLE)
    df = client.query_df(
        query,
        parameters={
            "run_id": run_id,
            "forecast_date": forecast_date,
            "bakery_id": bakery_id,
            "limit": limit,
        },
    )
    return _records(df)


def get_hourly_total(run_id: str, forecast_date: str, bakery_id: int) -> list[dict]:
    client = get_client()
    query = """
        select hour, sum(forecast_qty) as forecast_qty
        from {table}
        where run_id = %(run_id)s
          and forecast_date = %(forecast_date)s
          and bakery_id = %(bakery_id)s
        group by hour
        order by hour
        """.format(table=SKU_HOUR_TABLE)
    df = client.query_df(
        query,
        parameters={
            "run_id": run_id,
            "forecast_date": forecast_date,
            "bakery_id": bakery_id,
        },
    )
    return _records(df)


def get_sku_hour(run_id: str, forecast_date: str, bakery_id: int, product_id: int) -> list[dict]:
    client = get_client()
    query = """
        select h.hour, h.product_id, d.product_name, h.forecast_qty
        from {hour_table} h
        left join {day_table} d
          on d.run_id = h.run_id
         and d.forecast_date = h.forecast_date
         and d.bakery_id = h.bakery_id
         and d.product_id = h.product_id
        where h.run_id = %(run_id)s
          and h.forecast_date = %(forecast_date)s
          and h.bakery_id = %(bakery_id)s
          and h.product_id = %(product_id)s
        order by h.hour
        """.format(hour_table=SKU_HOUR_TABLE, day_table=SKU_DAY_TABLE)
    df = client.query_df(
        query,
        parameters={
            "run_id": run_id,
            "forecast_date": forecast_date,
            "bakery_id": bakery_id,
            "product_id": product_id,
        },
    )
    return _records(df)

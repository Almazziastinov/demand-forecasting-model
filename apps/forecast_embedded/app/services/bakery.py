from __future__ import annotations

# ruff: noqa: E501

from app.auth import AuthContext
from app.db import get_client


BAKERY_DAY_TABLE = "bakery_forecast_day_embedded"
CONTEXT_TABLE = "forecast_day_context_embedded"
SKU_DAY_TABLE = "sku_forecast_day_embedded"
SKU_HOUR_TABLE = "sku_forecast_hour_embedded"
ACCESS_TABLE = "bitrix_user_bakery_access_embedded"
MANAGEMENT_TABLE = "dim_management"
CLOSED_BAKERY_STATUS = "\u0417\u0430\u043a\u0440\u044b\u0442\u0430"


def _records(df):
    rows = []
    for record in df.to_dict("records"):
        normalized = {}
        for key, value in record.items():
            if value != value:
                normalized[key] = None
            elif hasattr(value, "isoformat"):
                normalized[key] = value.isoformat()
            else:
                normalized[key] = value
        rows.append(normalized)
    return rows


def _access_filter(auth: AuthContext, bakery_expr: str) -> tuple[str, dict]:
    if auth.is_admin:
        return "", {}
    return (
        f"""
          and {bakery_expr} in (
            select bakery_id
            from {ACCESS_TABLE}
            where bitrix_portal_id = %(bitrix_portal_id)s
              and (
                bitrix_user_id = %(bitrix_user_id)s
                or (%(bitrix_email)s != '' and bitrix_email = %(bitrix_email)s)
              )
          )
        """,
        {
            "bitrix_portal_id": auth.portal_id or "",
            "bitrix_user_id": auth.user_id or "",
            "bitrix_email": auth.email or "",
        },
    )


def _open_bakery_filter(bakery_expr: str) -> str:
    return f"""
          and {bakery_expr} in (
            select toInt64OrNull(toString(bakery_id))
            from {MANAGEMENT_TABLE}
            where coalesce(status, '') != %(closed_bakery_status)s
              and toInt64OrNull(toString(bakery_id)) is not null
          )
        """


def get_bakery_list(run_id: str, forecast_date: str, auth: AuthContext) -> list[dict]:
    client = get_client()
    access_sql, access_params = _access_filter(auth, "b.bakery_id")
    open_bakery_sql = _open_bakery_filter("b.bakery_id")
    query = """
        select
            b.bakery_id,
            b.bakery_name,
            b.city,
            b.forecast_final,
            c.temp_mean,
            c.precipitation,
            c.snowfall,
            c.is_bad_weather,
            c.holiday_name,
            c.is_holiday,
            c.is_pre_holiday,
            c.is_post_holiday,
            c.event_window_type
        from {table} b
        left join {context_table} c
          on c.run_id = b.run_id
         and c.forecast_date = b.forecast_date
         and c.city = b.city
        where b.run_id = %(run_id)s
          and b.forecast_date = %(forecast_date)s
          {open_bakery_sql}
          {access_sql}
        order by forecast_final desc, bakery_name asc
        """.format(
        table=BAKERY_DAY_TABLE,
        context_table=CONTEXT_TABLE,
        open_bakery_sql=open_bakery_sql,
        access_sql=access_sql,
    )
    df = client.query_df(
        query,
        parameters={
            "run_id": run_id,
            "forecast_date": forecast_date,
            "closed_bakery_status": CLOSED_BAKERY_STATUS,
            **access_params,
        },
    )
    return _records(df)


def get_bakery_day(
    run_id: str,
    forecast_date: str,
    bakery_id: int,
    auth: AuthContext,
) -> dict | None:
    client = get_client()
    access_sql, access_params = _access_filter(auth, "bakery_id")
    open_bakery_sql = _open_bakery_filter("bakery_id")
    query = """
        select bakery_id, bakery_name, city, forecast_base, forecast_final
        from {table}
        where run_id = %(run_id)s
          and forecast_date = %(forecast_date)s
          and bakery_id = %(bakery_id)s
          {open_bakery_sql}
          {access_sql}
        limit 1
        """.format(
        table=BAKERY_DAY_TABLE,
        open_bakery_sql=open_bakery_sql,
        access_sql=access_sql,
    )
    df = client.query_df(
        query,
        parameters={
            "run_id": run_id,
            "forecast_date": forecast_date,
            "bakery_id": bakery_id,
            "closed_bakery_status": CLOSED_BAKERY_STATUS,
            **access_params,
        },
    )
    if df.empty:
        return None
    return _records(df)[0]


def get_day_context(run_id: str, forecast_date: str, city: str | None) -> dict | None:
    client = get_client()
    query = """
        select
            city,
            temp_mean,
            precipitation,
            rain,
            snowfall,
            windspeed_max,
            is_bad_weather,
            weather_cat_code,
            holiday_name,
            is_holiday,
            is_pre_holiday,
            is_post_holiday,
            event_window_type,
            current_event_cluster,
            prev_event_cluster,
            next_event_cluster,
            days_since_prev_event,
            days_to_next_event
        from {table}
        where run_id = %(run_id)s
          and forecast_date = %(forecast_date)s
          and city = %(city)s
        limit 1
        """.format(table=CONTEXT_TABLE)
    df = client.query_df(
        query,
        parameters={
            "run_id": run_id,
            "forecast_date": forecast_date,
            "city": city or "unknown",
        },
    )
    if df.empty:
        return None
    return _records(df)[0]


def get_top_sku(
    run_id: str,
    forecast_date: str,
    bakery_id: int,
    auth: AuthContext,
    limit: int = 20,
) -> list[dict]:
    client = get_client()
    access_sql, access_params = _access_filter(auth, "bakery_id")
    open_bakery_sql = _open_bakery_filter("bakery_id")
    query = """
        select product_id, product_name, category_name, forecast_qty
        from {table}
        where run_id = %(run_id)s
          and forecast_date = %(forecast_date)s
          and bakery_id = %(bakery_id)s
          {open_bakery_sql}
          {access_sql}
        order by forecast_qty desc, product_name asc
        limit %(limit)s
        """.format(
        table=SKU_DAY_TABLE,
        open_bakery_sql=open_bakery_sql,
        access_sql=access_sql,
    )
    df = client.query_df(
        query,
        parameters={
            "run_id": run_id,
            "forecast_date": forecast_date,
            "bakery_id": bakery_id,
            "limit": limit,
            "closed_bakery_status": CLOSED_BAKERY_STATUS,
            **access_params,
        },
    )
    return _records(df)


def get_hourly_total(
    run_id: str,
    forecast_date: str,
    bakery_id: int,
    auth: AuthContext,
) -> list[dict]:
    client = get_client()
    access_sql, access_params = _access_filter(auth, "bakery_id")
    open_bakery_sql = _open_bakery_filter("bakery_id")
    query = """
        select hour, sum(forecast_qty) as forecast_qty
        from {table}
        where run_id = %(run_id)s
          and forecast_date = %(forecast_date)s
          and bakery_id = %(bakery_id)s
          {open_bakery_sql}
          {access_sql}
        group by hour
        order by hour
        """.format(
        table=SKU_HOUR_TABLE,
        open_bakery_sql=open_bakery_sql,
        access_sql=access_sql,
    )
    df = client.query_df(
        query,
        parameters={
            "run_id": run_id,
            "forecast_date": forecast_date,
            "bakery_id": bakery_id,
            "closed_bakery_status": CLOSED_BAKERY_STATUS,
            **access_params,
        },
    )
    return _records(df)


def get_sku_hour(
    run_id: str,
    forecast_date: str,
    bakery_id: int,
    product_id: int,
    auth: AuthContext,
) -> list[dict]:
    client = get_client()
    access_sql, access_params = _access_filter(auth, "h.bakery_id")
    open_bakery_sql = _open_bakery_filter("h.bakery_id")
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
          {open_bakery_sql}
          {access_sql}
        order by h.hour
        """.format(
        hour_table=SKU_HOUR_TABLE,
        day_table=SKU_DAY_TABLE,
        open_bakery_sql=open_bakery_sql,
        access_sql=access_sql,
    )
    df = client.query_df(
        query,
        parameters={
            "run_id": run_id,
            "forecast_date": forecast_date,
            "bakery_id": bakery_id,
            "product_id": product_id,
            "closed_bakery_status": CLOSED_BAKERY_STATUS,
            **access_params,
        },
    )
    return _records(df)

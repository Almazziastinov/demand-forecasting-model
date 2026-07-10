from __future__ import annotations

# ruff: noqa: E501
import logging

from app.auth import AuthContext
from app.db import get_client
from app.table_names import table_name

BAKERY_DAY_TABLE = table_name("bakery_forecast_day_embedded")
BAKERY_DAY_SNAPSHOT_TABLE = table_name("bakery_forecast_day_snapshots")
CONTEXT_TABLE = table_name("forecast_day_context_embedded")
SKU_DAY_TABLE = table_name("sku_forecast_day_embedded")
SKU_DAY_SNAPSHOT_TABLE = table_name("sku_forecast_day_snapshots")
SKU_HOUR_TABLE = table_name("sku_forecast_hour_embedded")
SKU_HOUR_SNAPSHOT_TABLE = table_name("sku_forecast_hour_snapshots")
SALES_LINE_TABLE = "mart_sales_60d"
RAW_SALES_LINE_TABLE = "Svezhar.fct_check_lines"
SALES_EVENT_HEX = "D09FD180D0BED0B4D0B0D0B6D0B0"
ACCESS_TABLE = table_name("bitrix_user_bakery_access_embedded")
BAKERIES_DIM_TABLE = "dim_bakeries"
logger = logging.getLogger(__name__)
CLOSED_BAKERY_STATUS = "\u0417\u0430\u043a\u0440\u044b\u0442\u0430"
ACTIVE_ROW_SORT_KEY = "tuple(2, toDateTime64('2100-01-01 00:00:00', 3))"
SNAPSHOT_ROW_SORT_KEY = "tuple(1, generated_at)"
DISCREPANCY_EPSILON = 0.000001
SCENARIO_SOURCE_PATTERNS: dict[str, str] = {
    "base_raw_uplift": "%_raw_uplift_%",
    "base_no_sku_uplift": "%_no_sku_uplift_%",
    "uplifted_norm": "%uplifted_bakery%",
}


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
            select distinct toInt64OrNull(toString(bakery_id))
            from {BAKERIES_DIM_TABLE}
            where toInt64OrNull(toString(bakery_id)) is not null
          )
        """


def _bakery_day_source(date_filter_sql: str, snapshot_scenario_like: str | None = None) -> str:
    scenario_clause = "and source_run_id LIKE %(snapshot_scenario)s" if snapshot_scenario_like else ""
    return """
        (
            select
                forecast_date,
                bakery_id,
                argMax(run_id, sort_key) as run_id,
                argMax(bakery_name, sort_key) as bakery_name,
                argMax(city, sort_key) as city,
                argMax(forecast_base, sort_key) as forecast_base,
                argMax(forecast_final, sort_key) as forecast_final
            from (
                select
                    run_id,
                    forecast_date,
                    bakery_id,
                    bakery_name,
                    city,
                    forecast_base,
                    forecast_final,
                    {active_sort_key} as sort_key
                from {active_table}
                where run_id = %(run_id)s
                  and {date_filter_sql}
                union all
                select
                    source_run_id as run_id,
                    forecast_date,
                    bakery_id,
                    bakery_name,
                    city,
                    forecast_base,
                    forecast_final,
                    {snapshot_sort_key} as sort_key
                from {snapshot_table}
                where lead_days = 1
                  and {date_filter_sql}
                  {scenario_clause}
            )
            group by forecast_date, bakery_id
        )
        """.format(
        active_table=BAKERY_DAY_TABLE,
        snapshot_table=BAKERY_DAY_SNAPSHOT_TABLE,
        active_sort_key=ACTIVE_ROW_SORT_KEY,
        snapshot_sort_key=SNAPSHOT_ROW_SORT_KEY,
        date_filter_sql=date_filter_sql,
        scenario_clause=scenario_clause,
    )


def _sku_bakery_total_source(date_filter_sql: str, snapshot_scenario_like: str | None = None) -> str:
    """Sum of SKU-level forecasts aggregated to bakery-day — used as forecast_final."""
    return """
        (
            select
                forecast_date,
                bakery_id,
                sum(forecast_qty) as forecast_sku_total
            from {sku_source}
            group by forecast_date, bakery_id
        )
        """.format(
        sku_source=_sku_day_source(date_filter_sql, snapshot_scenario_like=snapshot_scenario_like),
    )


def _sku_day_source(date_filter_sql: str, snapshot_scenario_like: str | None = None) -> str:
    scenario_clause = "and source_run_id LIKE %(snapshot_scenario)s" if snapshot_scenario_like else ""
    return """
        (
            select
                forecast_date,
                bakery_id,
                product_id,
                argMax(run_id, sort_key) as run_id,
                argMax(product_name, sort_key) as product_name,
                argMax(category_name, sort_key) as category_name,
                argMax(forecast_qty, sort_key) as forecast_qty
            from (
                select
                    run_id,
                    forecast_date,
                    bakery_id,
                    product_id,
                    product_name,
                    category_name,
                    forecast_qty,
                    {active_sort_key} as sort_key
                from {active_table}
                where run_id = %(run_id)s
                  and {date_filter_sql}
                union all
                select
                    source_run_id as run_id,
                    forecast_date,
                    bakery_id,
                    product_id,
                    product_name,
                    category_name,
                    forecast_qty,
                    {snapshot_sort_key} as sort_key
                from {snapshot_table}
                where lead_days = 1
                  and {date_filter_sql}
                  {scenario_clause}
            )
            group by forecast_date, bakery_id, product_id
        )
        """.format(
        active_table=SKU_DAY_TABLE,
        snapshot_table=SKU_DAY_SNAPSHOT_TABLE,
        active_sort_key=ACTIVE_ROW_SORT_KEY,
        snapshot_sort_key=SNAPSHOT_ROW_SORT_KEY,
        date_filter_sql=date_filter_sql,
        scenario_clause=scenario_clause,
    )


def _sku_hour_source(date_filter_sql: str, snapshot_scenario_like: str | None = None) -> str:
    scenario_clause = "and source_run_id LIKE %(snapshot_scenario)s" if snapshot_scenario_like else ""
    return """
        (
            select
                forecast_date,
                bakery_id,
                product_id,
                hour,
                argMax(run_id, sort_key) as run_id,
                argMax(forecast_qty, sort_key) as forecast_qty
            from (
                select
                    run_id,
                    forecast_date,
                    bakery_id,
                    product_id,
                    hour,
                    forecast_qty,
                    {active_sort_key} as sort_key
                from {active_table}
                where run_id = %(run_id)s
                  and {date_filter_sql}
                union all
                select
                    source_run_id as run_id,
                    forecast_date,
                    bakery_id,
                    product_id,
                    hour,
                    forecast_qty,
                    {snapshot_sort_key} as sort_key
                from {snapshot_table}
                where lead_days = 1
                  and {date_filter_sql}
                  {scenario_clause}
            )
            group by forecast_date, bakery_id, product_id, hour
        )
        """.format(
        active_table=SKU_HOUR_TABLE,
        snapshot_table=SKU_HOUR_SNAPSHOT_TABLE,
        active_sort_key=ACTIVE_ROW_SORT_KEY,
        snapshot_sort_key=SNAPSHOT_ROW_SORT_KEY,
        date_filter_sql=date_filter_sql,
        scenario_clause=scenario_clause,
    )


def _raw_sales_source(filter_sql: str) -> str:
    return """
        (
            select distinct
                fcl.check_datetime,
                fcl.check_date,
                fcl.bakery_id,
                fcl.product_id,
                fcl.quantity,
                fcl.price,
                fcl.line_amount,
                fcl.cash_event_type
            from {raw_sales_line_table} fcl
            where hex(fcl.cash_event_type) = %(sales_event_hex)s
              and {filter_sql}
        )
        """.format(
        raw_sales_line_table=SALES_LINE_TABLE,
        filter_sql=filter_sql,
    )


def get_bakery_list(run_id: str, forecast_date: str, auth: AuthContext, scenario: str | None = None) -> list[dict]:
    client = get_client()
    scenario_like = SCENARIO_SOURCE_PATTERNS.get(scenario) if scenario else None
    access_sql, access_params = _access_filter(auth, "b.bakery_id")
    open_bakery_sql = _open_bakery_filter("b.bakery_id")
    query = """
        with sales as (
            select
                toInt64OrNull(toString(fcl.bakery_id)) as sales_bakery_id,
                sum(toFloat64(fcl.quantity)) as actual_qty,
                sum(ifNull(toFloat64(fcl.line_amount), 0.0)) as actual_revenue
            from {sales_source} fcl
            group by sales_bakery_id
        )
        select
            b.bakery_id as bakery_id,
            b.bakery_name as bakery_name,
            b.city as city,
            coalesce(sku.forecast_sku_total, b.forecast_final) as forecast_final,
            sales.actual_qty as actual_qty,
            sales.actual_revenue as actual_revenue,
            c.temp_mean as temp_mean,
            c.precipitation as precipitation,
            c.snowfall as snowfall,
            c.is_bad_weather as is_bad_weather,
            c.holiday_name as holiday_name,
            c.is_holiday as is_holiday,
            c.is_pre_holiday as is_pre_holiday,
            c.is_post_holiday as is_post_holiday,
            c.event_window_type as event_window_type
        from {source} b
        left join {sku_total_source} sku
          on sku.forecast_date = b.forecast_date
         and sku.bakery_id = b.bakery_id
        left join {context_table} c
          on c.run_id = b.run_id
         and c.forecast_date = b.forecast_date
         and c.city = b.city
        left join sales
          on sales.sales_bakery_id = b.bakery_id
        where b.forecast_date = %(forecast_date)s
          {open_bakery_sql}
          {access_sql}
        order by forecast_final desc, bakery_name asc
        """.format(
        source=_bakery_day_source("forecast_date = %(forecast_date)s", snapshot_scenario_like=scenario_like),
        sku_total_source=_sku_bakery_total_source("forecast_date = %(forecast_date)s", snapshot_scenario_like=scenario_like),
        sales_source=_raw_sales_source("fcl.check_date = toDate(%(forecast_date)s)"),
        context_table=CONTEXT_TABLE,
        open_bakery_sql=open_bakery_sql,
        access_sql=access_sql,
    )
    df = client.query_df(
        query,
        parameters={
            "run_id": run_id,
            "forecast_date": forecast_date,
            "sales_event_hex": SALES_EVENT_HEX,
            "closed_bakery_status": CLOSED_BAKERY_STATUS,
            **({"snapshot_scenario": scenario_like} if scenario_like else {}),
            **access_params,
        },
    )
    return _records(df)


def get_bakery_week(
    run_id: str,
    start_date: str,
    end_date: str,
    bakery_id: int,
    auth: AuthContext,
    scenario: str | None = None,
) -> list[dict]:
    client = get_client()
    scenario_like = SCENARIO_SOURCE_PATTERNS.get(scenario) if scenario else None
    access_sql, access_params = _access_filter(auth, "b.bakery_id")
    open_bakery_sql = _open_bakery_filter("b.bakery_id")
    query = """
        with sales as (
            select
                fcl.check_date as forecast_date,
                toInt64OrNull(toString(fcl.bakery_id)) as sales_bakery_id,
                sum(toFloat64(fcl.quantity)) as actual_qty,
                sum(ifNull(toFloat64(fcl.line_amount), 0.0)) as actual_revenue
            from {sales_source} fcl
            group by forecast_date, sales_bakery_id
        )
        select
            b.bakery_id as bakery_id,
            b.bakery_name as bakery_name,
            b.city as city,
            b.forecast_date as forecast_date,
            b.forecast_base as forecast_base,
            coalesce(sku.forecast_sku_total, b.forecast_final) as forecast_final,
            sales.actual_qty as actual_qty,
            sales.actual_revenue as actual_revenue,
            c.temp_mean as temp_mean,
            c.precipitation as precipitation,
            c.rain as rain,
            c.snowfall as snowfall,
            c.windspeed_max as windspeed_max,
            c.is_bad_weather as is_bad_weather,
            c.holiday_name as holiday_name,
            c.is_holiday as is_holiday,
            c.is_pre_holiday as is_pre_holiday,
            c.is_post_holiday as is_post_holiday,
            c.event_window_type as event_window_type
        from {source} b
        left join {sku_total_source} sku
          on sku.forecast_date = b.forecast_date
         and sku.bakery_id = b.bakery_id
        left join {context_table} c
          on c.run_id = b.run_id
         and c.forecast_date = b.forecast_date
         and c.city = b.city
        left join sales
          on sales.forecast_date = b.forecast_date
         and sales.sales_bakery_id = b.bakery_id
        where b.forecast_date between %(start_date)s and %(end_date)s
          and b.bakery_id = %(bakery_id)s
          {open_bakery_sql}
          {access_sql}
        order by b.forecast_date
        """.format(
        source=_bakery_day_source("forecast_date between %(start_date)s and %(end_date)s", snapshot_scenario_like=scenario_like),
        sku_total_source=_sku_bakery_total_source("forecast_date between %(start_date)s and %(end_date)s", snapshot_scenario_like=scenario_like),
        sales_source=_raw_sales_source("fcl.check_date between toDate(%(start_date)s) and toDate(%(end_date)s)"),
        context_table=CONTEXT_TABLE,
        open_bakery_sql=open_bakery_sql,
        access_sql=access_sql,
    )
    df = client.query_df(
        query,
        parameters={
            "run_id": run_id,
            "start_date": start_date,
            "end_date": end_date,
            "bakery_id": bakery_id,
            "sales_event_hex": SALES_EVENT_HEX,
            "closed_bakery_status": CLOSED_BAKERY_STATUS,
            **({"snapshot_scenario": scenario_like} if scenario_like else {}),
            **access_params,
        },
    )
    return _records(df)


def get_bakery_day(
    run_id: str,
    forecast_date: str,
    bakery_id: int,
    auth: AuthContext,
    scenario: str | None = None,
) -> dict | None:
    client = get_client()
    scenario_like = SCENARIO_SOURCE_PATTERNS.get(scenario) if scenario else None
    access_sql, access_params = _access_filter(auth, "b.bakery_id")
    open_bakery_sql = _open_bakery_filter("b.bakery_id")
    query = """
        with sales as (
            select
                toInt64OrNull(toString(fcl.bakery_id)) as sales_bakery_id,
                sum(toFloat64(fcl.quantity)) as actual_qty,
                sum(ifNull(toFloat64(fcl.line_amount), 0.0)) as actual_revenue
            from {sales_source} fcl
            group by sales_bakery_id
        )
        select b.bakery_id as bakery_id, b.bakery_name as bakery_name, b.city as city,
               b.forecast_base as forecast_base,
               coalesce(sku.forecast_sku_total, b.forecast_final) as forecast_final,
               sales.actual_qty as actual_qty,
               sales.actual_revenue as actual_revenue
        from {source} b
        left join {sku_total_source} sku
          on sku.forecast_date = b.forecast_date
         and sku.bakery_id = b.bakery_id
        left join sales on sales.sales_bakery_id = b.bakery_id
        where b.forecast_date = %(forecast_date)s
          and b.bakery_id = %(bakery_id)s
          {open_bakery_sql}
          {access_sql}
        limit 1
        """.format(
        source=_bakery_day_source("forecast_date = %(forecast_date)s", snapshot_scenario_like=scenario_like),
        sku_total_source=_sku_bakery_total_source("forecast_date = %(forecast_date)s", snapshot_scenario_like=scenario_like),
        sales_source=_raw_sales_source("fcl.check_date = toDate(%(forecast_date)s)"),
        open_bakery_sql=open_bakery_sql,
        access_sql=access_sql,
    )
    df = client.query_df(
        query,
        parameters={
            "run_id": run_id,
            "forecast_date": forecast_date,
            "bakery_id": bakery_id,
            "sales_event_hex": SALES_EVENT_HEX,
            "closed_bakery_status": CLOSED_BAKERY_STATUS,
            **({"snapshot_scenario": scenario_like} if scenario_like else {}),
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
    category: str | None = None,
    scenario: str | None = None,
) -> list[dict]:
    client = get_client()
    scenario_like = SCENARIO_SOURCE_PATTERNS.get(scenario) if scenario else None
    access_sql, access_params = _access_filter(auth, "d.bakery_id")
    open_bakery_sql = _open_bakery_filter("d.bakery_id")
    category_sql = "and coalesce(d.category_name, '') = %(category)s" if category else ""
    query = """
        with sales as (
            select
                toInt64OrNull(toString(fcl.product_id)) as sales_product_id,
                sum(toFloat64(fcl.quantity)) as actual_qty,
                sum(ifNull(toFloat64(fcl.line_amount), 0.0)) as actual_revenue
            from {sales_source} fcl
            group by sales_product_id
        )
        select d.product_id as product_id,
               d.product_name as product_name,
               d.category_name as category_name,
               d.forecast_qty as forecast_qty,
               sales.actual_qty as actual_qty,
               sales.actual_revenue as actual_revenue
        from {source} d
        left join sales on sales.sales_product_id = d.product_id
        where d.forecast_date = %(forecast_date)s
          and d.bakery_id = %(bakery_id)s
          {category_sql}
          {open_bakery_sql}
          {access_sql}
        order by d.forecast_qty desc, d.product_name asc
        limit %(limit)s
        """.format(
        source=_sku_day_source("forecast_date = %(forecast_date)s", snapshot_scenario_like=scenario_like),
        sales_source=_raw_sales_source(
            "fcl.check_date = toDate(%(forecast_date)s) "
            "and toInt64OrNull(toString(fcl.bakery_id)) = %(bakery_id)s"
        ),
        category_sql=category_sql,
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
            "category": category or "",
            "sales_event_hex": SALES_EVENT_HEX,
            "closed_bakery_status": CLOSED_BAKERY_STATUS,
            **({"snapshot_scenario": scenario_like} if scenario_like else {}),
            **access_params,
        },
    )
    return _records(df)


def get_categories(run_id: str, forecast_date: str, bakery_id: int, auth: AuthContext, scenario: str | None = None) -> list[str]:
    client = get_client()
    scenario_like = SCENARIO_SOURCE_PATTERNS.get(scenario) if scenario else None
    access_sql, access_params = _access_filter(auth, "d.bakery_id")
    open_bakery_sql = _open_bakery_filter("d.bakery_id")
    query = """
        select distinct coalesce(category_name, 'Без группы') as category_name
        from {source} d
        where d.forecast_date = %(forecast_date)s
          and d.bakery_id = %(bakery_id)s
          {open_bakery_sql}
          {access_sql}
        order by category_name
        """.format(
        source=_sku_day_source("forecast_date = %(forecast_date)s", snapshot_scenario_like=scenario_like),
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
            **({"snapshot_scenario": scenario_like} if scenario_like else {}),
            **access_params,
        },
    )
    if df.empty:
        return []
    return [str(value) for value in df["category_name"].tolist()]


def get_hourly_total(
    run_id: str,
    forecast_date: str,
    bakery_id: int,
    auth: AuthContext,
    scenario: str | None = None,
) -> list[dict]:
    client = get_client()
    scenario_like = SCENARIO_SOURCE_PATTERNS.get(scenario) if scenario else None
    access_sql, access_params = _access_filter(auth, "bakery_id")
    open_bakery_sql = _open_bakery_filter("bakery_id")
    query = """
        with forecast as (
            select hour, sum(forecast_qty) as forecast_qty
            from {source}
            where forecast_date = %(forecast_date)s
              and bakery_id = %(bakery_id)s
              {open_bakery_sql}
              {access_sql}
            group by hour
        ),
        actual as (
            select
                toHour(fcl.check_datetime) as hour,
                sum(toFloat64(fcl.quantity)) as actual_qty
            from {sales_source} fcl
            group by hour
        )
        select
            coalesce(forecast.hour, actual.hour) as hour,
            forecast.forecast_qty as forecast_qty,
            actual.actual_qty as actual_qty
        from forecast
        full outer join actual on actual.hour = forecast.hour
        order by hour
        """.format(
        source=_sku_hour_source("forecast_date = %(forecast_date)s", snapshot_scenario_like=scenario_like),
        sales_source=_raw_sales_source(
            "fcl.check_date = toDate(%(forecast_date)s) "
            "and toInt64OrNull(toString(fcl.bakery_id)) = %(bakery_id)s"
        ),
        open_bakery_sql=open_bakery_sql,
        access_sql=access_sql,
    )
    df = client.query_df(
        query,
        parameters={
            "run_id": run_id,
            "forecast_date": forecast_date,
            "bakery_id": bakery_id,
            "sales_event_hex": SALES_EVENT_HEX,
            "closed_bakery_status": CLOSED_BAKERY_STATUS,
            **({"snapshot_scenario": scenario_like} if scenario_like else {}),
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
    scenario: str | None = None,
) -> list[dict]:
    client = get_client()
    scenario_like = SCENARIO_SOURCE_PATTERNS.get(scenario) if scenario else None
    access_sql, access_params = _access_filter(auth, "h.bakery_id")
    open_bakery_sql = _open_bakery_filter("h.bakery_id")
    query = """
        with forecast as (
            select h.hour, h.product_id, d.product_name, d.category_name, h.forecast_qty
            from {hour_source} h
            left join {day_source} d
              on d.run_id = h.run_id
             and d.forecast_date = h.forecast_date
             and d.bakery_id = h.bakery_id
             and d.product_id = h.product_id
            where h.forecast_date = %(forecast_date)s
              and h.bakery_id = %(bakery_id)s
              and h.product_id = %(product_id)s
              {open_bakery_sql}
              {access_sql}
        ),
        actual as (
            select
                toHour(fcl.check_datetime) as hour,
                toInt64OrNull(toString(fcl.product_id)) as product_id,
                sum(toFloat64(fcl.quantity)) as actual_qty
            from {sales_source} fcl
            group by hour, product_id
        )
        select
            coalesce(forecast.hour, actual.hour) as hour,
            coalesce(forecast.product_id, actual.product_id) as product_id,
            forecast.product_name as product_name,
            forecast.category_name as category_name,
            forecast.forecast_qty as forecast_qty,
            actual.actual_qty as actual_qty
        from forecast
        full outer join actual on actual.hour = forecast.hour
        order by hour
        """.format(
        hour_source=_sku_hour_source("forecast_date = %(forecast_date)s", snapshot_scenario_like=scenario_like),
        day_source=_sku_day_source("forecast_date = %(forecast_date)s", snapshot_scenario_like=scenario_like),
        sales_source=_raw_sales_source(
            "fcl.check_date = toDate(%(forecast_date)s) "
            "and toInt64OrNull(toString(fcl.bakery_id)) = %(bakery_id)s "
            "and toInt64OrNull(toString(fcl.product_id)) = %(product_id)s"
        ),
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
            "sales_event_hex": SALES_EVENT_HEX,
            "closed_bakery_status": CLOSED_BAKERY_STATUS,
            **({"snapshot_scenario": scenario_like} if scenario_like else {}),
            **access_params,
        },
    )
    return _records(df)


def get_sku_hour_forecast(
    run_id: str,
    forecast_date: str,
    bakery_id: int,
    auth: AuthContext,
    scenario: str | None = None,
) -> list[dict]:
    client = get_client()
    scenario_like = SCENARIO_SOURCE_PATTERNS.get(scenario) if scenario else None
    access_sql, access_params = _access_filter(auth, "h.bakery_id")
    open_bakery_sql = _open_bakery_filter("h.bakery_id")
    query = """
        select
            h.product_id as product_id,
            d.product_name as product_name,
            d.category_name as category_name,
            h.hour as hour,
            h.forecast_qty as forecast_qty
        from {hour_source} h
        left join {day_source} d
          on d.run_id = h.run_id
         and d.forecast_date = h.forecast_date
         and d.bakery_id = h.bakery_id
         and d.product_id = h.product_id
        where h.forecast_date = %(forecast_date)s
          and h.bakery_id = %(bakery_id)s
          {open_bakery_sql}
          {access_sql}
        order by d.product_name, h.product_id, h.hour
        """.format(
        hour_source=_sku_hour_source("forecast_date = %(forecast_date)s", snapshot_scenario_like=scenario_like),
        day_source=_sku_day_source("forecast_date = %(forecast_date)s", snapshot_scenario_like=scenario_like),
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
            **({"snapshot_scenario": scenario_like} if scenario_like else {}),
            **access_params,
        },
    )
    return _records(df)


def _build_hour_discrepancy_response(hour: int, rows: list[dict], limit: int) -> dict:
    prepared = []
    total_forecast = 0.0
    total_actual = 0.0
    for row in rows:
        forecast_qty = float(row.get("forecast_qty") or 0)
        actual_qty = float(row.get("actual_qty") or 0)
        delta = forecast_qty - actual_qty
        total_forecast += forecast_qty
        total_actual += actual_qty
        prepared.append(
            {
                **row,
                "forecast_qty": forecast_qty,
                "actual_qty": actual_qty,
                "delta": delta,
                "abs_delta": abs(delta),
            }
        )

    total_delta = total_forecast - total_actual
    if total_delta > DISCREPANCY_EPSILON:
        direction = "over"
        aligned = [row for row in prepared if row["delta"] > DISCREPANCY_EPSILON]
    elif total_delta < -DISCREPANCY_EPSILON:
        direction = "under"
        aligned = [row for row in prepared if row["delta"] < -DISCREPANCY_EPSILON]
    else:
        direction = "flat"
        aligned = [row for row in prepared if row["abs_delta"] > DISCREPANCY_EPSILON]

    aligned.sort(key=lambda row: row["abs_delta"], reverse=True)
    denominator = sum(row["abs_delta"] for row in aligned) or 1.0
    items = []
    for row in aligned[:limit]:
        items.append(
            {
                "product_id": row.get("product_id"),
                "product_name": row.get("product_name") or "SKU без названия",
                "category_name": row.get("category_name") or "Без группы",
                "forecast_qty": row["forecast_qty"],
                "actual_qty": row["actual_qty"],
                "delta": row["delta"],
                "abs_delta": row["abs_delta"],
                "contribution_pct": row["abs_delta"] / denominator,
            }
        )

    return {
        "hour": hour,
        "direction": direction,
        "forecast_qty": total_forecast,
        "actual_qty": total_actual,
        "delta": total_delta,
        "abs_delta": abs(total_delta),
        "items": items,
    }


def get_hour_discrepancy_contributors(
    run_id: str,
    forecast_date: str,
    bakery_id: int,
    hour: int,
    auth: AuthContext,
    limit: int = 10,
    scenario: str | None = None,
) -> dict:
    client = get_client()
    scenario_like = SCENARIO_SOURCE_PATTERNS.get(scenario) if scenario else None
    access_sql, access_params = _access_filter(auth, "h.bakery_id")
    open_bakery_sql = _open_bakery_filter("h.bakery_id")
    request_access_sql, request_access_params = _access_filter(auth, "toInt64(%(bakery_id)s)")
    request_open_bakery_sql = _open_bakery_filter("toInt64(%(bakery_id)s)")
    query = """
        with forecast as (
            select
                h.product_id as product_id,
                any(d.product_name) as product_name,
                any(d.category_name) as category_name,
                sum(h.forecast_qty) as forecast_qty
            from {hour_source} h
            left join {day_source} d
              on d.run_id = h.run_id
             and d.forecast_date = h.forecast_date
             and d.bakery_id = h.bakery_id
             and d.product_id = h.product_id
            where h.forecast_date = %(forecast_date)s
              and h.bakery_id = %(bakery_id)s
              and h.hour = %(hour)s
              {open_bakery_sql}
              {access_sql}
            group by h.product_id
        ),
        actual as (
            select
                toInt64OrNull(toString(fcl.product_id)) as product_id,
                sum(toFloat64(fcl.quantity)) as actual_qty
            from {sales_source} fcl
            group by product_id
        )
        select
            coalesce(forecast.product_id, actual.product_id) as product_id,
            forecast.product_name as product_name,
            forecast.category_name as category_name,
            forecast.forecast_qty as forecast_qty,
            actual.actual_qty as actual_qty
        from forecast
        full outer join actual
          on actual.product_id = forecast.product_id
        where coalesce(forecast.product_id, actual.product_id) is not null
          {request_open_bakery_sql}
          {request_access_sql}
        """.format(
        hour_source=_sku_hour_source("forecast_date = %(forecast_date)s", snapshot_scenario_like=scenario_like),
        day_source=_sku_day_source("forecast_date = %(forecast_date)s", snapshot_scenario_like=scenario_like),
        sales_source=_raw_sales_source(
            "fcl.check_date = toDate(%(forecast_date)s) "
            "and toInt64OrNull(toString(fcl.bakery_id)) = %(bakery_id)s "
            "and toHour(fcl.check_datetime) = %(hour)s"
        ),
        open_bakery_sql=open_bakery_sql,
        access_sql=access_sql,
        request_open_bakery_sql=request_open_bakery_sql,
        request_access_sql=request_access_sql,
    )
    df = client.query_df(
        query,
        parameters={
            "run_id": run_id,
            "forecast_date": forecast_date,
            "bakery_id": bakery_id,
            "hour": hour,
            "sales_event_hex": SALES_EVENT_HEX,
            "closed_bakery_status": CLOSED_BAKERY_STATUS,
            **({"snapshot_scenario": scenario_like} if scenario_like else {}),
            **access_params,
            **request_access_params,
        },
    )
    return _build_hour_discrepancy_response(hour, _records(df), limit)


def get_sku_day(
    run_id: str,
    forecast_date: str,
    bakery_id: int,
    product_id: int,
    auth: AuthContext,
    scenario: str | None = None,
) -> dict | None:
    client = get_client()
    scenario_like = SCENARIO_SOURCE_PATTERNS.get(scenario) if scenario else None
    access_sql, access_params = _access_filter(auth, "d.bakery_id")
    open_bakery_sql = _open_bakery_filter("d.bakery_id")
    query = """
        with sales as (
            select
                toInt64OrNull(toString(fcl.product_id)) as sales_product_id,
                sum(toFloat64(fcl.quantity)) as actual_qty,
                sum(ifNull(toFloat64(fcl.line_amount), 0.0)) as actual_revenue
            from {sales_source} fcl
            group by sales_product_id
        )
        select d.product_id as product_id,
               d.product_name as product_name,
               d.category_name as category_name,
               d.forecast_qty as forecast_qty,
               sales.actual_qty as actual_qty,
               sales.actual_revenue as actual_revenue
        from {source} d
        left join sales on sales.sales_product_id = d.product_id
        where d.forecast_date = %(forecast_date)s
          and d.bakery_id = %(bakery_id)s
          and d.product_id = %(product_id)s
          {open_bakery_sql}
          {access_sql}
        limit 1
        """.format(
        source=_sku_day_source("forecast_date = %(forecast_date)s", snapshot_scenario_like=scenario_like),
        sales_source=_raw_sales_source(
            "fcl.check_date = toDate(%(forecast_date)s) "
            "and toInt64OrNull(toString(fcl.bakery_id)) = %(bakery_id)s "
            "and toInt64OrNull(toString(fcl.product_id)) = %(product_id)s"
        ),
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
            "sales_event_hex": SALES_EVENT_HEX,
            "closed_bakery_status": CLOSED_BAKERY_STATUS,
            **({"snapshot_scenario": scenario_like} if scenario_like else {}),
            **access_params,
        },
    )
    if df.empty:
        return None
    return _records(df)[0]



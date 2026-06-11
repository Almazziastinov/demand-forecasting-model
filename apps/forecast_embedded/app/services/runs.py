from __future__ import annotations

from app.db import get_client
from app.settings import get_settings

RUNS_TABLE = "forecast_runs_embedded"


def _normalize_record_dates(record: dict) -> dict:
    normalized = dict(record)
    for key in ["horizon_start", "horizon_end", "generated_at"]:
        value = normalized.get(key)
        if value is not None:
            normalized[key] = getattr(value, "isoformat", lambda: str(value))()
    return normalized


def get_active_run() -> dict | None:
    settings = get_settings()
    client = get_client()

    if settings.active_run_id:
        query = """
            select run_id, model_version, profile_version, source_kind,
                   horizon_start, horizon_end, generated_at, status, is_bias_adjusted
            from {table}
            where run_id = %(run_id)s
            order by generated_at desc
            limit 1
            """.format(table=RUNS_TABLE)
        params = {"run_id": settings.active_run_id}
    else:
        query = """
            select run_id, model_version, profile_version, source_kind,
                   horizon_start, horizon_end, generated_at, status, is_bias_adjusted
            from {table}
            where status = 'active'
            order by generated_at desc
            limit 1
            """.format(table=RUNS_TABLE)
        params = {}

    df = client.query_df(query, parameters=params)
    if df.empty:
        return None
    row = df.iloc[0].to_dict()
    return _normalize_record_dates(dict(row))


def get_run(run_id: str) -> dict | None:
    client = get_client()
    query = """
        select run_id, model_version, profile_version, source_kind,
               horizon_start, horizon_end, generated_at, status, is_bias_adjusted
        from {table}
        where run_id = %(run_id)s
        order by generated_at desc
        limit 1
        """.format(table=RUNS_TABLE)
    df = client.query_df(query, parameters={"run_id": run_id})
    if df.empty:
        return None
    return _normalize_record_dates(dict(df.iloc[0].to_dict()))


def resolve_run(run_id: str | None = None) -> dict | None:
    if run_id:
        return get_run(run_id)
    return get_active_run()


def list_runs(limit: int = 50) -> list[dict]:
    client = get_client()
    query = """
        select
            run_id,
            argMax(model_version, generated_at) as model_version,
            argMax(profile_version, generated_at) as profile_version,
            argMax(source_kind, generated_at) as source_kind,
            argMax(horizon_start, generated_at) as horizon_start,
            argMax(horizon_end, generated_at) as horizon_end,
            max(generated_at) as latest_generated_at,
            groupUniqArray(status) as statuses,
            max(status = 'active') as is_active,
            argMax(is_bias_adjusted, generated_at) as is_bias_adjusted
        from {table}
        group by run_id
        order by is_active desc, latest_generated_at desc
        limit %(limit)s
        """.format(table=RUNS_TABLE)
    df = client.query_df(query, parameters={"limit": limit})
    if df.empty:
        return []

    rows = []
    for record in df.to_dict("records"):
        normalized = _normalize_record_dates(record)
        normalized["generated_at"] = normalized.pop("latest_generated_at", None)
        normalized["is_active"] = bool(normalized.get("is_active"))
        rows.append(normalized)
    return rows


def get_run_dates(run_id: str) -> list[str]:
    client = get_client()
    query = """
        select distinct forecast_date
        from (
            select forecast_date
            from bakery_forecast_day_embedded
            where run_id = %(run_id)s
            union all
            select forecast_date
            from bakery_forecast_day_snapshots
            where lead_days = 1
        )
        order by forecast_date
        """
    df = client.query_df(query, parameters={"run_id": run_id})
    if df.empty:
        return []
    return [
        getattr(value, "date", lambda: value)().isoformat()
        if hasattr(getattr(value, "date", None), "__call__")
        else getattr(value, "isoformat", lambda: str(value))()
        for value in df["forecast_date"].tolist()
    ]

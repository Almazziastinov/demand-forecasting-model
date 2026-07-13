"""Rolling (trailing-window) replacement for the static bakery-day bias table.

`models/bakery_day_bias.json` is a one-time snapshot of mean(actual - forecast)
per bakery computed on a single holdout window at training time. It never
refreshes, so any drift between that snapshot and live conditions (a model
retrain, a seasonal transition, an assortment change) becomes a permanent,
uncorrected error once baked into `forecast_final`.

This module recomputes the same style of correction every run from the
trailing window of actual lead-1 `forecast_base` vs `mart_sales_60d` (the
raw model's live behaviour, not a fixed backtest), falling back to the
static snapshot for bakeries without enough recent history.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipelines.forecast_publish.load_forecast_run import create_client
from src.experiments_v2.build_bakery_daily_dataset import BAKERY_ID_COL

DEFAULT_TRAILING_DAYS = 7
DEFAULT_MIN_DAYS = 3
DEFAULT_BAKERY_FORECAST_TABLE = "bakery_forecast_day_snapshots"
DEFAULT_ACTUAL_TABLE = "mart_sales_60d"


def load_recent_bakery_day_performance(
    client,
    bakery_ids: list[int],
    as_of_date: str | pd.Timestamp,
    *,
    trailing_days: int = DEFAULT_TRAILING_DAYS,
    bakery_forecast_table: str = DEFAULT_BAKERY_FORECAST_TABLE,
    actual_table: str = DEFAULT_ACTUAL_TABLE,
) -> pd.DataFrame:
    """Lead-1 forecast_base vs actual sales for the trailing window strictly
    before `as_of_date`. Only uses dates already in the past, so this is safe
    to call before generating the next forecast horizon."""
    as_of = pd.Timestamp(as_of_date).normalize()
    window_start = as_of - pd.Timedelta(days=trailing_days)

    fc = client.query_df(
        f"""
        with latest_run as (
            select forecast_date, max(source_run_id) as run_id
            from {bakery_forecast_table}
            where lead_days = 1
              and toInt64(bakery_id) in %(bak)s
              and forecast_date >= %(start)s and forecast_date < %(end)s
              and source_run_id not like 'dev_%%'
            group by forecast_date
        )
        select s.forecast_date, toInt64(s.bakery_id) as {BAKERY_ID_COL},
               any(s.forecast_base) as forecast_base
        from {bakery_forecast_table} s
        inner join latest_run l
            on s.forecast_date = l.forecast_date and s.source_run_id = l.run_id
        where s.lead_days = 1 and toInt64(s.bakery_id) in %(bak)s
        group by s.forecast_date, {BAKERY_ID_COL}
        """,
        parameters={
            "bak": [int(b) for b in bakery_ids],
            "start": window_start.date().isoformat(),
            "end": as_of.date().isoformat(),
        },
    )
    if fc.empty:
        return pd.DataFrame(
            columns=["forecast_date", BAKERY_ID_COL, "forecast_base", "actual_qty"]
        )

    act = client.query_df(
        f"""
        select check_date as forecast_date, toInt64(bakery_id) as {BAKERY_ID_COL},
               sum(quantity) as actual_qty
        from {actual_table}
        where toInt64(bakery_id) in %(bak)s
          and check_date >= %(start)s and check_date < %(end)s
        group by forecast_date, {BAKERY_ID_COL}
        """,
        parameters={
            "bak": [int(b) for b in bakery_ids],
            "start": window_start.date().isoformat(),
            "end": as_of.date().isoformat(),
        },
    )

    merged = fc.merge(act, on=["forecast_date", BAKERY_ID_COL], how="inner")
    merged["forecast_base"] = pd.to_numeric(merged["forecast_base"], errors="coerce")
    merged["actual_qty"] = pd.to_numeric(merged["actual_qty"], errors="coerce")
    return merged.dropna(subset=["forecast_base", "actual_qty"])


def compute_rolling_bias(
    performance_df: pd.DataFrame,
    *,
    min_days: int = DEFAULT_MIN_DAYS,
) -> pd.DataFrame:
    """mean(actual - forecast_base) per bakery, only for bakeries with enough
    trailing observations. Same sign convention as the static bias table
    (positive = raw model underforecasts, negative = raw model overforecasts)."""
    if performance_df.empty:
        return pd.DataFrame(columns=[BAKERY_ID_COL, "bias", "n_days"])

    work = performance_df.copy()
    work["resid"] = work["actual_qty"] - work["forecast_base"]
    agg = (
        work.groupby(BAKERY_ID_COL)
        .agg(bias=("resid", "mean"), n_days=("resid", "count"))
        .reset_index()
    )
    return agg[agg["n_days"] >= min_days].reset_index(drop=True)


def build_effective_bias_table(
    rolling_bias_df: pd.DataFrame,
    static_bias_df: pd.DataFrame,
    bakery_ids: list[int],
) -> pd.DataFrame:
    """Rolling bias where available, else the static (one-time holdout)
    value, else 0.0 for bakeries with neither."""
    base = pd.DataFrame({BAKERY_ID_COL: [int(b) for b in bakery_ids]})
    static = static_bias_df[[BAKERY_ID_COL, "bias"]].rename(
        columns={"bias": "static_bias"}
    )
    rolling = rolling_bias_df[[BAKERY_ID_COL, "bias"]].rename(
        columns={"bias": "rolling_bias"}
    )

    merged = base.merge(static, on=BAKERY_ID_COL, how="left").merge(
        rolling, on=BAKERY_ID_COL, how="left"
    )
    static_bias = pd.to_numeric(merged["static_bias"], errors="coerce")
    rolling_bias = pd.to_numeric(merged["rolling_bias"], errors="coerce")
    merged["bias"] = rolling_bias.where(rolling_bias.notna(), static_bias).fillna(0.0)
    return merged[[BAKERY_ID_COL, "bias"]]


def build_rolling_bias_table(
    *,
    env_file: str | Path,
    bakery_ids: list[int],
    as_of_date: str | pd.Timestamp,
    static_bias_df: pd.DataFrame,
    trailing_days: int = DEFAULT_TRAILING_DAYS,
    min_days: int = DEFAULT_MIN_DAYS,
    bakery_forecast_table: str = DEFAULT_BAKERY_FORECAST_TABLE,
    actual_table: str = DEFAULT_ACTUAL_TABLE,
    client=None,
) -> pd.DataFrame:
    """End-to-end: query recent performance, compute rolling bias, blend with
    the static fallback. Returns a DataFrame shaped like the static bias
    table (bakery_id, bias) so it's a drop-in replacement for
    `load_bias_table`'s output."""
    client = client if client is not None else create_client(env_file)
    perf = load_recent_bakery_day_performance(
        client,
        bakery_ids,
        as_of_date,
        trailing_days=trailing_days,
        bakery_forecast_table=bakery_forecast_table,
        actual_table=actual_table,
    )
    rolling = compute_rolling_bias(perf, min_days=min_days)
    return build_effective_bias_table(rolling, static_bias_df, bakery_ids)

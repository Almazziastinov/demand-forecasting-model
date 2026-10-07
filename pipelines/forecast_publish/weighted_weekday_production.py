"""Build a production-compatible weighted-weekday forecast run.

This module promotes the same core idea used by the pilot publisher into the
normal ClickHouse serving contract: forecast_runs_embedded,
bakery_forecast_day_embedded, sku_forecast_day_embedded, sku_forecast_hour_embedded
and their snapshot tables.

The run is draft by default. Pass --activate only after verifying the loaded
shape and totals.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date as date_type
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.forecast_publish.activate_run import activate_run
from pipelines.forecast_publish.direct_daily_to_hour import expand_direct_sku_day_to_hour
from pipelines.forecast_publish.load_forecast_run import (
    DEFAULT_ENV_PATH,
    DEFAULT_SCHEMA_PATH,
    create_client,
    load_forecast_run,
)
from pipelines.forecast_publish.table_names import get_table_suffix_from_env_file


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "weighted_weekday_production"
DEFAULT_HOUR_PROFILE = ROOT / "data" / "processed" / "bakery_hour_profile.csv"

WEIGHTS = [1.0, 1.15, 1.35, 1.65, 2.0]
PUBLISHABLE_PATTERN = "пирог|выпеч|фастфуд|хлеб|пирожн|маффин|печенье|донат|торт|рулет"
NORMALIZED_DEMAND_V1_LOOKBACK_DAYS = 98
NORMALIZED_DEMAND_V1_POLICIES = {
    "alm": {"policy": "alpha35_no_calibration", "lo": None, "hi": None},
    "irk": {"policy": "alpha35_bakery_calibration_085_160", "lo": 0.85, "hi": 1.60},
    "nov": {"policy": "alpha35_bakery_calibration_090_180", "lo": 0.90, "hi": 1.80},
}
REGIONAL_SOURCES = {
    "alm": {"offset": 100_000},
    "irk": {"offset": 200_000},
    "nov": {"offset": 300_000},
}
REGIONAL_RECENT_SCOPE_DAYS = 60
REGIONAL_ACTIVE_BAKERY_DAYS = 30


def _regional_offset(source_db: str) -> int:
    return int(REGIONAL_SOURCES[source_db]["offset"])


def _split_legacy_and_regional_ids(bakery_ids: list[int]) -> tuple[list[int], dict[str, list[int]]]:
    legacy: list[int] = []
    regional: dict[str, list[int]] = {source_db: [] for source_db in REGIONAL_SOURCES}
    for value in bakery_ids:
        bakery_id = int(value)
        matched = False
        for source_db in REGIONAL_SOURCES:
            offset = _regional_offset(source_db)
            if offset < bakery_id < offset + 100_000:
                regional[source_db].append(bakery_id)
                matched = True
                break
        if not matched:
            legacy.append(bakery_id)
    return legacy, regional


def _weighted_tail(values: pd.Series) -> float:
    tail = values.tail(len(WEIGHTS)).astype(float)
    if tail.empty:
        return 0.0
    weights = np.asarray(WEIGHTS[-len(tail) :], dtype=float)
    return float((tail.to_numpy(dtype=float) * weights).sum() / weights.sum())


def _active_run_id(client) -> str:
    df = client.query_df(
        """
        SELECT run_id
        FROM forecast_runs_embedded
        WHERE status = 'active'
        ORDER BY generated_at DESC
        LIMIT 1
        """
    )
    if df.empty:
        raise RuntimeError("No active forecast run found")
    return str(df.iloc[0]["run_id"])


def _latest_base_norm_recent_run_id(client) -> str:
    df = client.query_df(
        """
        SELECT run_id
        FROM forecast_runs_embedded
        WHERE model_version = 'bakery_day_lgbm_base'
          AND profile_version = 'clickhouse_norm_recent'
        ORDER BY generated_at DESC
        LIMIT 1
        """
    )
    if df.empty:
        raise RuntimeError("No latest base_norm_recent run found")
    return str(df.iloc[0]["run_id"])


def _active_bakery_ids(client, active_run_id: str, forecast_date: date_type) -> list[int]:
    df = client.query_df(
        """
        SELECT DISTINCT bakery_id
        FROM sku_forecast_day_embedded
        WHERE run_id = %(run_id)s
          AND forecast_date = toDate(%(forecast_date)s)
        ORDER BY bakery_id
        """,
        parameters={"run_id": active_run_id, "forecast_date": forecast_date.isoformat()},
    )
    if df.empty:
        raise RuntimeError(f"No active run SKU rows for {active_run_id} on {forecast_date}")
    return [int(value) for value in df["bakery_id"].tolist()]


def _regional_active_bakery_ids(client, forecast_date: date_type) -> list[int]:
    date_to = forecast_date - timedelta(days=1)
    date_from = forecast_date - timedelta(days=REGIONAL_ACTIVE_BAKERY_DAYS)
    frames = []
    for source_db in REGIONAL_SOURCES:
        offset = _regional_offset(source_db)
        table = f"fct_check_lines_{source_db}"
        df = client.query_df(
            f"""
            SELECT DISTINCT {offset} + toInt64OrZero(toString(bakery_id)) AS bakery_id
            FROM {table}
            WHERE check_date BETWEEN toDate(%(date_from)s) AND toDate(%(date_to)s)
              AND cash_event_type = 'Продажа'
              AND is_deleted NOT IN ('1','true','Да')
              AND toInt64OrZero(toString(bakery_id)) > 0
              AND toFloat64(quantity) > 0
            ORDER BY bakery_id
            """,
            parameters={"date_from": date_from.isoformat(), "date_to": date_to.isoformat()},
        )
        frames.append(df)
    if not frames:
        return []
    result = pd.concat(frames, ignore_index=True)
    return sorted({int(value) for value in result["bakery_id"].tolist()})


def _load_scope_for_date(client, forecast_date: date_type, bakery_ids: list[int]) -> pd.DataFrame:
    scope = client.query_df(
        """
        WITH latest AS (
            SELECT toInt64(bakery_id) AS bakery_id, max(valid_from) AS latest_valid_from
            FROM bakery_product_assortment_embedded FINAL
            WHERE valid_from <= toDate(%(forecast_date)s)
              AND toInt64(bakery_id) IN %(bids)s
            GROUP BY bakery_id
        ),
        products AS (
            SELECT
                toInt64OrZero(toString(product_id)) AS product_id,
                argMax(product_name, _updated_at) AS product_name,
                argMax(category_name, _updated_at) AS category_name
            FROM dim_products
            GROUP BY product_id
        )
        SELECT
            toInt64(a.bakery_id) AS bakery_id,
            toInt64OrZero(toString(a.product_id)) AS product_id,
            any(p.product_name) AS product_name,
            any(p.category_name) AS category_name
        FROM bakery_product_assortment_embedded AS a FINAL
        INNER JOIN latest l
          ON toInt64(a.bakery_id) = l.bakery_id
         AND a.valid_from = l.latest_valid_from
        INNER JOIN products p
          ON p.product_id = toInt64OrZero(toString(a.product_id))
        GROUP BY bakery_id, product_id
        """,
        parameters={"forecast_date": forecast_date.isoformat(), "bids": bakery_ids},
    )
    if scope.empty:
        raise RuntimeError(f"No assortment rows for {forecast_date}")
    scope["category_name"] = scope["category_name"].fillna("")
    scope = scope[
        scope["category_name"].str.lower().str.contains(PUBLISHABLE_PATTERN, regex=True, na=False)
    ].copy()
    if scope.empty:
        raise RuntimeError(f"No publishable assortment rows for {forecast_date}")
    return scope.drop_duplicates(["bakery_id", "product_id"]).reset_index(drop=True)


def _load_regional_scope_for_date(client, forecast_date: date_type, regional_ids: dict[str, list[int]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    date_to = forecast_date - timedelta(days=1)
    date_from = forecast_date - timedelta(days=REGIONAL_RECENT_SCOPE_DAYS)
    for source_db, ids in regional_ids.items():
        if not ids:
            continue
        offset = _regional_offset(source_db)
        source_bids = [int(value) - offset for value in ids]
        sales_table = f"fct_check_lines_{source_db}"
        products_table = f"dim_products_{source_db}"
        scope = client.query_df(
            f"""
            WITH products AS (
                SELECT
                    product_id,
                    argMax(product_name, _updated_at) AS product_name,
                    argMax(category_name, _updated_at) AS category_name
                FROM {products_table}
                GROUP BY product_id
            )
            SELECT
                {offset} + toInt64OrZero(toString(s.bakery_id)) AS bakery_id,
                {offset} + toInt64OrZero(toString(s.product_id)) AS product_id,
                any(p.product_name) AS product_name,
                any(p.category_name) AS category_name
            FROM (
                SELECT DISTINCT bakery_id, product_id
                FROM {sales_table}
                WHERE check_date BETWEEN toDate(%(date_from)s) AND toDate(%(date_to)s)
                  AND cash_event_type = 'Продажа'
                  AND is_deleted NOT IN ('1','true','Да')
                  AND toInt64OrZero(toString(bakery_id)) IN %(source_bids)s
                  AND toInt64OrZero(toString(product_id)) > 0
                  AND toFloat64(quantity) > 0
            ) AS s
            INNER JOIN products AS p ON toString(p.product_id) = toString(s.product_id)
            GROUP BY s.bakery_id, s.product_id
            """,
            parameters={
                "date_from": date_from.isoformat(),
                "date_to": date_to.isoformat(),
                "source_bids": source_bids,
            },
        )
        if scope.empty:
            continue
        scope["category_name"] = scope["category_name"].fillna("")
        scope = scope[
            scope["category_name"].str.lower().str.contains(PUBLISHABLE_PATTERN, regex=True, na=False)
        ].copy()
        frames.append(scope)
    if not frames:
        return pd.DataFrame(columns=["bakery_id", "product_id", "product_name", "category_name"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["bakery_id", "product_id"])


def load_production_scope(client, horizon: list[date_type], bakery_ids: list[int]) -> pd.DataFrame:
    legacy_ids, regional_ids = _split_legacy_and_regional_ids(bakery_ids)
    frames = []
    for forecast_date in horizon:
        daily_frames = []
        if legacy_ids:
            daily_frames.append(_load_scope_for_date(client, forecast_date, legacy_ids))
        regional = _load_regional_scope_for_date(client, forecast_date, regional_ids)
        if not regional.empty:
            daily_frames.append(regional)
        if not daily_frames:
            raise RuntimeError(f"No production scope rows for {forecast_date}")
        daily = pd.concat(daily_frames, ignore_index=True)
        daily["date"] = pd.Timestamp(forecast_date)
        frames.append(daily)
    return pd.concat(frames, ignore_index=True)


def _compute_demand_from_fact_parts(
    sales: pd.DataFrame,
    release: pd.DataFrame,
    moves: pd.DataFrame,
    writeoffs: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["date", "bakery_id", "product_id"]
    facts = sales
    for part in (release, moves, writeoffs):
        facts = facts.merge(part, on=keys, how="outer")
    if facts.empty:
        return pd.DataFrame(columns=keys + ["demand"])
    facts["date"] = pd.to_datetime(facts["date"]).dt.normalize()
    for column in ["observed_sales_qty", "release_qty", "incoming_move_qty", "outgoing_move_qty", "written_off_qty"]:
        facts[column] = pd.to_numeric(facts.get(column, 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    facts["available_proxy"] = (
        facts["release_qty"] + facts["incoming_move_qty"] - facts["outgoing_move_qty"]
    ).clip(lower=0.0)

    bakery_last = (
        facts.dropna(subset=["last_sale_time"])
        .groupby(["date", "bakery_id"], as_index=False)["last_sale_time"]
        .max()
        .rename(columns={"last_sale_time": "bakery_last_sale_time"})
    )
    facts = facts.merge(bakery_last, on=["date", "bakery_id"], how="left", validate="many_to_one")

    last_sale = pd.to_datetime(facts["last_sale_time"], errors="coerce", utc=True)
    bakery_end = pd.to_datetime(facts["bakery_last_sale_time"], errors="coerce", utc=True)
    opening = facts["date"].dt.tz_localize("Europe/Moscow").dt.tz_convert("UTC") + pd.Timedelta(hours=7.5)
    elapsed_hours = (last_sale - opening).dt.total_seconds() / 3600
    remaining_hours = (bakery_end - last_sale).dt.total_seconds().clip(lower=0) / 3600
    stockout_like = (
        facts["observed_sales_qty"].gt(0.0)
        & facts["available_proxy"].gt(0.0)
        & (facts["observed_sales_qty"] + facts["written_off_qty"]).ge(facts["available_proxy"] * 0.95)
        & elapsed_hours.ge(2.0)
        & remaining_hours.gt(0.0)
    )
    raw_lost = facts["observed_sales_qty"] / elapsed_hours.replace(0.0, np.nan) * remaining_hours
    cap = pd.concat([facts["observed_sales_qty"] * 1.5, pd.Series(15.0, index=facts.index)], axis=1).max(axis=1)
    facts["lost_demand_estimate"] = raw_lost.where(stockout_like, 0.0).fillna(0.0).clip(lower=0.0).clip(upper=cap)
    facts["demand"] = facts["observed_sales_qty"] + facts["lost_demand_estimate"]
    return facts[keys + ["demand"]].copy()


def _fetch_legacy_history_demand(client, history_dates: list[date_type], bakery_ids: list[int]) -> pd.DataFrame:
    if not bakery_ids:
        return pd.DataFrame(columns=["date", "bakery_id", "product_id", "demand"])
    date_strings = [d.isoformat() for d in sorted(set(history_dates))]
    params = {"dates": date_strings, "bids": bakery_ids}
    sales = client.query_df(
        """
        SELECT check_date AS date,
               toInt64OrZero(toString(bakery_id)) AS bakery_id,
               toInt64OrZero(toString(product_id)) AS product_id,
               sum(toFloat64(quantity)) AS observed_sales_qty,
               min(check_datetime) AS first_sale_time,
               max(check_datetime) AS last_sale_time
        FROM (
            SELECT DISTINCT check_datetime, check_date, bakery_id, product_id, quantity, line_amount
            FROM fct_check_lines
            WHERE hex(cash_event_type)='D09FD180D0BED0B4D0B0D0B6D0B0'
              AND check_date IN %(dates)s
              AND toInt64OrZero(toString(bakery_id)) IN %(bids)s
        )
        GROUP BY date, bakery_id, product_id
        """,
        parameters=params,
    )
    release = client.query_df(
        """
        SELECT rd AS date,
               toInt64OrZero(toString(bid)) AS bakery_id,
               toInt64OrZero(toString(pid)) AS product_id,
               sum(qty) AS release_qty
        FROM (
            SELECT argMax(release_date,_updated_at) rd,
                   argMax(bakery_id,_updated_at) bid,
                   argMax(product_id,_updated_at) pid,
                   toFloat64(argMax(quantity,_updated_at)) qty,
                   argMax(is_deleted,_updated_at) deleted
            FROM fct_production_release
            WHERE release_date IN %(dates)s
              AND toInt64OrZero(toString(bakery_id)) IN %(bids)s
            GROUP BY release_id,line_id
            HAVING deleted NOT IN ('1','true','Да')
        )
        GROUP BY date, bakery_id, product_id
        """,
        parameters=params,
    )
    moves = client.query_df(
        """
        SELECT date, bakery_id, product_id,
               sum(incoming_move_qty) AS incoming_move_qty,
               sum(outgoing_move_qty) AS outgoing_move_qty
        FROM (
            SELECT md AS date,
                   toInt64OrZero(toString(receiver)) AS bakery_id,
                   toInt64OrZero(toString(pid)) AS product_id,
                   qty AS incoming_move_qty,
                   0.0 AS outgoing_move_qty
            FROM (
                SELECT argMax(move_date,_updated_at) md,
                       argMax(receiver_id,_updated_at) receiver,
                       argMax(product_id,_updated_at) pid,
                       toFloat64(argMax(quantity,_updated_at)) qty,
                       argMax(is_deleted,_updated_at) deleted
                FROM fct_moves
                WHERE move_date IN %(dates)s
                GROUP BY move_id,line_id
                HAVING deleted NOT IN ('1','true','Да')
            )
            UNION ALL
            SELECT md AS date,
                   toInt64OrZero(toString(sender)) AS bakery_id,
                   toInt64OrZero(toString(pid)) AS product_id,
                   0.0 AS incoming_move_qty,
                   qty AS outgoing_move_qty
            FROM (
                SELECT argMax(move_date,_updated_at) md,
                       argMax(sender_id,_updated_at) sender,
                       argMax(product_id,_updated_at) pid,
                       toFloat64(argMax(quantity,_updated_at)) qty,
                       argMax(is_deleted,_updated_at) deleted
                FROM fct_moves
                WHERE move_date IN %(dates)s
                GROUP BY move_id,line_id
                HAVING deleted NOT IN ('1','true','Да')
            )
        )
        WHERE bakery_id IN %(bids)s
        GROUP BY date, bakery_id, product_id
        """,
        parameters=params,
    )
    writeoffs = client.query_df(
        """
        SELECT wd AS date,
               toInt64OrZero(toString(bid)) AS bakery_id,
               toInt64OrZero(toString(pid)) AS product_id,
               sum(qty) AS written_off_qty
        FROM (
            SELECT argMax(write_off_date,_updated_at) wd,
                   argMax(bakery_id,_updated_at) bid,
                   argMax(write_off_product_id,_updated_at) pid,
                   toFloat64(argMax(write_off_qty,_updated_at)) qty,
                   argMax(is_deleted,_updated_at) deleted
            FROM fct_write_offs
            WHERE write_off_date IN %(dates)s
              AND toInt64OrZero(toString(bakery_id)) IN %(bids)s
            GROUP BY write_off_doc_num,line_id
            HAVING deleted NOT IN ('1','true','Да')
        )
        GROUP BY date, bakery_id, product_id
        """,
        parameters=params,
    )

    return _compute_demand_from_fact_parts(sales, release, moves, writeoffs)


def _fetch_regional_history_demand(
    client,
    history_dates: list[date_type],
    regional_ids: dict[str, list[int]],
) -> pd.DataFrame:
    date_strings = [d.isoformat() for d in sorted(set(history_dates))]
    frames: list[pd.DataFrame] = []
    for source_db, ids in regional_ids.items():
        if not ids:
            continue
        offset = _regional_offset(source_db)
        source_bids = [int(value) - offset for value in ids]
        params = {"dates": date_strings, "source_bids": source_bids}
        sales_table = f"fct_check_lines_{source_db}"
        release_table = f"fct_production_release_{source_db}"
        moves_table = f"fct_moves_{source_db}"
        writeoffs_table = f"fct_write_offs_{source_db}"
        sales = client.query_df(
            f"""
            SELECT date,
                   {offset} + source_bakery_id AS bakery_id,
                   {offset} + source_product_id AS product_id,
                   sum(toFloat64(quantity)) AS observed_sales_qty,
                   min(check_datetime) AS first_sale_time,
                   max(check_datetime) AS last_sale_time
            FROM (
                SELECT DISTINCT
                    check_datetime,
                    check_date AS date,
                    toInt64OrZero(toString(bakery_id)) AS source_bakery_id,
                    toInt64OrZero(toString(product_id)) AS source_product_id,
                    quantity,
                    line_amount
                FROM {sales_table}
                WHERE cash_event_type = 'Продажа'
                  AND is_deleted NOT IN ('1','true','Да')
                  AND check_date IN %(dates)s
                  AND toInt64OrZero(toString(bakery_id)) IN %(source_bids)s
            )
            GROUP BY date, source_bakery_id, source_product_id
            """,
            parameters=params,
        )
        release = client.query_df(
            f"""
            SELECT rd AS date,
                   {offset} + toInt64OrZero(toString(bid)) AS bakery_id,
                   {offset} + toInt64OrZero(toString(pid)) AS product_id,
                   sum(qty) AS release_qty
            FROM (
                SELECT argMax(release_date,_updated_at) rd,
                       argMax(bakery_id,_updated_at) bid,
                       argMax(product_id,_updated_at) pid,
                       toFloat64(argMax(quantity,_updated_at)) qty,
                       argMax(is_deleted,_updated_at) deleted
                FROM {release_table}
                WHERE release_date IN %(dates)s
                  AND toInt64OrZero(toString(bakery_id)) IN %(source_bids)s
                GROUP BY release_id,line_id
                HAVING deleted NOT IN ('1','true','Да')
            )
            GROUP BY date, bakery_id, product_id
            """,
            parameters=params,
        )
        moves = client.query_df(
            f"""
            SELECT date, bakery_id, product_id,
                   sum(incoming_move_qty) AS incoming_move_qty,
                   sum(outgoing_move_qty) AS outgoing_move_qty
            FROM (
                SELECT md AS date,
                       {offset} + toInt64OrZero(toString(receiver)) AS bakery_id,
                       {offset} + toInt64OrZero(toString(pid)) AS product_id,
                       qty AS incoming_move_qty,
                       0.0 AS outgoing_move_qty
                FROM (
                    SELECT argMax(move_date,_updated_at) md,
                           argMax(receiver_id,_updated_at) receiver,
                           argMax(product_id,_updated_at) pid,
                           toFloat64(argMax(quantity,_updated_at)) qty,
                           argMax(is_deleted,_updated_at) deleted
                    FROM {moves_table}
                    WHERE move_date IN %(dates)s
                    GROUP BY move_id,line_id
                    HAVING deleted NOT IN ('1','true','Да')
                )
                UNION ALL
                SELECT md AS date,
                       {offset} + toInt64OrZero(toString(sender)) AS bakery_id,
                       {offset} + toInt64OrZero(toString(pid)) AS product_id,
                       0.0 AS incoming_move_qty,
                       qty AS outgoing_move_qty
                FROM (
                    SELECT argMax(move_date,_updated_at) md,
                           argMax(sender_id,_updated_at) sender,
                           argMax(product_id,_updated_at) pid,
                           toFloat64(argMax(quantity,_updated_at)) qty,
                           argMax(is_deleted,_updated_at) deleted
                    FROM {moves_table}
                    WHERE move_date IN %(dates)s
                    GROUP BY move_id,line_id
                    HAVING deleted NOT IN ('1','true','Да')
                )
            )
            WHERE bakery_id IN %(bids)s
            GROUP BY date, bakery_id, product_id
            """,
            parameters={**params, "bids": ids},
        )
        writeoffs = client.query_df(
            f"""
            SELECT wd AS date,
                   {offset} + toInt64OrZero(toString(bid)) AS bakery_id,
                   {offset} + toInt64OrZero(toString(pid)) AS product_id,
                   sum(qty) AS written_off_qty
            FROM (
                SELECT argMax(write_off_date,_updated_at) wd,
                       argMax(bakery_id,_updated_at) bid,
                       argMax(write_off_product_id,_updated_at) pid,
                       toFloat64(argMax(write_off_qty,_updated_at)) qty,
                       argMax(is_deleted,_updated_at) deleted
                FROM {writeoffs_table}
                WHERE write_off_date IN %(dates)s
                  AND toInt64OrZero(toString(bakery_id)) IN %(source_bids)s
                GROUP BY write_off_doc_num,line_id
                HAVING deleted NOT IN ('1','true','Да')
            )
            GROUP BY date, bakery_id, product_id
            """,
            parameters=params,
        )
        frames.append(_compute_demand_from_fact_parts(sales, release, moves, writeoffs))
    if not frames:
        return pd.DataFrame(columns=["date", "bakery_id", "product_id", "demand"])
    return pd.concat(frames, ignore_index=True)


def fetch_history_demand(client, history_dates: list[date_type], bakery_ids: list[int]) -> pd.DataFrame:
    legacy_ids, regional_ids = _split_legacy_and_regional_ids(bakery_ids)
    frames = [
        _fetch_legacy_history_demand(client, history_dates, legacy_ids),
        _fetch_regional_history_demand(client, history_dates, regional_ids),
    ]
    return pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)


def _load_new_region_policy(client, bakery_ids: list[int]) -> pd.DataFrame:
    _ = client
    policy_frames = []
    for source_db, config in NORMALIZED_DEMAND_V1_POLICIES.items():
        offset = _regional_offset(source_db)
        ids = [int(value) for value in bakery_ids if offset < int(value) < offset + 100_000]
        if not ids:
            continue
        mapped = pd.DataFrame({"bakery_id": ids})
        mapped["normalized_region"] = source_db
        mapped["normalized_policy"] = config["policy"]
        mapped["calibration_lo"] = config["lo"]
        mapped["calibration_hi"] = config["hi"]
        policy_frames.append(mapped)
    if not policy_frames:
        return pd.DataFrame(
            columns=["bakery_id", "normalized_region", "normalized_policy", "calibration_lo", "calibration_hi"]
        )
    return pd.concat(policy_frames, ignore_index=True).drop_duplicates("bakery_id")


def _rolling_by_pair(
    df: pd.DataFrame,
    column: str,
    window: int,
    min_periods: int,
    func: str,
    q: float | None = None,
) -> pd.Series:
    shifted = df.groupby(["bakery_id", "product_id"])[column].shift(1)
    grouped = shifted.groupby([df["bakery_id"], df["product_id"]])
    if func == "median":
        return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).median())
    if func == "quantile":
        if q is None:
            raise ValueError("q is required for quantile rolling")
        return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).quantile(q))
    raise ValueError(f"Unsupported rolling func: {func}")


def _weekday_expected(df: pd.DataFrame, value_col: str) -> pd.Series:
    work = df.sort_values(["bakery_id", "product_id", "dow", "date"]).copy()
    shifted = work.groupby(["bakery_id", "product_id", "dow"])[value_col].shift(1)
    work["weekday_expected"] = shifted.groupby([work["bakery_id"], work["product_id"], work["dow"]]).transform(
        lambda s: s.rolling(8, min_periods=3).median()
    )
    return work.sort_index()["weekday_expected"].reindex(df.index)


def _rolling_calibration_factor(daily: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    work = daily.sort_values(["bakery_id", "date"]).copy()
    raw_shifted = work.groupby("bakery_id")["raw"].shift(1)
    forecast_shifted = work.groupby("bakery_id")["forecast"].shift(1)
    work["raw_14"] = raw_shifted.groupby(work["bakery_id"]).transform(lambda s: s.rolling(14, min_periods=5).sum())
    work["forecast_14"] = forecast_shifted.groupby(work["bakery_id"]).transform(
        lambda s: s.rolling(14, min_periods=5).sum()
    )
    work["calibration_factor"] = (work["raw_14"] / work["forecast_14"].replace(0, np.nan)).clip(lo, hi)
    return work[["date", "bakery_id", "calibration_factor"]]


def apply_normalized_demand_v1(client, history: pd.DataFrame, bakery_ids: list[int]) -> tuple[pd.DataFrame, dict]:
    if history.empty:
        return history.assign(normalized_demand_v1=0.0), {
            "normalized_policy_rows": 0,
            "normalized_regions": {},
            "normalized_total": 0.0,
        }

    out = history.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["demand"] = pd.to_numeric(out["demand"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out = out.sort_values(["bakery_id", "product_id", "date"]).reset_index(drop=True)
    out["dow"] = out["date"].dt.dayofweek

    median_42 = _rolling_by_pair(out, "demand", 42, 10, "median")
    q10_42 = _rolling_by_pair(out, "demand", 42, 10, "quantile", 0.10)
    q90_42 = _rolling_by_pair(out, "demand", 42, 10, "quantile", 0.90)
    lower = np.minimum(q10_42, median_42 * 0.75).fillna(out["demand"])
    upper = np.maximum(q90_42, median_42 * 1.35).fillna(out["demand"])
    clipped = out["demand"].clip(lower=lower, upper=upper)
    weekday_expected = _weekday_expected(out, "demand").fillna(median_42).fillna(out["demand"]).clip(lower=0.0)
    out["target_a35"] = (0.65 * clipped + 0.35 * weekday_expected).clip(lower=0.0)

    policy = _load_new_region_policy(client, bakery_ids)
    out = out.merge(policy, on="bakery_id", how="left")
    out["normalized_region"] = out["normalized_region"].fillna("default")
    out["normalized_policy"] = out["normalized_policy"].fillna("alpha35_default")
    out["normalized_demand_v1"] = out["target_a35"]
    out["calibration_factor"] = 1.0

    for source_db, config in NORMALIZED_DEMAND_V1_POLICIES.items():
        lo = config["lo"]
        hi = config["hi"]
        if lo is None or hi is None:
            continue
        mask = out["normalized_region"].eq(source_db)
        if not bool(mask.any()):
            continue
        daily = (
            out.loc[mask]
            .groupby(["date", "bakery_id"], as_index=False)
            .agg(raw=("demand", "sum"), forecast=("target_a35", "sum"))
        )
        factors = _rolling_calibration_factor(daily, float(lo), float(hi))
        out = out.merge(
            factors.rename(columns={"calibration_factor": f"calibration_factor_{source_db}"}),
            on=["date", "bakery_id"],
            how="left",
        )
        factor_col = f"calibration_factor_{source_db}"
        region_factor = out[factor_col].where(mask).fillna(1.0)
        out.loc[mask, "calibration_factor"] = region_factor.loc[mask]
        out.loc[mask, "normalized_demand_v1"] = (
            out.loc[mask, "target_a35"] * out.loc[mask, "calibration_factor"].fillna(1.0)
        ).clip(lower=0.0)
        out = out.drop(columns=[factor_col])

    summary = {
        "normalized_policy_rows": int(len(out)),
        "normalized_regions": out.groupby("normalized_region")["bakery_id"].nunique().to_dict(),
        "raw_demand_total": round(float(out["demand"].sum()), 2),
        "target_a35_total": round(float(out["target_a35"].sum()), 2),
        "normalized_total": round(float(out["normalized_demand_v1"].sum()), 2),
    }
    return out.drop(columns=["dow"]), summary


def build_weighted_weekday(scope: pd.DataFrame, history: pd.DataFrame, demand_column: str = "demand") -> pd.DataFrame:
    hist = history.copy()
    hist["date"] = pd.to_datetime(hist["date"]).dt.normalize()
    hist["dow"] = hist["date"].dt.dayofweek
    if demand_column not in hist.columns:
        raise ValueError(f"History has no demand column {demand_column!r}")
    current = scope[["date", "bakery_id", "product_id", "product_name", "category_name"]].drop_duplicates().copy()
    current["dow"] = pd.to_datetime(current["date"]).dt.dayofweek

    rows: list[pd.DataFrame] = []
    for forecast_date, daily_scope in current.groupby("date", sort=True):
        daily_hist = hist[hist["dow"].eq(pd.Timestamp(forecast_date).dayofweek)].sort_values("date")
        if daily_hist.empty:
            out = daily_scope.copy()
            out["forecast_qty"] = 0.0
        else:
            pair_plan = (
                daily_hist.groupby(["bakery_id", "product_id"], as_index=False)[demand_column]
                .agg(forecast_qty=_weighted_tail)
            )
            out = daily_scope.merge(pair_plan, on=["bakery_id", "product_id"], how="left")
            out["forecast_qty"] = pd.to_numeric(out["forecast_qty"], errors="coerce").fillna(0.0).clip(lower=0.0)
        rows.append(out)
    result = pd.concat(rows, ignore_index=True)
    return result[["date", "bakery_id", "product_id", "product_name", "category_name", "forecast_qty"]]


def load_bakery_info(client, bakery_ids: list[int]) -> pd.DataFrame:
    legacy_ids, regional_ids = _split_legacy_and_regional_ids(bakery_ids)
    frames: list[pd.DataFrame] = []
    if legacy_ids:
        frames.append(
            client.query_df(
                """
                SELECT
                    toInt64OrZero(toString(bakery_id)) AS bakery_id,
                    any(bakery_name) AS bakery_name,
                    any(city) AS city
                FROM dim_bakeries
                WHERE toInt64OrZero(toString(bakery_id)) IN %(bids)s
                GROUP BY bakery_id
                """,
                parameters={"bids": legacy_ids},
            )
        )
    for source_db, ids in regional_ids.items():
        if not ids:
            continue
        offset = _regional_offset(source_db)
        source_bids = [int(value) - offset for value in ids]
        table = f"dim_bakeries_{source_db}"
        frames.append(
            client.query_df(
                f"""
                SELECT
                    {offset} + source_bakery_id AS bakery_id,
                    argMax(bakery_name, _updated_at) AS bakery_name,
                    argMax(city, _updated_at) AS city
                FROM (
                    SELECT
                        toInt64OrZero(toString(bakery_id)) AS source_bakery_id,
                        bakery_name,
                        city,
                        _updated_at
                    FROM {table}
                    WHERE toInt64OrZero(toString(bakery_id)) IN %(source_bids)s
                )
                GROUP BY source_bakery_id
                """,
                parameters={"source_bids": source_bids},
            )
        )
    if not frames:
        return pd.DataFrame(columns=["bakery_id", "bakery_name", "city"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["bakery_id"])


def _build_horizon(start_date: date_type, horizon_days: int) -> list[date_type]:
    return [start_date + timedelta(days=offset) for offset in range(horizon_days)]


def _history_dates_for_horizon(horizon: list[date_type], weeks: int) -> list[date_type]:
    return sorted({forecast_date - timedelta(days=7 * lag) for forecast_date in horizon for lag in range(1, weeks + 1)})


def _continuous_history_dates(start_date: date_type, lookback_days: int) -> list[date_type]:
    return [start_date - timedelta(days=offset) for offset in range(1, lookback_days + 1)]


def _hour_profile_with_fallback(sku_day: pd.DataFrame, path: Path) -> pd.DataFrame:
    profile = pd.read_csv(path, encoding="utf-8-sig")
    needed = sku_day[["date", "bakery_id"]].drop_duplicates().copy()
    needed["dow"] = pd.to_datetime(needed["date"]).dt.dayofweek
    available = set(map(tuple, profile[["bakery_id", "dow"]].drop_duplicates().to_numpy()))
    missing = [
        tuple(values)
        for values in needed[["bakery_id", "dow"]].to_numpy()
        if tuple(values) not in available
    ]
    if missing:
        network = profile.groupby(["dow", "hour"], as_index=False)["mean_hour_share_norm"].mean()
        fallback = pd.concat(
            [
                network[network["dow"].eq(dow)].assign(bakery_id=bakery_id)
                for bakery_id, dow in missing
            ],
            ignore_index=True,
        )
        profile = pd.concat([profile, fallback], ignore_index=True, sort=False)
    return profile


def run_weighted_weekday_production(
    *,
    client,
    env_file: str | Path,
    schema_path: str | Path,
    output_dir: str | Path,
    run_id: str,
    start_date: date_type,
    horizon_days: int,
    history_weeks: int,
    active_run_id: str | None,
    scope_source: str,
    demand_mode: str,
    activate: bool,
    hour_profile_path: str | Path = DEFAULT_HOUR_PROFILE,
) -> dict:
    if active_run_id:
        source_run_id = active_run_id
    elif scope_source == "latest-base":
        source_run_id = _latest_base_norm_recent_run_id(client)
    elif scope_source == "active":
        source_run_id = _active_run_id(client)
    else:
        raise ValueError(f"Unsupported scope_source: {scope_source}")
    bakery_ids = sorted(
        set(_active_bakery_ids(client, source_run_id, start_date))
        | set(_regional_active_bakery_ids(client, start_date))
    )
    horizon = _build_horizon(start_date, horizon_days)
    scope = load_production_scope(client, horizon, bakery_ids)
    if demand_mode == "normalized-demand-v1":
        history_dates = _continuous_history_dates(start_date, NORMALIZED_DEMAND_V1_LOOKBACK_DAYS)
    elif demand_mode == "weighted-weekday":
        history_dates = _history_dates_for_horizon(horizon, history_weeks)
    else:
        raise ValueError(f"Unsupported demand_mode: {demand_mode}")
    history = fetch_history_demand(client, history_dates, bakery_ids)
    normalization_summary: dict = {}
    demand_column = "demand"
    model_version = "weighted_weekday_calculated_demand_v1"
    if demand_mode == "normalized-demand-v1":
        history, normalization_summary = apply_normalized_demand_v1(client, history, bakery_ids)
        demand_column = "normalized_demand_v1"
        model_version = "normalized_demand_v1_weighted_weekday"
    sku = build_weighted_weekday(scope, history, demand_column=demand_column)

    bakery_totals = (
        sku.groupby(["date", "bakery_id"], as_index=False)["forecast_qty"]
        .sum()
        .rename(columns={"forecast_qty": "bakery_day_forecast"})
    )
    bakery_info = load_bakery_info(client, bakery_ids)
    bakery = bakery_totals.merge(bakery_info, on="bakery_id", how="left", validate="many_to_one")
    bakery["bakery_name"] = bakery["bakery_name"].fillna("")
    bakery["city"] = bakery["city"].fillna("unknown")
    bakery["bakery_day_forecast_bias_adj"] = bakery["bakery_day_forecast"]

    sku_day = sku.rename(columns={"forecast_qty": "sku_day_forecast"})[
        ["date", "bakery_id", "product_id", "product_name", "category_name", "sku_day_forecast"]
    ]
    profile = _hour_profile_with_fallback(sku_day, Path(hour_profile_path))
    sku_hour = expand_direct_sku_day_to_hour(sku_day, profile)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bakery_path = output / "bakery_day.csv"
    sku_day_path = output / "sku_day.csv"
    sku_hour_path = output / "sku_hour.csv"
    bakery.to_csv(bakery_path, index=False, encoding="utf-8-sig")
    sku_day.to_csv(sku_day_path, index=False, encoding="utf-8-sig")
    sku_hour.to_csv(sku_hour_path, index=False, encoding="utf-8-sig")

    loaded = load_forecast_run(
        env_file=env_file,
        schema_path=schema_path,
        bakery_path=bakery_path,
        sku_day_path=sku_day_path,
        sku_hour_path=sku_hour_path,
        lookup_source="clickhouse",
        run_id=run_id,
        model_version=model_version,
        profile_version="bakery_dow_timing_v1",
        notes=(
            f"Weighted weekday calculated demand; demand_mode={demand_mode}; "
            f"scope from {source_run_id}; history_weeks={history_weeks}; "
            f"history_rows={len(history)}"
        ),
        replace_existing=True,
    )
    if activate:
        activate_run(client, run_id, table_suffix=get_table_suffix_from_env_file(env_file))

    return {
        "run_id": run_id,
        "source_run_id": source_run_id,
        "horizon_start": horizon[0].isoformat(),
        "horizon_end": horizon[-1].isoformat(),
        "history_weeks": history_weeks,
        "demand_mode": demand_mode,
        "scope_rows": int(len(scope)),
        "history_rows": int(len(history)),
        "forecast_total": round(float(sku["forecast_qty"].sum()), 2),
        "bakery_count": int(sku["bakery_id"].nunique()),
        "product_count": int(sku["product_id"].nunique()),
        "loaded_rows": loaded,
        "activated": activate,
        "normalization_summary": normalization_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH))
    parser.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--date", default=None, help="Horizon start date YYYY-MM-DD; default today")
    parser.add_argument("--horizon-days", type=int, default=int(os.getenv("FORECAST_HORIZON_DAYS", "14")))
    parser.add_argument("--history-weeks", type=int, default=8)
    parser.add_argument("--active-run-id", default=None, help="Scope source run. Defaults to current active run")
    parser.add_argument("--scope-source", choices=["active", "latest-base"], default="active")
    parser.add_argument(
        "--demand-mode",
        choices=["weighted-weekday", "normalized-demand-v1"],
        default="weighted-weekday",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()

    start_date = date_type.fromisoformat(args.date) if args.date else date_type.today()
    if args.run_id:
        run_id = args.run_id
    elif args.demand_mode == "normalized-demand-v1":
        run_id = f"draft_normalized_demand_v1_{start_date.strftime('%Y%m%d')}_h14"
    else:
        run_id = f"prod_weighted_weekday_{start_date.strftime('%Y%m%d')}_h14"
    client = create_client(args.env_file)
    result = run_weighted_weekday_production(
        client=client,
        env_file=args.env_file,
        schema_path=args.schema_path,
        output_dir=args.output_dir,
        run_id=run_id,
        start_date=start_date,
        horizon_days=args.horizon_days,
        history_weeks=args.history_weeks,
        active_run_id=args.active_run_id,
        scope_source=args.scope_source,
        demand_mode=args.demand_mode,
        activate=args.activate,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

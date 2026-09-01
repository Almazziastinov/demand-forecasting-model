from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from pipelines.forecast_publish.load_forecast_run import DEFAULT_ENV_PATH
from pipelines.forecast_publish.load_forecast_run import create_client
from pipelines.forecast_publish.sku_hour_profile_store import UPLIFT_MULTIPLIER_TABLE
from pipelines.forecast_publish.table_names import (
    get_table_suffix_from_env_file,
    table_name,
)
from pipelines.forecast_publish.assortment_override_store import (
    TABLE_BASE as OVERRIDE_TABLE_BASE,
    load_active_overrides,
)
from scripts.build_bakery_product_assortment import (
    add_city_core_for_cold_start_bakeries,
    add_network_core_for_cold_start_bakeries,
    build_assortment_from_sales,
    build_cold_start_city_core,
    build_cold_start_network_core,
    carry_forward_bakeries_without_recent_sales,
    ensure_table as ensure_bakery_product_assortment_table,
    insert_to_clickhouse as insert_bakery_product_assortment,
    load_previous_assortment,
    _query_bakery_city_map,
)
from scripts.build_city_assortment_from_sales import build_layers
from scripts.build_city_assortment_from_sales import (
    insert_to_clickhouse as insert_assortment,
)
from scripts.build_city_assortment_from_sales import (
    DEFAULT_WINDOW_DAYS as ASSORTMENT_WINDOW_DAYS,
    DEFAULT_CITY_THRESHOLD as ASSORTMENT_CITY_THRESHOLD,
    DEFAULT_BAKEABLE_CATEGORY_PATTERNS as ASSORTMENT_CATEGORY_PATTERNS,
    _query_recent_sales,
    _query_bakery_count_per_city,
)
from scripts.export_clickhouse_bakery_daily import (
    DEFAULT_OUTPUT as DEFAULT_DAILY_AGGREGATE_OUTPUT,
)
from scripts.export_clickhouse_bakery_daily import (
    DEFAULT_SQL_TEMPLATE as DEFAULT_DAILY_SQL_TEMPLATE,
)
from scripts.export_clickhouse_bakery_daily import create_client as create_export_client
from scripts.export_clickhouse_bakery_daily import export_daily_windows
from src.experiments_v2.build_bakery_daily_dataset import BAKERY_ID_COL
from src.experiments_v2.build_bakery_daily_dataset import CHUNK_SIZE
from src.experiments_v2.build_bakery_daily_dataset import DATE_COL
from src.experiments_v2.build_bakery_daily_dataset import TARGET_COL
from src.experiments_v2.build_bakery_daily_dataset import (
    build_bakery_daily_dataset_from_aggregates,
)
from src.experiments_v2.build_bakery_daily_dataset import (
    build_summary as build_daily_summary,
)
from src.experiments_v2.build_bakery_daily_dataset import (
    save_outputs as save_daily_outputs,
)
from src.experiments_v2.build_bakery_hour_profile import DOW_COL
from src.experiments_v2.build_bakery_hour_profile import HOUR_COL
from src.experiments_v2.build_uplifted_bakery_daily_dataset import BASE_TARGET_COL
from src.experiments_v2.build_uplifted_bakery_daily_dataset import (
    DEFAULT_OUTPUT_PATH as DEFAULT_UPLIFTED_OUTPUT_PATH,
)


from src.experiments_v2.build_uplifted_bakery_daily_dataset import (
    DEFAULT_SUMMARY_PATH as DEFAULT_UPLIFTED_SUMMARY_PATH,
)
from src.experiments_v2.build_uplifted_bakery_daily_dataset import UPLIFTED_TARGET_COL
from src.experiments_v2.build_uplifted_bakery_daily_dataset import UPLIFT_DELTA_COL
from src.experiments_v2.build_uplifted_bakery_daily_dataset import UPLIFT_MULTIPLIER_COL
from src.experiments_v2.build_uplifted_bakery_daily_dataset import UPLIFT_RATE_COL
from src.experiments_v2.build_uplifted_bakery_daily_dataset import (
    rebuild_target_features,
)
from src.experiments_v2.build_uplifted_bakery_daily_dataset import (
    build_summary as build_uplifted_summary_base,
)


CLOSED_BAKERY_DAYS_WITHOUT_SALES = 30


def exclude_closed_bakeries(
    daily: pd.DataFrame,
    *,
    as_of_date: str | pd.Timestamp,
    max_days_without_sales: int = CLOSED_BAKERY_DAYS_WITHOUT_SALES,
) -> tuple[pd.DataFrame, list[int]]:
    """Exclude bakeries whose last positive sale is older than the threshold."""
    required = {DATE_COL, BAKERY_ID_COL, TARGET_COL}
    missing = sorted(required.difference(daily.columns))
    if missing:
        raise KeyError(f"Daily dataset is missing columns: {missing}")
    work = daily.copy()
    work[DATE_COL] = pd.to_datetime(work[DATE_COL], errors="raise").dt.normalize()
    positive = work[pd.to_numeric(work[TARGET_COL], errors="coerce").fillna(0).gt(0)]
    last_sale = positive.groupby(BAKERY_ID_COL)[DATE_COL].max()
    age = pd.Timestamp(as_of_date).normalize() - last_sale
    closed_ids = sorted(
        int(value)
        for value in age[age.dt.days.gt(max_days_without_sales)].index
    )
    return (
        work[~work[BAKERY_ID_COL].isin(closed_ids)].reset_index(drop=True),
        closed_ids,
    )


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SQL_TEMPLATE = DEFAULT_DAILY_SQL_TEMPLATE
DEFAULT_RAW_OUTPUT = DEFAULT_DAILY_AGGREGATE_OUTPUT
DEFAULT_PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_BAKERY_HOUR_PROFILE_PATH = DEFAULT_PROCESSED_DIR / "bakery_hour_profile.csv"
DEFAULT_WEATHER_PATH = DEFAULT_PROCESSED_DIR / "bakery_weather_features.csv"
DEFAULT_REFRESH_SUMMARY_PATH = (
    ROOT / "reports" / "production_dataset_refresh_summary.json"
)
DEFAULT_HISTORY_START_DATE = "2025-06-01"
DEFAULT_TIMEZONE = "Europe/Moscow"
DEFAULT_CLICKHOUSE_RETRY_ATTEMPTS = 3
DEFAULT_CLICKHOUSE_RETRY_SECONDS = 15.0
ALLOCATION_ASSORTMENT_SOURCE = "recent_sales_window"
MANAGED_ALLOCATION_ASSORTMENT_SOURCES = (
    ALLOCATION_ASSORTMENT_SOURCE,
    "carried_forward_no_recent_sales",
)


def build_allocation_assortment(
    sales: pd.DataFrame,
    *,
    valid_from: str,
) -> pd.DataFrame:
    """Build the full city/SKU allowlist consumed by forecast allocation."""
    if sales.empty:
        return pd.DataFrame()
    result = (
        sales.groupby(["city", "product_id"], as_index=False)
        .agg(
            product_name=("product_name", "first"),
            category_name=("category_name", "first"),
        )
    )
    result["product_id"] = result["product_id"].astype(str)
    result["product_name"] = result["product_name"].fillna("")
    result["category_name"] = result["category_name"].fillna("")
    result["is_required"] = 1
    result["is_top"] = 0
    result["top_rank"] = pd.NA
    result["source"] = ALLOCATION_ASSORTMENT_SOURCE
    result["source_priority"] = 1
    result["source_file"] = f"mart_sales_60d:window_{ASSORTMENT_WINDOW_DAYS}d"
    result["source_scope"] = result["city"]
    result["valid_from"] = pd.to_datetime(valid_from).date()
    result["valid_to"] = pd.NA
    result["is_active"] = 1
    result["loaded_at"] = pd.Timestamp.now(tz="UTC")
    result["comment"] = ""
    return result[
        [
            "city",
            "product_id",
            "product_name",
            "category_name",
            "is_required",
            "is_top",
            "top_rank",
            "source",
            "source_priority",
            "source_file",
            "source_scope",
            "valid_from",
            "valid_to",
            "is_active",
            "loaded_at",
            "comment",
        ]
    ]


def delete_older_allocation_snapshot_rows(
    client,
    *,
    table: str,
    valid_from: str,
    loaded_at_cutoff: pd.Timestamp,
) -> None:
    """Remove rows left by earlier attempts for the same effective snapshot."""
    cutoff = pd.Timestamp(loaded_at_cutoff)
    if cutoff.tzinfo is None:
        raise ValueError("loaded_at_cutoff must be timezone-aware")
    client.command(
        f"alter table {table} delete where valid_from = {{valid_from:Date}} "
        "and source in {managed_sources:Array(String)} "
        "and loaded_at < {loaded_at_cutoff:DateTime64(3)} "
        "settings mutations_sync = 2",
        parameters={
            "valid_from": valid_from,
            "managed_sources": list(MANAGED_ALLOCATION_ASSORTMENT_SOURCES),
            "loaded_at_cutoff": cutoff.to_pydatetime(),
        },
    )


def load_latest_allocation_assortment_products(
    client,
    *,
    table: str,
    cities: list[str],
    effective_date: str,
) -> pd.DataFrame:
    if not cities:
        return pd.DataFrame()
    return client.query_df(
        f"""
        select
            a.city,
            a.product_id,
            any(a.product_name) as product_name,
            any(a.category_name) as category_name
        from {table} as a final
        inner join (
            select city, max(valid_from) as latest_valid_from
            from {table} final
            where city in %(cities)s
              and valid_from <= toDate(%(effective_date)s)
            group by city
        ) as latest
          on a.city = latest.city and a.valid_from = latest.latest_valid_from
        where a.is_active = 1
          and (a.valid_to is null or a.valid_to >= toDate(%(effective_date)s))
        group by a.city, a.product_id
        """,
        parameters={"cities": cities, "effective_date": effective_date},
    )


@dataclass(frozen=True)
class RefreshDates:
    history_end_date: str
    forecast_start_date: str


def resolve_default_refresh_dates(
    now: pd.Timestamp | None = None,
    *,
    timezone: str = DEFAULT_TIMEZONE,
) -> RefreshDates:
    if now is None:
        current = pd.Timestamp.now(tz=ZoneInfo(timezone)).normalize()
    else:
        current = pd.Timestamp(now)
        if current.tzinfo is None:
            current = current.tz_localize(ZoneInfo(timezone))
        else:
            current = current.tz_convert(ZoneInfo(timezone))
        current = current.normalize()
    return RefreshDates(
        history_end_date=str((current - pd.Timedelta(days=1)).date()),
        forecast_start_date=str(current.date()),
    )


def _normalize_key_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    work = df.copy()
    for col in columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(-1).astype("int64")
    return work


def create_client_with_retry(
    factory,
    env_file: str | Path,
    *,
    attempts: int = DEFAULT_CLICKHOUSE_RETRY_ATTEMPTS,
    sleep_seconds: float = DEFAULT_CLICKHOUSE_RETRY_SECONDS,
):
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return factory(env_file)
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            print(
                "ClickHouse client connection failed "
                f"(attempt {attempt}/{attempts}): {exc}; "
                f"retrying in {sleep_seconds:g}s",
                flush=True,
            )
            time.sleep(sleep_seconds)
    assert last_exc is not None
    raise last_exc


def load_uplift_multipliers_from_clickhouse(
    *,
    env_file: str | Path = DEFAULT_ENV_PATH,
    uplift_table: str = UPLIFT_MULTIPLIER_TABLE,
    profile_version: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    client = create_client_with_retry(create_client, env_file)
    where = ""
    if profile_version:
        safe_version = profile_version.replace("'", "''")
        where = f"where profile_version = '{safe_version}'"

    exact = client.query_df(
        f"""
        select
            bakery_id,
            dow,
            hour,
            argMax(sku_uplift_multiplier, generated_at) as sku_uplift_multiplier
        from {uplift_table}
        {where}
        {"and" if where else "where"} dow >= 0
        group by bakery_id, dow, hour
        """
    )
    fallback = client.query_df(
        f"""
        select
            bakery_id,
            hour,
            argMax(sku_uplift_multiplier, generated_at) as sku_uplift_multiplier
        from {uplift_table}
        {where}
        {"and" if where else "where"} dow = -1
        group by bakery_id, hour
        """
    )
    # clickhouse-connect can return a columnless empty DataFrame for GROUP BY on 0 rows
    if BAKERY_ID_COL not in exact.columns:
        exact = pd.DataFrame(
            columns=[BAKERY_ID_COL, DOW_COL, HOUR_COL, "sku_uplift_multiplier"]
        )
    if BAKERY_ID_COL not in fallback.columns:
        fallback = pd.DataFrame(
            columns=[BAKERY_ID_COL, HOUR_COL, "sku_uplift_multiplier"]
        )
    exact = _normalize_key_columns(exact, [BAKERY_ID_COL, DOW_COL, HOUR_COL])
    fallback = _normalize_key_columns(fallback, [BAKERY_ID_COL, HOUR_COL])
    return exact, fallback


def prepare_bakery_hour_profile_for_refresh(
    profile: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {BAKERY_ID_COL, DOW_COL, HOUR_COL, "mean_hour_share_norm"}
    missing = sorted(required.difference(profile.columns))
    if missing:
        raise ValueError("Bakery hour profile missing columns: " + ", ".join(missing))

    exact = profile[[BAKERY_ID_COL, DOW_COL, HOUR_COL, "mean_hour_share_norm"]].copy()
    exact = _normalize_key_columns(exact, [BAKERY_ID_COL, DOW_COL, HOUR_COL])
    exact["mean_hour_share_norm"] = pd.to_numeric(
        exact["mean_hour_share_norm"],
        errors="coerce",
    ).fillna(0.0)

    fallback = (
        exact.groupby([BAKERY_ID_COL, HOUR_COL], as_index=False)["mean_hour_share_norm"]
        .mean()
        .rename(columns={"mean_hour_share_norm": "fallback_hour_share"})
    )
    totals = (
        fallback.groupby(BAKERY_ID_COL, as_index=False)["fallback_hour_share"]
        .sum()
        .rename(columns={"fallback_hour_share": "_fallback_sum"})
    )
    fallback = fallback.merge(totals, on=BAKERY_ID_COL, how="left")
    fallback["fallback_hour_share"] = (
        fallback["fallback_hour_share"] / fallback["_fallback_sum"].replace(0.0, pd.NA)
    ).fillna(0.0)
    return exact, fallback.drop(columns=["_fallback_sum"])


def build_uplifted_daily_from_clickhouse_multipliers(
    daily: pd.DataFrame,
    bakery_hour_profile: pd.DataFrame,
    exact_multipliers: pd.DataFrame,
    fallback_multipliers: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    base = daily.copy()
    base[DATE_COL] = pd.to_datetime(base[DATE_COL], errors="coerce").dt.normalize()
    base[BAKERY_ID_COL] = pd.to_numeric(base[BAKERY_ID_COL], errors="coerce")
    base = base.dropna(subset=[DATE_COL, BAKERY_ID_COL]).copy()
    base[BAKERY_ID_COL] = base[BAKERY_ID_COL].astype("int64")
    base[DOW_COL] = base[DATE_COL].dt.dayofweek.astype("int64")
    base[TARGET_COL] = pd.to_numeric(base[TARGET_COL], errors="coerce").fillna(0.0)

    exact_profile, fallback_profile = prepare_bakery_hour_profile_for_refresh(
        bakery_hour_profile,
    )
    hourly = base[[DATE_COL, BAKERY_ID_COL, DOW_COL, TARGET_COL]].merge(
        exact_profile,
        on=[BAKERY_ID_COL, DOW_COL],
        how="left",
        validate="many_to_many",
    )
    missing_profile = hourly["mean_hour_share_norm"].isna()
    if missing_profile.any():
        missing = hourly.loc[
            missing_profile,
            [DATE_COL, BAKERY_ID_COL, DOW_COL, TARGET_COL],
        ].drop_duplicates()
        fallback_hourly = missing.merge(fallback_profile, on=BAKERY_ID_COL, how="left")
        fallback_hourly = fallback_hourly.rename(
            columns={"fallback_hour_share": "mean_hour_share_norm"},
        )
        hourly = pd.concat(
            [hourly.loc[~missing_profile], fallback_hourly],
            ignore_index=True,
        )

    hourly["mean_hour_share_norm"] = pd.to_numeric(
        hourly["mean_hour_share_norm"],
        errors="coerce",
    ).fillna(0.0)
    hourly["bakery_hour_sales"] = hourly[TARGET_COL] * hourly["mean_hour_share_norm"]

    hourly = hourly.merge(
        exact_multipliers,
        on=[BAKERY_ID_COL, DOW_COL, HOUR_COL],
        how="left",
        validate="many_to_one",
    )
    if not fallback_multipliers.empty:
        hourly = hourly.merge(
            fallback_multipliers.rename(
                columns={"sku_uplift_multiplier": "_fallback_uplift_multiplier"}
            ),
            on=[BAKERY_ID_COL, HOUR_COL],
            how="left",
            validate="many_to_one",
        )
    else:
        hourly["_fallback_uplift_multiplier"] = pd.NA
    hourly[UPLIFT_MULTIPLIER_COL] = pd.to_numeric(
        hourly["sku_uplift_multiplier"],
        errors="coerce",
    ).fillna(
        pd.to_numeric(hourly["_fallback_uplift_multiplier"], errors="coerce"),
    ).fillna(1.0)
    hourly[UPLIFT_MULTIPLIER_COL] = hourly[UPLIFT_MULTIPLIER_COL].clip(lower=1.0)
    hourly["bakery_hour_sales_uplifted"] = (
        hourly["bakery_hour_sales"] * hourly[UPLIFT_MULTIPLIER_COL]
    )

    uplift = (
        hourly.groupby([DATE_COL, BAKERY_ID_COL], as_index=False)
        .agg(
            bakery_sales_from_hours=("bakery_hour_sales", "sum"),
            bakery_sales_uplifted_from_hours=("bakery_hour_sales_uplifted", "sum"),
        )
    )
    uplift[UPLIFT_MULTIPLIER_COL] = (
        uplift["bakery_sales_uplifted_from_hours"]
        / uplift["bakery_sales_from_hours"].replace(0.0, pd.NA)
    ).fillna(1.0).clip(lower=1.0)

    work = base.merge(uplift, on=[DATE_COL, BAKERY_ID_COL], how="left")
    work[BASE_TARGET_COL] = pd.to_numeric(work[TARGET_COL], errors="coerce").fillna(0.0)
    work[UPLIFT_MULTIPLIER_COL] = pd.to_numeric(
        work[UPLIFT_MULTIPLIER_COL],
        errors="coerce",
    ).fillna(1.0).clip(lower=1.0)
    work[UPLIFTED_TARGET_COL] = work[BASE_TARGET_COL] * work[UPLIFT_MULTIPLIER_COL]
    work[UPLIFT_DELTA_COL] = work[UPLIFTED_TARGET_COL] - work[BASE_TARGET_COL]
    work[UPLIFT_RATE_COL] = (
        work[UPLIFT_DELTA_COL] / work[BASE_TARGET_COL].replace(0.0, pd.NA)
    ).fillna(0.0)
    work[TARGET_COL] = work[UPLIFTED_TARGET_COL].clip(lower=0.0)
    work = rebuild_target_features(work)

    summary = build_uplifted_summary_base(work)
    summary.update(
        {
            "target": TARGET_COL,
            "base_target_col": BASE_TARGET_COL,
            "uplifted_target_col": UPLIFTED_TARGET_COL,
            "mean_base_bakery_sales": round(float(work[BASE_TARGET_COL].mean()), 6),
            "mean_uplifted_bakery_sales": round(
                float(work[UPLIFTED_TARGET_COL].mean()),
                6,
            ),
            "mean_uplift_delta": round(float(work[UPLIFT_DELTA_COL].mean()), 6),
            "mean_uplift_rate": round(float(work[UPLIFT_RATE_COL].mean()), 6),
            "p95_uplift_rate": round(float(work[UPLIFT_RATE_COL].quantile(0.95)), 6),
            "max_uplift_rate": round(float(work[UPLIFT_RATE_COL].max()), 6),
            "uplifted_rows": int((work[UPLIFT_DELTA_COL] > 1e-9).sum()),
            "uplift_source": "clickhouse_uplift_multipliers",
        }
    )
    return work, summary


def refresh_weather_features_with_fallback(
    *,
    dataset_paths: list[Path],
    horizon_days: int,
    weather_path: str | Path,
) -> dict[str, object]:
    from src.experiments_v2.build_bakery_weather_features import (
        fetch_weather_features,
    )
    from src.experiments_v2.build_bakery_weather_features import (
        infer_weather_request,
    )

    weather_output = Path(weather_path)
    weather_cities, weather_start, weather_end = infer_weather_request(
        dataset_paths,
        horizon_days=int(horizon_days),
    )
    weather_start_value = str(pd.Timestamp(weather_start).date())
    weather_end_value = str(pd.Timestamp(weather_end).date())
    try:
        weather_df = fetch_weather_features(
            weather_cities,
            start_date=weather_start,
            end_date=weather_end,
        )
    except Exception as exc:
        if not weather_output.exists():
            raise
        fallback = pd.read_csv(weather_output, encoding="utf-8-sig")
        return {
            "weather_rows": int(len(fallback)),
            "weather_status": "existing_file_fallback",
            "weather_error": str(exc),
            "weather_start_date": weather_start_value,
            "weather_end_date": weather_end_value,
        }

    weather_output.parent.mkdir(parents=True, exist_ok=True)
    weather_df.to_csv(weather_output, index=False, encoding="utf-8-sig")
    return {
        "weather_rows": int(len(weather_df)),
        "weather_status": "refreshed",
        "weather_error": None,
        "weather_start_date": weather_start_value,
        "weather_end_date": weather_end_value,
    }


def refresh_production_datasets(
    *,
    env_file: str | Path = DEFAULT_ENV_PATH,
    sql_template: str | Path = DEFAULT_SQL_TEMPLATE,
    raw_output: str | Path = DEFAULT_RAW_OUTPUT,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    history_start_date: str = DEFAULT_HISTORY_START_DATE,
    history_end_date: str,
    horizon_days: int,
    bakery_hour_profile_path: str | Path = DEFAULT_BAKERY_HOUR_PROFILE_PATH,
    uplifted_output_path: str | Path = DEFAULT_UPLIFTED_OUTPUT_PATH,
    uplifted_summary_path: str | Path = DEFAULT_UPLIFTED_SUMMARY_PATH,
    weather_path: str | Path = DEFAULT_WEATHER_PATH,
    uplift_table: str = UPLIFT_MULTIPLIER_TABLE,
    uplift_profile_version: str | None = None,
    refresh_weather: bool = True,
    chunk_size: int = CHUNK_SIZE,
) -> dict:
    sql_template_text = Path(sql_template).read_text(encoding="utf-8")
    aggregate_export = export_daily_windows(
        client=create_client_with_retry(create_export_client, env_file),
        sql_template_text=sql_template_text,
        output_path=Path(raw_output),
        date_from=history_start_date,
        date_to=history_end_date,
        batch_mode="monthly",
        limit=None,
    )

    daily_aggregate = pd.read_csv(raw_output, encoding="utf-8-sig")
    daily_df = build_bakery_daily_dataset_from_aggregates(daily_aggregate)
    daily_df, closed_bakery_ids = exclude_closed_bakeries(
        daily_df,
        as_of_date=history_end_date,
    )
    active_bakery_ids = set(daily_df[BAKERY_ID_COL].astype(int).unique())
    print(
        "Closed bakery filter: "
        f"excluded={len(closed_bakery_ids)} threshold_days="
        f"{CLOSED_BAKERY_DAYS_WITHOUT_SALES}",
        flush=True,
    )
    daily_summary = build_daily_summary(daily_df)
    daily_paths = save_daily_outputs(processed_dir, daily_df, daily_summary)

    bakery_hour_profile = pd.read_csv(bakery_hour_profile_path, encoding="utf-8-sig")
    exact_multipliers, fallback_multipliers = load_uplift_multipliers_from_clickhouse(
        env_file=env_file,
        uplift_table=uplift_table,
        profile_version=uplift_profile_version,
    )
    uplifted_df, uplifted_summary = build_uplifted_daily_from_clickhouse_multipliers(
        daily_df,
        bakery_hour_profile,
        exact_multipliers,
        fallback_multipliers,
    )
    uplifted_output = Path(uplifted_output_path)
    uplifted_summary_output = Path(uplifted_summary_path)
    uplifted_output.parent.mkdir(parents=True, exist_ok=True)
    uplifted_df.to_csv(uplifted_output, index=False, encoding="utf-8-sig")
    uplifted_summary_output.write_text(
        json.dumps(uplifted_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    weather_result: dict[str, object] = {
        "weather_rows": None,
        "weather_status": "skipped",
        "weather_error": None,
        "weather_start_date": None,
        "weather_end_date": None,
    }
    if refresh_weather:
        weather_result = refresh_weather_features_with_fallback(
            dataset_paths=[Path(daily_paths["dataset"]), uplifted_output],
            horizon_days=int(horizon_days),
            weather_path=weather_path,
        )

    # Refresh bakeable assortment from recent sales facts (city + bakery layers)
    assortment_result: dict[str, object] = {
        "assortment_city_rows": None,
        "assortment_bakery_rows": None,
        "assortment_status": "skipped",
        "assortment_error": None,
        "allocation_assortment_rows": None,
    }
    try:
        assortment_client = create_client_with_retry(create_client, env_file)
        suffix = get_table_suffix_from_env_file(env_file)
        bakery_tbl = table_name("bakery_forecast_day_embedded", suffix=suffix)
        sku_day_tbl = table_name("sku_forecast_day_embedded", suffix=suffix)
        bakeable_tbl = table_name("bakeable_products", suffix=suffix)
        allocation_assortment_tbl = table_name(
            "assortment_city_products", suffix=suffix
        )
        sales = _query_recent_sales(
            assortment_client,
            window_days=ASSORTMENT_WINDOW_DAYS,
            bakery_table=bakery_tbl,
            sku_day_table=sku_day_tbl,
            sales_table="mart_sales_60d",
        )
        bakery_counts = _query_bakery_count_per_city(
            assortment_client, bakery_table=bakery_tbl
        )
        bakery_city_map = _query_bakery_city_map(
            assortment_client, bakery_table=bakery_tbl
        )
        bakery_city_map = bakery_city_map[
            bakery_city_map["bakery_id"].astype(int).isin(active_bakery_ids)
        ].copy()
        bakery_counts = (
            bakery_city_map.groupby("city", as_index=False)["bakery_id"]
            .nunique()
            .rename(columns={"bakery_id": "total_bakeries"})
        )
        valid_from = history_end_date
        assortment_df = build_layers(
            sales,
            bakery_counts,
            city_threshold=ASSORTMENT_CITY_THRESHOLD,
            category_patterns=ASSORTMENT_CATEGORY_PATTERNS,
            valid_from=valid_from,
        )
        inserted = insert_assortment(
            assortment_client,
            assortment_df,
            target_table=bakeable_tbl,
        )
        allocation_assortment_df = build_allocation_assortment(
            sales,
            valid_from=valid_from,
        )
        expected_cities = set(bakery_counts["city"].dropna().astype(str))
        refreshed_cities = set(allocation_assortment_df["city"].astype(str))
        carried_cities = sorted(expected_cities - refreshed_cities)
        if carried_cities:
            carried_products = load_latest_allocation_assortment_products(
                assortment_client,
                table=allocation_assortment_tbl,
                cities=carried_cities,
                effective_date=valid_from,
            )
            found_cities = set(carried_products["city"].dropna().astype(str))
            unavailable = sorted(set(carried_cities) - found_cities)
            if unavailable:
                raise RuntimeError(
                    "No recent sales or previous allocation assortment for cities: "
                    f"{unavailable}"
                )
            carried_assortment = build_allocation_assortment(
                carried_products,
                valid_from=valid_from,
            )
            carried_assortment["source"] = "carried_forward_no_recent_sales"
            carried_assortment["source_file"] = "previous_allocation_assortment"
            carried_assortment["comment"] = "No rows in current mart_sales_60d window"
            allocation_assortment_df = pd.concat(
                [allocation_assortment_df, carried_assortment],
                ignore_index=True,
            )
        allocation_inserted = insert_assortment(
            assortment_client,
            allocation_assortment_df,
            target_table=allocation_assortment_tbl,
        )
        delete_older_allocation_snapshot_rows(
            assortment_client,
            table=allocation_assortment_tbl,
            valid_from=valid_from,
            loaded_at_cutoff=pd.to_datetime(
                allocation_assortment_df["loaded_at"], errors="raise"
            ).min(),
        )
        city_rows = (
            int((assortment_df["scope"] == "city").sum())
            if not assortment_df.empty
            else 0
        )
        bakery_rows = (
            int((assortment_df["scope"] == "bakery").sum())
            if not assortment_df.empty
            else 0
        )
        assortment_result = {
            "assortment_city_rows": city_rows,
            "assortment_bakery_rows": bakery_rows,
            "assortment_status": "refreshed",
            "assortment_error": None,
            "allocation_assortment_rows": int(len(allocation_assortment_df)),
            "allocation_assortment_carried_cities": carried_cities,
        }
        print(
            f"Assortment refresh: city={city_rows} bakery={bakery_rows} "
            f"inserted={inserted} allocation_inserted={allocation_inserted} "
            f"valid_from={valid_from}",
            flush=True,
        )

        bakery_product_tbl = table_name(
            "bakery_product_assortment_embedded", suffix=suffix
        )
        bakery_product_df = build_assortment_from_sales(
            sales,
            valid_from=valid_from,
            overrides=load_active_overrides(
                assortment_client,
                table=table_name(OVERRIDE_TABLE_BASE, suffix=suffix),
                effective_date=valid_from,
            ),
        )
        required_bakery_ids = sorted(active_bakery_ids)
        missing_bakery_ids = sorted(
            set(required_bakery_ids)
            - set(bakery_product_df["bakery_id"].astype(int))
        )
        previous_bakery_product_df = load_previous_assortment(
            assortment_client,
            table=bakery_product_tbl,
            bakery_ids=missing_bakery_ids,
            before_date=valid_from,
        )
        bakery_product_df, carried_bakery_ids = (
            carry_forward_bakeries_without_recent_sales(
                bakery_product_df,
                previous_bakery_product_df,
                required_bakery_ids=required_bakery_ids,
                valid_from=valid_from,
            )
        )
        bakery_product_df, cold_start_bakery_ids = (
            add_city_core_for_cold_start_bakeries(
                bakery_product_df,
                bakery_city_map,
                build_cold_start_city_core(sales),
                required_bakery_ids=required_bakery_ids,
                valid_from=valid_from,
            )
        )
        bakery_product_df, network_cold_start_bakery_ids = (
            add_network_core_for_cold_start_bakeries(
                bakery_product_df,
                build_cold_start_network_core(sales),
                required_bakery_ids=required_bakery_ids,
                valid_from=valid_from,
            )
        )
        ensure_bakery_product_assortment_table(
            assortment_client, bakery_product_tbl
        )
        insert_bakery_product_assortment(
            assortment_client,
            bakery_product_df,
            target_table=bakery_product_tbl,
        )
        bp_rows = len(bakery_product_df)
        bp_bakeries = int(bakery_product_df["bakery_id"].nunique())
        bp_products = int(bakery_product_df["product_id"].nunique())
        assortment_result["bakery_product_assortment_rows"] = bp_rows
        assortment_result["bakery_product_assortment_bakeries"] = bp_bakeries
        assortment_result["bakery_product_assortment_carried_bakeries"] = (
            carried_bakery_ids
        )
        assortment_result["bakery_product_assortment_cold_start_bakeries"] = (
            cold_start_bakery_ids
        )
        assortment_result[
            "bakery_product_assortment_network_cold_start_bakeries"
        ] = network_cold_start_bakery_ids
        print(
            f"Bakery-product assortment: bakeries={bp_bakeries} "
            f"products={bp_products} rows={bp_rows}",
            flush=True,
        )
    except Exception as exc:
        assortment_result["assortment_error"] = str(exc)
        assortment_result["assortment_status"] = "failed"
        print(f"Assortment refresh FAILED: {exc}", flush=True)

    return {
        "history_start_date": history_start_date,
        "history_end_date": history_end_date,
        "closed_bakery_days_without_sales": CLOSED_BAKERY_DAYS_WITHOUT_SALES,
        "closed_bakery_ids": closed_bakery_ids,
        "daily_aggregate_output": str(Path(raw_output)),
        "daily_aggregate_rows": aggregate_export["rows"],
        "base_dataset_path": str(daily_paths["dataset"]),
        "base_summary_path": str(daily_paths["summary"]),
        "uplifted_dataset_path": str(uplifted_output),
        "uplifted_summary_path": str(uplifted_summary_output),
        "weather_path": str(Path(weather_path)),
        **weather_result,
        **assortment_result,
        "daily_summary": daily_summary,
        "uplifted_summary": uplifted_summary,
    }

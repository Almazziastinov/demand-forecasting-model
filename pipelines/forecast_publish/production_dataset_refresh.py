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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SQL_TEMPLATE = DEFAULT_DAILY_SQL_TEMPLATE
DEFAULT_RAW_OUTPUT = DEFAULT_DAILY_AGGREGATE_OUTPUT
DEFAULT_PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_BAKERY_HOUR_PROFILE_PATH = DEFAULT_PROCESSED_DIR / "bakery_hour_profile.csv"
DEFAULT_WEATHER_PATH = DEFAULT_PROCESSED_DIR / "bakery_weather_features.csv"
DEFAULT_REFRESH_SUMMARY_PATH = (
    ROOT / "reports" / "production_dataset_refresh_summary.json"
)
DEFAULT_HISTORY_START_DATE = "2025-12-01"
DEFAULT_TIMEZONE = "Europe/Moscow"
DEFAULT_CLICKHOUSE_RETRY_ATTEMPTS = 3
DEFAULT_CLICKHOUSE_RETRY_SECONDS = 15.0


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

    return {
        "history_start_date": history_start_date,
        "history_end_date": history_end_date,
        "daily_aggregate_output": str(Path(raw_output)),
        "daily_aggregate_rows": aggregate_export["rows"],
        "base_dataset_path": str(daily_paths["dataset"]),
        "base_summary_path": str(daily_paths["summary"]),
        "uplifted_dataset_path": str(uplifted_output),
        "uplifted_summary_path": str(uplifted_summary_output),
        "weather_path": str(Path(weather_path)),
        **weather_result,
        "daily_summary": daily_summary,
        "uplifted_summary": uplifted_summary,
    }

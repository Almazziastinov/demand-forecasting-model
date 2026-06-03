from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.experiments_v2.bakery_day_forecast import CITY_COL
from src.experiments_v2.bakery_day_forecast import DATE_COL
from src.experiments_v2.bakery_day_forecast import DEFAULT_WEATHER_PATH
from src.experiments_v2.bakery_day_forecast import WEATHER_DEFAULTS
from src.experiments_v2.bakery_day_forecast import add_enriched_event_features
from src.experiments_v2.bakery_day_forecast import add_event_cluster_features
from src.experiments_v2.bakery_day_forecast import add_holiday_features
from src.experiments_v2.bakery_day_forecast import attach_weather_features
from src.experiments_v2.bakery_day_forecast import load_weather_features


CONTEXT_TABLE = "forecast_day_context_embedded"


def _load_optional_weather(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    weather_path = Path(path)
    if not weather_path.exists():
        raise FileNotFoundError(f"Weather features not found: {weather_path}")
    return load_weather_features(weather_path)


def prepare_forecast_context(
    bakery_df: pd.DataFrame,
    run_id: str,
    *,
    weather_path: str | Path | None = DEFAULT_WEATHER_PATH,
) -> pd.DataFrame:
    work = bakery_df[[DATE_COL, CITY_COL]].drop_duplicates().copy()
    work[DATE_COL] = pd.to_datetime(work[DATE_COL], errors="coerce").dt.normalize()
    work[CITY_COL] = work[CITY_COL].fillna("unknown").astype(str)
    work = work.dropna(subset=[DATE_COL])
    work = work.drop_duplicates([DATE_COL, CITY_COL]).reset_index(drop=True)
    work["dow"] = work[DATE_COL].dt.dayofweek.astype(int)

    weather_df = _load_optional_weather(weather_path)
    work = attach_weather_features(work, weather_df)
    work = add_holiday_features(work)
    work = add_event_cluster_features(work)
    work = add_enriched_event_features(work)

    for col, default in WEATHER_DEFAULTS.items():
        if col not in work.columns:
            work[col] = default
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(default)

    work["run_id"] = run_id
    work["forecast_date"] = work[DATE_COL].dt.date
    int_defaults = {
        "is_holiday": 0,
        "is_pre_holiday": 0,
        "is_post_holiday": 0,
        "is_bad_weather": 0,
        "days_to_next_event": 999,
        "days_since_prev_event": 999,
    }
    for col, default in int_defaults.items():
        work[col] = (
            pd.to_numeric(work[col], errors="coerce")
            .fillna(default)
            .astype(int)
        )

    return work[
        [
            "run_id",
            "forecast_date",
            CITY_COL,
            "temp_mean",
            "precipitation",
            "rain",
            "snowfall",
            "windspeed_max",
            "is_bad_weather",
            "weather_cat_code",
            "holiday_name",
            "is_holiday",
            "is_pre_holiday",
            "is_post_holiday",
            "event_window_type",
            "current_event_cluster",
            "prev_event_cluster",
            "next_event_cluster",
            "days_since_prev_event",
            "days_to_next_event",
        ]
    ].drop_duplicates(["run_id", "forecast_date", CITY_COL])

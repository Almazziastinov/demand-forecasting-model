from __future__ import annotations

import argparse
from pathlib import Path

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

from src.experiments_v2.bakery_day_forecast import CITY_COL
from src.experiments_v2.bakery_day_forecast import DATE_COL
from src.experiments_v2.bakery_day_forecast import DEFAULT_DATASET_PATH
from src.experiments_v2.bakery_day_forecast import DEFAULT_HORIZON_DAYS
from src.experiments_v2.bakery_day_forecast import DEFAULT_WEATHER_PATH
from src.experiments_v2.bakery_day_forecast import WEATHER_FEATURES
from src.experiments_v2.bakery_day_forecast import normalize_weather_frame


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_PATH = ROOT / "data" / ".weather_cache"
DEFAULT_UPLIFTED_DATASET_PATH = (
    ROOT / "data" / "processed" / "bakery_daily_sales_uplifted.csv"
)

CITY_COORDS = {
    "Бугульма": {"lat": 54.5393, "lon": 52.7959},
    "Заинск": {"lat": 55.2978, "lon": 52.0046},
    "Зеленодольск": {"lat": 55.8430, "lon": 48.5206},
    "Казань": {"lat": 55.7879, "lon": 49.1233},
    "Курск": {"lat": 51.7373, "lon": 36.1874},
    "Москва": {"lat": 55.7558, "lon": 37.6173},
    "Набережные Челны": {"lat": 55.7439, "lon": 52.4042},
    "Нижнекамск": {"lat": 55.6386, "lon": 51.8221},
    "Новокузнецк": {"lat": 53.7576, "lon": 87.1361},
    "Чебоксары": {"lat": 56.1439, "lon": 47.2489},
}

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "windspeed_10m_max",
    "weathercode",
]


def _weather_category(code: float) -> str:
    if pd.isna(code):
        return "clear"
    value = int(code)
    if value == 0:
        return "clear"
    if value <= 3:
        return "cloudy"
    if value <= 48:
        return "fog"
    if value <= 67:
        return "rain"
    if value <= 77:
        return "snow"
    if value <= 82:
        return "rain"
    return "storm"


def _response_to_frame(response: object, city: str) -> pd.DataFrame:
    daily = response.Daily()
    values = {
        "temp_max": daily.Variables(0).ValuesAsNumpy(),
        "temp_min": daily.Variables(1).ValuesAsNumpy(),
        "temp_mean": daily.Variables(2).ValuesAsNumpy(),
        "precipitation": daily.Variables(3).ValuesAsNumpy(),
        "rain": daily.Variables(4).ValuesAsNumpy(),
        "snowfall": daily.Variables(5).ValuesAsNumpy(),
        "windspeed_max": daily.Variables(6).ValuesAsNumpy(),
        "weathercode": daily.Variables(7).ValuesAsNumpy(),
    }
    days = len(values["temp_max"])
    start = (
        pd.to_datetime(daily.Time(), unit="s", utc=True)
        .tz_convert("Europe/Moscow")
        .tz_localize(None)
    )
    dates = pd.date_range(
        start=start,
        periods=days,
        freq="D",
    )
    frame = pd.DataFrame({CITY_COL: city, DATE_COL: dates})
    for col, values_array in values.items():
        frame[col] = values_array[:days]
    return frame


def _enrich_weather(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    temp_max = pd.to_numeric(work["temp_max"], errors="coerce")
    temp_min = pd.to_numeric(work["temp_min"], errors="coerce")
    temp_mean = pd.to_numeric(work["temp_mean"], errors="coerce").fillna(10.0)
    precipitation = pd.to_numeric(work["precipitation"], errors="coerce").fillna(0.0)
    rain = pd.to_numeric(work["rain"], errors="coerce").fillna(0.0)
    snowfall = pd.to_numeric(work["snowfall"], errors="coerce").fillna(0.0)
    windspeed = pd.to_numeric(work["windspeed_max"], errors="coerce").fillna(0.0)
    categories = work["weathercode"].apply(_weather_category)

    work["temp_range"] = temp_max - temp_min
    work["is_rainy"] = (
        (rain > 1.0)
        | ((precipitation > 1.0) & categories.isin(["rain", "storm"]))
    ).astype(int)
    work["is_snowy"] = (snowfall > 0.5).astype(int)
    work["is_cold"] = (temp_mean < 0.0).astype(int)
    work["is_warm"] = (temp_mean >= 15.0).astype(int)
    work["is_windy"] = (windspeed > 30.0).astype(int)
    work["is_bad_weather"] = (
        (work["is_rainy"] == 1)
        | (work["is_snowy"] == 1)
        | (work["is_windy"] == 1)
        | (precipitation > 10.0)
        | categories.eq("storm")
    ).astype(int)
    work["weather_cat_code"] = categories.map(
        {"clear": 0, "cloudy": 1, "fog": 2, "rain": 3, "snow": 4, "storm": 5}
    ).fillna(0)
    return normalize_weather_frame(work[[DATE_COL, CITY_COL, *WEATHER_FEATURES]])


def _create_client(cache_path: Path) -> openmeteo_requests.Client:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_session = requests_cache.CachedSession(str(cache_path), expire_after=3600)
    retry_session = retry(cache_session, retries=3, backoff_factor=0.3)
    return openmeteo_requests.Client(session=retry_session)


def _fetch_endpoint(
    client: openmeteo_requests.Client,
    *,
    endpoint: str,
    city: str,
    coords: dict[str, float],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    params: dict[str, object] = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "daily": DAILY_VARS,
        "timezone": "Europe/Moscow",
    }
    if endpoint == "archive":
        params["start_date"] = start_date.strftime("%Y-%m-%d")
        params["end_date"] = end_date.strftime("%Y-%m-%d")
        url = "https://archive-api.open-meteo.com/v1/archive"
    else:
        today = pd.Timestamp.now(tz="Europe/Moscow").normalize().tz_localize(None)
        forecast_days = max(1, int((end_date.normalize() - today).days) + 1)
        params["forecast_days"] = forecast_days
        url = "https://api.open-meteo.com/v1/forecast"

    responses = client.weather_api(url, params=params)
    frame = _response_to_frame(responses[0], city)
    mask = (frame[DATE_COL] >= start_date.normalize()) & (
        frame[DATE_COL] <= end_date.normalize()
    )
    return frame[mask]


def fetch_weather_features(
    cities: list[str],
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
) -> pd.DataFrame:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError("end_date must be greater than or equal to start_date")

    known_cities = [city for city in cities if city in CITY_COORDS]
    unknown_cities = sorted(set(cities) - set(known_cities) - {"unknown"})
    if unknown_cities:
        raise KeyError(f"Missing weather coordinates for cities: {unknown_cities}")

    today = pd.Timestamp.now(tz="Europe/Moscow").normalize().tz_localize(None)
    archive_end = min(end, today - pd.Timedelta(days=1))
    forecast_start = max(start, today)
    client = _create_client(Path(cache_path))

    frames: list[pd.DataFrame] = []
    for city in known_cities:
        coords = CITY_COORDS[city]
        if start <= archive_end:
            frames.append(
                _fetch_endpoint(
                    client,
                    endpoint="archive",
                    city=city,
                    coords=coords,
                    start_date=start,
                    end_date=archive_end,
                )
            )
        if forecast_start <= end:
            frames.append(
                _fetch_endpoint(
                    client,
                    endpoint="forecast",
                    city=city,
                    coords=coords,
                    start_date=forecast_start,
                    end_date=end,
                )
            )

    if not frames:
        return pd.DataFrame(columns=[DATE_COL, CITY_COL, *WEATHER_FEATURES])
    return _enrich_weather(pd.concat(frames, ignore_index=True))


def infer_weather_request(
    dataset_paths: list[str | Path],
    *,
    horizon_days: int,
) -> tuple[list[str], pd.Timestamp, pd.Timestamp]:
    frames: list[pd.DataFrame] = []
    for path in dataset_paths:
        dataset_path = Path(path)
        if dataset_path.exists():
            frames.append(
                pd.read_csv(
                    dataset_path,
                    usecols=[DATE_COL, CITY_COL],
                    encoding="utf-8-sig",
                )
            )
    if not frames:
        raise FileNotFoundError("No existing dataset paths were provided")

    data = pd.concat(frames, ignore_index=True)
    dates = pd.to_datetime(data[DATE_COL], errors="coerce").dropna()
    cities = sorted(data[CITY_COL].fillna("unknown").astype(str).unique())
    start_date = dates.min().normalize()
    end_date = dates.max().normalize() + pd.Timedelta(days=int(horizon_days))
    return cities, start_date, end_date


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build production bakery weather feature CSV"
    )
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument(
        "--uplifted-dataset-path",
        default=str(DEFAULT_UPLIFTED_DATASET_PATH),
    )
    parser.add_argument("--output-path", default=str(DEFAULT_WEATHER_PATH))
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--cache-path", default=str(DEFAULT_CACHE_PATH))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cities, inferred_start, inferred_end = infer_weather_request(
        [args.dataset_path, args.uplifted_dataset_path],
        horizon_days=args.horizon_days,
    )
    start_date = (
        pd.Timestamp(args.start_date).normalize() if args.start_date else inferred_start
    )
    end_date = (
        pd.Timestamp(args.end_date).normalize() if args.end_date else inferred_end
    )

    weather = fetch_weather_features(
        cities,
        start_date=start_date,
        end_date=end_date,
        cache_path=args.cache_path,
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(output_path, index=False, encoding="utf-8-sig")

    known_city_count = len([city for city in cities if city in CITY_COORDS])
    expected_rows = known_city_count * (int((end_date - start_date).days) + 1)
    missing_ratio = float(1.0 - len(weather) / max(1, expected_rows))
    city_count = weather[CITY_COL].nunique() if not weather.empty else 0
    print(f"weather rows: {len(weather):,}")
    print(f"cities: {city_count}/{len(cities)}")
    print(f"date range: {start_date.date()}..{end_date.date()}")
    print(f"missing ratio vs known city/date grid: {missing_ratio:.4f}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()

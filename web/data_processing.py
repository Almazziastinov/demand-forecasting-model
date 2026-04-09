"""
data_processing.py -- Transform raw XLSX check files from web/data/
into daily feature DataFrames for the V3 demand model.

Pipeline:
  1. Read all XLSX files from web/data/
  2. Filter "Продажа" events, aggregate to daily level
  3. Extract city, map categories
  4. Calendar + holiday features
  5. Sales lags & rolling stats
  6. Price features from check-level data
  7. Demand features (sales as proxy)
  8. Weather (Open-Meteo or defaults)
  9. Cache results to parquet

Also builds hourly stats and indicators for the UI.
"""

import os
import sys
import time
import json
import hashlib
from pathlib import Path
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.experiments_v2.common import (
    FEATURES_V3, DEMAND_FEATURES, CATEGORICAL_COLS_V2,
    CITY_COORDS, CITY_SUFFIXES, extract_city,
)
from src.features.fetch_weather import fetch_weather, enrich_weather

# --- Paths ---
WEB_DATA_DIR = Path(__file__).resolve().parent / "data"
CACHE_DIR = WEB_DATA_DIR / "cache"
DAILY_CACHE = CACHE_DIR / "daily_features.parquet"
HOURLY_CACHE = CACHE_DIR / "hourly_sales.parquet"
INDICATORS_CACHE = CACHE_DIR / "indicators.parquet"
PROFILES_PATH = ROOT / "data" / "processed" / "demand_profiles.json"
DEMAND_8M_PATH = ROOT / "data" / "processed" / "daily_sales_8m_demand.csv"

# Raw historical checks (30M rows)
HISTORICAL_RAW_PATH = Path(__file__).resolve().parent / "data" / "historical data" / "sales_hrs_all.csv"

HISTORICAL_COLS = [
    "Дата", "Пекарня", "Номенклатура", "Категория", "Город", "Продано",
    "sales_lag1", "sales_lag2", "sales_lag3", "sales_lag7", "sales_lag14", "sales_lag30",
    "sales_roll_mean3", "sales_roll_mean7", "sales_roll_std7", "sales_roll_mean14", "sales_roll_mean30",
    "demand_lag1", "demand_lag2", "demand_lag3", "demand_lag7", "demand_lag14", "demand_lag30",
    "demand_roll_mean3", "demand_roll_mean7", "demand_roll_mean14", "demand_roll_mean30", "demand_roll_std7",
]

# Russian holidays
RU_HOLIDAYS = {
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
    (1, 6), (1, 7), (1, 8),
    (2, 23), (3, 8), (5, 1), (5, 9), (6, 12), (11, 4),
}


# ── Category mapping ─────────────────────────────────────────────

def get_category_mapping():
    """Build product -> category mapping from historical data."""
    mapping = {}
    if DEMAND_8M_PATH.exists():
        df = pd.read_csv(DEMAND_8M_PATH, usecols=["Номенклатура", "Категория"],
                         encoding="utf-8-sig", nrows=500_000)
        mapping = df.groupby("Номенклатура")["Категория"].first().to_dict()
    return mapping


def load_historical_data():
    """Process historical raw checks in batches to compute daily lags."""
    if not HISTORICAL_RAW_PATH.exists():
        print(f"  Warning: historical raw data not found at {HISTORICAL_RAW_PATH}")
        return pd.DataFrame()
    
    print(f"  Processing historical raw data (batched)...")
    
    # Read first batch to get column names and find indices
    first_chunk = pd.read_csv(HISTORICAL_RAW_PATH, nrows=1000, encoding="utf-8-sig")
    col_names = first_chunk.columns.tolist()
    print(f"    Raw columns: {col_names}")
    
    # Find column indices: 0=date, 5=bakery (Касса), 5=product (Номенклатура), 8=qty (Кол-во)
    date_idx = 0
    bakery_idx = None
    product_idx = None
    qty_idx = None
    
    for i, c in enumerate(col_names):
        if "Касса" in c:
            bakery_idx = i
        elif c == "Номенклатура":
            product_idx = i
        elif c == "Кол-во":
            qty_idx = i
    
    usecols = [date_idx, bakery_idx, product_idx, qty_idx]
    names = ["Дата", "Пекарня", "Номенклатура", "Кол-во"]
    print(f"    Using indices: date={date_idx}, bakery={bakery_idx}, product={product_idx}, qty={qty_idx}")
    
    batch_size = 200_000
    daily_agg = []
    
    for batch_num, chunk in enumerate(pd.read_csv(
        HISTORICAL_RAW_PATH, 
        chunksize=batch_size,
        encoding="utf-8-sig",
        usecols=usecols,
        names=names,
        header=0
    )):
        # Parse date
        chunk["Дата"] = pd.to_datetime(chunk["Дата"], format="%d.%m.%Y", errors="coerce")
        chunk = chunk.dropna(subset=["Дата", "Пекарня", "Номенклатура"])
        
        # Aggregate to daily
        daily = (chunk.groupby(["Дата", "Пекарня", "Номенклатура"])
                 .agg(Продано=("Кол-во", "sum"))
                 .reset_index())
        daily["Продано"] = daily["Продано"].clip(lower=0)
        
        daily_agg.append(daily)
        
        if (batch_num + 1) % 10 == 0:
            print(f"    Batch {batch_num + 1}...")
    
    # Combine all batches
    print(f"    Combining {len(daily_agg)} batches...")
    all_daily = pd.concat(daily_agg, ignore_index=True)
    
    # Re-aggregate in case of overlaps
    all_daily = (all_daily.groupby(["Дата", "Пекарня", "Номенклатура"])
                 .agg(Продано=("Продано", "sum"))
                 .reset_index())
    
    print(f"    Historical daily: {len(all_daily):,} rows, dates {all_daily['Дата'].min().date()} to {all_daily['Дата'].max().date()}")
    return all_daily


def normalize_bakery_name(name: str) -> str:
    """Normalize bakery name by removing (n) suffixes like (2), (3), etc."""
    import re
    if pd.isna(name):
        return name
    name = str(name)
    # Remove patterns like " (2)", "(2)", " 2", etc. at the end
    name = re.sub(r'\s*\(\d+\)\s*$', '', name)
    name = re.sub(r'\s+\d+\s*$', '', name)
    return name.strip()


# ── XLSX reading ─────────────────────────────────────────────────

def list_xlsx_files(data_dir=None):
    """List all XLSX files in web/data/ sorted by date."""
    d = data_dir or WEB_DATA_DIR
    files = sorted(glob(str(Path(d) / "*_sales_hrs_kzn.xlsx")))
    return files


def _file_hash(files):
    """Quick hash of file list + sizes for cache validation."""
    parts = []
    for f in sorted(files):
        st = os.stat(f)
        parts.append(f"{f}:{st.st_size}:{st.st_mtime}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def load_all_xlsx(data_dir=None, progress_callback=None):
    """Read all XLSX files from web/data/, return raw DataFrame.

    Columns: Дата, Время, Пекарня, Номенклатура, Свежесть, Цена, Кол-во, Сумма
    """
    files = list_xlsx_files(data_dir)
    if not files:
        return pd.DataFrame()

    print(f"Reading {len(files)} XLSX files with {min(8, os.cpu_count() or 4)} threads...")
    
    def read_single_file(filepath):
        """Read one XLSX file and return DataFrame."""
        try:
            df = pd.read_excel(filepath, engine="openpyxl")

            col_map = {
                "Дата продажи": "Дата",
                "Дата время чека": "Время",
                "Касса": "Пекарня",
                "Касса.Торговая точка": "Пекарня",
                "Кол-во": "Кол-во",
            }
            df.rename(columns=col_map, inplace=True)

            keep = ["Дата", "Время", "Пекарня", "Номенклатура", "Свежесть",
                    "Цена", "Кол-во", "Сумма", "Вид события по кассе"]
            existing = [c for c in keep if c in df.columns]
            return df[existing]
        except Exception as e:
            print(f"  Warning: failed to read {os.path.basename(filepath)}: {e}")
            return pd.DataFrame()

    frames = []
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as executor:
        futures = {executor.submit(read_single_file, f): f for f in files}
        for i, future in enumerate(as_completed(futures)):
            filename = os.path.basename(futures[future])
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i + 1}/{len(files)}] {filename}")
            frames.append(future.result())

    print(f"  Done: {len(frames)} files loaded, {sum(len(f) for f in frames if hasattr(f, 'shape')):,} rows")

    raw = pd.concat(frames, ignore_index=True)
    return raw


# ── Aggregation ──────────────────────────────────────────────────

def aggregate_daily(raw_df):
    """Aggregate check-level data to daily: bakery x product x date -> total sold."""
    df = raw_df.copy()

    # Filter: only sales
    if "Вид события по кассе" in df.columns:
        df = df[df["Вид события по кассе"] == "Продажа"].copy()

    # Parse date
    df["Дата"] = pd.to_datetime(df["Дата"], format="%d.%m.%Y", errors="coerce")
    df = df.dropna(subset=["Дата"])

    # Aggregate daily sales
    daily = (df.groupby(["Дата", "Пекарня", "Номенклатура"])
             .agg(Продано=("Кол-во", "sum"))
             .reset_index())

    daily["Продано"] = daily["Продано"].clip(lower=0)

    return daily


def aggregate_daily_prices(raw_df):
    """Aggregate check-level data to daily prices per bakery x product."""
    df = raw_df.copy()
    if "Вид события по кассе" in df.columns:
        df = df[df["Вид события по кассе"] == "Продажа"].copy()

    df["Дата"] = pd.to_datetime(df["Дата"], format="%d.%m.%Y", errors="coerce")
    df = df.dropna(subset=["Дата", "Цена", "Кол-во"])
    df = df[df["Кол-во"] > 0]

    # Weighted average price
    daily_price = (df.groupby(["Дата", "Пекарня", "Номенклатура"])
                   .apply(lambda g: np.average(g["Цена"], weights=g["Кол-во"]),
                          include_groups=False)
                   .reset_index(name="avg_price"))

    return daily_price


# ── Feature engineering ──────────────────────────────────────────

def add_calendar_features(df):
    """Add calendar and holiday features."""
    df["ДеньНедели"] = df["Дата"].dt.dayofweek
    df["День"] = df["Дата"].dt.day
    df["IsWeekend"] = (df["ДеньНедели"] >= 5).astype(int)
    df["Месяц"] = df["Дата"].dt.month
    df["НомерНедели"] = df["Дата"].dt.isocalendar().week.astype(int)

    date_md = list(zip(df["Дата"].dt.month, df["Дата"].dt.day))
    df["is_holiday"] = [int((m, d) in RU_HOLIDAYS) for m, d in date_md]

    next_day = df["Дата"] + pd.Timedelta(days=1)
    prev_day = df["Дата"] - pd.Timedelta(days=1)
    df["is_pre_holiday"] = [int((m, d) in RU_HOLIDAYS)
                            for m, d in zip(next_day.dt.month, next_day.dt.day)]
    df["is_post_holiday"] = [int((m, d) in RU_HOLIDAYS)
                             for m, d in zip(prev_day.dt.month, prev_day.dt.day)]

    df["is_payday_week"] = df["День"].isin([4, 5, 6, 19, 20, 21]).astype(int)
    df["is_month_start"] = (df["День"] <= 5).astype(int)
    df["is_month_end"] = (df["День"] >= 25).astype(int)

    return df


def add_lag_features(df, col="Продано", prefix="sales"):
    """Add lag and rolling features for a given column."""
    df = df.sort_values(["Пекарня", "Номенклатура", "Дата"]).reset_index(drop=True)
    grouped = df.groupby(["Пекарня", "Номенклатура"])[col]

    for lag in [1, 2, 3, 7, 14, 30]:
        df[f"{prefix}_lag{lag}"] = grouped.shift(lag)

    for window, min_p in [(3, 1), (7, 1), (14, 7), (30, 14)]:
        df[f"{prefix}_roll_mean{window}"] = grouped.transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=min_p).mean()
        )

    df[f"{prefix}_roll_std7"] = grouped.transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=2).std()
    )

    return df


def add_price_features(daily_df, daily_prices):
    """Merge and compute price features."""
    df = daily_df.merge(daily_prices, on=["Дата", "Пекарня", "Номенклатура"], how="left")

    # Price vs product median
    product_medians = df.groupby("Номенклатура")["avg_price"].transform("median")
    df["price_vs_median"] = df["avg_price"] / (product_medians + 1e-8)

    # Price lag and rolling
    df = df.sort_values(["Пекарня", "Номенклатура", "Дата"]).reset_index(drop=True)
    grouped = df.groupby(["Пекарня", "Номенклатура"])["avg_price"]

    df["price_lag7"] = grouped.shift(7)
    df["price_change_7d"] = (df["avg_price"] - df["price_lag7"]) / (df["price_lag7"] + 1e-8)
    df["price_roll_mean7"] = grouped.transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
    )
    df["price_roll_std7"] = grouped.transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=2).std()
    )

    # Fill NaN prices with 0
    price_cols = ["avg_price", "price_vs_median", "price_lag7",
                  "price_change_7d", "price_roll_mean7", "price_roll_std7"]
    for col in price_cols:
        df[col] = df[col].fillna(0)

    return df


def add_weather_features(df):
    """Fetch and merge weather data from Open-Meteo."""
    start_date = df["Дата"].min().strftime("%Y-%m-%d")
    end_date = df["Дата"].max().strftime("%Y-%m-%d")

    try:
        weather_df = fetch_weather(CITY_COORDS, start_date, end_date)
        if weather_df.empty:
            raise ValueError("Empty weather response")
        weather_df = enrich_weather(weather_df)

        # weather_category -> weather_cat_code
        cat_map = {"clear": 0, "cloudy": 1, "fog": 2, "rain": 3,
                   "showers": 4, "snow": 5, "storm": 6}
        weather_df["weather_cat_code"] = (weather_df["weather_category"]
                                          .map(cat_map).fillna(0).astype(int))
        weather_df.drop(columns=["weather_category"], inplace=True)
        if "weathercode" in weather_df.columns:
            weather_df.drop(columns=["weathercode"], inplace=True)

        df["Город"] = df["Город"].astype(str)
        weather_df["Город"] = weather_df["Город"].astype(str)
        weather_df["Дата"] = pd.to_datetime(weather_df["Дата"]).dt.normalize()

        df = df.merge(weather_df, on=["Дата", "Город"], how="left")

        weather_cols = [c for c in weather_df.columns if c not in ["Дата", "Город"]]
        for col in weather_cols:
            df[col] = df[col].fillna(0)

    except Exception as e:
        print(f"  Weather error: {e}. Filling with defaults.")
        weather_cols = ["temp_max", "temp_min", "temp_mean", "temp_range",
                        "precipitation", "rain", "snowfall", "windspeed_max",
                        "is_rainy", "is_snowy", "is_cold", "is_warm",
                        "is_windy", "is_bad_weather", "weather_cat_code"]
        for col in weather_cols:
            if col not in df.columns:
                df[col] = 0

    return df


# ── Indicators (for UI) ─────────────────────────────────────────

def build_indicators(raw_df):
    """Build per-bakery-product indicators from raw check data.

    Returns DataFrame with columns:
      Дата, Пекарня, Номенклатура,
      остаток_вчера (yesterday's product sold today),
      время_последней_продажи (last sale time),
    """
    df = raw_df.copy()
    if "Вид события по кассе" in df.columns:
        df = df[df["Вид события по кассе"] == "Продажа"].copy()

    df["Дата"] = pd.to_datetime(df["Дата"], format="%d.%m.%Y", errors="coerce")
    df["Время"] = pd.to_datetime(df["Время"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["Дата"])

    # Yesterday's product sold today (Свежесть = "Вчерашний")
    stale = df[df["Свежесть"] == "Вчерашний"].copy()
    stale_daily = (stale.groupby(["Дата", "Пекарня", "Номенклатура"])["Кол-во"]
                   .sum().reset_index()
                   .rename(columns={"Кол-во": "остаток_вчера"}))

    # Last sale time per bakery x product x date
    last_sale = (df.dropna(subset=["Время"])
                 .groupby(["Дата", "Пекарня", "Номенклатура"])["Время"]
                 .max().reset_index()
                 .rename(columns={"Время": "время_последней_продажи"}))

    # Merge
    indicators = last_sale.merge(stale_daily,
                                 on=["Дата", "Пекарня", "Номенклатура"],
                                 how="left")
    indicators["остаток_вчера"] = indicators["остаток_вчера"].fillna(0).astype(int)

    return indicators


# ── Hourly stats (for daily plan) ────────────────────────────────

def build_hourly_stats(raw_df):
    """Build hourly sales aggregation for demand profiles.

    Returns DataFrame: Дата, Пекарня, Номенклатура, Час, Кол-во
    """
    df = raw_df.copy()
    if "Вид события по кассе" in df.columns:
        df = df[df["Вид события по кассе"] == "Продажа"].copy()

    df["Дата"] = pd.to_datetime(df["Дата"], format="%d.%m.%Y", errors="coerce")
    df["Время"] = pd.to_datetime(df["Время"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["Дата", "Время"])

    df["Час"] = df["Время"].dt.hour

    hourly = (df.groupby(["Дата", "Пекарня", "Номенклатура", "Час"])["Кол-во"]
              .sum().reset_index())

    return hourly


def build_hourly_profile(hourly_df, bakery, product, n_days=30, min_days=7):
    """Build hourly sales profile for a bakery+product from recent data.

    Returns dict: {hour: share_of_daily_total} (6-22 hours)
    Returns None if fewer than min_days of data available (fallback to demand_profiles.json).
    """
    df = hourly_df[
        (hourly_df["Пекарня"] == bakery) &
        (hourly_df["Номенклатура"] == product)
    ].copy()

    if df.empty:
        return None

    # Use last n_days
    recent_dates = sorted(df["Дата"].unique())[-n_days:]

    # Not enough days for a reliable profile — let caller use fallback
    if len(recent_dates) < min_days:
        return None

    df = df[df["Дата"].isin(recent_dates)]

    # Average hourly sales
    avg_hourly = df.groupby("Час")["Кол-во"].mean()
    total = avg_hourly.sum()
    if total == 0:
        return None

    profile = {}
    for hour in range(6, 23):
        profile[hour] = round(float(avg_hourly.get(hour, 0) / total), 4)

    return profile


def get_hourly_profile(hourly_df, bakery, product, category=None):
    """Get hourly profile with fallback chain:
    1. From web/data (recent data for this bakery+product)
    2. From demand_profiles.json
    3. Average profile for the category
    4. Default even distribution
    """
    # Try from recent data
    profile = build_hourly_profile(hourly_df, bakery, product)
    if profile:
        return profile

    # Try demand_profiles.json
    if PROFILES_PATH.exists():
        with open(PROFILES_PATH, "r", encoding="utf-8") as f:
            profiles = json.load(f)
        key = f"{bakery}|{product}"
        if key in profiles:
            # Convert cumulative to incremental
            cum = profiles[key]
            prev = 0
            profile = {}
            for hour in range(6, 23):
                val = cum.get(str(hour), prev)
                profile[hour] = round(val - prev, 4)
                prev = val
            return profile

    # Try category average from hourly_df
    if category and not hourly_df.empty:
        # We don't have category in hourly_df directly, skip this fallback
        pass

    # Default: even distribution across working hours (7-21)
    hours = list(range(7, 22))
    share = round(1.0 / len(hours), 4)
    return {h: share for h in hours}


# ── Full pipeline ────────────────────────────────────────────────

def process_and_cache(data_dir=None, force=False, progress_callback=None):
    """Full processing pipeline: XLSX -> features + hourly + indicators.

    Returns (daily_df, hourly_df, indicators_df)
    """
    data_dir = data_dir or WEB_DATA_DIR
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    files = list_xlsx_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No XLSX files in {data_dir}")

    current_hash = _file_hash(files)
    hash_file = CACHE_DIR / "data_hash.txt"

    # Check cache validity
    if not force and DAILY_CACHE.exists() and hash_file.exists():
        cached_hash = hash_file.read_text().strip()
        if cached_hash == current_hash:
            print("Cache is valid, loading from parquet...")
            daily = pd.read_parquet(DAILY_CACHE)
            daily["Дата"] = pd.to_datetime(daily["Дата"])
            hourly = pd.read_parquet(HOURLY_CACHE) if HOURLY_CACHE.exists() else pd.DataFrame()
            if not hourly.empty:
                hourly["Дата"] = pd.to_datetime(hourly["Дата"])
            indicators = pd.read_parquet(INDICATORS_CACHE) if INDICATORS_CACHE.exists() else pd.DataFrame()
            if not indicators.empty:
                indicators["Дата"] = pd.to_datetime(indicators["Дата"])
            return daily, hourly, indicators

    print(f"Processing {len(files)} XLSX files...")
    t0 = time.time()

    # Step 1: Read all XLSX
    raw = load_all_xlsx(data_dir)
    print(f"  Raw: {len(raw):,} rows ({time.time() - t0:.0f}s)")

    # Step 2: Build hourly stats (before filtering)
    hourly = build_hourly_stats(raw)
    print(f"  Hourly: {len(hourly):,} rows")

    # Step 3: Build indicators
    indicators = build_indicators(raw)
    print(f"  Indicators: {len(indicators):,} rows")

    # Step 4: Aggregate to daily
    daily = aggregate_daily(raw)
    print(f"  Daily: {len(daily):,} rows")

    # Step 5: Daily prices
    daily_prices = aggregate_daily_prices(raw)
    print(f"  Prices: {len(daily_prices):,} rows")

    # Normalize bakery names in all dataframes
    for df_part in [daily, hourly, indicators]:
        if not df_part.empty and "Пекарня" in df_part.columns:
            df_part["Пекарня"] = df_part["Пекарня"].apply(normalize_bakery_name)

    # Step 5b: Load historical data for lags
    historical = load_historical_data()
    
    # Free raw data
    del raw

    # Step 6: Normalize bakery names in daily (XLSX) data
    daily["Пекарня"] = daily["Пекарня"].apply(normalize_bakery_name)
    
    # Step 6b: Combine XLSX data with historical for lag computation
    if not historical.empty:
        # Normalize historical bakery names too
        historical["Пекарня"] = historical["Пекарня"].apply(normalize_bakery_name)
        
        # Append historical data (keep only needed columns)
        hist_cols = ["Дата", "Пекарня", "Номенклатура", "Категория", "Город", "Продано"]
        hist_subset = historical[[c for c in hist_cols if c in historical.columns]]
        # Ensure same dtypes
        for col in ["Пекарня", "Номенклатура"]:
            if col in daily.columns and col in hist_subset.columns:
                daily[col] = daily[col].astype(str)
                hist_subset[col] = hist_subset[col].astype(str)
        combined = pd.concat([hist_subset, daily], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Дата", "Пекарня", "Номенклатура"], keep="last")
    else:
        combined = daily
    
    # Step 7: Category mapping + city extraction
    cat_map = get_category_mapping()
    combined["Категория"] = combined["Номенклатура"].map(cat_map).fillna("Прочее")
    combined["Город"] = combined["Пекарня"].apply(extract_city)
    print(f"  Categories mapped: {combined['Категория'].nunique()} categories")
    print(f"  Cities: {sorted(combined['Город'].unique())}")

    # Step 8: Calendar features
    combined = add_calendar_features(combined)

    # Step 9: Sales lags & rolling (computed on combined data including historical)
    combined = add_lag_features(combined, col="Продано", prefix="sales")

    # Step 10: Demand features (using sales as proxy)
    combined = add_lag_features(combined, col="Продано", prefix="demand")

    # Step 11: Price features
    combined = add_price_features(combined, daily_prices)

    # Step 12: Weather
    combined = add_weather_features(combined)
    
    # Filter back to only XLSX dates (current data)
    xlsx_dates = daily["Дата"].unique()
    print(f"  XLSX dates to keep: {len(xlsx_dates)} - {sorted(xlsx_dates)[:5]}...")
    daily = combined[combined["Дата"].isin(xlsx_dates)].copy()
    print(f"  After filter: {len(daily)} rows, dates: {daily['Дата'].nunique()}")

    # Step 12: Fill remaining NaN in lag/rolling with 0
    lag_cols = [c for c in daily.columns
                if c.startswith("sales_lag") or c.startswith("sales_roll")
                or c.startswith("demand_lag") or c.startswith("demand_roll")]
    for col in lag_cols:
        daily[col] = daily[col].fillna(0)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s. Shape: {daily.shape}")

    # Save cache
    daily.to_parquet(DAILY_CACHE, index=False)
    hourly.to_parquet(HOURLY_CACHE, index=False)
    indicators.to_parquet(INDICATORS_CACHE, index=False)
    hash_file.write_text(current_hash)
    print(f"  Cache saved to {CACHE_DIR}")

    return daily, hourly, indicators


def load_or_process(data_dir=None, force=False):
    """Load cached data or process from scratch. Returns (daily, hourly, indicators)."""
    return process_and_cache(data_dir, force=force)


# ── CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv
    daily, hourly, indicators = process_and_cache(force=force)

    print(f"\nDaily: {daily.shape}")
    print(f"  Dates: {daily['Дата'].nunique()}")
    print(f"  Bakeries: {daily['Пекарня'].nunique()}")
    print(f"  Products: {daily['Номенклатура'].nunique()}")
    print(f"  Columns: {daily.columns.tolist()}")
    print(f"\nHourly: {hourly.shape}")
    print(f"Indicators: {indicators.shape}")

    # Check V3 feature coverage
    missing = [f for f in FEATURES_V3 if f not in daily.columns]
    if missing:
        print(f"\nWARNING: Missing V3 features: {missing}")
    else:
        print(f"\nAll {len(FEATURES_V3)} V3 features present!")

import pandas as pd
import numpy as np
import openmeteo_requests
import requests_cache
from retry_requests import retry
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Координаты городов датасета
# ─────────────────────────────────────────────
CITIES = {
    'Казань':             {'lat': 55.7879, 'lon': 49.1233},
    'Чебоксары':          {'lat': 56.1439, 'lon': 47.2489},
    'Набережные Челны':   {'lat': 55.7439, 'lon': 52.4042},
    'Нижнекамск':         {'lat': 55.6386, 'lon': 51.8221},
    'Зеленодольск':       {'lat': 55.8430, 'lon': 48.5206},
    'Бугульма':           {'lat': 54.5393, 'lon': 52.7959},
    'Заинск':             {'lat': 55.2978, 'lon': 52.0046},
}

# ─────────────────────────────────────────────
# 1. Скачиваем погоду с Open-Meteo
# ─────────────────────────────────────────────
def fetch_weather(cities: dict, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Скачивает исторические дневные данные о погоде для каждого города.
    Возвращает DataFrame с колонками:
        Город, Дата, temp_max, temp_min, temp_mean,
        precipitation, rain, snowfall, windspeed_max, weathercode
    """
    cache_session = requests_cache.CachedSession('data/.weather_cache', expire_after=-1)
    retry_session = retry(cache_session, retries=3, backoff_factor=0.3)
    om = openmeteo_requests.Client(session=retry_session)

    daily_vars = [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
        "windspeed_10m_max",
        "weathercode",
    ]

    all_frames = []

    for city, coords in cities.items():
        print(f"  Загрузка погоды для: {city}...")
        try:
            responses = om.weather_api(
                "https://archive-api.open-meteo.com/v1/archive",
                params={
                    "latitude":   coords['lat'],
                    "longitude":  coords['lon'],
                    "start_date": start_date,
                    "end_date":   end_date,
                    "daily":      daily_vars,
                    "timezone":   "Europe/Moscow",
                }
            )
            resp  = responses[0]
            daily = resp.Daily()

            values = {
                'temp_max':      daily.Variables(0).ValuesAsNumpy(),
                'temp_min':      daily.Variables(1).ValuesAsNumpy(),
                'temp_mean':     daily.Variables(2).ValuesAsNumpy(),
                'precipitation': daily.Variables(3).ValuesAsNumpy(),
                'rain':          daily.Variables(4).ValuesAsNumpy(),
                'snowfall':      daily.Variables(5).ValuesAsNumpy(),
                'windspeed_max': daily.Variables(6).ValuesAsNumpy(),
                'weathercode':   daily.Variables(7).ValuesAsNumpy(),
            }
            n_days = len(values['temp_max'])
            dates = pd.date_range(
                start=pd.to_datetime(daily.Time(), unit='s', utc=True)
                        .tz_convert('Europe/Moscow')
                        .tz_localize(None),
                periods=n_days,
                freq='D'
            )

            df = pd.DataFrame({'Город': city, 'Дата': dates})
            for col, arr in values.items():
                df[col] = arr[:n_days]
            all_frames.append(df)

        except Exception as e:
            print(f"    ⚠️  Ошибка для {city}: {e}")

    if not all_frames:
        return pd.DataFrame()

    weather_df = pd.concat(all_frames, ignore_index=True)
    weather_df['Дата'] = pd.to_datetime(weather_df['Дата']).dt.normalize()
    return weather_df


def enrich_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет производные погодные признаки.
    """
    # Диапазон температур за день
    df['temp_range']       = df['temp_max'] - df['temp_min']

    # Бинарные флаги
    df['is_rainy']         = (df['rain']     > 1.0).astype(int)
    df['is_snowy']         = (df['snowfall'] > 0.5).astype(int)
    df['is_cold']          = (df['temp_mean'] < 0).astype(int)
    df['is_warm']          = (df['temp_mean'] >= 15).astype(int)
    df['is_windy']         = (df['windspeed_max'] > 30).astype(int)   # > 30 км/ч
    df['is_bad_weather']   = ((df['is_rainy'] == 1) | (df['is_snowy'] == 1) | (df['is_windy'] == 1)).astype(int)

    # WMO weather code → категория
    # 0: ясно, 1-3: облачно, 45-48: туман, 51-67: дождь, 71-77: снег, 80-82: ливень, 95+: гроза
    def wmo_to_category(code):
        code = int(code) if not np.isnan(code) else 0
        if code == 0:
            return 'clear'
        elif code <= 3:
            return 'cloudy'
        elif code <= 48:
            return 'fog'
        elif code <= 67:
            return 'rain'
        elif code <= 77:
            return 'snow'
        elif code <= 82:
            return 'showers'
        else:
            return 'storm'

    df['weather_category'] = df['weathercode'].apply(wmo_to_category)

    return df


# ─────────────────────────────────────────────
# 2. Российские праздники и специальные дни
# ─────────────────────────────────────────────
def build_calendar_features(dates: pd.Series) -> pd.DataFrame:
    """
    Создаёт календарные признаки для каждой даты:
    - Российские праздники
    - Эффект зарплаты (5-е и 20-е числа)
    - Канун праздника / день после праздника
    """
    # Российские государственные праздники (для любого года)
    # Формат: (месяц, день)
    RU_HOLIDAYS = {
        (1, 1):  'Новый год',
        (1, 2):  'Новогодние каникулы',
        (1, 3):  'Новогодние каникулы',
        (1, 4):  'Новогодние каникулы',
        (1, 5):  'Новогодние каникулы',
        (1, 6):  'Новогодние каникулы',
        (1, 7):  'Рождество',
        (1, 8):  'Новогодние каникулы',
        (2, 23): 'День защитника Отечества',
        (3, 8):  'Международный женский день',
        (5, 1):  'Праздник Весны и Труда',
        (5, 9):  'День Победы',
        (6, 12): 'День России',
        (11, 4): 'День народного единства',
    }

    unique_dates = pd.Series(pd.to_datetime(dates.unique())).sort_values().reset_index(drop=True)

    rows = []
    for date in unique_dates:
        key = (date.month, date.day)
        is_holiday = int(key in RU_HOLIDAYS)
        holiday_name = RU_HOLIDAYS.get(key, '')

        # Канун праздника (день перед праздником)
        next_day_key = ((date + pd.Timedelta(days=1)).month,
                        (date + pd.Timedelta(days=1)).day)
        is_pre_holiday = int(next_day_key in RU_HOLIDAYS)

        # День после праздника
        prev_day_key = ((date - pd.Timedelta(days=1)).month,
                        (date - pd.Timedelta(days=1)).day)
        is_post_holiday = int(prev_day_key in RU_HOLIDAYS)

        # Эффект зарплаты: в России традиционно выплачивают 5-го и 20-го
        is_payday_week = int(date.day in [4, 5, 6, 19, 20, 21])

        # Начало / конец месяца (люди тратят больше / меньше)
        is_month_start = int(date.day <= 5)
        is_month_end   = int(date.day >= 25)

        rows.append({
            'Дата':            date.normalize(),
            'is_holiday':      is_holiday,
            'holiday_name':    holiday_name,
            'is_pre_holiday':  is_pre_holiday,
            'is_post_holiday': is_post_holiday,
            'is_payday_week':  is_payday_week,
            'is_month_start':  is_month_start,
            'is_month_end':    is_month_end,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 3. Основной пайплайн
# ─────────────────────────────────────────────
def main():
    import sys

    print("=" * 60)
    print("  ЗАГРУЗКА ДОПОЛНИТЕЛЬНЫХ ПРИЗНАКОВ")
    print("=" * 60)

    # Путь к входному CSV: можно передать аргументом
    input_path = sys.argv[1] if len(sys.argv) > 1 else 'data/processed/preprocessed_data.csv'

    # Выходной путь: заменяем .csv на _enriched.csv
    if input_path.endswith('.csv'):
        output_path = input_path.replace('.csv', '_enriched.csv')
    else:
        output_path = input_path + '_enriched.csv'

    # Загружаем препроцессированные данные, чтобы знать диапазон дат
    print(f"\nЧитаем {input_path}...")
    df = pd.read_csv(input_path)
    df['Дата'] = pd.to_datetime(df['Дата'])

    start_date = df['Дата'].min().strftime('%Y-%m-%d')
    end_date   = df['Дата'].max().strftime('%Y-%m-%d')

    print(f"Диапазон дат: {start_date} — {end_date}")
    print(f"Города: {df['Город'].unique().tolist()}")

    # ── 3.1 Погода ──────────────────────────────────────────────
    print("\n[1/3] Скачиваем погоду с Open-Meteo...")
    weather_df = fetch_weather(CITIES, start_date, end_date)

    if weather_df.empty:
        print("  ❌ Погода не загружена. Проверьте соединение с интернетом.")
        return

    weather_df = enrich_weather(weather_df)
    print(f"  ✓ Загружено {len(weather_df)} строк погоды для {weather_df['Город'].nunique()} городов")

    # ── 3.2 Календарь ───────────────────────────────────────────
    print("\n[2/3] Создаём календарные признаки...")
    calendar_df = build_calendar_features(df['Дата'])
    print(f"  ✓ Создано {len(calendar_df)} строк календарных признаков")
    holidays_in_period = calendar_df[calendar_df['is_holiday'] == 1]
    if not holidays_in_period.empty:
        print(f"  Праздники в периоде: {holidays_in_period[['Дата','holiday_name']].to_string(index=False)}")
    else:
        print("  Праздников в данном периоде не обнаружено")

    # ── 3.3 Объединяем с основными данными ──────────────────────
    print("\n[3/3] Объединяем признаки с основным датасетом...")

    # Приводим типы к str для джойна
    weather_df['Город'] = weather_df['Город'].astype(str)
    df['Город']         = df['Город'].astype(str)

    # Джойн с погодой по Дата + Город
    df_enriched = df.merge(
        weather_df.drop(columns=['weathercode']),
        on=['Дата', 'Город'],
        how='left'
    )

    # Джойн с календарём только по Дате (убираем колонки, которые уже есть в данных)
    calendar_to_merge = calendar_df.drop(columns=['holiday_name'])
    overlap_cols = [c for c in calendar_to_merge.columns if c in df_enriched.columns and c != 'Дата']
    if overlap_cols:
        calendar_to_merge = calendar_to_merge.drop(columns=overlap_cols)
        print(f"  Календарные признаки уже есть в данных (пропускаем): {overlap_cols}")
    df_enriched = df_enriched.merge(calendar_to_merge, on='Дата', how='left')

    # Кодируем weather_category как числовой признак
    weather_cat_map = {
        'clear':   0,
        'cloudy':  1,
        'fog':     2,
        'rain':    3,
        'showers': 4,
        'snow':    5,
        'storm':   6,
    }
    df_enriched['weather_cat_code'] = df_enriched['weather_category'].map(weather_cat_map).fillna(0).astype(int)
    df_enriched = df_enriched.drop(columns=['weather_category'])

    # ── Проверка ─────────────────────────────────────────────────
    print(f"\n  Итоговый размер датасета: {df_enriched.shape}")
    new_cols = [c for c in df_enriched.columns if c not in df.columns]
    print(f"  Новых признаков добавлено: {len(new_cols)}")
    print(f"  Список новых признаков:")
    for col in new_cols:
        n_null = df_enriched[col].isnull().sum()
        print(f"    {'⚠️ ' if n_null > 0 else '✓ '}{col:<22}  nulls={n_null}")

    # ── Статистика по погоде ──────────────────────────────────────
    print(f"\n  Статистика погодных признаков:")
    weather_cols = ['temp_max', 'temp_min', 'temp_mean', 'precipitation',
                    'snowfall', 'windspeed_max', 'is_rainy', 'is_snowy', 'is_bad_weather']
    available = [c for c in weather_cols if c in df_enriched.columns]
    print(df_enriched[available].describe().round(2).to_string())

    # ── Сохранение ───────────────────────────────────────────────
    df_enriched.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n  ✓ Данные сохранены в {output_path}")

    # ── Краткая сводка по новым фичам ────────────────────────────
    print("\n" + "=" * 60)
    print("  ИТОГОВЫЙ СПИСОК ВСЕХ НОВЫХ ПРИЗНАКОВ")
    print("=" * 60)
    feature_descriptions = {
        'temp_max':         'Максимальная температура дня (°C)',
        'temp_min':         'Минимальная температура дня (°C)',
        'temp_mean':        'Средняя температура дня (°C)',
        'temp_range':       'Суточный перепад температур (°C)',
        'precipitation':    'Осадки за день (мм)',
        'rain':             'Дождь (мм)',
        'snowfall':         'Снег (мм)',
        'windspeed_max':    'Максимальная скорость ветра (км/ч)',
        'is_rainy':         'Флаг: дождливый день (>1 мм)',
        'is_snowy':         'Флаг: снежный день (>0.5 мм)',
        'is_cold':          'Флаг: холодно (temp_mean < 0°C)',
        'is_warm':          'Флаг: тепло (temp_mean >= 15°C)',
        'is_windy':         'Флаг: ветрено (>30 км/ч)',
        'is_bad_weather':   'Флаг: плохая погода (дождь/снег/ветер)',
        'weather_cat_code': 'Тип погоды: 0=ясно..6=гроза',
        'is_holiday':       'Флаг: государственный праздник',
        'is_pre_holiday':   'Флаг: день перед праздником',
        'is_post_holiday':  'Флаг: день после праздника',
        'is_payday_week':   'Флаг: неделя зарплаты (4-6 и 19-21)',
        'is_month_start':   'Флаг: начало месяца (1-5 число)',
        'is_month_end':     'Флаг: конец месяца (25-31 число)',
    }
    for col, desc in feature_descriptions.items():
        in_data = '✓' if col in df_enriched.columns else '✗'
        print(f"  {in_data}  {col:<22}  — {desc}")

    return df_enriched


if __name__ == "__main__":
    result = main()

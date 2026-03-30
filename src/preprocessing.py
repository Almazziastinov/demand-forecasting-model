import pandas as pd
import numpy as np
import os

def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Умная загрузка файла: определяет нужен ли сдвиг заголовка.
    - beigl_data.xlsx      -- заголовок в строке 0 (стандартный)
    - 3_month_data.xlsx    -- служебная шапка из 10 строк, заголовок в строке 10
    """
    # Читаем первые 12 строк без заголовка, чтобы понять структуру
    probe = pd.read_excel(file_path, header=None, nrows=12)

    # Если строка 10 содержит 'Дата' -- у файла нестандартная шапка
    row10_values = probe.iloc[10].astype(str).tolist()
    if any('Дата' in v for v in row10_values if isinstance(v, str)):
        print(f"  Обнаружена служебная шапка (10 строк) -- читаем с header=10")
        df = pd.read_excel(file_path, header=10)
        # Убираем первые пустые колонки (артефакт Excel-шапки)
        df = df.dropna(axis=1, how='all')
    else:
        print(f"  Стандартный формат -- читаем с header=0")
        df = pd.read_excel(file_path)

    return df


def merge_raw_files(old_path, new_path, cutoff_date='2026-03-01'):
    """
    Объединяет два сырых файла: из старого берем данные до cutoff_date,
    из нового -- с cutoff_date и далее. Убирает дубликаты.
    """
    print(f"Объединение файлов:")
    print(f"  Старый: {old_path}")
    print(f"  Новый:  {new_path}")

    old = load_raw_data(old_path)
    new = load_raw_data(new_path)

    old['Дата'] = pd.to_datetime(old['Дата'])
    new['Дата'] = pd.to_datetime(new['Дата'])

    cutoff = pd.to_datetime(cutoff_date)
    old_part = old[old['Дата'] < cutoff]
    new_part = new[new['Дата'] >= cutoff]

    print(f"  Из старого (до {cutoff_date}): {len(old_part):,} строк, "
          f"{old_part['Дата'].nunique()} дней")
    print(f"  Из нового (с {cutoff_date}):   {len(new_part):,} строк, "
          f"{new_part['Дата'].nunique()} дней")

    # Берем только общие колонки
    common_cols = sorted(set(old_part.columns) & set(new_part.columns))
    merged = pd.concat([old_part[common_cols], new_part[common_cols]], ignore_index=True)
    merged = merged.sort_values('Дата').reset_index(drop=True)

    print(f"  Итого: {len(merged):,} строк, {merged['Дата'].nunique()} дней "
          f"({merged['Дата'].min().date()} -- {merged['Дата'].max().date()})")

    return merged


def preprocess_data(file_path='data/raw/beigl_data.xlsx', merge_with=None):
    """
    Загружает данные, обрабатывает аномалии и создает признаки:
    - лаги продаж: lag1, lag2, lag3, lag7
    - скользящие статистики: mean3, mean7, std7
    - вчерашний остаток: stock_lag1

    merge_with: если указан, объединяет file_path (старый) с merge_with (новый)
    """
    try:
        if merge_with:
            df = merge_raw_files(file_path, merge_with)
        else:
            if not os.path.exists(file_path):
                print(f"Ошибка: Файл {file_path} не найден.")
                return pd.DataFrame()

            print(f"Загрузка данных из: {file_path}...")
            df = load_raw_data(file_path)

        print(f"Загружено строк: {len(df):,}")

        # 1. Обработка аномалий
        neg_sales = (df['Продано'] < 0).sum()
        neg_stock  = (df['Остаток'] < 0).sum()
        neg_produced = (df['Выпуск'] < 0).sum()
        df['Продано'] = df['Продано'].clip(lower=0)
        df['Остаток']  = df['Остаток'].clip(lower=0)
        df['Выпуск']   = df['Выпуск'].clip(lower=0)
        print(f"Исправлено отрицательных значений - Продано: {neg_sales}, Остаток: {neg_stock}, Выпуск: {neg_produced}")

        # 2. Временные признаки
        df['Дата'] = pd.to_datetime(df['Дата'])
        df['ДеньНедели']  = df['Дата'].dt.dayofweek   # 0=Пн .. 6=Вс
        df['День']        = df['Дата'].dt.day
        df['IsWeekend']   = (df['ДеньНедели'] >= 5).astype(int)

        # 3. Агрегация до уровня Дата-Пекарня-Номенклатура
        group_cols = [
            'Дата', 'Пекарня', 'Номенклатура', 'Категория',
            'Город', 'ДеньНедели', 'День', 'IsWeekend'
        ]
        aggregated_df = df.groupby(group_cols, as_index=False).agg({
            'Продано':  'sum',
            'Выпуск':   'sum',
            'Остаток':  'sum',
        })

        # 4. Feature Engineering -- лаги и скользящие статистики
        # Сортируем, чтобы shift() работал корректно
        aggregated_df = aggregated_df.sort_values(
            by=['Пекарня', 'Номенклатура', 'Дата']
        ).reset_index(drop=True)

        grouped = aggregated_df.groupby(['Пекарня', 'Номенклатура'])

        # Лаги продаж (lag7 нужен всегда, lag14/lag30 -- только если данных достаточно)
        n_days_available = aggregated_df['Дата'].nunique()
        lags = [1, 2, 3, 7]
        if n_days_available >= 21:
            lags.append(14)
            print(f"  Добавляем lag14 (дней в данных: {n_days_available})")
        if n_days_available >= 37:
            lags.append(30)
            print(f"  Добавляем lag30 (дней в данных: {n_days_available})")

        for lag in lags:
            aggregated_df[f'sales_lag{lag}'] = grouped['Продано'].shift(lag)

        # Скользящие статистики (не включают текущий день -> shift(1) перед rolling)
        aggregated_df['sales_roll_mean3'] = grouped['Продано'].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        )
        aggregated_df['sales_roll_mean7'] = grouped['Продано'].transform(
            lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
        )
        aggregated_df['sales_roll_std7'] = grouped['Продано'].transform(
            lambda x: x.shift(1).rolling(window=7, min_periods=2).std()
        )
        if n_days_available >= 21:
            aggregated_df['sales_roll_mean14'] = grouped['Продано'].transform(
                lambda x: x.shift(1).rolling(window=14, min_periods=7).mean()
            )
        if n_days_available >= 37:
            aggregated_df['sales_roll_mean30'] = grouped['Продано'].transform(
                lambda x: x.shift(1).rolling(window=30, min_periods=14).mean()
            )

        # Вчерашний остаток на конец дня
        aggregated_df['stock_lag1'] = grouped['Остаток'].shift(1)

        # Stock-to-sales ratio (how much stock relative to recent demand)
        aggregated_df['stock_sales_ratio'] = aggregated_df['stock_lag1'] / (aggregated_df['sales_lag1'] + 1)

        # Stock deficit (unmet demand signal)
        aggregated_df['stock_deficit'] = (aggregated_df['sales_lag1'] - aggregated_df['stock_lag1']).clip(lower=0)

        # 5. Дополнительные календарные признаки
        aggregated_df['Месяц']       = aggregated_df['Дата'].dt.month
        aggregated_df['НомерНедели'] = aggregated_df['Дата'].dt.isocalendar().week.astype(int)

        # Российские праздники (месяц, день)
        RU_HOLIDAYS = {
            (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
            (1, 6), (1, 7), (1, 8),   # Новый год + Рождество
            (2, 23),                   # День защитника
            (3, 8),                    # 8 марта
            (5, 1), (5, 9),            # Труд, Победа
            (6, 12),                   # День России
            (11, 4),                   # Народное единство
        }
        def is_holiday(date):
            return int((date.month, date.day) in RU_HOLIDAYS)
        def is_pre_holiday(date):
            nxt = date + pd.Timedelta(days=1)
            return int((nxt.month, nxt.day) in RU_HOLIDAYS)
        def is_post_holiday(date):
            prv = date - pd.Timedelta(days=1)
            return int((prv.month, prv.day) in RU_HOLIDAYS)

        aggregated_df['is_holiday']      = aggregated_df['Дата'].apply(is_holiday)
        aggregated_df['is_pre_holiday']  = aggregated_df['Дата'].apply(is_pre_holiday)
        aggregated_df['is_post_holiday'] = aggregated_df['Дата'].apply(is_post_holiday)
        aggregated_df['is_payday_week']  = aggregated_df['Дата'].dt.day.isin([4,5,6,19,20,21]).astype(int)

        holidays_found = aggregated_df[aggregated_df['is_holiday'] == 1]['Дата'].dt.date.unique()
        print(f"  Праздничных дней в данных: {len(holidays_found)} -> {sorted(holidays_found)}")

        # 6. Удаляем строки с NaN (первые дни каждого ряда после максимального лага)
        initial_len = len(aggregated_df)
        aggregated_df = aggregated_df.dropna().reset_index(drop=True)
        print(f"Удалено {initial_len - len(aggregated_df)} строк с NaN (первые дни каждого ряда).")
        print(f"Итоговый размер данных: {aggregated_df.shape}")

        return aggregated_df

    except Exception as e:
        print(f"Произошла ошибка при обработке: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


if __name__ == "__main__":
    import sys

    # Варианты запуска:
    #   python preprocessing.py                              -- beigl_data.xlsx
    #   python preprocessing.py data/raw/3_month_data.xlsx   -- один файл
    #   python preprocessing.py --merge                      -- объединить 3_month + beigl
    args = sys.argv[1:]

    if '--merge' in args:
        old_file = 'data/raw/3_month_data.xlsx'
        new_file = 'data/raw/beigl_data.xlsx'
        data = preprocess_data(old_file, merge_with=new_file)
        output_path = 'data/processed/preprocessed_data_merged.csv'
    else:
        file = args[0] if args else 'data/raw/beigl_data.xlsx'
        data = preprocess_data(file)
        if '3_month' in file:
            output_path = 'data/processed/preprocessed_data_3month.csv'
        else:
            output_path = 'data/processed/preprocessed_data.csv'

    if not data.empty:
        print(f"\n--- Первые 5 строк ---")
        print(data.head())
        print(f"\n--- Список всех признаков ---")
        print(data.columns.tolist())
        print(f"\nПериод: {data['Дата'].min().date()} -- {data['Дата'].max().date()}")
        print(f"Уникальных дней:    {data['Дата'].nunique()}")
        print(f"Уникальных пекарен: {data['Пекарня'].nunique()}")
        print(f"Уникальных товаров: {data['Номенклатура'].nunique()}")

        data.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\nДанные сохранены в {output_path}")

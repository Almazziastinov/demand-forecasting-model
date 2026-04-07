"""
DataLoader - загрузка и кэширование данных для веб-приложения.
Поддерживает:
1. Предобработанный CSV (daily_sales_8m_demand.csv)
2. Чеки Excel/CSV с автоагрегацией до дня
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd


DAILY_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "daily_sales_8m_demand.csv"

# Russian holidays
RU_HOLIDAYS = {
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),
    (2, 23), (3, 8), (5, 1), (5, 9), (6, 12), (11, 4),
}

# City extraction from bakery name
CITY_KEYWORDS = {
    "казань": "Казань", "казанский": "Казань", "казан": "Казань",
    "москва": "Москва", "московский": "Москва",
    "петербург": "Санкт-Петербург", "спб": "Санкт-Петербург",
    "екатеринбург": "Екатеринбург", "екб": "Екатеринбург",
    "краснодар": "Краснодар", "ростов": "Ростов", "ростовский": "Ростов",
    "тольятти": "Тольятти", "самара": "Самара",
    "казан": "Казань",
}


def extract_city(bakery_name: str) -> str:
    """Извлечь город из названия пекарни."""
    name_lower = str(bakery_name).lower()
    for keyword, city in CITY_KEYWORDS.items():
        if keyword in name_lower:
            return city
    return "Казань"  # default


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить календарные фичи."""
    df["ДеньНедели"] = df["Дата"].dt.dayofweek
    df["День"] = df["Дата"].dt.day
    df["IsWeekend"] = (df["ДеньНедели"] >= 5).astype(int)
    df["Месяц"] = df["Дата"].dt.month
    df["НомерНедели"] = df["Дата"].dt.isocalendar().week.astype(int)
    
    # Праздники
    date_md = list(zip(df["Дата"].dt.month, df["Дата"].dt.day))
    df["is_holiday"] = [int((m, d) in RU_HOLIDAYS) for m, d in date_md]
    
    next_md = list(zip((df["Дата"] + pd.Timedelta(days=1)).dt.month, (df["Дата"] + pd.Timedelta(days=1)).dt.day))
    prev_md = list(zip((df["Дата"] - pd.Timedelta(days=1)).dt.month, (df["Дата"] - pd.Timedelta(days=1)).dt.day))
    df["is_pre_holiday"] = [int((m, d) in RU_HOLIDAYS) for m, d in next_md]
    df["is_post_holiday"] = [int((m, d) in RU_HOLIDAYS) for m, d in prev_md]
    
    # Зарплатные недели
    df["is_payday_week"] = df["День"].isin([4, 5, 6, 19, 20, 21]).astype(int)
    df["is_month_start"] = (df["День"] <= 5).astype(int)
    df["is_month_end"] = (df["День"] >= 25).astype(int)
    
    return df


class DataLoader:
    """Загрузчик данных с автоопределением формата."""
    
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path or str(DAILY_DATA_PATH)
        self._df: Optional[pd.DataFrame] = None
        self._source_type: Optional[str] = None
    
    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self.load()
        return self._df
    
    @property
    def source_type(self) -> str:
        return self._source_type or "unknown"
    
    def load(self, path: Optional[str] = None) -> pd.DataFrame:
        """Загрузить данные с автоопределением формата."""
        if path:
            self.data_path = path
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Файл не найден: {self.data_path}")
        
        print(f"  Загрузка: {self.data_path}")
        
        # Определяем формат по расширению и структуре
        if self.data_path.endswith(".xlsx") or self.data_path.endswith(".xls"):
            self._source_type = "excel_checks"
            self._df = self._load_excel_checks(self.data_path)
        elif self.data_path.endswith(".csv"):
            # Пробуем определить по колонкам
            df_sample = pd.read_csv(self.data_path, nrows=5, encoding="utf-8-sig")
            
            if "Дата продажи" in df_sample.columns or "Дата время чека" in df_sample.columns:
                self._source_type = "csv_checks"
                self._df = self._load_csv_checks(self.data_path)
            else:
                self._source_type = "daily_csv"
                self._df = pd.read_csv(self.data_path, encoding="utf-8-sig")
                self._df["Дата"] = pd.to_datetime(self._df["Дата"])
        else:
            raise ValueError(f"Неподдерживаемый формат: {self.data_path}")
        
        print(f"  Загружено: {len(self._df):,} строк ({self._source_type})")
        print(f"  Период: {self._df['Дата'].min().date()} -- {self._df['Дата'].max().date()}")
        
        return self._df
    
    def _load_excel_checks(self, path: str) -> pd.DataFrame:
        """Загрузка Excel с чеками и агрегация до дня."""
        print("  Чтение Excel...")
        df = pd.read_excel(path, engine="openpyxl")
        return self._aggregate_checks(df)
    
    def _load_csv_checks(self, path: str) -> pd.DataFrame:
        """Загрузка CSV с чеками и агрегация до дня."""
        print("  Чтение CSV...")
        
        # Пробуем разные кодировки
        for enc in ["utf-8-sig", "utf-8", "cp1251"]:
            try:
                df = pd.read_csv(path, encoding=enc)
                print(f"  Кодировка: {enc}")
                break
            except UnicodeDecodeError:
                continue
        
        return self._aggregate_checks(df)
    
    def _aggregate_checks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Агрегация чеков до дня."""
        print("  Определение колонок...")
        
        # Определяем колонки
        col_map = {}
        for col in df.columns:
            col_lower = str(col).lower()
            if "дата продажи" in col_lower:
                col_map[col] = "Дата"
            elif "дата время" in col_lower:
                col_map[col] = "ДатаВремя"
            elif "торговая точка" in col_lower or "касса" in col_lower:
                col_map[col] = "Пекарня"
            elif "номенклатура" in col_lower or "продукт" in col_lower:
                col_map[col] = "Номенклатура"
            elif "категория" in col_lower:
                col_map[col] = "Категория"
            elif "свежесть" in col_lower:
                col_map[col] = "Свежесть"
            elif "цена" in col_lower:
                col_map[col] = "Цена"
            elif "кол" in col_lower or "количество" in col_lower:
                col_map[col] = "Кол-во"
            elif "событие" in col_lower or "вид" in col_lower:
                col_map[col] = "Событие"
        
        # Переименовываем
        df = df.rename(columns=col_map)
        
        # Парсим дату
        if "Дата" in df.columns:
            df["Дата"] = pd.to_datetime(df["Дата"], format="%d.%m.%Y", errors="coerce")
        elif "ДатаВремя" in df.columns:
            df["ДатаВремя"] = pd.to_datetime(df["ДатаВремя"], errors="coerce")
            df["Дата"] = df["ДатаВремя"].dt.date
            df["Дата"] = pd.to_datetime(df["Дата"])
        
        # Фильтруем продажи (не возвраты)
        if "Событие" in df.columns:
            sales_mask = df["Событие"].astype(str).str.lower().str.contains("продаж", na=False)
            df = df[sales_mask]
            print(f"  Отфильтровано продаж: {len(df):,}")
        
        # Парсим числовые
        if "Кол-во" in df.columns:
            df["Кол-во"] = pd.to_numeric(df["Кол-во"], errors="coerce").fillna(0)
            df["Кол-во"] = df["Кол-во"].clip(lower=0)
        
        if "Цена" in df.columns:
            df["Цена"] = pd.to_numeric(df["Цена"], errors="coerce").fillna(0)
        
        # Извлекаем город
        if "Пекарня" in df.columns:
            df["Город"] = df["Пекарня"].apply(extract_city)
        
        print("  Агрегация до дня...")
        
        # Агрегация
        agg_cols = ["Дата", "Пекарня", "Номенклатура", "Категория", "Город"]
        if "Свежесть" in df.columns:
            agg_cols.append("Свежесть")
        
        agg = df.groupby(agg_cols, as_index=False).agg({
            "Кол-во": "sum",
            "Цена": "mean",
        })
        
        agg = agg.rename(columns={"Кол-во": "Продано"})
        
        # Добавляем календарные фичи
        agg = add_calendar_features(agg)
        
        # Сортируем
        agg = agg.sort_values(["Пекарня", "Номенклатура", "Дата"]).reset_index(drop=True)
        
        print(f"  Агрегировано: {len(agg):,} строк")
        
        return agg
    
    def get_sales_history(
        self,
        bakery: str,
        product: str,
        days: int = 30
    ) -> pd.DataFrame:
        """Получить историю для пары."""
        mask = (
            (self.df["Пекарня"] == bakery) &
            (self.df["Номенклатура"] == product)
        )
        history = self.df[mask].sort_values("Дата")
        
        if days > 0:
            cutoff = history["Дата"].max() - pd.Timedelta(days=days)
            history = history[history["Дата"] > cutoff]
        
        return history
    
    def get_latest_date(self) -> pd.Timestamp:
        return self.df["Дата"].max()
    
    def get_bakeries(self) -> list:
        return sorted(self.df["Пекарня"].unique().tolist())
    
    def get_products(self, bakery: Optional[str] = None) -> list:
        if bakery:
            mask = self.df["Пекарня"] == bakery
            return sorted(self.df.loc[mask, "Номенклатура"].unique().tolist())
        return sorted(self.df["Номенклатура"].unique().tolist())


_loader: Optional[DataLoader] = None


def get_loader() -> DataLoader:
    global _loader
    if _loader is None:
        _loader = DataLoader()
    return _loader

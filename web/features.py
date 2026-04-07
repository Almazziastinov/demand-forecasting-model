"""
Feature Generator для предсказания на любую дату.
Берёт исторические данные и вычисляет фичи "на лету".
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from src.experiments_v2.common import CITY_COORDS


# Russian holidays (month, day)
RU_HOLIDAYS = {
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
    (1, 6), (1, 7), (1, 8),
    (2, 23), (3, 8), (5, 1), (5, 9), (6, 12), (11, 4),
}


class FeatureGenerator:
    """
    Генератор фичей для любой даты на основе исторических данных.
    
    Usage:
        fg = FeatureGenerator(history_df)  # исторические данные
        features = fg.generate(bakery, product, target_date)
    """
    
    def __init__(self, daily_sales: pd.DataFrame, weather_cache: Optional[pd.DataFrame] = None):
        """
        Args:
            daily_sales: DataFrame с колонками 
                Дата, Пекарня, Номенклатура, Категория, Город, Продано, Цена
            weather_cache: DataFrame с погодой (опционально)
        """
        self.sales = daily_sales.copy()
        self.weather = weather_cache
        
        # Индекс для быстрого доступа
        self.sales["Дата"] = pd.to_datetime(self.sales["Дата"])
        
        # Предвычисленные rolling statistics для скорости
        self._precompute_rolling()
    
    def _precompute_rolling(self):
        """Предвычисляем rolling statistics для каждой пары."""
        print("  Предвычисление rolling statistics...")
        
        # Сортируем
        self.sales = self.sales.sort_values(
            ["Пекарня", "Номенклатура", "Дата"]
        ).reset_index(drop=True)
        
        grouped = self.sales.groupby(["Пекарня", "Номенклатура"])["Продано"]
        
        # Rolling means
        for window in [3, 7, 14, 30]:
            col = f"roll_mean{window}"
            self.sales[col] = grouped.transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        
        # Rolling std
        self.sales["roll_std7"] = grouped.transform(
            lambda x: x.rolling(window=7, min_periods=2).std()
        )
        
        print(f"  Rolling precomputed: {len(self.sales):,} rows")
    
    def generate(
        self, 
        bakery: str, 
        product: str, 
        target_date: datetime,
        include_price: bool = True,
    ) -> Dict[str, Any]:
        """
        Сгенерировать вектор фичей для пары пекарня × продукт на дату.
        
        Args:
            bakery: Название пекарни
            product: Название продукта
            target_date: Целевая дата предсказания
            include_price: Включать ли ценовые фичи
            
        Returns:
            Dict с фичами для model.predict()
        """
        target_date = pd.to_datetime(target_date)
        
        # Берём историю до target_date (включительно)
        mask = (
            (self.sales["Пекарня"] == bakery) &
            (self.sales["Номенклатура"] == product) &
            (self.sales["Дата"] <= target_date)
        )
        history = self.sales[mask].sort_values("Дата")
        
        # Получаем метаданные
        if len(history) > 0:
            category = history["Категория"].iloc[-1]
            city = history["Город"].iloc[-1]
        else:
            # Нет истории - дефолты
            return self._default_features(bakery, product, target_date)
        
        features = {}
        
        # === Категориальные фичи ===
        features["Пекарня"] = bakery
        features["Номенклатура"] = product
        features["Категория"] = category
        features["Город"] = city
        features["Месяц"] = target_date.month
        
        # === Календарные фичи ===
        dow = target_date.dayofweek
        features["ДеньНедели"] = dow
        features["День"] = target_date.day
        features["IsWeekend"] = 1 if dow >= 5 else 0
        features["НомерНедели"] = target_date.isocalendar()[1]
        
        # Праздники
        features["is_holiday"] = 1 if (target_date.month, target_date.day) in RU_HOLIDAYS else 0
        
        next_day = target_date + timedelta(days=1)
        prev_day = target_date - timedelta(days=1)
        features["is_pre_holiday"] = 1 if (next_day.month, next_day.day) in RU_HOLIDAYS else 0
        features["is_post_holiday"] = 1 if (prev_day.month, prev_day.day) in RU_HOLIDAYS else 0
        
        # Зарплатные недели
        features["is_payday_week"] = 1 if target_date.day in [4, 5, 6, 19, 20, 21] else 0
        features["is_month_start"] = 1 if target_date.day <= 5 else 0
        features["is_month_end"] = 1 if target_date.day >= 25 else 0
        
        # === Sales Lag фичи ===
        sales_values = history["Продано"].values
        
        # lag1 = продажи за вчера (или последний день)
        for lag, days_ago in [(1, 1), (2, 2), (3, 3), (7, 7), (14, 14), (30, 30)]:
            target_day = target_date - timedelta(days=days_ago)
            day_mask = history["Дата"] == target_day
            if day_mask.any():
                features[f"sales_lag{lag}"] = history.loc[day_mask, "Продано"].values[0]
            else:
                # Берём последнее доступное
                idx = len(history) - days_ago - 1
                if idx >= 0:
                    features[f"sales_lag{lag}"] = sales_values[idx]
                else:
                    features[f"sales_lag{lag}"] = np.nan
        
        # === Rolling Statistics ===
        if len(history) >= 1:
            features["sales_roll_mean3"] = history["roll_mean3"].iloc[-1] if "roll_mean3" in history.columns else np.nan
            features["sales_roll_mean7"] = history["roll_mean7"].iloc[-1] if "roll_mean7" in history.columns else np.nan
            features["sales_roll_mean14"] = history["roll_mean14"].iloc[-1] if "roll_mean14" in history.columns else np.nan
            features["sales_roll_mean30"] = history["roll_mean30"].iloc[-1] if "roll_mean30" in history.columns else np.nan
            features["sales_roll_std7"] = history["roll_std7"].iloc[-1] if "roll_std7" in history.columns else np.nan
        else:
            for col in ["sales_roll_mean3", "sales_roll_mean7", "sales_roll_mean14", 
                       "sales_roll_mean30", "sales_roll_std7"]:
                features[col] = np.nan
        
        # === Ценовые фичи ===
        if include_price and "Цена" in history.columns and len(history) > 0:
            prices = history["Цена"].values
            features["avg_price"] = prices[-1] if len(prices) > 0 else 0
            
            # price_vs_median
            if len(prices) > 0:
                median_price = np.median(prices)
                features["price_vs_median"] = prices[-1] / (median_price + 1e-8)
            else:
                features["price_vs_median"] = 1.0
            
            # price_lag7
            if len(prices) >= 7:
                features["price_lag7"] = prices[-7]
            else:
                features["price_lag7"] = prices[-1] if len(prices) > 0 else 0
            
            # price_change_7d
            if len(prices) >= 7:
                features["price_change_7d"] = (prices[-1] - prices[-7]) / (prices[-7] + 1e-8)
            else:
                features["price_change_7d"] = 0
            
            # price_roll_mean7, price_roll_std7
            if len(prices) >= 7:
                features["price_roll_mean7"] = np.mean(prices[-7:])
                features["price_roll_std7"] = np.std(prices[-7:])
            else:
                features["price_roll_mean7"] = np.mean(prices) if len(prices) > 0 else 0
                features["price_roll_std7"] = np.std(prices) if len(prices) > 0 else 0
        else:
            features["avg_price"] = 0
            features["price_vs_median"] = 1.0
            features["price_lag7"] = 0
            features["price_change_7d"] = 0
            features["price_roll_mean7"] = 0
            features["price_roll_std7"] = 0
        
        # === Погодные фичи ===
        if self.weather is not None and city in CITY_COORDS:
            weather_row = self.weather[
                (self.weather["Город"] == city) &
                (self.weather["Дата"] == target_date)
            ]
            if len(weather_row) > 0:
                w = weather_row.iloc[0]
                features["temp_max"] = w.get("temp_max", 0)
                features["temp_min"] = w.get("temp_min", 0)
                features["temp_mean"] = w.get("temp_mean", 0)
                features["temp_range"] = w.get("temp_range", 0)
                features["precipitation"] = w.get("precipitation", 0)
                features["rain"] = w.get("rain", 0)
                features["snowfall"] = w.get("snowfall", 0)
                features["windspeed_max"] = w.get("windspeed_max", 0)
                features["is_rainy"] = w.get("is_rainy", 0)
                features["is_snowy"] = w.get("is_snowy", 0)
                features["is_cold"] = w.get("is_cold", 0)
                features["is_warm"] = w.get("is_warm", 0)
                features["is_windy"] = w.get("is_windy", 0)
                features["is_bad_weather"] = w.get("is_bad_weather", 0)
                features["weather_cat_code"] = w.get("weather_cat_code", 0)
            else:
                features.update(self._default_weather())
        else:
            features.update(self._default_weather())
        
        return features
    
    def generate_batch(
        self,
        bakery_product_pairs: List[tuple],
        target_date: datetime,
    ) -> pd.DataFrame:
        """
        Генерировать фичи для батча пар (пекарня, продукт) на одну дату.
        
        Args:
            bakery_product_pairs: List of (bakery, product) tuples
            target_date: Целевая дата
            
        Returns:
            DataFrame с фичами
        """
        records = []
        for bakery, product in bakery_product_pairs:
            feat = self.generate(bakery, product, target_date)
            records.append(feat)
        
        return pd.DataFrame(records)
    
    def _default_features(self, bakery: str, product: str, date: datetime) -> Dict[str, Any]:
        """Дефолтные фичи если нет истории."""
        return {
            "Пекарня": bakery,
            "Номенклатура": product,
            "Категория": "unknown",
            "Город": "unknown",
            "Месяц": date.month,
            "ДеньНедели": date.dayofweek,
            "День": date.day,
            "IsWeekend": 1 if date.dayofweek >= 5 else 0,
            "НомерНедели": date.isocalendar()[1],
            "is_holiday": 0,
            "is_pre_holiday": 0,
            "is_post_holiday": 0,
            "is_payday_week": 0,
            "is_month_start": 0,
            "is_month_end": 0,
            "sales_lag1": 0, "sales_lag2": 0, "sales_lag3": 0,
            "sales_lag7": 0, "sales_lag14": 0, "sales_lag30": 0,
            "sales_roll_mean3": 0, "sales_roll_mean7": 0,
            "sales_roll_mean14": 0, "sales_roll_mean30": 0,
            "sales_roll_std7": 0,
            "avg_price": 0, "price_vs_median": 1.0,
            "price_lag7": 0, "price_change_7d": 0,
            "price_roll_mean7": 0, "price_roll_std7": 0,
        }
    
    def _default_weather(self) -> Dict[str, float]:
        """Дефолтные погодные фичи."""
        return {
            "temp_max": 0, "temp_min": 0, "temp_mean": 0,
            "temp_range": 0, "precipitation": 0, "rain": 0,
            "snowfall": 0, "windspeed_max": 0,
            "is_rainy": 0, "is_snowy": 0, "is_cold": 0,
            "is_warm": 0, "is_windy": 0, "is_bad_weather": 0,
            "weather_cat_code": 0,
        }


# Фичи которые нужны модели (в порядке важности)
DEMAND_MODEL_FEATURES = [
    # Categorical
    "Пекарня", "Номенклатура", "Категория", "Город", "Месяц",
    # Calendar
    "ДеньНедели", "День", "IsWeekend", "НомерНедели",
    # Sales lags
    "sales_lag1", "sales_lag2", "sales_lag3", "sales_lag7", "sales_lag14", "sales_lag30",
    # Rolling
    "sales_roll_mean3", "sales_roll_mean7", "sales_roll_std7",
    "sales_roll_mean14", "sales_roll_mean30",
    # Holidays
    "is_holiday", "is_pre_holiday", "is_post_holiday",
    "is_payday_week", "is_month_start", "is_month_end",
    # Weather
    "temp_max", "temp_min", "temp_mean", "temp_range",
    "precipitation", "rain", "snowfall", "windspeed_max",
    "is_rainy", "is_snowy", "is_cold", "is_warm",
    "is_windy", "is_bad_weather", "weather_cat_code",
    # Price
    "avg_price", "price_vs_median", "price_lag7",
    "price_change_7d", "price_roll_mean7", "price_roll_std7",
]

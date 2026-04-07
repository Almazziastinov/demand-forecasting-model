"""
Model Manager - загрузка и управление моделью.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class ModelManager:
    """
    Управление моделью для предсказаний.
    """
    
    CATEGORICAL_COLS = ["Пекарня", "Номенклатура", "Категория", "Город", "Месяц"]
    
    def __init__(self, model_path: Optional[str] = None):
        # V3: demand + price + quantile P50
        self.model_path = model_path or str(MODELS_DIR / "demand_model_v3.pkl")
        self.meta_path = str(MODELS_DIR / "model_meta_v3.pkl")
        self.model: Optional[LGBMRegressor] = None
        self.meta: Optional[Dict] = None
        self._loaded = False
    
    def load(self) -> bool:
        """
        Загрузить модель и метаданные.
        Returns:
            True если успешно, False если не найдено
        """
        if not os.path.exists(self.model_path):
            print(f"  Модель не найдена: {self.model_path}")
            return False
        
        print(f"  Загрузка модели: {self.model_path}")
        self.model = joblib.load(self.model_path)
        
        if os.path.exists(self.meta_path):
            self.meta = joblib.load(self.meta_path)
            print(f"  Метаданные загружены")
        else:
            self.meta = {}
            print(f"  Метаданные не найдены, будут сгенерированы")
        
        self._loaded = True
        return True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказание модели.
        
        Args:
            X: DataFrame с фичами
            
        Returns:
            Массив предсказаний
        """
        if not self._loaded:
            raise RuntimeError("Модель не загружена. Вызовите load()")
        
        # Приводим категориальные колонки к правильному типу
        X = X.copy()
        for col in self.CATEGORICAL_COLS:
            if col in X.columns:
                X[col] = X[col].astype("category")
        
        # Предсказание
        pred = self.model.predict(X)
        return np.maximum(pred, 0)  # Clip отрицательных
    
    def predict_single(self, features: Dict[str, Any]) -> float:
        """
        Предсказание для одной комбинации.
        
        Args:
            features: Dict с фичами
            
        Returns:
            Предсказание (float)
        """
        df = pd.DataFrame([features])
        pred = self.predict(df)
        return float(pred[0])
    
    def get_metrics(self) -> Dict[str, float]:
        """Получить метрики модели."""
        if self.meta and "metrics" in self.meta:
            return self.meta["metrics"]
        return {}
    
    def get_feature_names(self) -> list:
        """Получить список фичей модели."""
        if self.meta and "features" in self.meta:
            return self.meta["features"]
        # Fallback
        return []
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded


# Singleton
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Глобальный менеджер модели."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager

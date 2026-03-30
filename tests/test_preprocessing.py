"""Tests for preprocessing on synthetic data (no real files needed)."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import preprocess_data


def _make_synthetic_excel(tmp_path):
    """Create a minimal valid Excel file for preprocessing."""
    dates = pd.date_range("2026-01-01", periods=45, freq="D")
    rows = []
    for date in dates:
        for bakery in ["B1", "B2"]:
            for product in ["P1", "P2"]:
                rows.append({
                    "Дата": date,
                    "Пекарня": bakery,
                    "Номенклатура": product,
                    "Категория": "Cat1",
                    "Город": "Казань",
                    "Продано": np.random.randint(0, 20),
                    "Выпуск": np.random.randint(5, 25),
                    "Остаток": np.random.randint(0, 10),
                })
    df = pd.DataFrame(rows)
    path = os.path.join(str(tmp_path), "test_data.xlsx")
    df.to_excel(path, index=False)
    return path


def test_preprocess_returns_dataframe(tmp_path):
    path = _make_synthetic_excel(tmp_path)
    result = preprocess_data(path)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_preprocess_has_lag_columns(tmp_path):
    path = _make_synthetic_excel(tmp_path)
    result = preprocess_data(path)
    assert "sales_lag1" in result.columns
    assert "sales_lag7" in result.columns
    assert "sales_roll_mean7" in result.columns


def test_preprocess_no_negative_sales(tmp_path):
    path = _make_synthetic_excel(tmp_path)
    result = preprocess_data(path)
    assert (result["Продано"] >= 0).all()

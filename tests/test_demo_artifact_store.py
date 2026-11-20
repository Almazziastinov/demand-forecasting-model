"""Tests for the artifact-based demo store."""

from __future__ import annotations

import os
import sys
import pandas as pd


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_demo_store_loads_and_builds_series(tmp_path, monkeypatch):
    from web import demo_artifact_store as store_module

    best_by_sku_path = tmp_path / "merged_best_by_sku.csv"
    history_path = tmp_path / "daily_sales_8m_demand.csv"
    prediction_path = tmp_path / "predictions.csv"

    pd.DataFrame(
        {
            "bakery": ["Test bakery"],
            "product": ["Test SKU"],
            "best_r2": [0.5],
        }
    ).to_csv(best_by_sku_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "Дата": ["2026-01-01", "2026-01-02"],
            "Пекарня": ["Test bakery", "Test bakery"],
            "Номенклатура": ["Test SKU", "Test SKU"],
            "Спрос": [10, 12],
        }
    ).to_csv(history_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02"],
            "bakery": ["Test bakery", "Test bakery"],
            "product": ["Test SKU", "Test SKU"],
            "fact": [10, 12],
            "pred": [11, 11],
        }
    ).to_csv(prediction_path, index=False, encoding="utf-8-sig")

    monkeypatch.setattr(store_module, "MERGED_BEST_BY_SKU_PATH", best_by_sku_path)
    monkeypatch.setattr(store_module, "DEMAND_8M_PATH", history_path)
    monkeypatch.setattr(
        store_module,
        "MODEL_SPECS",
        [{"name": "avg_lag_7_14_core", "path": prediction_path, "fact_col": "fact", "pred_col": "pred"}],
    )
    store_module.load_store.cache_clear()

    store = store_module.load_store()

    assert "avg_lag_7_14_core" in store["models"]
    assert store["wide"].shape[0] > 0

    sample = store["best_by_sku"].dropna(subset=["bakery", "product"]).iloc[0]
    payload = store_module.build_series(
        sample["bakery"], sample["product"], "avg_lag_7_14_core", "avg_lag_7_14_core"
    )

    assert payload["series"]
    assert payload["metrics"]["main"]["n_rows"] > 0
    assert payload["metrics"]["compare"]["n_rows"] > 0
    store_module.load_store.cache_clear()


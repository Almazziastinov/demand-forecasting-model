"""Causal target and frozen-reference checks for the guarded LoRA experiment."""

import pandas as pd
import pytest

from scripts.finetune_chronos2_volatility_guard import _checked_existing, guarded_flows


def test_guarded_flows_uses_only_finalized_positive_uplifts() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "bakery_id": [1, 1],
            "product_id": [2, 2],
            "release_qty": [20.0, 20.0],
            "observed_sales_qty": [10.0, 12.0],
        }
    )
    registry = panel[["date", "bakery_id", "product_id", "observed_sales_qty"]].copy()
    registry["guarded_total_uplift"] = [6.0, 8.0]
    result = guarded_flows(panel, registry, pd.Timestamp("2026-01-08"))
    assert result["observed_sales_qty"].tolist() == [13.0, 12.0]
    result = guarded_flows(panel, registry, pd.Timestamp("2026-01-09"))
    assert result["observed_sales_qty"].tolist() == [13.0, 16.0]


def test_guarded_flows_rejects_registry_sales_mismatch() -> None:
    panel = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-01")],
            "bakery_id": [1],
            "product_id": [2],
            "release_qty": [20.0],
            "observed_sales_qty": [10.0],
        }
    )
    registry = panel[["date", "bakery_id", "product_id", "observed_sales_qty"]].copy()
    registry["observed_sales_qty"] = 9.0
    registry["guarded_total_uplift"] = 6.0
    with pytest.raises(ValueError, match="do not match"):
        guarded_flows(panel, registry, pd.Timestamp("2026-01-08"))


def test_reference_predictions_require_exact_truth(tmp_path) -> None:
    rows = pd.DataFrame(
        {
            "forecast_origin": [pd.Timestamp("2026-07-06")],
            "date": [pd.Timestamp("2026-07-07")],
            "lead_days": [1],
            "bakery_id": [1],
            "product_id": [2],
            "actual": [10.0],
        }
    )
    old = rows.copy()
    old["model"] = "raw"
    old["prediction"] = 11.0
    path = tmp_path / "old.parquet"
    old.to_parquet(path)
    assert len(_checked_existing(path, rows, "raw")) == 1
    old["actual"] = 9.0
    old.to_parquet(path)
    with pytest.raises(ValueError, match="truth differs"):
        _checked_existing(path, rows, "raw")

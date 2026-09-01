from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.run_direct_alpha_shadow import run_shadow


def test_shadow_runner_writes_only_local_artifacts(tmp_path: Path) -> None:
    source = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-09-01", "2026-09-01"]),
            "bakery_id": [23, 23],
            "product_id": [108, 1071],
            "direct_p50": [20.0, 200.0],
            "predictive_uplift": [5.0, 0.0],
            "loss_scale": [1.0, 1.0],
            "broad_56_mean": [20.0, 200.0],
            "floor_history_n": [10, 10],
            "floor_demand_p67": [25.0, 210.0],
            "historical_stockout_rate": [0.8, 0.1],
            "historical_lost_mean": [5.0, 0.0],
        }
    )
    input_path = tmp_path / "input.parquet"
    output = tmp_path / "shadow"
    source.to_parquet(input_path, index=False)

    summary = run_shadow(input_path, output)

    assert summary["database_write"] is False
    assert summary["activation"] is False
    assert (output / "shadow_rows.parquet").exists()
    assert (output / "shadow_sku_day.csv").exists()
    saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert saved["contract"] == "direct_alpha_025_floor_tail_v1"


def test_shadow_uses_incumbent_for_bakery_without_sales_evidence(
    tmp_path: Path,
) -> None:
    source = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-09-01", "2026-09-01"]),
            "bakery_id": [233, 233],
            "product_id": [10760, 10909],
            "incumbent_sku_forecast": [180.0, 80.0],
            "direct_p50": [130.0, 130.0],
            "predictive_uplift": [0.0, 0.0],
            "loss_scale": [1.0, 1.0],
            "broad_56_mean": [0.0, 0.0],
            "floor_history_n": [0, 0],
            "floor_demand_p67": [0.0, 0.0],
            "historical_stockout_rate": [0.0, 0.0],
            "historical_lost_mean": [0.0, 0.0],
        }
    )
    input_path = tmp_path / "input.parquet"
    output = tmp_path / "shadow"
    source.to_parquet(input_path, index=False)
    summary = run_shadow(input_path, output)
    result = pd.read_parquet(output / "shadow_rows.parquet")
    assert summary["cold_start_fallback_bakery_days"] == 1
    assert result["selected_sku_forecast"].tolist() == [180.0, 80.0]

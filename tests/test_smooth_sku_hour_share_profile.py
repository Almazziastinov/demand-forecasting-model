"""Tests for smoothing applied SKU hour-share profiles."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from uuid import uuid4

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments_v2.smooth_sku_hour_share_profile import build_adjusted_profile  # noqa: E402
from src.experiments_v2.smooth_sku_hour_share_profile import deduplicate_profile_means  # noqa: E402
from src.experiments_v2.smooth_sku_hour_share_profile import smooth_applied_chunk  # noqa: E402


def _tmp_dir() -> Path:
    path = Path("tests") / "_tmp_smooth_profiles" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_smooth_applied_chunk_lifts_below_mean_and_renormalizes():
    applied = pd.DataFrame(
        [
            {
                "date": "2026-01-05",
                "dow": 0,
                "hour": 8,
                "bakery_id": "B1",
                "bakery_name": "Bakery 1",
                "product_id": "P1",
                "product_name": "Product 1",
                "category_name": "Cat",
                "sku_hour_sales": 2.0,
                "bakery_hour_sales": 8.0,
                "sku_share_in_hour": 0.25,
            },
            {
                "date": "2026-01-05",
                "dow": 0,
                "hour": 8,
                "bakery_id": "B1",
                "bakery_name": "Bakery 1",
                "product_id": "P2",
                "product_name": "Product 2",
                "category_name": "Cat",
                "sku_hour_sales": 6.0,
                "bakery_hour_sales": 8.0,
                "sku_share_in_hour": 0.75,
            },
        ]
    )
    profile_means = pd.DataFrame(
        [
            {"bakery_id": "B1", "product_id": "P1", "dow": 0, "hour": 8, "mean_sku_share_in_hour": 0.50},
            {"bakery_id": "B1", "product_id": "P2", "dow": 0, "hour": 8, "mean_sku_share_in_hour": 0.75},
        ]
    )

    result = smooth_applied_chunk(applied, profile_means)
    p1 = result[result["product_id"] == "P1"].iloc[0]
    p2 = result[result["product_id"] == "P2"].iloc[0]

    assert round(float(p1["sku_share_in_hour_adj"]), 4) == 0.5
    assert round(float(p2["sku_share_in_hour_adj"]), 4) == 0.75
    assert round(float(result["sku_share_in_hour_adj_norm"].sum()), 4) == 1.0
    assert round(float(p1["sku_share_in_hour_adj_norm"]), 4) == 0.4
    assert round(float(p2["sku_share_in_hour_adj_norm"]), 4) == 0.6
    assert round(float(p1["sku_share_uplift_raw"]), 4) == 0.25
    assert int(p1["sku_share_uplift_flag"]) == 1
    assert round(float(p1["sku_share_uplift_norm_delta"]), 4) == 0.15


def test_load_profile_means_deduplicates_keys():
    result = deduplicate_profile_means(
        pd.DataFrame(
            [
                {"bakery_id": "B1", "product_id": "P1", "dow": 0, "hour": 8, "mean_sku_share_in_hour": 0.4},
                {"bakery_id": "B1", "product_id": "P1", "dow": 0, "hour": 8, "mean_sku_share_in_hour": 0.6},
            ]
        )
    )
    assert len(result) == 1
    assert round(float(result.iloc[0]["mean_sku_share_in_hour"]), 4) == 0.5


def test_build_adjusted_profile_preserves_base_audit_columns():
    tmp_path = _tmp_dir()
    applied_path = tmp_path / "applied_smoothed.csv"
    pd.DataFrame(
        [
            {
                "date": "2026-01-05",
                "dow": 0,
                "hour": 8,
                "bakery_id": "B1",
                "bakery_name": "Bakery 1",
                "product_id": "P1",
                "product_name": "Product 1",
                "category_name": "Cat",
                "sku_share_in_hour_adj_norm": 1.0,
                "sku_hour_sales_adj": 10.0,
                "sku_share_uplift_raw": 0.25,
                "sku_share_uplift_norm_delta": 0.15,
                "sku_share_uplift_flag": 1,
            }
        ]
    ).to_csv(applied_path, index=False, encoding="utf-8-sig")
    profile_audit = pd.DataFrame(
        [
            {
                "bakery_id": "B1",
                "product_id": "P1",
                "dow": 0,
                "hour": 8,
                "base_recent_n_days": 4,
                "base_reliability_score": 0.75,
            }
        ]
    )

    profile = build_adjusted_profile(
        applied_path,
        profile_audit=profile_audit,
        chunk_size=10,
    )

    row = profile.iloc[0]
    assert round(float(row["mean_share_uplift_raw"]), 4) == 0.25
    assert round(float(row["mean_share_uplift_norm_delta"]), 4) == 0.15
    assert round(float(row["uplifted_row_rate"]), 4) == 1.0
    assert int(row["base_recent_n_days"]) == 4
    assert round(float(row["base_reliability_score"]), 4) == 0.75

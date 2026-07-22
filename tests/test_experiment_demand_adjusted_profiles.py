from __future__ import annotations

import pandas as pd
import pytest

from scripts.experiment_demand_adjusted_profiles import (
    apply_hourly_adjustments,
    build_scored_rows,
    compact_profile,
)


def _hourly() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-01", "2026-06-01"]),
            "dow": [0, 0],
            "hour": [10, 10],
            "bakery_id": [1, 1],
            "bakery_name": ["B", "B"],
            "city": ["C", "C"],
            "product_id": [10, 20],
            "product_name": ["P1", "P2"],
            "category_name": ["Cat", "Cat"],
            "sku_hour_sales": [2.0, 8.0],
        }
    )


def test_apply_hourly_adjustments_adds_existing_and_synthetic_hours() -> None:
    adjustments = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-01", "2026-06-01"]),
            "bakery_id": [1, 1],
            "product_id": [10, 10],
            "hour": [10, 11],
            "imputed_demand": [1.0, 3.0],
        }
    )

    result = apply_hourly_adjustments(_hourly(), adjustments)

    sku = result[result["product_id"].eq(10)].sort_values("hour")
    assert sku["sku_hour_sales"].tolist() == [3.0, 3.0]
    assert result["sku_hour_sales"].sum() == 14.0


def test_compact_profile_renormalizes_duplicate_metadata_rows() -> None:
    profile = pd.DataFrame(
        {
            "bakery_id": [1, 1, 1],
            "product_id": [10, 10, 20],
            "dow": [0, 0, 0],
            "hour": [10, 10, 10],
            "mean_sku_share_in_hour_norm": [0.2, 0.1, 0.7],
            "n_days": [2, 1, 3],
        }
    )

    compact = compact_profile(profile)

    assert compact["profile_share"].sum() == 1.0
    assert compact.loc[
        compact["product_id"].eq(10), "profile_share"
    ].iloc[0] == pytest.approx(0.3)


def test_guarded_routing_keeps_new_exact_triple_on_fallback() -> None:
    profile = pd.DataFrame(
        {
            "bakery_id": [1, 1, 1, 1],
            "product_id": [10, 20, 10, 20],
            "dow": [0, 0, 1, 1],
            "hour": [10, 10, 10, 10],
            "mean_sku_share_in_hour_norm": [0.8, 0.2, 0.2, 0.8],
            "n_days": [8, 8, 8, 8],
        }
    )
    holdout = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-01", "2026-06-01"]),
            "dow": [0, 0],
            "hour": [10, 10],
            "bakery_id": [1, 1],
            "product_id": [10, 20],
            "sku_hour_sales": [2.0, 8.0],
        }
    )

    exact = build_scored_rows(profile, holdout)
    guarded = build_scored_rows(profile, holdout, allowed_exact_triples=set())

    exact_pred = exact.set_index("product_id")["predicted_qty"]
    guarded_pred = guarded.set_index("product_id")["predicted_qty"]
    assert exact_pred.to_dict() == {10: 8.0, 20: 2.0}
    assert guarded_pred.to_dict() == {10: 5.0, 20: 5.0}

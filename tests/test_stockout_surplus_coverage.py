from __future__ import annotations

import pandas as pd
import pytest

from scripts.analyze_stockout_surplus_coverage import (
    build_day_coverage,
    build_surplus_context_comparison,
    classify_coverage,
    prepare_surplus_rows,
)


def test_classify_coverage_separates_volume_mixed_and_allocation() -> None:
    assert classify_coverage(10.0, 0.0) == "volume_shortage_supported"
    assert classify_coverage(10.0, 5.0) == "mixed_supported"
    assert classify_coverage(10.0, 10.0) == "allocation_balanced_supported"
    assert classify_coverage(10.0, 20.0) == "allocation_plus_excess_supported"


def test_prepare_surplus_excludes_stockouts_two_day_and_unreliable_rows() -> None:
    balance = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-01"] * 4),
            "bakery_id": [1] * 4,
            "product_id": [10, 20, 30, 40],
            "stock_balance": [8.0, 7.0, 6.0, 5.0],
            "balance_is_consistent": [True] * 4,
            "hourly_daily_sales_agree": [True, True, True, False],
            "last_sale_time": [None] * 4,
        }
    )
    stockouts = pd.DataFrame(
        {"date": [pd.Timestamp("2026-06-01")], "bakery_id": [1], "product_id": [10]}
    )
    two_day = pd.DataFrame({"product_id": [30], "is_two_day": [1]})

    result = prepare_surplus_rows(balance, stockouts, two_day, reserve_units=1.0)
    surplus = result.set_index("product_id")["strict_usable_surplus"]

    assert surplus.to_dict() == {10: 0.0, 20: 6.0, 30: 0.0, 40: 0.0}


def test_build_day_coverage_decomposes_deficit() -> None:
    adjustments = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-01", "2026-06-01"]),
            "bakery_id": [1, 1],
            "product_id": [10, 11],
            "last_sale_hour": [15, 16],
            "imputed_demand": [6.0, 4.0],
        }
    )
    donors = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-01")],
            "bakery_id": [1],
            "strict_usable_surplus": [7.0],
            "balance_only_surplus": [7.0],
            "all_product_surplus": [7.0],
            "donor_last_sale_hour": [18.0],
        }
    )

    result = build_day_coverage(adjustments, donors).iloc[0]

    assert result["reconstructed_deficit"] == 10.0
    assert result["strict_usable_allocation_component"] == 7.0
    assert result["strict_usable_volume_gap"] == 3.0
    assert result["strict_usable_coverage"] == pytest.approx(0.7)
    assert result["strict_usable_mechanism"] == "mixed_supported"
    assert result["late_confirmed_surplus"] == 7.0


def test_context_comparison_separates_stockout_days() -> None:
    surplus_rows = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-01", "2026-06-02"]),
            "bakery_id": [1, 1],
            "strict_usable_surplus": [8.0, 2.0],
        }
    )
    adjustments = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-01", "2026-06-02"]),
            "bakery_id": [1, 2],
        }
    )

    result = build_surplus_context_comparison(surplus_rows, adjustments)

    assert len(result) == 2
    indexed = result.set_index("has_reconstructed_stockout")
    assert indexed.loc[True, "mean_surplus"] == 8.0

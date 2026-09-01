from __future__ import annotations

import pandas as pd
import pytest

from src.experiments_v2.effective_assortment import (
    apply_emergency_overrides,
    build_automatic_assortment,
    diagnose_baking_meta_gaps,
)


def test_case_270_uses_seven_day_sales_without_hardcoded_products() -> None:
    sales = pd.DataFrame(
        {
            "date": ["2026-08-10", "2026-08-19", "2026-08-19"],
            "bakery_id": [270, 270, 270],
            "product_id": [11573, 11615, 11575],
            "sold_qty": [10.0, 2.0, 2.0],
        }
    )

    result = build_automatic_assortment(sales, as_of_date="2026-08-20")

    assert set(result["product_id"]) == {11575, 11615}


def test_temporary_override_expires_back_to_automatic_result() -> None:
    automatic = pd.DataFrame(
        {"bakery_id": [270], "product_id": [11615], "source": ["recent_sales_7d"]}
    )
    overrides = pd.DataFrame(
        {
            "bakery_id": [270, 270],
            "product_id": [11615, 11575],
            "action": ["force_exclude", "force_include"],
            "valid_from": ["2026-08-20", "2026-08-20"],
            "valid_to": ["2026-08-21", "2026-08-21"],
            "reason": ["incident", "incident"],
            "created_by": ["operator", "operator"],
        }
    )

    active = apply_emergency_overrides(
        automatic, overrides, effective_date="2026-08-20"
    )
    expired = apply_emergency_overrides(
        automatic, overrides, effective_date="2026-08-22"
    )

    assert active[["bakery_id", "product_id"]].values.tolist() == [[270, 11575]]
    assert expired[["bakery_id", "product_id"]].values.tolist() == [[270, 11615]]


def test_override_requires_end_date() -> None:
    overrides = pd.DataFrame(
        {
            "bakery_id": [270],
            "product_id": [11575],
            "action": ["force_include"],
            "valid_from": ["2026-08-20"],
            "valid_to": [None],
            "reason": ["incident"],
            "created_by": ["operator"],
        }
    )
    with pytest.raises(ValueError, match="require valid_to"):
        apply_emergency_overrides(
            pd.DataFrame(), overrides, effective_date="2026-08-20"
        )


def test_missing_baking_meta_is_reported_per_bakery_sku() -> None:
    assortment = pd.DataFrame(
        {"bakery_id": [270, 270], "product_id": [11615, 11575], "source": ["a", "a"]}
    )
    meta = pd.DataFrame(
        {
            "bakery_id": [pd.NA],
            "product_id": [11615],
            "scope": ["base"],
            "is_active": [1],
        }
    )

    gaps = diagnose_baking_meta_gaps(assortment, meta)

    assert gaps[["bakery_id", "product_id", "reason"]].values.tolist() == [
        [270, 11575, "missing_baking_sku_meta"]
    ]

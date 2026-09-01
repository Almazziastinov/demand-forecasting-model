from __future__ import annotations

import pandas as pd

from scripts.build_bakery_product_assortment import (
    add_city_core_for_cold_start_bakeries,
    add_network_core_for_cold_start_bakeries,
    build_assortment,
    build_assortment_from_sales,
    build_cold_start_city_core,
    build_cold_start_network_core,
    carry_forward_bakeries_without_recent_sales,
)


def test_flat_assortment_uses_each_bakery_recent_sales() -> None:
    sales = pd.DataFrame(
        {
            "bakery_id": [1, 1, 2, 2],
            "product_id": [10, 20, 10, 30],
            "category_name": [
                "Выпечка сладкая",
                "Пирог сладкий",
                "Выпечка сладкая",
                "Напитки горячие",
            ],
        }
    )

    result = build_assortment_from_sales(sales, valid_from="2026-08-27")

    assert result[["bakery_id", "product_id"]].values.tolist() == [
        [1, "000000010"],
        [1, "000000020"],
        [2, "000000010"],
    ]


def test_zero_sales_bakery_carries_only_its_previous_snapshot() -> None:
    current = pd.DataFrame(
        {
            "bakery_id": [1],
            "product_id": ["000000010"],
            "valid_from": [pd.Timestamp("2026-08-27").date()],
            "loaded_at": [pd.Timestamp("2026-08-27")],
        }
    )
    previous = pd.DataFrame(
        {
            "bakery_id": [2, 2, 3],
            "product_id": [20, 30, 40],
        }
    )

    result, carried = carry_forward_bakeries_without_recent_sales(
        current,
        previous,
        required_bakery_ids=[1, 2],
        valid_from="2026-08-27",
    )

    assert carried == [2]
    assert result[["bakery_id", "product_id"]].values.tolist() == [
        [1, "000000010"],
        [2, "000000020"],
        [2, "000000030"],
    ]


def test_never_seen_bakery_uses_only_its_city_core() -> None:
    current = pd.DataFrame(
        {
            "bakery_id": [1],
            "product_id": ["000000010"],
            "valid_from": [pd.Timestamp("2026-08-27").date()],
            "loaded_at": [pd.Timestamp("2026-08-27")],
        }
    )
    bakery_city = pd.DataFrame(
        {"bakery_id": [1, 2], "city": ["Kazan", "Kursk"]}
    )
    bakeable = pd.DataFrame(
        {
            "city": ["Kursk", "Kursk", "Kursk"],
            "product_id": [20, 30, 40],
            "scope": ["city", "city", "bakery"],
        }
    )

    result, cold_start = add_city_core_for_cold_start_bakeries(
        current,
        bakery_city,
        bakeable,
        required_bakery_ids=[1, 2],
        valid_from="2026-08-27",
    )

    assert cold_start == [2]
    assert result[["bakery_id", "product_id"]].values.tolist() == [
        [1, "000000010"],
        [2, "000000020"],
        [2, "000000030"],
    ]


def test_cold_start_city_core_uses_participating_bakeries_denominator() -> None:
    sales = pd.DataFrame(
        {
            "city": ["Kursk"] * 4,
            "bakery_id": [1, 1, 2, 2],
            "product_id": [10, 20, 10, 30],
            "category_name": ["Выпечка"] * 4,
        }
    )

    core = build_cold_start_city_core(sales, city_threshold=0.8)

    assert core[["city", "product_id", "scope"]].values.tolist() == [
        ["Kursk", 10, "city"]
    ]


def test_new_city_uses_common_network_core() -> None:
    sales = pd.DataFrame(
        {
            "bakery_id": [1, 1, 2, 2],
            "product_id": [10, 20, 10, 30],
            "category_name": ["Выпечка"] * 4,
        }
    )
    core = build_cold_start_network_core(sales, network_threshold=0.8)
    current = pd.DataFrame(
        columns=["bakery_id", "product_id", "valid_from", "loaded_at"]
    )

    result, cold_start = add_network_core_for_cold_start_bakeries(
        current,
        core,
        required_bakery_ids=[233, 236],
        valid_from="2026-08-27",
    )

    assert cold_start == [233, 236]
    assert result[["bakery_id", "product_id"]].values.tolist() == [
        [233, "000000010"],
        [236, "000000010"],
    ]


def test_flat_assortment_applies_temporary_emergency_overrides() -> None:
    bakeries = pd.DataFrame({"bakery_id": [270], "city": ["Новочебоксарск"]})
    bakeable = pd.DataFrame(
        {
            "city": ["Новочебоксарск"],
            "product_id": ["11573"],
            "scope": ["city"],
            "bakery_id": [pd.NA],
        }
    )
    overrides = pd.DataFrame(
        {
            "bakery_id": [270, 270],
            "product_id": ["11573", "11575"],
            "action": ["force_exclude", "force_include"],
            "valid_from": ["2026-08-20", "2026-08-20"],
            "valid_to": ["2026-08-21", "2026-08-21"],
            "reason": ["incident", "incident"],
            "created_by": ["operator", "operator"],
        }
    )

    result = build_assortment(
        bakeries,
        bakeable,
        valid_from="2026-08-20",
        overrides=overrides,
    )

    assert result[["bakery_id", "product_id"]].values.tolist() == [
        [270, "000011575"]
    ]

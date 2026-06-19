from __future__ import annotations

import pandas as pd

from scripts.audit_baking_plan_templates_assortment import is_service_row
from scripts.audit_baking_plan_templates_assortment import match_product


def _dim_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "product_id": "000004629",
                "product_name": "Конвертик курица",
                "category_name": "Выпечка сытная",
                "is_inactive_product": "False",
            },
            {
                "product_id": "000004944",
                "product_name": "Пирожок капуста и курица",
                "category_name": "Выпечка сытная",
                "is_inactive_product": "False",
            },
            {
                "product_id": "000001071",
                "product_name": "Треугольник курица безд",
                "category_name": "Выпечка сытная",
                "is_inactive_product": "False",
            },
        ]
    )


def test_template_aliases_keep_pie_and_small_pastry_distinct() -> None:
    from scripts.audit_baking_plan_templates_assortment import build_lookup

    exact, products = build_lookup(_dim_rows())

    pastry = match_product("Пирожок капуста курица", exact, products)
    envelope = match_product("Конвертик с курицей", exact, products)

    assert pastry["product_id"] == "000004944"
    assert envelope["product_id"] == "000004629"


def test_template_alias_matches_current_assortment_name() -> None:
    from scripts.audit_baking_plan_templates_assortment import build_lookup

    exact, products = build_lookup(_dim_rows())

    match = match_product(
        "Треугольник курица (тесто ночного брожжения)",
        exact,
        products,
    )

    assert match["product_id"] == "000001071"


def test_zero_quantity_row_without_role_is_service_row() -> None:
    source = pd.Series(
        {
            "table_role": "",
            "qty_sum_in_template": "0.0",
            "product_name": "Ассортимент пирогов РТ",
        }
    )

    assert is_service_row(source)


def test_scheduled_product_row_is_not_service_row() -> None:
    source = pd.Series(
        {
            "table_role": "Пекарь Стол 1",
            "qty_sum_in_template": "20.0",
            "product_name": "Конвертик с курицей",
        }
    )

    assert not is_service_row(source)

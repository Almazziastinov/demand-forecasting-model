import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "forecast_embedded"))

from apps.baking_plan.simple_plan import (  # noqa: E402
    MISSING_KRATNOST_LABEL,
    calculate_plan_rows,
)


def test_missing_meta_is_kept_with_explicit_kratnost_label() -> None:
    forecast_rows = [
        {
            "product_id": 999,
            "product_name": "Новая позиция",
            "category_name": "Выпечка сытная",
            "forecast_qty": 15.8,
        }
    ]

    rows = calculate_plan_rows(
        forecast_rows,
        stock_by_product={},
        base_meta={},
        bakery_meta={},
        bakery_name="Тестовая пекарня",
    )

    assert len(rows) == 1
    assert rows[0]["production_plan"] == 16
    assert rows[0]["kratnost"] == MISSING_KRATNOST_LABEL


def test_temporary_pletenka_names_override_dimension_variants() -> None:
    forecast_rows = [
        {
            "product_id": 11615,
            "product_name": "Сдоба Кленовый пекан",
            "category_name": "Выпечка сладкая",
            "forecast_qty": 2.3,
        },
        {
            "product_id": 11616,
            "product_name": "Сдоба с черникой",
            "category_name": "Выпечка сладкая",
            "forecast_qty": 2.4,
        },
        {
            "product_id": 11617,
            "product_name": "Сдоба с земляникой",
            "category_name": "Выпечка сладкая",
            "forecast_qty": 1.7,
        },
    ]
    meta = {
        product_id: {"dough_group": "Тесто сдобное", "kratnost": 10}
        for product_id in (11615, 11616, 11617)
    }

    rows = calculate_plan_rows(
        forecast_rows,
        stock_by_product={},
        base_meta=meta,
        bakery_meta={},
        bakery_name="Пекарня",
    )

    assert [row["product_name"] for row in rows] == [
        "Плетенка кленовая",
        "Плетенка с земляникой",
        "Плетенка с черникой",
    ]

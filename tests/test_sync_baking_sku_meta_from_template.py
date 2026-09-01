from pathlib import Path

import pandas as pd

from scripts.sync_baking_sku_meta_from_template import (
    build_deactivation_rows,
    build_sync_rows,
)


TEMPLATE = Path(r"C:\Users\dns\Downloads\Шаблон плана выпекания для ИИ (1).xlsx")


def _template_product_names() -> list[str]:
    return [
        "Треугольник курица безд",
        "Треугольник говядина безд",
        "Треугольник острый",
        "Элеш с курицей",
        "Пирог Ханский",
        "Вак-бэлиш",
        "Жар пицца с курицей",
        "Пицца с колбасой",
        "Пирожок с горбушей",
        "Конвертик курица",
        "Пирожок с картошкой",
        "Пицца закрытая",
        "мини Бургер с котлетой",
        "Сосиска в тесте",
        "Беккен капуста",
        "Сосиска под шубой",
        "Киш грибы курица",
        "Шафран творог",
        "Ватрушка",
        "Пирожок яблоко",
        "Маковка",
        "Яблочный",
        "Пирог с черносливом и грец орехом",
        "Капустный",
        "Капуста и мясо",
        "Горбуша саго",
        "Круассан миндальный",
        "Круассан с малиной",
        "Круассан с шоколадом",
        "Киш курица",
        "Корзинка ягодная",
        "Сметанник мини",
        "Губадия мини",
        "Трехслойник НОВЫЙ",
        "Клубника и банан НОВЫЙ",
        "Тропический",
        "Сметанник",
        "Губадия",
        "Пирог с Манго",
        "Пирог с киви",
        "Пирог Ягодный",
        "Сметанник маковый",
        "Жар Киш курица",
        "Жар Киш грибы курица",
        "ЖарПицца Пикантная",
        "ЖарПицца Оригинальная",
        "Бейгл курица",
        "Сэндвич Мюнхенский",
        "Сэндвич курица",
        "Они гири с креветкой",
        "Они гири с курицей",
        "Они гири с лососем",
        "Поке с креветкой",
        "Поке с лососем",
        "Роллы Вулкан с курицей",
        "Роллы Запеченные с креветкой",
        "Роллы запеченные с курицей",
        "Роллы лосось запеч",
        "Роллы Филадельфия",
        "Хот-дог Баварский говяжий",
        "Хот-дог Баварский куриный",
        "Хот-дог Датский говяжий",
        "Хот-дог Датский куриный",
        "Кыстыбый П",
        "Сочень",
        "Королевская ватрушка",
        "Пицца Маргарита кусок",
        "Пицца Мясная кусок",
    ]


def test_build_sync_rows_uses_aliases_and_new_multiples() -> None:
    if not TEMPLATE.exists():
        return
    names = _template_product_names()
    products = pd.DataFrame(
        {
            "product_id": list(range(100000, 100000 + len(names)))
            + [
                11613,
                11640,
                11251,
                11567,
                11568,
                11566,
                11565,
                11575,
                10625,
                10628,
                5106,
                10627,
            ],
            "product_name": names
            + [
                "Пирог с Киви",
                "Пирог с киви",
                "ЖарПицца с колбасками",
                "Хэнд ролл ветчина",
                "Хэнд ролл краб",
                "Роллы тубус Вулкан с курицей",
                "Роллы тубус Филадельфия",
                "Пирог Кексовый с манго",
                "Пицца Маргарита П",
                "Пицца с колбасками П",
                "Пицца с колбасками кусок",
                "Пицца Мясная П",
            ],
        }
    )

    rows, unresolved = build_sync_rows(
        TEMPLATE,
        products,
        valid_from=pd.Timestamp("2026-08-31"),
        loaded_at=pd.Timestamp("2026-08-31 18:00:00"),
    )

    by_name = rows.set_index("product_name")
    assert len(rows) == 78
    assert unresolved == ["Мексиканский ролл"]
    assert by_name.loc["Сосиска под шубой", "kratnost"] == 10
    assert by_name.loc["Сэндвич курица", "kratnost"] == 6
    assert by_name.loc["Хэнд ролл ветчина", "kratnost"] == 2
    assert by_name.loc["Пирог с Киви"].name == "Пирог с Киви"
    assert "000011613" in set(rows["product_id"])
    assert "000011640" not in set(rows["product_id"])
    assert "Основа чиабатта покупная" not in set(rows["product_name"])


def test_build_deactivation_rows_closes_only_older_versions() -> None:
    current = pd.DataFrame(
        [
            {
                "product_id": "000000127",
                "valid_from": "2026-07-09",
                "is_active": 1,
                "loaded_at": pd.Timestamp("2026-07-09"),
                "comment": "old",
            },
            {
                "product_id": "000000127",
                "valid_from": "2026-08-31",
                "is_active": 1,
                "loaded_at": pd.Timestamp("2026-08-31"),
                "comment": "current",
            },
        ]
    )

    result = build_deactivation_rows(
        current,
        replacement_valid_from=pd.Timestamp("2026-08-31"),
        loaded_at=pd.Timestamp("2026-09-01"),
    )

    assert result["valid_from"].tolist() == [pd.Timestamp("2026-07-09").date()]
    assert result["is_active"].tolist() == [0]

"""Metadata for purchased products used by pilot forecast publishers.

The source documents are operational spreadsheets supplied by the business:

- ``price_list_2026-08-04 1 (1).xlsx`` — confectionery/order coefficients.
- ``price_list_2026-08-04 (2).xlsx`` — purchased bread/order coefficients.
- Bitrix shelf-life workbook ``Сроки годности и условия хранения ГИ и ПФ по
  группам 02.09.2026.xlsx`` — shelf-life labels.

Rows with no reliable match are deliberately kept in the publication and
rendered with explicit "нет данных ..." labels by the caller.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


PURCHASED_BREAD_CATEGORIES = {"Хлеб"}
PURCHASED_CONFECTIONERY_CATEGORIES = {
    "Пирожные",
    "Маффин Печенье Донатс",
    "Торты Рулеты",
}
PURCHASED_CATEGORIES = PURCHASED_BREAD_CATEGORIES | PURCHASED_CONFECTIONERY_CATEGORIES

MISSING_KRATNOST_LABEL = "нет данных по кратности"
MISSING_SHELF_LIFE_LABEL = "нет данных по сроку хранения"


@dataclass(frozen=True)
class PurchasedProductMetadata:
    kratnost: int | None = None
    shelf_life: str | None = None


def _normalize_product_name(value: object) -> str:
    text = str(value or "").lower().replace("ё", "е")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\d+[,.]?\d*\s*(?:г|гр|кг|мл)\b", " ", text)
    text = re.sub(r"[^а-яa-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _metadata(
    kratnost: int | None = None,
    shelf_life: str | None = None,
) -> PurchasedProductMetadata:
    return PurchasedProductMetadata(kratnost=kratnost, shelf_life=shelf_life)


# Ordered from more specific to more generic aliases.
_PURCHASED_PRODUCT_ALIASES: tuple[tuple[str, PurchasedProductMetadata], ...] = tuple(
    (_normalize_product_name(alias), metadata)
    for alias, metadata in [
        # Purchased bread. Shelf-life business rule: one day, no carry-over stock.
        ("Хлеб Ржано-пшеничный безд", _metadata(1, "1 день")),
        ("Хлеб Боярский", _metadata(1, "1 день")),
        ("Хлеб Тартин со злаками", _metadata(1, "1 день")),
        ("Хлеб Тартин", _metadata(1, "1 день")),
        ("Хлеб Пшеничный", _metadata(1, "1 день")),
        ("Хлеб Картофельный", _metadata(1, "1 день")),
        ("Хлеб тостовый пшеничный", _metadata(1, "1 день")),
        ("Хлеб Ржано-пшеничный", _metadata(1, "1 день")),
        ("Батон нарезной", _metadata(1, "1 день")),
        ("Хлеб Фитнес", _metadata(1, "1 день")),
        ("Хлебец Чиабатта пшен 90", _metadata(12, "1 день")),
        ("Хлебец Чиабатта пшен 50", _metadata(10, "1 день")),
        ("Хлеб Домашний безд", _metadata(1, "1 день")),
        ("Булочка Бейгл", _metadata(6, "1 день")),
        ("Булочка для Хот дога", _metadata(10, "1 день")),
        ("Хлебушек Заварной", _metadata(18, "1 день")),
        ("Хлебушек Бородино", _metadata(1, "1 день")),
        ("Хлебушек Зерновой", _metadata(1, "1 день")),
        ("Хлеб Здоровье", _metadata(1, "1 день")),
        # Confectionery.
        ("Чизкейк класс", _metadata(4, "48 часов")),
        ("Чизкейк Брауни", _metadata(4, "48 часов")),
        ("Медовик", _metadata(3, "5 суток")),
        ("Пирожное со сливками", _metadata(4, "5 суток")),
        ("Пирожное Малина", _metadata(4, "5 суток")),
        ("Графские развалины", _metadata(6, "5 суток")),
        ("Кольцо творог", _metadata(5, "5 суток")),
        ("Кольцо заварное с творожным кремом", _metadata(5, "5 суток")),
        ("Кейк попс", _metadata(6, "5 суток")),
        ("Кейк-попс", _metadata(6, "5 суток")),
        ("Каприз", _metadata(6, "5 суток")),
        ("Картошка", _metadata(8, "5 суток")),
        ("Шоколадно карамельное", _metadata(4, "5 суток")),
        ("Шоколадно-карамельное", _metadata(4, "5 суток")),
        ("Рожок с кремом", _metadata(6, "5 суток")),
        ("Школьное", _metadata(4, "5 суток")),
        ("Наполеон с бананом", _metadata(1, "5 суток")),
        ("Наполеон", _metadata(1, "5 суток")),
        ("Торт меренговый с абрикос", _metadata(1, "72 часа")),
        ("Торт меренговый с виш", _metadata(1, "72 часа")),
        ("Меренговый с абрикос", _metadata(1, "72 часа")),
        ("Меренговый с виш", _metadata(1, "72 часа")),
        ("Малиновое сердце", _metadata(1, "5 суток")),
        ("Печенье Имбирное", _metadata(8, "30 суток")),
        ("Печенье Детское", _metadata(None, "10 суток")),
        ("Печенье Красный Бархат", _metadata(2, "5 суток")),
        ("Печенье фисташковое", _metadata(2, "5 суток")),
        ("Печенье цитрус", _metadata(2, "5 суток")),
        ("Печенье Шоколадное", _metadata(2, "5 суток")),
        ("Печенье Лимонное", _metadata(2, None)),
        ("Пончик белый шоколад", _metadata(4, "72 часа")),
        ("Пончик малина", _metadata(4, "72 часа")),
        ("Пончик шоколад", _metadata(4, "72 часа")),
        ("Эклер с шок с посыпкой", _metadata(3, "5 суток")),
        ("Эклер Шоколадный посыпка", _metadata(3, "5 суток")),
        ("Эклер слив", _metadata(3, "5 суток")),
        ("Эклер сливочный", _metadata(3, "5 суток")),
        ("Эклер класс", _metadata(3, "5 суток")),
        ("Эклер классический", _metadata(3, "5 суток")),
        ("Трубочка вафельная с сгущенкой", _metadata(12, None)),
        ("Маффин шоколад", _metadata(None, "7 суток")),
        ("Маффин морковь", _metadata(None, "7 суток")),
        ("Маффин ваниль", _metadata(None, "5 суток")),
        ("Маффин Праздничный", _metadata(None, "5 суток")),
        ("Макаронсы", _metadata(None, "5 суток")),
    ]
)


def get_purchased_product_metadata(product_name: object) -> PurchasedProductMetadata:
    normalized = _normalize_product_name(product_name)
    if not normalized:
        return PurchasedProductMetadata()
    for alias, metadata in _PURCHASED_PRODUCT_ALIASES:
        if alias and alias in normalized:
            return metadata
    return PurchasedProductMetadata()

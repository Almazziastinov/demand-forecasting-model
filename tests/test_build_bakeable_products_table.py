from __future__ import annotations

from pathlib import Path

import openpyxl
import pandas as pd
import pytest
from openpyxl.styles import Font

from scripts.build_bakeable_products_table import (
    MARKUP_NAME_COLUMN,
    MARKUP_START_ROW,
    build_bakeable_table,
    read_red_markup_names,
    resolve_markup_ids,
)


def _write_markup(path: Path, rows: list[tuple[str, bool]]) -> None:
    """rows: list of (product_name, is_red)."""
    wb = openpyxl.Workbook()
    ws = wb.active
    for offset, (name, is_red) in enumerate(rows):
        cell = ws.cell(row=MARKUP_START_ROW + offset, column=MARKUP_NAME_COLUMN)
        cell.value = name
        if is_red:
            cell.font = Font(color="FFFF0000")
    wb.save(path)


def _write_assortment(path: Path) -> None:
    pd.DataFrame(
        {
            "city": ["Казань", "Казань", "Казань"],
            "product_id": ["1", "2", "3"],
            "product_name": ["Треугольник", "Эклер", "Кыстыбый"],
            "category_name": ["Выпечка сытная", "Пирожные", "Выпечка сытная"],
            "is_active": [1, 1, 1],
        }
    ).to_csv(path, index=False, encoding="utf-8-sig")


def _write_dim(path: Path) -> None:
    pd.DataFrame(
        {
            "product_id": ["1", "2", "3"],
            "product_name": ["Треугольник", "Эклер", "Кыстыбый"],
            "category_name": ["Выпечка сытная", "Пирожные", "Выпечка сытная"],
        }
    ).to_csv(path, index=False, encoding="utf-8-sig")


def test_read_red_markup_only_returns_red_font_rows(tmp_path: Path) -> None:
    markup = tmp_path / "markup.xlsx"
    _write_markup(markup, [("Треугольник", False), ("Эклер", True)])
    assert read_red_markup_names(markup) == ["Эклер"]


def test_meringue_cake_markup_aliases_resolve_to_dim_product_ids() -> None:
    exact = {
        "меренговый с абрикосом": {"000011471"},
        "меренговый с вишней": {"000011472"},
    }

    matched, not_found = resolve_markup_ids(
        ["Торт Меренговый с абрикосом", "Торт Меренговый с вишней"],
        exact,
        {},
    )

    assert matched == {"000011471", "000011472"}
    assert not_found == []


def test_bakeable_allowlist_is_active_minus_red(tmp_path: Path) -> None:
    markup = tmp_path / "markup.xlsx"
    assortment = tmp_path / "assortment.csv"
    dim = tmp_path / "dim.csv"
    _write_markup(markup, [("Треугольник", False), ("Эклер", True)])
    _write_assortment(assortment)
    _write_dim(dim)

    table, excluded_ids, not_found = build_bakeable_table(
        assortment_csv=assortment,
        markup_xlsx=markup,
        dim_products_path=dim,
        valid_from="2026-06-18",
    )

    assert excluded_ids == {"2"}
    assert not_found == []
    assert set(table["product_id"]) == {"1", "3"}
    assert set(table["city"]) == {"Казань"}
    assert (table["is_bakeable"] == 1).all()
    assert "2" not in set(table["product_id"])


def test_red_blacklist_is_applied_to_each_city_and_unlisted_products_remain(
    tmp_path: Path,
) -> None:
    markup = tmp_path / "markup.xlsx"
    assortment = tmp_path / "assortment.csv"
    dim = tmp_path / "dim.csv"
    _write_markup(markup, [("Эклер", True)])
    pd.DataFrame(
        {
            "city": ["Казань", "Казань", "Чебоксары", "Чебоксары"],
            "product_id": ["1", "2", "2", "3"],
            "product_name": ["Треугольник", "Эклер", "Эклер", "Кыстыбый"],
            "category_name": ["Выпечка", "Пирожные", "Пирожные", "Выпечка"],
            "is_active": [1, 1, 1, 1],
        }
    ).to_csv(assortment, index=False, encoding="utf-8-sig")
    _write_dim(dim)

    table, excluded_ids, _ = build_bakeable_table(
        assortment_csv=assortment,
        markup_xlsx=markup,
        dim_products_path=dim,
        valid_from="2026-06-18",
    )

    assert excluded_ids == {"2"}
    assert set(map(tuple, table[["city", "product_id"]].to_numpy())) == {
        ("Казань", "1"),
        ("Чебоксары", "3"),
    }


def test_unmatched_red_product_blocks_allowlist_build(tmp_path: Path) -> None:
    markup = tmp_path / "markup.xlsx"
    assortment = tmp_path / "assortment.csv"
    dim = tmp_path / "dim.csv"
    _write_markup(markup, [("Неизвестный красный товар", True)])
    _write_assortment(assortment)
    _write_dim(dim)

    with pytest.raises(ValueError, match="Неизвестный красный товар"):
        build_bakeable_table(
            assortment_csv=assortment,
            markup_xlsx=markup,
            dim_products_path=dim,
            valid_from="2026-06-18",
        )

from __future__ import annotations

from scripts.build_city_assortment_table import DEFAULT_MANUAL_PATH
from scripts.build_city_assortment_table import TATARSTAN_SCOPE
from scripts.build_city_assortment_table import read_ocr_tatarstan
from scripts.build_required_assortment_contract import SCOPE_TO_CITIES


def test_tatarstan_city_assortment_uses_78_screenshot_products() -> None:
    source = read_ocr_tatarstan(DEFAULT_MANUAL_PATH)
    expected_cities = set(SCOPE_TO_CITIES[TATARSTAN_SCOPE])

    assert set(source["city"]) == expected_cities
    assert source["source"].eq("ocr_tatarstan").all()
    assert source.groupby("city")["product_key"].nunique().eq(78).all()
    assert len(source) == 78 * len(expected_cities)

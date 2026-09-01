from datetime import date

import pandas as pd

from scripts.seed_assortment_sku_meta_202608 import build_rows


def test_build_rows_contains_confirmed_metadata() -> None:
    loaded_at = pd.Timestamp("2026-08-27 12:00:00")
    rows = build_rows(date(2026, 8, 27), loaded_at=loaded_at).set_index("product_id")

    assert set(rows.index) == {"000011575", "000011615", "000011616", "000011617"}
    assert rows.loc["000011575", "dough_group"] == "Тесто Песочка"
    assert rows.loc["000011575", "kratnost"] == 1
    assert rows.loc["000011615", "kratnost"] == 10
    assert rows["station"].eq("Пекарь").all()
    assert rows["is_two_day"].eq(0).all()
    assert rows["is_on_demand"].eq(0).all()
    assert rows["scope"].eq("base").all()
    assert rows["valid_from"].eq(date(2026, 8, 27)).all()

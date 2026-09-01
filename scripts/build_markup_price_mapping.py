"""Map markup workbook product names to ClickHouse product identifiers."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_relaxed_stockout_demand import client_from_env  # noqa: E402

VALUES = ROOT / ".codex_tmp/markup_workbook_inspection/values.json"
OUTPUT = ROOT / "reports/markup_price_mapping_20260826"


def normalize(value: object) -> str:
    text = str(value or "").strip().lower().replace("ё", "е")
    return re.sub(r"[^0-9a-zа-я]+", " ", text).strip()


def main() -> None:
    sheets = json.loads(VALUES.read_text(encoding="utf-8"))
    analytics = next(sheet for sheet in sheets if sheet["name"] == "Table_Analytics")
    values = analytics["values"]
    headers = values[0]
    workbook = pd.DataFrame(values[1:], columns=headers)
    workbook = workbook[
        ["Номенклатура", "Категория", "Себес новый", "Цена текущая", "Наценка текущая"]
    ].rename(
        columns={
            "Номенклатура": "workbook_product_name",
            "Категория": "workbook_category",
            "Себес новый": "unit_cost",
            "Цена текущая": "unit_price",
            "Наценка текущая": "markup_current",
        }
    )
    workbook = workbook.dropna(subset=["workbook_product_name"]).copy()
    workbook["product_key"] = workbook["workbook_product_name"].map(normalize)
    for column in ["unit_cost", "unit_price", "markup_current"]:
        workbook[column] = pd.to_numeric(workbook[column], errors="coerce")

    dim = client_from_env(ROOT / ".env").query_df(
        """
        select toInt64OrZero(toString(product_id)) product_id,
               argMax(product_name, _updated_at) product_name,
               argMax(category_name, _updated_at) category_name
        from Svezhar.dim_products
        group by product_id
        """
    )
    dim["product_key"] = dim["product_name"].map(normalize)
    key_counts = dim.groupby("product_key")["product_id"].nunique()
    unique_dim = dim[dim["product_key"].map(key_counts).eq(1)].drop_duplicates("product_key")
    mapped = workbook.merge(unique_dim, on="product_key", how="left", validate="many_to_one")
    mapped["valid_economics"] = (
        mapped["product_id"].notna()
        & mapped["unit_price"].gt(0)
        & mapped["unit_cost"].ge(0)
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    mapped.to_csv(OUTPUT / "mapped_products.csv", index=False, encoding="utf-8-sig")
    mapped[~mapped["valid_economics"]].to_csv(
        OUTPUT / "unmapped_or_invalid.csv", index=False, encoding="utf-8-sig"
    )
    print(
        f"workbook_rows={len(mapped)} mapped={mapped['product_id'].notna().sum()} "
        f"valid={mapped['valid_economics'].sum()}"
    )
    print(mapped[mapped["workbook_product_name"].eq("Кыстыбый П")].to_string(index=False))


if __name__ == "__main__":
    main()

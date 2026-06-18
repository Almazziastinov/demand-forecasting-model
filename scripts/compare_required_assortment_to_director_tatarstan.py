"""Compare OCR required Татарстан assortment with director assortment workbook."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_required_assortment_contract import (  # noqa: E402
    DEFAULT_MANUAL_PATH,
    DEFAULT_OUTPUT_DIR,
    expand_manual_by_city,
    normalize_category,
    normalize_product,
    read_manual,
)


DEFAULT_DIRECTOR_PATH = (
    ROOT / "tmp_assortment_work" / "director_tatarstan_assortment.xlsx"
)
DEFAULT_OUTPUT_PATH = (
    DEFAULT_OUTPUT_DIR / "director_tatarstan_assortment_comparison.csv"
)
TATARSTAN_SCOPE = "kazan_zelenodolsk_zakamye"


def read_director_assortment(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Table_Analytics", header=3).dropna(how="all")
    columns = {
        raw.columns[2]: "director_product_name",
        raw.columns[3]: "director_supercategory",
        raw.columns[4]: "director_category",
        raw.columns[5]: "director_qty",
        raw.columns[7]: "director_bakery_count",
        raw.columns[18]: "director_current_price",
    }
    director = raw[list(columns)].rename(columns=columns).copy()
    director["product_key"] = director["director_product_name"].map(normalize_product)
    director["director_supercategory_norm"] = director["director_supercategory"].map(
        normalize_category
    )
    return director.drop_duplicates("product_key")


def build_comparison(*, manual_path: Path, director_path: Path) -> pd.DataFrame:
    manual = expand_manual_by_city(read_manual(manual_path))
    required = manual[manual["market_scope"].eq(TATARSTAN_SCOPE)].copy()
    required = required[
        [
            "market_scope",
            "category",
            "category_norm",
            "product_name",
            "product_key",
            "is_required",
            "is_top",
            "top_rank",
            "source_note",
        ]
    ].drop_duplicates(["product_key", "category_norm"])

    director = read_director_assortment(director_path)
    compared = required.merge(director, on="product_key", how="left")
    compared["present_in_director_tatarstan"] = compared[
        "director_product_name"
    ].notna()
    compared["director_category_mismatch"] = (
        compared["present_in_director_tatarstan"]
        & (compared["category_norm"] != compared["director_supercategory_norm"])
    )
    compared["director_status"] = "not_found"
    compared.loc[
        compared["present_in_director_tatarstan"], "director_status"
    ] = "found"
    compared.loc[
        compared["director_category_mismatch"], "director_status"
    ] = "category_mismatch"
    return compared


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-path", default=DEFAULT_MANUAL_PATH, type=Path)
    parser.add_argument("--director-path", default=DEFAULT_DIRECTOR_PATH, type=Path)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH, type=Path)
    args = parser.parse_args()

    compared = build_comparison(
        manual_path=args.manual_path,
        director_path=args.director_path,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    compared.to_csv(args.output_path, index=False, encoding="utf-8-sig")

    print("required unique OCR Татарстан:", len(compared))
    print("present in director:", int(compared["present_in_director_tatarstan"].sum()))
    print(
        "missing in director:",
        int((~compared["present_in_director_tatarstan"]).sum()),
    )
    print(
        "category mismatches:",
        int(compared["director_category_mismatch"].sum()),
    )
    print("output:", args.output_path)


if __name__ == "__main__":
    main()

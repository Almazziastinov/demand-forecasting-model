"""Build city-specific bakeable-product allowlists from partner markup.

The baking plan must only contain products that are actually baked on site.
Partner feedback marks non-baked items (bought-in confectionery, doughnuts,
cookies, eclairs, etc.) with **red font** in a baking-plan preview workbook. This
script takes each city's active assortment, subtracts the globally red-marked
products, and emits the resulting city-specific allowlist. Products absent from
the markup workbook are retained: the workbook is a blacklist, not a complete
list of bakeable products.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_city_assortment_table import (  # noqa: E402
    DEFAULT_VALID_FROM,
)
from scripts.build_required_assortment_contract import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    normalize_text,
)
from scripts.compare_required_assortment_to_dim_products import (  # noqa: E402
    normalize_product_for_dim_lookup,
)


DEFAULT_ASSORTMENT_CSV = DEFAULT_OUTPUT_DIR / "assortment_city_products.csv"
DEFAULT_DIM_PRODUCTS_PATH = DEFAULT_OUTPUT_DIR / "dim_products_lookup.csv"
DEFAULT_MARKUP_XLSX = (
    ROOT
    / "reports"
    / "baking_plan_templates"
    / "preview_sibirskiy_25_screenshot_assortment_v2.xlsx"
)
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "bakeable_products.csv"
SOURCE_NAME = "partner_baking_markup"
# Plan rows start at row 6 in the baking-plan workbooks; product names live in
# column 2. A red font marks a product that is NOT baked on site.
MARKUP_START_ROW = 6
MARKUP_NAME_COLUMN = 2
RED_FONT_RGBS = {"FFFF0000", "FF0000"}
MARKUP_NAME_ALIASES = {
    normalize_text("Торт Меренговый с абрикосом"): "Меренговый с абрикосом",
    normalize_text("Торт Меренговый с вишней"): "Меренговый с вишней",
}


def read_red_markup_names(markup_xlsx: Path) -> list[str]:
    """Return product names rendered with red font in the markup workbook."""
    workbook = openpyxl.load_workbook(markup_xlsx)
    sheet = workbook.active
    names: list[str] = []
    for row in range(MARKUP_START_ROW, sheet.max_row + 1):
        cell = sheet.cell(row=row, column=MARKUP_NAME_COLUMN)
        name = cell.value
        if not name:
            continue
        font = cell.font
        color = font.color if font else None
        if (
            color is not None
            and color.type == "rgb"
            and str(color.rgb).upper() in RED_FONT_RGBS
        ):
            names.append(str(name).strip())
    return names


def build_dim_id_lookups(
    dim_products_path: Path,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    dim = pd.read_csv(dim_products_path, dtype={"product_id": str})
    dim["product_key_exact"] = dim["product_name"].map(normalize_text)
    dim["product_key_lookup"] = dim["product_name"].map(
        normalize_product_for_dim_lookup
    )
    exact = (
        dim.groupby("product_key_exact")["product_id"].apply(lambda s: set(s)).to_dict()
    )
    alias = (
        dim.groupby("product_key_lookup")["product_id"]
        .apply(lambda s: set(s))
        .to_dict()
    )
    return exact, alias


def resolve_markup_ids(
    names: list[str],
    exact_lookup: dict[str, set[str]],
    alias_lookup: dict[str, set[str]],
) -> tuple[set[str], list[str]]:
    """Map red product names to product_id; return (matched_ids, not_found_names)."""
    matched: set[str] = set()
    not_found: list[str] = []
    for name in names:
        lookup_name = MARKUP_NAME_ALIASES.get(normalize_text(name), name)
        ids = exact_lookup.get(normalize_text(lookup_name)) or alias_lookup.get(
            normalize_product_for_dim_lookup(lookup_name)
        )
        if ids:
            matched |= ids
        else:
            not_found.append(name)
    return matched, not_found


def build_bakeable_table(
    *,
    assortment_csv: Path,
    markup_xlsx: Path,
    dim_products_path: Path,
    valid_from: str,
) -> tuple[pd.DataFrame, set[str], list[str]]:
    assortment = pd.read_csv(assortment_csv, dtype={"product_id": str})
    active = assortment[
        pd.to_numeric(assortment["is_active"], errors="coerce").fillna(0).eq(1)
    ].copy()
    required_columns = {"city", "product_id", "product_name", "category_name"}
    missing_columns = required_columns - set(active.columns)
    if missing_columns:
        raise ValueError(
            "Assortment CSV is missing columns: " + ", ".join(sorted(missing_columns))
        )
    universe = active.dropna(subset=["city", "product_id"]).copy()
    universe["city"] = universe["city"].astype(str).str.strip()
    universe["product_id"] = universe["product_id"].astype(str).str.strip()
    universe = universe[
        universe["city"].ne("") & universe["product_id"].ne("")
    ].drop_duplicates(["city", "product_id"])

    red_names = read_red_markup_names(markup_xlsx)
    exact_lookup, alias_lookup = build_dim_id_lookups(dim_products_path)
    excluded_ids, not_found = resolve_markup_ids(red_names, exact_lookup, alias_lookup)
    if not_found:
        names = ", ".join(sorted(not_found))
        raise ValueError(f"Red markup products not found in dim_products: {names}")

    table = universe[~universe["product_id"].isin(excluded_ids)][
        ["city", "product_id", "product_name", "category_name"]
    ].copy()
    loaded_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    table["is_bakeable"] = 1
    table["source"] = SOURCE_NAME
    table["source_file"] = markup_xlsx.name
    table["valid_from"] = pd.to_datetime(valid_from).date().isoformat()
    table["valid_to"] = pd.NA
    table["is_active"] = 1
    table["loaded_at"] = loaded_at
    table["comment"] = ""
    table = table.sort_values(["city", "product_id"]).reset_index(drop=True)
    return table, excluded_ids, not_found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assortment-csv", default=DEFAULT_ASSORTMENT_CSV, type=Path)
    parser.add_argument("--markup-xlsx", default=DEFAULT_MARKUP_XLSX, type=Path)
    parser.add_argument(
        "--dim-products-path", default=DEFAULT_DIM_PRODUCTS_PATH, type=Path
    )
    parser.add_argument("--valid-from", default=DEFAULT_VALID_FROM)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH, type=Path)
    args = parser.parse_args()

    table, excluded_ids, not_found = build_bakeable_table(
        assortment_csv=args.assortment_csv,
        markup_xlsx=args.markup_xlsx,
        dim_products_path=args.dim_products_path,
        valid_from=args.valid_from,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_path, index=False, encoding="utf-8-sig")

    print("bakeable allowlist rows:", len(table))
    print("excluded (red) product_ids:", len(excluded_ids))
    print("red names not matched in dim_products:", len(not_found))
    for name in not_found:
        print("  not_found:", name)
    print("output:", args.output_path)


if __name__ == "__main__":
    main()

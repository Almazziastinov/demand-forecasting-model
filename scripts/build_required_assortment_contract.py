"""Build required assortment contract reports from manual OCR and city tops.

The source workbook ``Топы по городам.xlsx`` contains useful sales-derived tops,
but the screenshots carry an explicit "required assortment" flag. This script
keeps those two signals separate and produces audit-ready CSVs.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANUAL_PATH = ROOT / "config" / "required_assortment_manual.csv"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "required_assortment"

WATCH_SHEET_INDEX = 4
FULL_TOPS_SHEET_INDEX = 3
SALES_SHEET_INDEX = 1
STORES_SHEET_INDEX = 2

WATCH_HEADER_ROW = 1

CATEGORY_PREFIX_RE = re.compile(r"^\s*\d+[а-яa-z]?\.\s*", flags=re.IGNORECASE)

CATEGORY_ALIASES = {
    "пирожные десерты": "Кондитерка",
    "пирожные премиум": "Кондитерка",
    "пирожные": "Кондитерка",
    "выпечка сладкая": "Выпечка сладкая",
    "выпечка сытная": "Выпечка сытная",
    "пироги сладкие": "Пироги сладкие",
    "пироги сытные": "Пироги сытные",
    "маффин печенье донатс": "Маффин Печенье Донатс",
    "торты рулеты": "Торты Рулеты",
    "торты рулеты печенье": "Торты Рулеты",
}

PRODUCT_ALIASES = {
    "жарпицца курица": "жар пицца с курицей",
    "кейкпопс": "кейк попс",
    "кейк попс": "кейк попс",
    "киш гриб кур": "киш грибы курица",
    "киш грибы кур": "киш грибы курица",
    "кыстыбый цп": "кыстыбый п",
    "пир чизкейк брауни": "чизкейк брауни",
    "пицца колбаса п": "пицца с колбасой",
    "треуг гов безд": "треугольник говядина безд",
    "треуг кур": "треугольник курица безд",
    "треуг острый": "треугольник острый",
    "эклер класс": "эклер классический",
    "эклер слив": "эклер сливочный",
    "эклер шок посып": "эклер шоколадный посыпка",
    "элеш": "элеш с курицей",
}

SCOPE_TO_CITIES = {
    "kazan_zelenodolsk_zakamye": [
        "Казань",
        "Зеленодольск",
        "Набережные Челны",
        "Нижнекамск",
        "Альметьевск",
        "Заинск",
    ],
    "cheboksary": ["Чебоксары"],
}


def normalize_text(value: object) -> str:
    text = str(value or "").casefold().replace("ё", "е")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    return " ".join(text.split())


def normalize_product(value: object) -> str:
    key = normalize_text(value)
    return PRODUCT_ALIASES.get(key, key)


def normalize_category(value: object) -> str:
    text = CATEGORY_PREFIX_RE.sub("", str(value or "")).strip()
    key = normalize_text(text)
    return CATEGORY_ALIASES.get(key, text)


def read_tops(path: Path, *, sheet_index: int) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet_index, header=WATCH_HEADER_ROW)
    raw = raw.dropna(how="all")
    raw = raw.iloc[:, 3:10].copy()
    raw.columns = [
        "city",
        "category",
        "product_name",
        "qty",
        "revenue",
        "qty_per_bakery_day",
        "rank_qty",
    ]
    raw = raw.dropna(subset=["city", "category", "product_name"])
    raw["category_norm"] = raw["category"].map(normalize_category)
    raw["product_key_raw"] = raw["product_name"].map(normalize_text)
    raw["product_key"] = raw["product_name"].map(normalize_product)
    return raw


def read_sales(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=SALES_SHEET_INDEX, header=0)
    raw.columns = [
        "city",
        "op_dir",
        "regional_dir",
        "partner",
        "bakery_name",
        "category",
        "product_name",
        "qty",
        "revenue",
        "specific",
        "qty_per_bakery_day",
    ]
    raw = raw.dropna(subset=["city", "bakery_name", "category", "product_name"])
    raw["category_norm"] = raw["category"].map(normalize_category)
    raw["product_key_raw"] = raw["product_name"].map(normalize_text)
    raw["product_key"] = raw["product_name"].map(normalize_product)
    return raw


def read_stores(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=STORES_SHEET_INDEX, header=0)
    raw.columns = [
        "database",
        "city",
        "op_dir",
        "regional_dir",
        "partner",
        "bakery_name",
        "format",
        "status",
    ]
    return raw.dropna(subset=["city", "bakery_name"])


def read_manual(path: Path) -> pd.DataFrame:
    manual = pd.read_csv(path, encoding="utf-8")
    manual["category_norm"] = manual["category"].map(normalize_category)
    manual["product_key_raw"] = manual["product_name"].map(normalize_text)
    manual["product_key"] = manual["product_name"].map(normalize_product)
    manual["is_required"] = manual["is_required"].astype(int)
    manual["is_top"] = manual["is_top"].astype(int)
    return manual


def expand_manual_by_city(manual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in manual.to_dict("records"):
        cities = SCOPE_TO_CITIES.get(str(row["market_scope"]), [])
        for city in cities:
            expanded = dict(row)
            expanded["city"] = city
            rows.append(expanded)
    return pd.DataFrame(rows)


def build_contract(
    *,
    manual: pd.DataFrame,
    watch_tops: pd.DataFrame,
    full_tops: pd.DataFrame,
    sales: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manual_city = expand_manual_by_city(manual)

    strict_keys = ["city", "category_norm", "product_key"]
    product_keys = ["city", "product_key"]
    watch_cols = strict_keys + [
        "product_name",
        "category",
        "qty",
        "revenue",
        "qty_per_bakery_day",
        "rank_qty",
    ]
    contract = manual_city.merge(
        watch_tops[watch_cols].rename(
            columns={
                "product_name": "tops_product_name",
                "category": "tops_category",
                "qty": "tops_qty",
                "revenue": "tops_revenue",
                "qty_per_bakery_day": "tops_qty_per_bakery_day",
                "rank_qty": "tops_rank_qty",
            }
        ),
        on=strict_keys,
        how="left",
    )
    contract["present_in_watch_tops_strict_category"] = contract[
        "tops_product_name"
    ].notna()

    watch_product_hit = watch_tops[
        product_keys + ["product_name", "category", "category_norm", "rank_qty"]
    ].drop_duplicates(product_keys)
    contract = contract.merge(
        watch_product_hit.rename(
            columns={
                "product_name": "tops_product_name_any_category",
                "category": "tops_category_any_category",
                "category_norm": "tops_category_norm_any_category",
                "rank_qty": "tops_rank_qty_any_category",
            }
        ),
        on=product_keys,
        how="left",
    )
    contract["present_in_watch_tops"] = contract[
        "tops_product_name_any_category"
    ].notna()
    contract["watch_tops_category_mismatch"] = (
        contract["present_in_watch_tops"]
        & (contract["category_norm"] != contract["tops_category_norm_any_category"])
    )

    full_hit = full_tops[
        product_keys + ["product_name", "category", "category_norm", "rank_qty"]
    ].drop_duplicates(product_keys)
    contract = contract.merge(
        full_hit.rename(
            columns={
                "product_name": "full_tops_product_name_any_category",
                "category": "full_tops_category_any_category",
                "category_norm": "full_tops_category_norm_any_category",
                "rank_qty": "full_rank_qty_any_category",
            }
        ),
        on=product_keys,
        how="left",
    )
    contract["present_in_full_tops"] = contract[
        "full_tops_product_name_any_category"
    ].notna()
    contract["full_tops_category_mismatch"] = (
        contract["present_in_full_tops"]
        & (
            contract["category_norm"]
            != contract["full_tops_category_norm_any_category"]
        )
    )

    sales_agg = (
        sales.groupby(product_keys, as_index=False)
        .agg(
            sales_rows=("product_name", "size"),
            sales_bakeries=("bakery_name", "nunique"),
            sales_qty=("qty", "sum"),
            sales_revenue=("revenue", "sum"),
            sales_categories=("category_norm", lambda s: " | ".join(sorted(set(s)))),
        )
    )
    contract = contract.merge(sales_agg, on=product_keys, how="left")
    for col in ["sales_rows", "sales_bakeries", "sales_qty", "sales_revenue"]:
        contract[col] = pd.to_numeric(contract[col], errors="coerce").fillna(0)
    contract["present_in_sales_detail"] = contract["sales_rows"] > 0

    watch_manual_keys = manual_city[product_keys].drop_duplicates()
    watch_extra = watch_tops.merge(
        watch_manual_keys.assign(in_manual_required=True),
        on=product_keys,
        how="left",
    )
    watch_extra = watch_extra[watch_extra["in_manual_required"].isna()].drop(
        columns=["in_manual_required"]
    )

    summary = (
        contract.groupby(["market_scope", "city", "category_norm"], as_index=False)
        .agg(
            required_rows=("product_name", "size"),
            top_rows=("is_top", "sum"),
            present_in_watch_tops_strict_category=(
                "present_in_watch_tops_strict_category",
                "sum",
            ),
            present_in_watch_tops=("present_in_watch_tops", "sum"),
            watch_tops_category_mismatch=("watch_tops_category_mismatch", "sum"),
            present_in_full_tops=("present_in_full_tops", "sum"),
            full_tops_category_mismatch=("full_tops_category_mismatch", "sum"),
            present_in_sales_detail=("present_in_sales_detail", "sum"),
        )
    )
    for col in [
        "present_in_watch_tops_strict_category",
        "present_in_watch_tops",
        "watch_tops_category_mismatch",
        "present_in_full_tops",
        "full_tops_category_mismatch",
        "present_in_sales_detail",
    ]:
        summary[col] = summary[col].astype(int)
    summary["missing_from_watch_tops"] = (
        summary["required_rows"] - summary["present_in_watch_tops"]
    )
    summary["missing_from_full_tops"] = (
        summary["required_rows"] - summary["present_in_full_tops"]
    )
    summary["missing_from_sales_detail"] = (
        summary["required_rows"] - summary["present_in_sales_detail"]
    )
    return contract, watch_extra, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tops-path", required=True, type=Path)
    parser.add_argument("--manual-path", default=DEFAULT_MANUAL_PATH, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    args = parser.parse_args()

    manual = read_manual(args.manual_path)
    watch_tops = read_tops(args.tops_path, sheet_index=WATCH_SHEET_INDEX)
    full_tops = read_tops(args.tops_path, sheet_index=FULL_TOPS_SHEET_INDEX)
    sales = read_sales(args.tops_path)
    stores = read_stores(args.tops_path)

    contract, watch_extra, summary = build_contract(
        manual=manual,
        watch_tops=watch_tops,
        full_tops=full_tops,
        sales=sales,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract.to_csv(
        args.output_dir / "required_assortment_contract.csv",
        index=False,
        encoding="utf-8-sig",
    )
    contract[~contract["present_in_full_tops"]].to_csv(
        args.output_dir / "required_missing_from_full_tops.csv",
        index=False,
        encoding="utf-8-sig",
    )
    contract[
        contract["full_tops_category_mismatch"]
        | contract["watch_tops_category_mismatch"]
    ].to_csv(
        args.output_dir / "required_category_mismatches.csv",
        index=False,
        encoding="utf-8-sig",
    )
    watch_extra.to_csv(
        args.output_dir / "watch_tops_not_in_manual_required.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        args.output_dir / "required_assortment_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stores.to_csv(
        args.output_dir / "stores_from_tops_workbook.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("required rows:", len(contract))
    print("watch tops extra rows:", len(watch_extra))
    print("outputs:", args.output_dir)


if __name__ == "__main__":
    main()

"""Build city assortment table rows from approved source rules."""

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
    SCOPE_TO_CITIES,
    normalize_category,
    normalize_text,
    normalize_product,
    read_manual,
)
from scripts.compare_required_assortment_to_dim_products import (  # noqa: E402
    is_inactive_name,
    normalize_product_for_dim_lookup,
)


DEFAULT_DIRECTOR_PATH = (
    ROOT / "tmp_assortment_work" / "director_tatarstan_assortment.xlsx"
)
DEFAULT_DIM_PRODUCTS_PATH = DEFAULT_OUTPUT_DIR / "dim_products_lookup.csv"
DEFAULT_OUTPUT_TABLE_PATH = DEFAULT_OUTPUT_DIR / "assortment_city_products.csv"
DEFAULT_OUTPUT_AUDIT_PATH = DEFAULT_OUTPUT_DIR / "assortment_source_audit.csv"
DEFAULT_VALID_FROM = "2026-06-18"
TATARSTAN_SCOPE = "kazan_zelenodolsk_zakamye"
CHEBOKSARY_SCOPE = "cheboksary"
MANUAL_EXCLUDED_ACTIVE_PRODUCT_KEYS = {
    # Partner feedback confirmed this active product must be removed from assortment.
    normalize_text("\u0412\u0438\u0448\u043d\u0435\u0432\u044b\u0439"),
}
MANUAL_EXCLUDED_ACTIVE_PRODUCT_TOKEN_GROUPS = (
    (
        normalize_text("\u043a\u0430\u043f\u0443\u0441\u0442\u0430"),
        normalize_text("\u043a\u0443\u0440\u0438\u0446\u0430"),
    ),
    (normalize_text("\u0432\u0438\u0448\u043d\u0435\u0432"),),
)


def join_unique(values: pd.Series) -> str:
    return " | ".join(sorted(set(map(str, values.dropna()))))


def is_manual_excluded_active_name(product_name: str, city: str = "") -> bool:
    product_key = normalize_text(product_name)
    if product_key in MANUAL_EXCLUDED_ACTIVE_PRODUCT_KEYS:
        return True
    city_key = normalize_text(city)
    for tokens in MANUAL_EXCLUDED_ACTIVE_PRODUCT_TOKEN_GROUPS:
        if (
            city_key == normalize_text("Чебоксары")
            and set(tokens)
            == {
                normalize_text("капуста"),
                normalize_text("курица"),
            }
        ):
            continue
        if all(token in product_key for token in tokens):
            return True
    return False


def read_dim_products(path: Path) -> pd.DataFrame:
    dim = pd.read_csv(path, dtype={"product_id": str})
    dim["product_key_exact"] = dim["product_name"].map(normalize_text)
    dim["product_key_lookup"] = dim["product_name"].map(
        normalize_product_for_dim_lookup
    )
    dim["category_norm"] = dim["category_name"].map(normalize_category)
    dim["is_inactive_product"] = dim["product_name"].map(is_inactive_name) | dim[
        "category_name"
    ].map(is_inactive_name)
    dim = dim[~dim["is_inactive_product"]].copy()
    dim = dim.dropna(subset=["product_key_lookup", "product_id"])
    dim = dim[dim["product_key_lookup"].ne("")]
    return dim.drop_duplicates(["product_key_lookup", "product_id"])


def _aggregate_dim(dim: pd.DataFrame, key_column: str) -> pd.DataFrame:
    return (
        dim.groupby(key_column, as_index=False)
        .agg(
            matched_product_ids=("product_id", lambda s: " | ".join(sorted(set(s)))),
            matched_product_names=("product_name", join_unique),
            matched_category_names=("category_name", join_unique),
            matched_rows=("product_id", "nunique"),
        )
        .rename(columns={key_column: "match_key"})
    )


def build_dim_lookup(dim: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    exact = _aggregate_dim(dim, "product_key_exact")
    alias = _aggregate_dim(dim, "product_key_lookup")
    return exact, alias


def read_director_tatarstan(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Table_Analytics", header=3).dropna(how="all")
    columns = {
        raw.columns[2]: "raw_product_name",
        raw.columns[3]: "raw_category_name",
    }
    source = raw[list(columns)].rename(columns=columns).copy()
    source = source.dropna(subset=["raw_product_name"])
    source["product_key_exact"] = source["raw_product_name"].map(normalize_text)
    source["product_key"] = source["raw_product_name"].map(normalize_product)
    source["source_is_inactive"] = source["raw_product_name"].map(is_inactive_name)
    source = source.drop_duplicates("product_key")

    rows = []
    for city in SCOPE_TO_CITIES[TATARSTAN_SCOPE]:
        work = source.copy()
        work["city"] = city
        rows.append(work)
    result = pd.concat(rows, ignore_index=True)
    result["source"] = "director_tatarstan"
    result["source_priority"] = 1
    result["source_file"] = "Асорт для Алмаза.xlsx"
    result["source_scope"] = TATARSTAN_SCOPE
    result["is_required"] = 1
    result["is_top"] = 0
    result["top_rank"] = pd.NA
    return result


def read_ocr_cheboksary(path: Path) -> pd.DataFrame:
    manual = read_manual(path)
    source = manual[manual["market_scope"].eq(CHEBOKSARY_SCOPE)].copy()
    source = source.rename(
        columns={
            "product_name": "raw_product_name",
            "category": "raw_category_name",
        }
    )
    source = source.drop_duplicates(["product_key", "category_norm"])
    source["product_key_exact"] = source["raw_product_name"].map(normalize_text)
    source["source_is_inactive"] = source["raw_product_name"].map(is_inactive_name)
    source["city"] = SCOPE_TO_CITIES[CHEBOKSARY_SCOPE][0]
    source["source"] = "ocr_cheboksary"
    source["source_priority"] = 20
    source["source_file"] = "OCR screenshots 2026-05-15"
    source["source_scope"] = CHEBOKSARY_SCOPE
    return source[
        [
            "city",
            "raw_product_name",
            "raw_category_name",
            "product_key",
            "product_key_exact",
            "source_is_inactive",
            "source",
            "source_priority",
            "source_file",
            "source_scope",
            "is_required",
            "is_top",
            "top_rank",
        ]
    ]


def match_sources(
    sources: pd.DataFrame,
    exact_lookup: pd.DataFrame,
    alias_lookup: pd.DataFrame,
) -> pd.DataFrame:
    exact = exact_lookup.add_prefix("exact_").rename(
        columns={"exact_match_key": "product_key_exact"}
    )
    alias = alias_lookup.add_prefix("alias_").rename(
        columns={"alias_match_key": "product_key"}
    )
    matched = sources.merge(exact, on="product_key_exact", how="left")
    matched = matched.merge(alias, on="product_key", how="left")
    for column in ["product_ids", "product_names", "category_names", "rows"]:
        matched[f"matched_{column}"] = matched[f"exact_matched_{column}"].fillna(
            matched[f"alias_matched_{column}"]
        )

    matched["match_status"] = "matched"
    matched.loc[matched["matched_rows"].isna(), "match_status"] = "not_found"
    matched.loc[matched["matched_rows"].fillna(0).gt(1), "match_status"] = (
        "duplicate_match"
    )
    matched.loc[matched["source_is_inactive"], "match_status"] = "source_inactive"
    matched["issue"] = ""
    matched.loc[matched["match_status"].eq("not_found"), "issue"] = (
        "Product was not found in active dim_products"
    )
    matched.loc[matched["match_status"].eq("duplicate_match"), "issue"] = (
        "Product name matched multiple active product_id values"
    )
    matched.loc[matched["match_status"].eq("source_inactive"), "issue"] = (
        "Source row is marked as inactive / not used"
    )
    return matched


def build_outputs(
    *,
    director_path: Path,
    manual_path: Path,
    dim_products_path: Path,
    valid_from: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dim = read_dim_products(dim_products_path)
    exact_lookup, alias_lookup = build_dim_lookup(dim)
    sources = pd.concat(
        [
            read_director_tatarstan(director_path),
            read_ocr_cheboksary(manual_path),
        ],
        ignore_index=True,
    )
    matched = match_sources(sources, exact_lookup, alias_lookup)

    ok = matched[matched["match_status"].eq("matched")].copy()
    ok["product_id"] = ok["matched_product_ids"]
    ok["product_name"] = ok["matched_product_names"]
    ok["category_name"] = ok["matched_category_names"]
    manual_excluded = ok.apply(
        lambda row: is_manual_excluded_active_name(
            row["product_name"],
            row["city"],
        ),
        axis=1,
    )
    ok = ok[~manual_excluded].copy()
    ok["valid_from"] = pd.to_datetime(valid_from).date().isoformat()
    ok["valid_to"] = pd.NA
    ok["is_active"] = 1
    ok["loaded_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    ok["comment"] = ""
    table_key = ["city", "product_id", "source", "valid_from"]
    duplicate_table_key = ok.duplicated(table_key, keep=False)
    ok.loc[duplicate_table_key, "comment"] = (
        "Multiple source rows matched the same city/product_id; see audit table"
    )
    ok = ok.sort_values(
        ["city", "product_id", "source_priority", "is_top", "top_rank"],
        ascending=[True, True, True, False, True],
        na_position="last",
    ).drop_duplicates(table_key, keep="first")
    table = ok[
        [
            "city",
            "product_id",
            "product_name",
            "category_name",
            "is_required",
            "is_top",
            "top_rank",
            "source",
            "source_priority",
            "source_file",
            "source_scope",
            "valid_from",
            "valid_to",
            "is_active",
            "loaded_at",
            "comment",
        ]
    ].copy()

    audit = matched.copy()
    audit["matched_product_id"] = audit["matched_product_ids"]
    audit["matched_product_name"] = audit["matched_product_names"]
    audit["matched_category_name"] = audit["matched_category_names"]
    audit["loaded_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    audit = audit[
        [
            "source",
            "source_file",
            "source_scope",
            "city",
            "raw_product_name",
            "raw_category_name",
            "matched_product_id",
            "matched_product_name",
            "matched_category_name",
            "match_status",
            "issue",
            "is_required",
            "is_top",
            "top_rank",
            "loaded_at",
        ]
    ].copy()
    return table, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--director-path", default=DEFAULT_DIRECTOR_PATH, type=Path)
    parser.add_argument("--manual-path", default=DEFAULT_MANUAL_PATH, type=Path)
    parser.add_argument(
        "--dim-products-path",
        default=DEFAULT_DIM_PRODUCTS_PATH,
        type=Path,
    )
    parser.add_argument("--valid-from", default=DEFAULT_VALID_FROM)
    parser.add_argument(
        "--output-table-path",
        default=DEFAULT_OUTPUT_TABLE_PATH,
        type=Path,
    )
    parser.add_argument(
        "--output-audit-path",
        default=DEFAULT_OUTPUT_AUDIT_PATH,
        type=Path,
    )
    args = parser.parse_args()

    table, audit = build_outputs(
        director_path=args.director_path,
        manual_path=args.manual_path,
        dim_products_path=args.dim_products_path,
        valid_from=args.valid_from,
    )
    args.output_table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_table_path, index=False, encoding="utf-8-sig")
    audit.to_csv(args.output_audit_path, index=False, encoding="utf-8-sig")

    print("assortment rows:", len(table))
    print("audit rows:", len(audit))
    print("audit status counts:")
    print(audit["match_status"].value_counts().to_string())
    print("table:", args.output_table_path)
    print("audit:", args.output_audit_path)


if __name__ == "__main__":
    main()

"""Compare required assortment contract with ClickHouse ``Svezhar.dim_products``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipelines.forecast_publish.load_forecast_run import create_client  # noqa: E402
from scripts.build_required_assortment_contract import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    normalize_category,
    normalize_text,
    normalize_product,
)


DEFAULT_ENV_PATH = ROOT / ".env"
DEFAULT_CONTRACT_PATH = DEFAULT_OUTPUT_DIR / "required_assortment_contract.csv"
INACTIVE_PREFIXES = (
    "я не использую",
    "не использовать",
    "я не исп",
    "я не ипс",
    "я не сип",
    "яя не сип",
    "не испол",
    "не испп",
    "не исп",
    "я не",
)


def join_unique(values: pd.Series) -> str:
    return " | ".join(sorted(set(map(str, values))))


def is_inactive_name(value: object) -> bool:
    key = normalize_text(value)
    return any(
        key == prefix or key.startswith(f"{prefix} ") for prefix in INACTIVE_PREFIXES
    )


def normalize_product_for_dim_lookup(value: object) -> str:
    key = normalize_text(value)
    for prefix in INACTIVE_PREFIXES:
        if key == prefix:
            key = ""
            break
        if key.startswith(f"{prefix} "):
            key = key[len(prefix) + 1 :]
            break
    return normalize_product(key)


def load_dim_products(env_path: Path) -> pd.DataFrame:
    client = create_client(env_path)
    rows = client.query(
        """
        SELECT
            product_id,
            product_name,
            category_name
        FROM Svezhar.dim_products
        WHERE notEmpty(product_name)
        """
    )
    dim = pd.DataFrame(rows.result_rows, columns=rows.column_names)
    dim["product_key"] = dim["product_name"].map(normalize_product)
    dim["product_key_lookup"] = dim["product_name"].map(
        normalize_product_for_dim_lookup
    )
    dim["category_norm"] = dim["category_name"].map(normalize_category)
    dim["is_inactive_product"] = dim["product_name"].map(is_inactive_name) | dim[
        "category_name"
    ].map(is_inactive_name)
    return dim


def aggregate_dim_products(dim: pd.DataFrame) -> pd.DataFrame:
    active_dim = dim[~dim["is_inactive_product"]].copy()
    inactive_dim = dim[dim["is_inactive_product"]].copy()
    active_agg = (
        active_dim.groupby("product_key_lookup", as_index=False)
        .agg(
            active_dim_product_ids=("product_id", join_unique),
            active_dim_product_names=("product_name", join_unique),
            active_dim_categories=("category_norm", join_unique),
            active_dim_rows=("product_id", "size"),
        )
        .rename(columns={"product_key_lookup": "product_key"})
    )
    inactive_agg = (
        inactive_dim.groupby("product_key_lookup", as_index=False)
        .agg(
            inactive_dim_product_ids=("product_id", join_unique),
            inactive_dim_product_names=("product_name", join_unique),
            inactive_dim_categories=("category_norm", join_unique),
            inactive_dim_rows=("product_id", "size"),
        )
        .rename(columns={"product_key_lookup": "product_key"})
    )
    return (
        active_agg.merge(inactive_agg, on="product_key", how="outer")
        .assign(
            active_dim_rows=lambda df: pd.to_numeric(
                df["active_dim_rows"], errors="coerce"
            ).fillna(0),
            inactive_dim_rows=lambda df: pd.to_numeric(
                df["inactive_dim_rows"], errors="coerce"
            ).fillna(0),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-path", default=DEFAULT_ENV_PATH, type=Path)
    parser.add_argument("--contract-path", default=DEFAULT_CONTRACT_PATH, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    args = parser.parse_args()

    contract = pd.read_csv(args.contract_path)
    dim = load_dim_products(args.env_path)
    dim_agg = aggregate_dim_products(dim)

    compared = contract.merge(dim_agg, on="product_key", how="left")
    compared["has_active_dim_product"] = compared["active_dim_rows"].fillna(0) > 0
    compared["has_inactive_dim_product"] = compared["inactive_dim_rows"].fillna(0) > 0
    compared["present_in_dim_products"] = (
        compared["has_active_dim_product"] | compared["has_inactive_dim_product"]
    )
    compared["dim_product_status"] = "not_found"
    compared.loc[compared["has_inactive_dim_product"], "dim_product_status"] = (
        "only_inactive"
    )
    compared.loc[compared["has_active_dim_product"], "dim_product_status"] = (
        "active_found"
    )
    compared["dim_product_ids"] = compared["active_dim_product_ids"].fillna(
        compared["inactive_dim_product_ids"]
    )
    compared["dim_product_names"] = compared["active_dim_product_names"].fillna(
        compared["inactive_dim_product_names"]
    )
    compared["dim_categories"] = compared["active_dim_categories"].fillna(
        compared["inactive_dim_categories"]
    )
    compared["dim_rows"] = compared["active_dim_rows"].fillna(0) + compared[
        "inactive_dim_rows"
    ].fillna(0)
    compared["dim_category_matches"] = compared.apply(
        lambda row: bool(row["has_active_dim_product"])
        and str(row["category_norm"]) in str(row["dim_categories"]).split(" | "),
        axis=1,
    )
    compared["dim_category_mismatch"] = (
        compared["has_active_dim_product"] & ~compared["dim_category_matches"]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dim.to_csv(
        args.output_dir / "dim_products_lookup.csv",
        index=False,
        encoding="utf-8-sig",
    )
    compared.to_csv(
        args.output_dir / "required_vs_dim_products.csv",
        index=False,
        encoding="utf-8-sig",
    )
    compared[~compared["present_in_dim_products"]].to_csv(
        args.output_dir / "required_missing_from_dim_products.csv",
        index=False,
        encoding="utf-8-sig",
    )
    compared[compared["dim_product_status"] == "only_inactive"].to_csv(
        args.output_dir / "required_inactive_in_dim_products.csv",
        index=False,
        encoding="utf-8-sig",
    )
    compared[compared["dim_category_mismatch"]].to_csv(
        args.output_dir / "required_dim_category_mismatches.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("required rows:", len(compared))
    missing_count = int((~compared["present_in_dim_products"]).sum())
    print("missing from dim_products:", missing_count)
    print(
        "only inactive in dim_products:",
        int((compared["dim_product_status"] == "only_inactive").sum()),
    )
    print("dim category mismatches:", int(compared["dim_category_mismatch"].sum()))
    print("outputs:", args.output_dir)


if __name__ == "__main__":
    main()

"""Audit partner feedback against dim_products, assortment and prod forecast."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipelines.forecast_publish.load_forecast_run import create_client  # noqa: E402
from scripts.build_required_assortment_contract import normalize_product  # noqa: E402
from scripts.compare_required_assortment_to_dim_products import (  # noqa: E402
    is_inactive_name,
    normalize_product_for_dim_lookup,
)


DEFAULT_FEEDBACK_PATH = ROOT / "config" / "assortment_partner_feedback.csv"
DEFAULT_DIM_PRODUCTS_PATH = (
    ROOT / "reports" / "required_assortment" / "dim_products_lookup.csv"
)
DEFAULT_ENV_PATH = ROOT / ".env"
DEFAULT_OUTPUT_PATH = (
    ROOT / "reports" / "required_assortment" / "partner_feedback_audit.csv"
)
DEFAULT_PROD_RUN_ID = "prod_uplifted_bakery_norm_uplift_sku_20260618_h14"


def join_unique(values: pd.Series) -> str:
    return " | ".join(sorted(set(map(str, values.dropna()))))


def build_dim_lookup(path: Path) -> pd.DataFrame:
    dim = pd.read_csv(path, dtype={"product_id": str})
    dim["product_key"] = dim["product_name"].map(normalize_product_for_dim_lookup)
    dim["is_inactive_product"] = dim["product_name"].map(is_inactive_name) | dim[
        "category_name"
    ].map(is_inactive_name)
    active = dim[~dim["is_inactive_product"]].copy()
    return (
        active.groupby("product_key", as_index=False)
        .agg(
            matched_product_ids=("product_id", join_unique),
            matched_product_names=("product_name", join_unique),
            matched_category_names=("category_name", join_unique),
            matched_rows=("product_id", "nunique"),
        )
    )


def load_prod_forecast_products(client, run_id: str) -> pd.DataFrame:
    forecast = client.query_df(
        """
        select
          toString(product_id) as product_id_norm,
          any(product_name) as prod_product_name,
          any(category_name) as prod_category_name,
          countDistinct(bakery_id) as prod_bakery_count,
          sum(forecast_qty) as prod_horizon_forecast_qty
        from sku_forecast_day_embedded
        where run_id = %(run_id)s
        group by product_id
        having prod_horizon_forecast_qty > 0
        """,
        parameters={"run_id": run_id},
    )
    return forecast


def load_current_assortment_products(client) -> pd.DataFrame:
    assortment = client.query_df(
        """
        select
          replaceRegexpOne(product_id, '^0+', '') as product_id_norm,
          any(product_name) as assortment_product_name,
          any(category_name) as assortment_category_name,
          groupUniqArray(city) as assortment_cities,
          countDistinct(city) as assortment_city_count
        from assortment_city_products
        where is_active = 1
        group by product_id
        """
    )
    assortment["assortment_cities"] = assortment["assortment_cities"].map(
        lambda values: " | ".join(sorted(map(str, values)))
    )
    return assortment


def build_audit(
    *,
    feedback_path: Path,
    dim_products_path: Path,
    env_path: Path,
    prod_run_id: str,
) -> pd.DataFrame:
    feedback = pd.read_csv(feedback_path)
    feedback["product_key"] = feedback["raw_product_name"].map(normalize_product)
    dim_lookup = build_dim_lookup(dim_products_path)
    audit = feedback.merge(dim_lookup, on="product_key", how="left")
    audit["match_status"] = "matched"
    audit.loc[audit["matched_rows"].isna(), "match_status"] = "not_found"
    audit.loc[audit["matched_rows"].fillna(0).gt(1), "match_status"] = (
        "duplicate_match"
    )

    client = create_client(env_path)
    prod = load_prod_forecast_products(client, prod_run_id)
    assortment = load_current_assortment_products(client)

    expanded_rows = []
    for row in audit.to_dict("records"):
        ids = str(row.get("matched_product_ids") or "").split(" | ")
        if not ids or ids == ["nan"] or ids == [""]:
            expanded_rows.append(row)
            continue
        for product_id in ids:
            expanded = dict(row)
            expanded["product_id_norm"] = product_id.lstrip("0") or "0"
            expanded_rows.append(expanded)
    expanded_audit = pd.DataFrame(expanded_rows)

    expanded_audit = expanded_audit.merge(prod, on="product_id_norm", how="left")
    expanded_audit = expanded_audit.merge(
        assortment,
        on="product_id_norm",
        how="left",
    )
    expanded_audit["present_in_prod_forecast"] = expanded_audit[
        "prod_horizon_forecast_qty"
    ].notna()
    expanded_audit["present_in_new_assortment"] = expanded_audit[
        "assortment_city_count"
    ].notna()
    return expanded_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback-path", default=DEFAULT_FEEDBACK_PATH, type=Path)
    parser.add_argument(
        "--dim-products-path",
        default=DEFAULT_DIM_PRODUCTS_PATH,
        type=Path,
    )
    parser.add_argument("--env-path", default=DEFAULT_ENV_PATH, type=Path)
    parser.add_argument("--prod-run-id", default=DEFAULT_PROD_RUN_ID)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH, type=Path)
    args = parser.parse_args()

    audit = build_audit(
        feedback_path=args.feedback_path,
        dim_products_path=args.dim_products_path,
        env_path=args.env_path,
        prod_run_id=args.prod_run_id,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output_path, index=False, encoding="utf-8-sig")

    print("rows:", len(audit))
    print("status counts:")
    print(audit["status"].value_counts().to_string())
    print("match counts:")
    print(audit["match_status"].value_counts().to_string())
    print("present in prod forecast:", int(audit["present_in_prod_forecast"].sum()))
    print("present in new assortment:", int(audit["present_in_new_assortment"].sum()))
    print("output:", args.output_path)


if __name__ == "__main__":
    main()

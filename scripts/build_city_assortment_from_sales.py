"""Build city-level and bakery-level bakeable assortment from recent sales facts.

Two-layer logic:

- scope='city'   — products sold in >= CITY_THRESHOLD share of bakeries in the
                   city over the last WINDOW_DAYS days.  Stable core assortment.
- scope='bakery' — products sold in a specific bakery over the last WINDOW_DAYS
                   days that are NOT already in the city layer for that city.
                   Catches bakery-specific specialties and new launches.

Both layers filter to bakeable categories (пирог / выпечка / фастфуд) before
inserting, consistent with build_bakeable_products_table.py.

Usage
-----
# Production (uses .env, reads mart_sales_60d)
.venv\\Scripts\\python.exe scripts\\build_city_assortment_from_sales.py

# Dry-run (print summary, do NOT write to ClickHouse)
.venv\\Scripts\\python.exe scripts\\build_city_assortment_from_sales.py --dry-run

# Override window or threshold
.venv\\Scripts\\python.exe scripts\\build_city_assortment_from_sales.py \\
    --window-days 14 --city-threshold 0.7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipelines.forecast_publish.load_forecast_run import (  # noqa: E402
    DEFAULT_ENV_PATH,
    create_client,
)
from pipelines.forecast_publish.table_names import (  # noqa: E402
    get_table_suffix_from_env_file,
    table_name,
)

SALES_TABLE = "mart_sales_60d"
DEFAULT_WINDOW_DAYS = 7
DEFAULT_CITY_THRESHOLD = 0.80
DEFAULT_BAKEABLE_CATEGORY_PATTERNS: list[str] = ["пирог", "выпечка", "фастфуд"]
SOURCE_NAME = "sales_window"


# ---------------------------------------------------------------------------
# ClickHouse queries
# ---------------------------------------------------------------------------


def _query_recent_sales(
    client,
    *,
    window_days: int,
    bakery_table: str,
    sku_day_table: str,
    sales_table: str,
) -> pd.DataFrame:
    """Return (city, bakery_id, product_id, product_name, category_name) for
    products that sold at least once in the last window_days days."""
    return client.query_df(
        f"""
        SELECT
            ifNull(nullIf(s.city, ''), b.city) AS city,
            toInt64(s.bakery_id)         AS bakery_id,
            toString(s.product_id)       AS product_id,
            any(s.product_name)          AS product_name,
            any(s.category_name)         AS category_name
        FROM {sales_table} AS s
        INNER JOIN (
            SELECT
                bakery_id,
                if(
                    countIf(city IS NOT NULL AND city != '' AND city != 'unknown') > 0,
                    anyIf(city, city IS NOT NULL AND city != '' AND city != 'unknown'),
                    any(city)
                ) AS city
            FROM {bakery_table}
            GROUP BY bakery_id
        ) AS b ON toInt64(s.bakery_id) = toInt64(b.bakery_id)
        WHERE s.check_date >= today() - {window_days}
          AND s.check_date < today()
          AND s.quantity > 0
          AND NOT startsWith(s.product_name, 'я_не_исп')
          AND NOT startsWith(s.product_name, 'я не исп')
        GROUP BY city, s.bakery_id, s.product_id
        ORDER BY city, s.bakery_id, s.product_id
        """
    )


def _query_bakery_count_per_city(client, *, bakery_table: str) -> pd.DataFrame:
    """Return total number of distinct bakeries per city (denominator for threshold)."""
    return client.query_df(
        f"""
        SELECT city, uniqExact(bakery_id) AS total_bakeries
        FROM (
            SELECT
                bakery_id,
                anyIf(
                    city,
                    city IS NOT NULL AND city != '' AND city != 'unknown'
                ) AS city
            FROM {bakery_table}
            GROUP BY bakery_id
            HAVING city != ''
        )
        GROUP BY city
        """
    )


# ---------------------------------------------------------------------------
# Build layers
# ---------------------------------------------------------------------------


def _apply_category_filter(
    df: pd.DataFrame,
    patterns: list[str],
) -> pd.DataFrame:
    pattern_re = "|".join(p.lower() for p in patterns)
    mask = (
        df["category_name"]
        .fillna("")
        .str.lower()
        .str.contains(pattern_re, regex=True)
    )
    return df[mask].copy()


def build_layers(
    sales: pd.DataFrame,
    bakery_counts: pd.DataFrame,
    *,
    city_threshold: float,
    category_patterns: list[str],
    valid_from: str,
) -> pd.DataFrame:
    """Return combined city + bakery layer table ready for ClickHouse insert."""
    sales = _apply_category_filter(sales, category_patterns)
    if sales.empty:
        return pd.DataFrame()

    # count how many distinct bakeries in each city sold each product
    sold_by = (
        sales.groupby(["city", "product_id"], as_index=False)["bakery_id"]
        .nunique()
        .rename(columns={"bakery_id": "bakeries_selling"})
    )
    sold_by = sold_by.merge(bakery_counts, on="city", how="left")
    sold_by["share"] = sold_by["bakeries_selling"] / sold_by[
        "total_bakeries"
    ].replace(0, pd.NA)

    city_mask = sold_by["share"] >= city_threshold
    city_products = sold_by.loc[city_mask, ["city", "product_id"]].copy()
    city_products["scope"] = "city"
    city_products["bakery_id"] = pd.NA

    # bakery layer: sold in this bakery AND not already in city layer
    city_set = set(zip(city_products["city"], city_products["product_id"]))
    bakery_rows = sales[["city", "bakery_id", "product_id"]].drop_duplicates()
    not_in_city = ~bakery_rows.apply(
        lambda r: (r["city"], r["product_id"]) in city_set, axis=1
    )
    bakery_products = bakery_rows[not_in_city].copy()
    bakery_products["scope"] = "bakery"

    combined = pd.concat([city_products, bakery_products], ignore_index=True)

    # attach product_name + category_name (any from sales)
    name_lookup = (
        sales.groupby(["city", "product_id"], as_index=False)
        .agg(
            product_name=("product_name", "first"),
            category_name=("category_name", "first"),
        )
    )
    combined = combined.merge(name_lookup, on=["city", "product_id"], how="left")

    loaded_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    combined["is_bakeable"] = 1
    combined["source"] = SOURCE_NAME
    combined["source_file"] = f"mart_sales_60d:window_{DEFAULT_WINDOW_DAYS}d"
    # Keep this a real date object, not a string: this frame is inserted
    # directly into ClickHouse via client.insert_df (unlike
    # build_bakeable_products_table.py's CSV-only sibling, where a string
    # is fine). clickhouse-connect's Date serializer does
    # `(value - epoch).days` on each cell, which raises
    # "unsupported operand type(s) for -: 'str' and 'datetime.date'" if
    # value is a str instead of a date.
    combined["valid_from"] = pd.to_datetime(valid_from).date()
    combined["valid_to"] = pd.NA
    combined["is_active"] = 1
    combined["loaded_at"] = loaded_at
    combined["comment"] = ""

    cols = [
        "city", "product_id", "product_name", "category_name",
        "scope", "bakery_id",
        "is_bakeable", "source", "source_file",
        "valid_from", "valid_to", "is_active", "loaded_at", "comment",
    ]
    return (
        combined[cols]
        .sort_values(["city", "scope", "product_id"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# ClickHouse write
# ---------------------------------------------------------------------------


def insert_to_clickhouse(client, df: pd.DataFrame, *, target_table: str) -> int:
    """Append rows to the ReplacingMergeTree-backed bakeable table."""
    if df.empty:
        return 0
    client.insert_df(target_table, df)
    return len(df)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build bakeable assortment from recent sales (city + bakery layers)."
        )
    )
    parser.add_argument("--env-file", default=DEFAULT_ENV_PATH, type=Path)
    parser.add_argument("--window-days", default=DEFAULT_WINDOW_DAYS, type=int)
    parser.add_argument(
        "--city-threshold",
        default=DEFAULT_CITY_THRESHOLD,
        type=float,
        help="Min share of bakeries that must have sold the product (default 0.80).",
    )
    parser.add_argument(
        "--bakeable-categories",
        nargs="+",
        default=DEFAULT_BAKEABLE_CATEGORY_PATTERNS,
        metavar="PATTERN",
    )
    parser.add_argument(
        "--valid-from",
        default=pd.Timestamp.now().date().isoformat(),
        help="valid_from date for inserted rows (ISO format).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary only, do not write to ClickHouse.",
    )
    args = parser.parse_args()

    client = create_client(args.env_file)
    suffix = get_table_suffix_from_env_file(args.env_file)
    bakery_table = table_name("bakery_forecast_day_embedded", suffix=suffix)
    sku_day_table = table_name("sku_forecast_day_embedded", suffix=suffix)
    bakeable_table = table_name("bakeable_products", suffix=suffix)

    print(f"window_days      : {args.window_days}")
    print(f"city_threshold   : {args.city_threshold}")
    print(f"patterns         : {args.bakeable_categories}")
    print(f"valid_from       : {args.valid_from}")
    print(f"target table     : {bakeable_table}")
    print("Querying recent sales...", flush=True)

    sales = _query_recent_sales(
        client,
        window_days=args.window_days,
        bakery_table=bakery_table,
        sku_day_table=sku_day_table,
        sales_table=SALES_TABLE,
    )
    print(f"Sales rows       : {len(sales)}")

    bakery_counts = _query_bakery_count_per_city(client, bakery_table=bakery_table)
    print(f"Cities           : {len(bakery_counts)}")

    result = build_layers(
        sales,
        bakery_counts,
        city_threshold=args.city_threshold,
        category_patterns=args.bakeable_categories,
        valid_from=args.valid_from,
    )

    city_rows = (result["scope"] == "city").sum() if not result.empty else 0
    bakery_rows = (result["scope"] == "bakery").sum() if not result.empty else 0
    print(f"city layer rows  : {city_rows}")
    print(f"bakery layer rows: {bakery_rows}")
    print(f"total rows       : {len(result)}")

    if not result.empty:
        print("\ncity breakdown:")
        summary = (
            result.groupby(["city", "scope"])["product_id"]
            .nunique()
            .unstack(fill_value=0)
        )
        print(summary.to_string())

    if args.dry_run:
        print("\n[dry-run] skipping ClickHouse insert")
        return

    inserted = insert_to_clickhouse(client, result, target_table=bakeable_table)
    print(f"\nInserted {inserted} rows into {bakeable_table}")


if __name__ == "__main__":
    main()

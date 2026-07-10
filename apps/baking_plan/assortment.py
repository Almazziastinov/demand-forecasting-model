"""Bakeable-products allowlist: which SKUs a given bakery is allowed to bake.

Two layers, read from the `bakeable_products` ClickHouse table:
- scope='city'   — products sold in >=80% of bakeries in the city
- scope='bakery' — products unique to one bakery, not already in the city layer
"""

from __future__ import annotations

# ruff: noqa: E501
import logging

from ._clickhouse import get_client, records, table_name

logger = logging.getLogger(__name__)

BAKEABLE_TABLE = table_name("bakeable_products")


def get_bakeable_products(
    city: str,
    effective_date: str,
    bakery_id: int | None = None,
) -> list[dict]:
    """Return the bakeable assortment effective on the forecast date.

    When `bakery_id` is given, returns the union of `scope='city'` rows for
    the city and `scope='bakery'` rows for that specific bakery. When it is
    `None`, returns all rows for the city regardless of scope.

    Missing or empty data is treated as an error: silently disabling this
    filter would put bought-in products back into the baking plan.
    """
    client = get_client()

    if bakery_id is not None:
        scope_filter = """
          and (
              (ifNull(scope, 'city') = 'city')
              or (ifNull(scope, '') = 'bakery' and toInt64OrNull(toString(b.bakery_id)) = %(bakery_id)s)
          )"""
        params: dict = {"city": city, "effective_date": effective_date, "bakery_id": bakery_id}
    else:
        scope_filter = ""
        params = {"city": city, "effective_date": effective_date}

    query = """
        select product_id, any(product_name) as product_name,
               any(category_name) as category_name
        from {table} as b final
        where city = %(city)s
          and is_bakeable = 1
          and is_active = 1
          and valid_from = (
              select max(valid_from)
              from {table} final
              where city = %(city)s
                and valid_from <= toDate(%(effective_date)s)
          )
          and (valid_to is null or valid_to >= toDate(%(effective_date)s))
          {scope_filter}
        group by product_id
        order by category_name, product_name, product_id
        """.format(table=BAKEABLE_TABLE, scope_filter=scope_filter)
    try:
        df = client.query_df(query, parameters=params)
    except Exception:
        logger.exception("Failed to load bakeable products for city %s", city)
        raise
    rows = records(df)
    if not rows:
        raise RuntimeError(f"Bakeable-products allowlist is empty for city {city!r}")
    # bakeable_products mixes sources with inconsistent product_id padding
    # (older `forecast_category_filter` rows are unpadded, e.g. "67" instead
    # of "000000067") — normalize to the 9-digit width used everywhere else
    # (dim_products, baking_sku_meta) so downstream joins match.
    for row in rows:
        row["product_id"] = str(row["product_id"]).zfill(9)
    return rows

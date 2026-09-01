"""Build flat (bakery_id, product_id) assortment from seven-day sales.

The resulting table `bakery_product_assortment_embedded` is a fast
lookup for downstream consumers (baking plan, analytics) that need
per-bakery assortment without writing the city-expansion join themselves.

Usage
-----
# Production
.venv\\Scripts\\python.exe scripts\\build_bakery_product_assortment.py

# Dry-run (print summary, do NOT write to ClickHouse)
.venv\\Scripts\\python.exe scripts\\build_bakery_product_assortment.py --dry-run
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
from pipelines.forecast_publish.assortment_override_store import (  # noqa: E402
    TABLE_BASE as OVERRIDE_TABLE_BASE,
    load_active_overrides,
)
from src.experiments_v2.effective_assortment import apply_emergency_overrides  # noqa: E402
from scripts.build_city_assortment_from_sales import (  # noqa: E402
    DEFAULT_BAKEABLE_CATEGORY_PATTERNS,
    DEFAULT_WINDOW_DAYS,
    SALES_TABLE,
    _apply_category_filter,
    _query_recent_sales,
)

TARGET_TABLE_BASE = "bakery_product_assortment_embedded"
BAKERY_DIMENSION_TABLE = "dim_bakeries"

CREATE_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
    bakery_id   Int64,
    product_id  String,
    valid_from  Date,
    loaded_at   DateTime64(3)
)
ENGINE = ReplacingMergeTree(loaded_at)
ORDER BY (bakery_id, product_id, valid_from)
"""


def ensure_table(client, table: str) -> None:
    client.command(CREATE_DDL.format(table=table))


def _query_bakery_city_map(client, *, bakery_table: str) -> pd.DataFrame:
    """Return (bakery_id, city) for every bakery with a known city."""
    return client.query_df(
        f"""
        SELECT
            toInt64(bakery_id) AS bakery_id,
            anyIf(city, city IS NOT NULL AND city != '' AND city != 'unknown') AS city
        FROM {bakery_table}
        GROUP BY bakery_id
        HAVING city != ''
        ORDER BY bakery_id
        """
    )


def _query_latest_bakeable(
    client, *, bakeable_table: str, cities: list[str]
) -> pd.DataFrame:
    """Return the latest active bakeable_products batch for the given cities."""
    return client.query_df(
        f"""
        SELECT
            b.city,
            b.product_id,
            b.scope,
            toInt64OrNull(toString(b.bakery_id)) AS bakery_id
        FROM {bakeable_table} AS b FINAL
        INNER JOIN (
            SELECT city, max(valid_from) AS max_valid_from
            FROM {bakeable_table} FINAL
            WHERE city IN %(cities)s
              AND valid_from <= today()
            GROUP BY city
        ) AS latest ON b.city = latest.city AND b.valid_from = latest.max_valid_from
        WHERE b.city IN %(cities)s
          AND b.is_bakeable = 1
          AND b.is_active = 1
          AND (b.valid_to IS NULL OR b.valid_to >= today())
        ORDER BY b.city, b.scope, b.product_id
        """,
        parameters={"cities": cities},
    )


def build_assortment(
    bakery_city_map: pd.DataFrame,
    bakeable: pd.DataFrame,
    *,
    valid_from: str,
    overrides: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return flat (bakery_id, product_id) DataFrame ready for ClickHouse insert."""
    if bakery_city_map.empty:
        return pd.DataFrame(
            columns=["bakery_id", "product_id", "valid_from", "loaded_at"]
        )

    if bakeable.empty:
        combined = pd.DataFrame(columns=["bakery_id", "product_id"])
    else:
        city_scope = bakeable[bakeable["scope"] == "city"][
            ["city", "product_id"]
        ].drop_duplicates()
        bakery_scope = bakeable[bakeable["scope"] == "bakery"].dropna(
            subset=["bakery_id"]
        )[["bakery_id", "product_id"]].drop_duplicates()
        bakery_scope = bakery_scope.copy()
        bakery_scope["bakery_id"] = bakery_scope["bakery_id"].astype("int64")
        expanded = bakery_city_map.merge(city_scope, on="city", how="inner")[
            ["bakery_id", "product_id"]
        ].drop_duplicates()
        known_bakeries = set(bakery_city_map["bakery_id"].tolist())
        bakery_extra = bakery_scope[
            bakery_scope["bakery_id"].isin(known_bakeries)
        ].copy()
        combined = pd.concat(
            [expanded, bakery_extra], ignore_index=True
        ).drop_duplicates(subset=["bakery_id", "product_id"])
    combined["bakery_id"] = combined["bakery_id"].astype("int64")
    combined["product_id"] = combined["product_id"].astype(str).str.zfill(9)
    if overrides is not None and not overrides.empty:
        normalized = overrides.copy()
        normalized["product_id"] = normalized["product_id"].astype(str).str.zfill(9)
        combined = apply_emergency_overrides(
            combined.assign(source="recent_sales_7d"),
            normalized,
            effective_date=valid_from,
        ).drop(columns="source")
    combined["valid_from"] = pd.to_datetime(valid_from).date()
    combined["loaded_at"] = pd.Timestamp.now()
    return combined[["bakery_id", "product_id", "valid_from", "loaded_at"]].sort_values(
        ["bakery_id", "product_id"]
    ).reset_index(drop=True)


def build_assortment_from_sales(
    sales: pd.DataFrame,
    *,
    valid_from: str,
    overrides: pd.DataFrame | None = None,
    category_patterns: list[str] = DEFAULT_BAKEABLE_CATEGORY_PATTERNS,
) -> pd.DataFrame:
    """Return effective bakery/SKU pairs sold in the prior seven-day window."""
    filtered = _apply_category_filter(sales, category_patterns)
    combined = filtered[["bakery_id", "product_id"]].drop_duplicates().copy()
    combined["bakery_id"] = combined["bakery_id"].astype("int64")
    combined["product_id"] = combined["product_id"].astype(str).str.zfill(9)
    if overrides is not None and not overrides.empty:
        normalized = overrides.copy()
        normalized["product_id"] = normalized["product_id"].astype(str).str.zfill(9)
        combined = apply_emergency_overrides(
            combined.assign(source="recent_sales_7d"),
            normalized,
            effective_date=valid_from,
        ).drop(columns="source")
    combined["valid_from"] = pd.to_datetime(valid_from).date()
    combined["loaded_at"] = pd.Timestamp.now()
    return combined[["bakery_id", "product_id", "valid_from", "loaded_at"]].sort_values(
        ["bakery_id", "product_id"]
    ).reset_index(drop=True)


def insert_to_clickhouse(client, df: pd.DataFrame, *, target_table: str) -> int:
    if df.empty:
        return 0
    client.insert_df(target_table, df)
    return len(df)


def load_previous_assortment(
    client,
    *,
    table: str,
    bakery_ids: list[int],
    before_date: str,
) -> pd.DataFrame:
    """Load each bakery's latest snapshot strictly before the new batch."""
    if not bakery_ids:
        return pd.DataFrame(columns=["bakery_id", "product_id"])
    return client.query_df(
        f"""
        select toInt64(a.bakery_id) as bakery_id, toString(a.product_id) as product_id
        from {table} as a final
        inner join (
            select bakery_id, max(valid_from) as latest_valid_from
            from {table} final
            where bakery_id in %(bakery_ids)s
              and valid_from < toDate(%(before_date)s)
            group by bakery_id
        ) latest on a.bakery_id = latest.bakery_id
            and a.valid_from = latest.latest_valid_from
        group by bakery_id, product_id
        """,
        parameters={"bakery_ids": bakery_ids, "before_date": before_date},
    )


def carry_forward_bakeries_without_recent_sales(
    current: pd.DataFrame,
    previous: pd.DataFrame,
    *,
    required_bakery_ids: list[int],
    valid_from: str,
) -> tuple[pd.DataFrame, list[int]]:
    """Carry the prior snapshot only when a required bakery has zero current rows."""
    present = set(current["bakery_id"].astype(int)) if not current.empty else set()
    missing = sorted(set(map(int, required_bakery_ids)) - present)
    if not missing or previous.empty:
        return current, []
    carried = previous[previous["bakery_id"].astype(int).isin(missing)].copy()
    carried_ids = sorted(carried["bakery_id"].astype(int).unique().tolist())
    if carried.empty:
        return current, []
    carried["bakery_id"] = carried["bakery_id"].astype("int64")
    carried["product_id"] = carried["product_id"].astype(str).str.zfill(9)
    carried["valid_from"] = pd.to_datetime(valid_from).date()
    carried["loaded_at"] = pd.Timestamp.now()
    result = pd.concat([current, carried[current.columns]], ignore_index=True)
    result = result.sort_values(["bakery_id", "product_id"]).reset_index(drop=True)
    return result, carried_ids


def add_city_core_for_cold_start_bakeries(
    current: pd.DataFrame,
    bakery_city_map: pd.DataFrame,
    bakeable: pd.DataFrame,
    *,
    required_bakery_ids: list[int],
    valid_from: str,
) -> tuple[pd.DataFrame, list[int]]:
    """Seed never-seen bakeries from the automatic seven-day city core."""
    present = set(current["bakery_id"].astype(int)) if not current.empty else set()
    missing = sorted(set(map(int, required_bakery_ids)) - present)
    if not missing or bakery_city_map.empty or bakeable.empty:
        return current, []
    city_core = bakeable[bakeable["scope"].eq("city")][
        ["city", "product_id"]
    ].drop_duplicates()
    cold_start = bakery_city_map[
        bakery_city_map["bakery_id"].astype(int).isin(missing)
    ].merge(city_core, on="city", how="inner")
    if cold_start.empty:
        return current, []
    cold_start = cold_start[["bakery_id", "product_id"]].drop_duplicates()
    cold_start["bakery_id"] = cold_start["bakery_id"].astype("int64")
    cold_start["product_id"] = cold_start["product_id"].astype(str).str.zfill(9)
    cold_start["valid_from"] = pd.to_datetime(valid_from).date()
    cold_start["loaded_at"] = pd.Timestamp.now()
    cold_start_ids = sorted(
        cold_start["bakery_id"].astype(int).unique().tolist()
    )
    result = pd.concat(
        [current, cold_start[current.columns]], ignore_index=True
    ).drop_duplicates(["bakery_id", "product_id"], keep="last")
    result = result.sort_values(["bakery_id", "product_id"]).reset_index(drop=True)
    return result, cold_start_ids


def build_cold_start_city_core(
    sales: pd.DataFrame,
    *,
    city_threshold: float = 0.8,
    category_patterns: list[str] = DEFAULT_BAKEABLE_CATEGORY_PATTERNS,
) -> pd.DataFrame:
    """Build a stable city core using only bakeries participating in the window."""
    filtered = _apply_category_filter(sales, category_patterns)
    if filtered.empty:
        return pd.DataFrame(columns=["city", "product_id", "scope"])
    participating = (
        filtered.groupby("city", as_index=False)["bakery_id"]
        .nunique()
        .rename(columns={"bakery_id": "participating_bakeries"})
    )
    sold_by = (
        filtered.groupby(["city", "product_id"], as_index=False)["bakery_id"]
        .nunique()
        .rename(columns={"bakery_id": "bakeries_selling"})
        .merge(participating, on="city", how="left")
    )
    sold_by["share"] = (
        sold_by["bakeries_selling"] / sold_by["participating_bakeries"]
    )
    core = sold_by[sold_by["share"].ge(city_threshold)][
        ["city", "product_id"]
    ].copy()
    core["scope"] = "city"
    return core


def build_cold_start_network_core(
    sales: pd.DataFrame,
    *,
    network_threshold: float = 0.8,
    category_patterns: list[str] = DEFAULT_BAKEABLE_CATEGORY_PATTERNS,
) -> pd.DataFrame:
    """Return the common network core for a city with no participating bakery."""
    filtered = _apply_category_filter(sales, category_patterns)
    if filtered.empty:
        return pd.DataFrame(columns=["product_id"])
    participating = filtered["bakery_id"].nunique()
    sold_by = filtered.groupby("product_id", as_index=False)["bakery_id"].nunique()
    sold_by["share"] = sold_by["bakery_id"] / participating
    return sold_by[sold_by["share"].ge(network_threshold)][
        ["product_id"]
    ].copy()


def add_network_core_for_cold_start_bakeries(
    current: pd.DataFrame,
    network_core: pd.DataFrame,
    *,
    required_bakery_ids: list[int],
    valid_from: str,
) -> tuple[pd.DataFrame, list[int]]:
    """Seed a never-seen city from the automatic common network core."""
    present = set(current["bakery_id"].astype(int)) if not current.empty else set()
    missing = sorted(set(map(int, required_bakery_ids)) - present)
    if not missing or network_core.empty:
        return current, []
    additions = pd.MultiIndex.from_product(
        [missing, network_core["product_id"].drop_duplicates().tolist()],
        names=["bakery_id", "product_id"],
    ).to_frame(index=False)
    additions["bakery_id"] = additions["bakery_id"].astype("int64")
    additions["product_id"] = additions["product_id"].astype(str).str.zfill(9)
    additions["valid_from"] = pd.to_datetime(valid_from).date()
    additions["loaded_at"] = pd.Timestamp.now()
    result = pd.concat(
        [current, additions[current.columns]], ignore_index=True
    ).drop_duplicates(["bakery_id", "product_id"], keep="last")
    result = result.sort_values(["bakery_id", "product_id"]).reset_index(drop=True)
    return result, missing


def build_and_insert(
    client,
    *,
    bakery_table: str,
    bakeable_table: str,
    target_table: str,
    valid_from: str,
    override_table: str | None = None,
    dry_run: bool = False,
) -> dict:
    sales = _query_recent_sales(
        client,
        window_days=DEFAULT_WINDOW_DAYS,
        bakery_table=bakery_table,
        sku_day_table="sku_forecast_day_embedded",
        sales_table=SALES_TABLE,
    )
    overrides = (
        load_active_overrides(
            client,
            table=override_table,
            effective_date=valid_from,
        )
        if override_table
        else pd.DataFrame()
    )
    result = build_assortment_from_sales(
        sales,
        valid_from=valid_from,
        overrides=overrides,
    )
    bakery_city_map = _query_bakery_city_map(client, bakery_table=bakery_table)
    required_bakery_ids = bakery_city_map["bakery_id"].astype(int).tolist()
    missing_ids = sorted(
        set(required_bakery_ids) - set(result["bakery_id"].astype(int))
    )
    previous = load_previous_assortment(
        client,
        table=target_table,
        bakery_ids=missing_ids,
        before_date=valid_from,
    )
    result, carried_ids = carry_forward_bakeries_without_recent_sales(
        result,
        previous,
        required_bakery_ids=required_bakery_ids,
        valid_from=valid_from,
    )
    present_ids = set(result["bakery_id"].astype(int))
    cold_start_missing = sorted(set(required_bakery_ids) - present_ids)
    bakeable = build_cold_start_city_core(sales)
    result, cold_start_ids = add_city_core_for_cold_start_bakeries(
        result,
        bakery_city_map,
        bakeable,
        required_bakery_ids=cold_start_missing,
        valid_from=valid_from,
    )
    result, network_cold_start_ids = add_network_core_for_cold_start_bakeries(
        result,
        build_cold_start_network_core(sales),
        required_bakery_ids=required_bakery_ids,
        valid_from=valid_from,
    )

    summary = {
        "bakeries": int(result["bakery_id"].nunique()) if not result.empty else 0,
        "products": int(result["product_id"].nunique()) if not result.empty else 0,
        "rows": len(result),
        "carried_bakeries": carried_ids,
        "cold_start_bakeries": cold_start_ids,
        "network_cold_start_bakeries": network_cold_start_ids,
        "status": "dry_run" if dry_run else "inserted",
    }

    if not dry_run:
        ensure_table(client, target_table)
        insert_to_clickhouse(client, result, target_table=target_table)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build flat (bakery_id, product_id) assortment table."
    )
    parser.add_argument("--env-file", default=DEFAULT_ENV_PATH, type=Path)
    parser.add_argument(
        "--valid-from",
        default=pd.Timestamp.now().date().isoformat(),
        help="valid_from date for inserted rows (ISO format).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    client = create_client(args.env_file)
    suffix = get_table_suffix_from_env_file(args.env_file)
    # The expansion source must not depend on the active forecast run. Using
    # bakery_forecast_day_embedded creates a circular cold-start failure: a
    # bakery absent from one run never receives assortment rows and therefore
    # cannot enter later runs. dim_bakeries is the authoritative network map.
    bakery_tbl = BAKERY_DIMENSION_TABLE
    bakeable_tbl = table_name("bakeable_products", suffix=suffix)
    target_tbl = table_name(TARGET_TABLE_BASE, suffix=suffix)
    override_tbl = table_name(OVERRIDE_TABLE_BASE, suffix=suffix)

    print(f"bakery table    : {bakery_tbl}")
    print(f"bakeable table  : {bakeable_tbl}")
    print(f"target table    : {target_tbl}")
    print(f"valid_from      : {args.valid_from}")

    summary = build_and_insert(
        client,
        bakery_table=bakery_tbl,
        bakeable_table=bakeable_tbl,
        target_table=target_tbl,
        valid_from=args.valid_from,
        override_table=override_tbl,
        dry_run=args.dry_run,
    )

    print(f"bakeries        : {summary['bakeries']}")
    print(f"products        : {summary['products']}")
    print(f"rows            : {summary['rows']}")
    print(f"status          : {summary['status']}")


if __name__ == "__main__":
    main()

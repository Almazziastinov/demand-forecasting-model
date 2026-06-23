"""Build city assortment table from the active ClickHouse forecast run.

Replaces the static OCR / director-file source with a dynamic source:
all SKU positions that appear in the current active production forecast
with forecast_qty > 0.

A (city, product_id) pair is included if it is forecasted for at least one
bakery in that city on any date within the active run horizon.

Usage
-----
# Production (uses .env)
.venv\\Scripts\\python.exe scripts\\build_city_assortment_from_forecast.py

# Dev (uses .env.dev, reads _dev tables)
.venv\\Scripts\\python.exe scripts\\build_city_assortment_from_forecast.py --env-file .env.dev

# Override active run
.venv\\Scripts\\python.exe scripts\\build_city_assortment_from_forecast.py --run-id prod_weatherfix2_uplifted_bakery_norm_uplift_sku_20260623_h14
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipelines.forecast_publish.load_forecast_run import (  # noqa: E402
    create_client,
    load_env_file,
)
from pipelines.forecast_publish.table_names import (  # noqa: E402
    get_table_suffix_from_env_file,
    table_name,
)

DEFAULT_ENV_PATH = ROOT / ".env"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "required_assortment"
DEFAULT_OUTPUT_TABLE_PATH = DEFAULT_OUTPUT_DIR / "assortment_city_products.csv"
DEFAULT_OUTPUT_AUDIT_PATH = DEFAULT_OUTPUT_DIR / "assortment_source_audit.csv"
SOURCE_NAME = "forecast_snapshot"


# ---------------------------------------------------------------------------
# ClickHouse helpers
# ---------------------------------------------------------------------------


def get_active_run_id(client, runs_table: str) -> str:
    """Return run_id of the currently active run, or raise if none found."""
    df = client.query_df(
        f"""
        SELECT run_id
        FROM {runs_table}
        WHERE status = 'active'
        ORDER BY generated_at DESC
        LIMIT 1
        """
    )
    if df.empty:
        raise RuntimeError(
            f"No active run found in {runs_table}. "
            "Activate a run first or pass --run-id explicitly."
        )
    return str(df.iloc[0]["run_id"])


def query_forecast_assortment(
    client,
    *,
    run_id: str,
    sku_table: str,
    bakery_table: str,
) -> pd.DataFrame:
    """Return DISTINCT (city, product_id, product_name, category_name) for the run.

    Only rows with forecast_qty > 0 are considered.
    Bakeries without a city value are excluded.
    Products whose name starts with inactive markers (я_не_исп, я не исп) are excluded.
    """
    return client.query_df(
        f"""
        SELECT
            b.city                      AS city,
            toString(s.product_id)      AS product_id,
            any(s.product_name)         AS product_name,
            any(s.category_name)        AS category_name
        FROM {sku_table} AS s
        INNER JOIN (
            SELECT DISTINCT bakery_id, city
            FROM {bakery_table}
            WHERE run_id = {{run_id:String}}
              AND city IS NOT NULL
              AND city != ''
        ) AS b ON s.bakery_id = b.bakery_id
        WHERE s.run_id = {{run_id:String}}
          AND s.forecast_qty > 0
          AND NOT startsWith(s.product_name, 'я_не_исп')
          AND NOT startsWith(s.product_name, 'я не исп')
        GROUP BY b.city, s.product_id
        ORDER BY b.city, s.product_id
        """,
        parameters={"run_id": run_id},
    )


# ---------------------------------------------------------------------------
# Output builders
# ---------------------------------------------------------------------------


def build_assortment_table(
    df: pd.DataFrame,
    *,
    run_id: str,
    valid_from: str,
) -> pd.DataFrame:
    """Convert raw query result into the assortment_city_products schema."""
    loaded_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    table = df[["city", "product_id", "product_name", "category_name"]].copy()
    # product_name and category_name are non-nullable String in ClickHouse schema.
    table["product_name"] = table["product_name"].fillna("")
    table["category_name"] = table["category_name"].fillna("")
    table["is_required"] = 1
    table["is_top"] = 0
    table["top_rank"] = pd.NA
    table["source"] = SOURCE_NAME
    table["source_priority"] = 1
    table["source_file"] = run_id
    table["source_scope"] = table["city"]
    table["valid_from"] = pd.to_datetime(valid_from).date().isoformat()
    table["valid_to"] = pd.NA
    table["is_active"] = 1
    table["loaded_at"] = loaded_at
    table["comment"] = ""
    return table[[
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
    ]]


def build_audit_table(
    df: pd.DataFrame,
    *,
    run_id: str,
) -> pd.DataFrame:
    """Build a flat audit table (mirrors assortment_source_audit schema)."""
    loaded_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    audit = df[["city", "product_id", "product_name", "category_name"]].copy()
    audit["source"] = SOURCE_NAME
    audit["source_file"] = run_id
    audit["source_scope"] = audit["city"]
    audit["raw_product_name"] = audit["product_name"]
    audit["raw_category_name"] = audit["category_name"]
    audit["matched_product_id"] = audit["product_id"]
    audit["matched_product_name"] = audit["product_name"]
    audit["matched_category_name"] = audit["category_name"]
    audit["match_status"] = "matched"
    audit["issue"] = ""
    audit["is_required"] = 1
    audit["is_top"] = 0
    audit["top_rank"] = pd.NA
    audit["loaded_at"] = loaded_at
    return audit[[
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
    ]]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build city assortment CSV from active ClickHouse forecast run."
    )
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_PATH),
        type=Path,
        help="Path to .env file (default: .env for prod, use .env.dev for dev tables).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Forecast run_id to use. Defaults to the currently active run.",
    )
    parser.add_argument(
        "--valid-from",
        default=None,
        help="valid_from date for assortment rows (ISO format). Defaults to today.",
    )
    parser.add_argument(
        "--output-table-path",
        default=DEFAULT_OUTPUT_TABLE_PATH,
        type=Path,
        help="Output path for assortment_city_products CSV.",
    )
    parser.add_argument(
        "--output-audit-path",
        default=DEFAULT_OUTPUT_AUDIT_PATH,
        type=Path,
        help="Output path for assortment_source_audit CSV.",
    )
    args = parser.parse_args()

    env_path = Path(args.env_file)
    table_suffix = get_table_suffix_from_env_file(env_path)
    client = create_client(env_path)

    runs_table = table_name("forecast_runs_embedded", table_suffix)
    sku_table = table_name("sku_forecast_day_embedded", table_suffix)
    bakery_table = table_name("bakery_forecast_day_embedded", table_suffix)

    run_id = args.run_id or get_active_run_id(client, runs_table)
    valid_from = args.valid_from or pd.Timestamp.now().date().isoformat()

    print(f"run_id        : {run_id}")
    print(f"sku_table     : {sku_table}")
    print(f"bakery_table  : {bakery_table}")
    print(f"valid_from    : {valid_from}")

    df = query_forecast_assortment(
        client,
        run_id=run_id,
        sku_table=sku_table,
        bakery_table=bakery_table,
    )

    if df.empty:
        print("WARNING: no rows returned — check that the run_id exists and has forecast_qty > 0.")
        return

    print(f"\nforecast positions : {len(df)}")
    print(f"cities             : {df['city'].nunique()}")
    print(f"unique products    : {df['product_id'].nunique()}")

    # Show category breakdown so the user can verify category patterns
    if "category_name" in df.columns:
        print("\ncategory breakdown (unique products per category_name):")
        breakdown = (
            df.groupby(df["category_name"].fillna("<null>"))["product_id"]
            .nunique()
            .sort_values(ascending=False)
        )
        print(breakdown.to_string())

    table = build_assortment_table(df, run_id=run_id, valid_from=valid_from)
    audit = build_audit_table(df, run_id=run_id)

    args.output_table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_table_path, index=False, encoding="utf-8-sig")
    audit.to_csv(args.output_audit_path, index=False, encoding="utf-8-sig")

    print(f"\nassortment rows : {len(table)}")
    print(f"table  : {args.output_table_path}")
    print(f"audit  : {args.output_audit_path}")


if __name__ == "__main__":
    main()

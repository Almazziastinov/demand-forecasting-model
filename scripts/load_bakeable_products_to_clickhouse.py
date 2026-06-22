"""Load the bakeable-products allowlist CSV into ClickHouse."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipelines.forecast_publish.load_forecast_run import (  # noqa: E402
    create_client,
)
from scripts.load_city_assortment_to_clickhouse import (  # noqa: E402
    normalize_nullable_columns,
    normalize_string_columns,
    quote_sql_string,
    wait_for_mutations,
)


DEFAULT_ENV_PATH = ROOT / ".env"
DEFAULT_SCHEMA_PATH = (
    ROOT / "apps" / "forecast_embedded" / "sql" / "assortment_schema.sql"
)
DEFAULT_CSV = ROOT / "reports" / "required_assortment" / "bakeable_products.csv"
DEFAULT_TABLE = "bakeable_products"
SCHEMA_BASE_TABLE = "bakeable_products"


def apply_bakeable_schema(client, schema_path: Path, table: str) -> None:
    """Create the bakeable_products table under the requested (possibly suffixed)
    name. The shared schema file is not registered in EMBEDDED_TABLES, so the
    generic suffix rewriter does not touch it; rewrite the create statement here.
    """
    sql_text = schema_path.read_text(encoding="utf-8")
    for statement in sql_text.split(";"):
        stmt = statement.strip()
        if not stmt or SCHEMA_BASE_TABLE not in stmt:
            continue
        if re.search(rf"create table[^(]*\b{SCHEMA_BASE_TABLE}\b", stmt, re.IGNORECASE):
            stmt = re.sub(rf"\b{SCHEMA_BASE_TABLE}\b", table, stmt, count=1)
            client.command(stmt)


def load_bakeable_products(
    *,
    env_path: Path,
    schema_path: Path,
    csv_path: Path,
    table: str,
    apply_schema: bool,
    replace_current: bool,
) -> None:
    client = create_client(env_path)
    if apply_schema:
        apply_bakeable_schema(client, schema_path, table)

    df = pd.read_csv(csv_path, dtype={"product_id": str})
    df["valid_from"] = pd.to_datetime(df["valid_from"]).dt.date
    valid_to = pd.to_datetime(df["valid_to"], errors="coerce").dt.date
    df["valid_to"] = valid_to.where(valid_to.notna(), None)
    df["loaded_at"] = pd.to_datetime(df["loaded_at"])
    df = normalize_nullable_columns(df, ["valid_to"])
    df = normalize_string_columns(
        df,
        [
            "city",
            "product_id",
            "product_name",
            "category_name",
            "source",
            "source_file",
            "comment",
        ],
    )
    df["comment"] = df["comment"].fillna("")

    if replace_current and not df.empty:
        sources = sorted(set(df["source"].astype(str)))
        valid_from_values = sorted(set(df["valid_from"].astype(str)))
        source_sql = ", ".join(f"'{quote_sql_string(source)}'" for source in sources)
        valid_from_sql = ", ".join(
            f"toDate('{quote_sql_string(valid_from)}')"
            for valid_from in valid_from_values
        )
        client.command(
            f"""
            alter table {table}
            delete where source in ({source_sql})
              and valid_from in ({valid_from_sql})
            """
        )
        wait_for_mutations(client, [table])

    client.insert_df(table, df)

    table_count = client.query(f"select count() from {table}").result_rows[0][0]
    print("inserted bakeable rows:", len(df))
    print("current bakeable table rows:", table_count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-path", default=DEFAULT_ENV_PATH, type=Path)
    parser.add_argument("--schema-path", default=DEFAULT_SCHEMA_PATH, type=Path)
    parser.add_argument("--csv", default=DEFAULT_CSV, type=Path)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--apply-schema", action="store_true")
    parser.add_argument("--replace-current", action="store_true")
    args = parser.parse_args()

    load_bakeable_products(
        env_path=args.env_path,
        schema_path=args.schema_path,
        csv_path=args.csv,
        table=args.table,
        apply_schema=args.apply_schema,
        replace_current=args.replace_current,
    )


if __name__ == "__main__":
    main()

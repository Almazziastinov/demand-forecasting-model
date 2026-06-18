"""Load city assortment CSV outputs into ClickHouse."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipelines.forecast_publish.load_forecast_run import (  # noqa: E402
    create_client,
    load_schema,
)


DEFAULT_ENV_PATH = ROOT / ".env"
DEFAULT_SCHEMA_PATH = (
    ROOT / "apps" / "forecast_embedded" / "sql" / "assortment_schema.sql"
)
DEFAULT_TABLE_CSV = (
    ROOT / "reports" / "required_assortment" / "assortment_city_products.csv"
)
DEFAULT_AUDIT_CSV = (
    ROOT / "reports" / "required_assortment" / "assortment_source_audit.csv"
)
DEFAULT_ASSORTMENT_TABLE = "assortment_city_products"
DEFAULT_AUDIT_TABLE = "assortment_source_audit"


def quote_sql_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "''")


def normalize_nullable_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    work = df.copy()
    for column in columns:
        if column in work.columns:
            work[column] = work[column].where(work[column].notna(), None)
    return work


def normalize_string_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    work = df.copy()
    for column in columns:
        if column in work.columns:
            work[column] = work[column].where(work[column].notna(), None)
            work[column] = work[column].map(
                lambda value: None if value is None else str(value)
            )
    return work


def wait_for_mutations(
    client,
    tables: list[str],
    *,
    timeout_seconds: int = 120,
    poll_seconds: float = 2.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    table_sql = ", ".join(f"'{quote_sql_string(table)}'" for table in tables)
    while True:
        pending = client.query(
            f"""
            select count()
            from system.mutations
            where database = currentDatabase()
              and table in ({table_sql})
              and is_done = 0
            """
        ).result_rows[0][0]
        if int(pending) == 0:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for ClickHouse mutations: {tables}")
        time.sleep(poll_seconds)


def load_assortment(
    *,
    env_path: Path,
    schema_path: Path,
    table_csv: Path,
    audit_csv: Path,
    assortment_table: str,
    audit_table: str,
    apply_schema: bool,
    replace_current: bool,
) -> None:
    client = create_client(env_path)
    if apply_schema:
        load_schema(client, schema_path)

    table_df = pd.read_csv(table_csv, dtype={"product_id": str})
    audit_df = pd.read_csv(audit_csv, dtype={"matched_product_id": str})
    table_df["valid_from"] = pd.to_datetime(table_df["valid_from"]).dt.date
    valid_to = pd.to_datetime(table_df["valid_to"], errors="coerce").dt.date
    table_df["valid_to"] = valid_to.where(valid_to.notna(), None)
    table_df["loaded_at"] = pd.to_datetime(table_df["loaded_at"])
    audit_df["loaded_at"] = pd.to_datetime(audit_df["loaded_at"])
    table_df = normalize_nullable_columns(table_df, ["top_rank", "valid_to"])
    table_df = normalize_string_columns(
        table_df,
        [
            "city",
            "product_id",
            "product_name",
            "category_name",
            "source",
            "source_file",
            "source_scope",
            "comment",
        ],
    )
    table_df["comment"] = table_df["comment"].fillna("")
    audit_df = normalize_nullable_columns(
        audit_df,
        [
            "matched_product_id",
            "matched_product_name",
            "matched_category_name",
            "top_rank",
        ],
    )
    audit_df = normalize_string_columns(
        audit_df,
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
        ],
    )
    audit_df["issue"] = audit_df["issue"].fillna("")

    if replace_current and not table_df.empty:
        sources = sorted(set(table_df["source"].astype(str)))
        valid_from_values = sorted(set(table_df["valid_from"].astype(str)))
        source_sql = ", ".join(f"'{quote_sql_string(source)}'" for source in sources)
        valid_from_sql = ", ".join(
            f"toDate('{quote_sql_string(valid_from)}')"
            for valid_from in valid_from_values
        )
        client.command(
            f"""
            alter table {assortment_table}
            delete where source in ({source_sql})
              and valid_from in ({valid_from_sql})
            """
        )
        source_files = sorted(set(table_df["source_file"].astype(str)))
        source_file_sql = ", ".join(
            f"'{quote_sql_string(source_file)}'" for source_file in source_files
        )
        client.command(
            f"""
            alter table {audit_table}
            delete where source in ({source_sql})
              and source_file in ({source_file_sql})
            """
        )
        wait_for_mutations(client, [assortment_table, audit_table])

    client.insert_df(assortment_table, table_df)
    client.insert_df(audit_table, audit_df)

    table_count = client.query(
        f"select count() from {assortment_table}"
    ).result_rows[0][0]
    audit_count = client.query(f"select count() from {audit_table}").result_rows[0][0]
    print("inserted assortment rows:", len(table_df))
    print("inserted audit rows:", len(audit_df))
    print("current assortment table rows:", table_count)
    print("current audit table rows:", audit_count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-path", default=DEFAULT_ENV_PATH, type=Path)
    parser.add_argument("--schema-path", default=DEFAULT_SCHEMA_PATH, type=Path)
    parser.add_argument("--table-csv", default=DEFAULT_TABLE_CSV, type=Path)
    parser.add_argument("--audit-csv", default=DEFAULT_AUDIT_CSV, type=Path)
    parser.add_argument("--assortment-table", default=DEFAULT_ASSORTMENT_TABLE)
    parser.add_argument("--audit-table", default=DEFAULT_AUDIT_TABLE)
    parser.add_argument("--apply-schema", action="store_true")
    parser.add_argument("--replace-current", action="store_true")
    args = parser.parse_args()

    load_assortment(
        env_path=args.env_path,
        schema_path=args.schema_path,
        table_csv=args.table_csv,
        audit_csv=args.audit_csv,
        assortment_table=args.assortment_table,
        audit_table=args.audit_table,
        apply_schema=args.apply_schema,
        replace_current=args.replace_current,
    )


if __name__ == "__main__":
    main()

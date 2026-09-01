"""Append-only emergency overrides for the automatic assortment."""

from __future__ import annotations

from uuid import UUID, uuid4

import pandas as pd

TABLE_BASE = "assortment_emergency_overrides"

CREATE_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
    override_id UUID,
    bakery_id Int64,
    product_id String,
    action LowCardinality(String),
    valid_from Date,
    valid_to Date,
    reason String,
    created_by String,
    created_at DateTime64(3),
    is_cancelled UInt8 DEFAULT 0
)
ENGINE = ReplacingMergeTree(created_at)
ORDER BY (override_id)
"""


def ensure_table(client, table: str) -> None:
    client.command(CREATE_DDL.format(table=table))


def build_override_row(
    *,
    bakery_id: int,
    product_id: int | str,
    action: str,
    valid_from: str,
    valid_to: str,
    reason: str,
    created_by: str,
    override_id: UUID | None = None,
) -> pd.DataFrame:
    if action not in {"force_include", "force_exclude"}:
        raise ValueError("action must be force_include or force_exclude")
    start = pd.Timestamp(valid_from).normalize()
    end = pd.Timestamp(valid_to).normalize()
    if end < start:
        raise ValueError("valid_to must be on or after valid_from")
    if not reason.strip() or not created_by.strip():
        raise ValueError("reason and created_by are required")
    return pd.DataFrame(
        [
            {
                "override_id": override_id or uuid4(),
                "bakery_id": int(bakery_id),
                "product_id": str(product_id).zfill(9),
                "action": action,
                "valid_from": start.date(),
                "valid_to": end.date(),
                "reason": reason.strip(),
                "created_by": created_by.strip(),
                "created_at": pd.Timestamp.now(),
                "is_cancelled": 0,
            }
        ]
    )


def append_override(client, *, table: str, row: pd.DataFrame) -> str:
    ensure_table(client, table)
    client.insert_df(table, row)
    return str(row["override_id"].iat[0])


def load_active_overrides(
    client,
    *,
    table: str,
    effective_date: str,
) -> pd.DataFrame:
    ensure_table(client, table)
    return client.query_df(
        f"""
        SELECT bakery_id, product_id, action, valid_from, valid_to,
               reason, created_by
        FROM {table} FINAL
        WHERE is_cancelled = 0
          AND valid_from <= toDate(%(effective_date)s)
          AND valid_to >= toDate(%(effective_date)s)
        """,
        parameters={"effective_date": effective_date},
    )

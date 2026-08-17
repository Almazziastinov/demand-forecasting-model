"""Service for managing pilot bakery scope via ClickHouse.

Stores each add/exclude action as an immutable event in
Svezhar.pilot_scope_events.  Current state = latest event per bakery.
Bakeries with no events fall back to the hardcoded PILOT_IDS seed.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import pandas as pd

# Hardcoded seed list (expanded_pilot_38) — used only when the events table
# has no record for a given bakery so the UI reflects the current state on
# first launch without requiring a migration step.
_SEED_PILOT_IDS: frozenset[int] = frozenset({
    1, 3, 12, 13, 14, 20, 21, 22, 26, 27, 28, 39, 41, 56, 57, 66, 67, 69,
    80, 89, 99, 107, 113, 125, 149, 153, 155, 160, 191, 221, 222, 229, 230,
    246, 257, 260, 268, 270,
})

_EVENTS_TABLE = "Svezhar.pilot_scope_events"
_DIM_STORES_TABLE = "Svezhar.dim_stores"

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS Svezhar.pilot_scope_events (
    event_id   String,
    scope_name LowCardinality(String),
    bakery_id  Int64,
    action     LowCardinality(String),
    changed_at DateTime64(3, 'UTC'),
    changed_by String,
    reason     String
) ENGINE = MergeTree()
ORDER BY (scope_name, bakery_id, changed_at)
"""


def _to_py(value: Any) -> Any:
    """Convert numpy/pandas scalars to plain Python types."""
    if value != value:  # NaN
        return None
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


class PilotConfigService:
    def __init__(self, client) -> None:
        self._client = client
        self._ensure_table()

    # ------------------------------------------------------------------ setup

    def _ensure_table(self) -> None:
        self._client.command(_CREATE_SQL)

    # ------------------------------------------------------------------ reads

    def get_all_bakeries(self, scope_name: str) -> list[dict]:
        """All dim_stores bakeries with current pilot status and last-event info."""
        stores_df = self._client.query_df(
            f"""
            SELECT toInt64(s.bakery_id)    AS bakery_id,
                   any(b.bakery_name)      AS bakery_name,
                   any(s.city)             AS city,
                   any(s.partner)          AS partner,
                   any(s.regional_director) AS regional_director
            FROM {_DIM_STORES_TABLE} s
            LEFT JOIN Svezhar.dim_bakeries b ON toInt64(b.bakery_id) = toInt64(s.bakery_id)
            GROUP BY s.bakery_id
            ORDER BY s.bakery_id
            """
        )
        if stores_df.empty:
            return []

        latest_df = self._client.query_df(
            f"""
            SELECT bakery_id, action, changed_by, changed_at, reason
            FROM {_EVENTS_TABLE}
            WHERE scope_name = %(scope_name)s
              AND (scope_name, bakery_id, changed_at) IN (
                  SELECT scope_name, bakery_id, max(changed_at)
                  FROM {_EVENTS_TABLE}
                  WHERE scope_name = %(scope_name)s
                  GROUP BY scope_name, bakery_id
              )
            """,
            parameters={"scope_name": scope_name},
        )

        if not latest_df.empty:
            stores_df = stores_df.merge(latest_df, on="bakery_id", how="left")
        else:
            stores_df["action"] = None
            stores_df["changed_by"] = None
            stores_df["changed_at"] = None
            stores_df["reason"] = None

        rows: list[dict] = []
        for record in stores_df.to_dict("records"):
            bid = int(record["bakery_id"])
            has_event = pd.notna(record.get("action"))
            if has_event:
                is_active = record["action"] == "add"
                status_source = "event"
            else:
                is_active = bid in _SEED_PILOT_IDS
                status_source = "seed"
            rows.append({
                "bakery_id": bid,
                "bakery_name": _to_py(record.get("bakery_name")),
                "city": _to_py(record.get("city")),
                "partner": _to_py(record.get("partner")),
                "regional_director": _to_py(record.get("regional_director")),
                "is_active": is_active,
                "status_source": status_source,
                "last_action": _to_py(record.get("action")),
                "last_changed_by": _to_py(record.get("changed_by")),
                "last_changed_at": _to_py(record.get("changed_at")),
                "last_reason": _to_py(record.get("reason")),
            })

        # Sort: active first, then by bakery_id
        rows.sort(key=lambda r: (0 if r["is_active"] else 1, r["bakery_id"]))
        return rows

    def get_history(self, scope_name: str, limit: int = 200) -> list[dict]:
        """Full event log newest-first, enriched with bakery_name."""
        df = self._client.query_df(
            f"""
            SELECT e.event_id,
                   e.bakery_id,
                   any(s.bakery_name) AS bakery_name,
                   any(s.city)        AS city,
                   e.action,
                   e.changed_at,
                   e.changed_by,
                   e.reason
            FROM {_EVENTS_TABLE} e
            LEFT JOIN (
                SELECT toInt64(s.bakery_id) AS bakery_id,
                       any(b.bakery_name)   AS bakery_name,
                       any(s.city)          AS city
                FROM {_DIM_STORES_TABLE} s
                LEFT JOIN Svezhar.dim_bakeries b ON toInt64(b.bakery_id) = toInt64(s.bakery_id)
                GROUP BY s.bakery_id
            ) s ON s.bakery_id = e.bakery_id
            WHERE e.scope_name = %(scope_name)s
            GROUP BY e.event_id, e.bakery_id, e.action, e.changed_at, e.changed_by, e.reason
            ORDER BY e.changed_at DESC
            LIMIT %(limit)s
            """,
            parameters={"scope_name": scope_name, "limit": limit},
        )
        if df.empty:
            return []
        rows = []
        for record in df.to_dict("records"):
            rows.append({k: _to_py(v) for k, v in record.items()})
        return rows

    def get_summary(self, scope_name: str) -> dict:
        """Quick counts: total bakeries, active in pilot, excluded."""
        all_bakeries = self.get_all_bakeries(scope_name)
        active = sum(1 for b in all_bakeries if b["is_active"])
        return {
            "scope_name": scope_name,
            "total": len(all_bakeries),
            "active": active,
            "inactive": len(all_bakeries) - active,
        }

    # ----------------------------------------------------------------- writes

    def add_bakery(self, scope_name: str, bakery_id: int, changed_by: str, reason: str) -> None:
        self._write_event(scope_name, bakery_id, "add", changed_by, reason)

    def exclude_bakery(self, scope_name: str, bakery_id: int, changed_by: str, reason: str) -> None:
        self._write_event(scope_name, bakery_id, "exclude", changed_by, reason)

    def _write_event(
        self,
        scope_name: str,
        bakery_id: int,
        action: str,
        changed_by: str,
        reason: str,
    ) -> None:
        self._client.insert(
            _EVENTS_TABLE,
            [[
                str(uuid.uuid4()),
                scope_name,
                int(bakery_id),
                action,
                datetime.now(tz=timezone.utc),
                changed_by or "unknown",
                reason or "",
            ]],
            column_names=["event_id", "scope_name", "bakery_id", "action",
                          "changed_at", "changed_by", "reason"],
        )

"""Thin ClickHouse helpers shared across this package.

Only wraps `app.db`/`app.table_names` (the shared infra this package is
allowed to depend on per `README.md`) plus a DataFrame -> list[dict]
normalizer used by every query function in this package.
"""

from __future__ import annotations

from app.db import get_client as get_client
from app.table_names import table_name as table_name


def records(df) -> list[dict]:
    rows = []
    for record in df.to_dict("records"):
        normalized = {}
        for key, value in record.items():
            if value != value:  # NaN
                normalized[key] = None
            elif hasattr(value, "isoformat"):
                normalized[key] = value.isoformat()
            else:
                normalized[key] = value
        rows.append(normalized)
    return rows

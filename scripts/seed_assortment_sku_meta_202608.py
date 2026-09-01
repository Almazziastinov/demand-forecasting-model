"""Seed confirmed baking metadata for the August 2026 assortment update.

Dry-run is the default. Use ``--apply`` only from the controlled production
writer after reviewing the displayed rows.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
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


CONFIRMED_SKUS = (
    {
        "product_id": "000011575",
        "product_name": "Кексовый с манго",
        "dough_group": "Тесто Песочка",
        "kratnost": 1,
        "comment": (
            "confirmed assortment update; category=Пирог сладкий; "
            "max_per_tray=4"
        ),
    },
    {
        "product_id": "000011615",
        "product_name": "Плетенка кленовая",
        "dough_group": "Тесто сдобное на заварке НОВОЕ",
        "kratnost": 10,
        "comment": (
            "confirmed assortment update; category=Выпечка сладкая; "
            "max_per_tray=10"
        ),
    },
    {
        "product_id": "000011616",
        "product_name": "Плетенка с черникой",
        "dough_group": "Тесто сдобное на заварке НОВОЕ",
        "kratnost": 10,
        "comment": (
            "confirmed assortment update; category=Выпечка сладкая; "
            "max_per_tray=10"
        ),
    },
    {
        "product_id": "000011617",
        "product_name": "Плетенка с земляникой",
        "dough_group": "Тесто сдобное на заварке НОВОЕ",
        "kratnost": 10,
        "comment": (
            "confirmed assortment update; category=Выпечка сладкая; "
            "max_per_tray=10"
        ),
    },
)


def build_rows(
    valid_from: date,
    *,
    loaded_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    timestamp = loaded_at or pd.Timestamp.now()
    rows = []
    for sku in CONFIRMED_SKUS:
        rows.append(
            {
                **sku,
                "dough_group_source": "business_confirmed",
                "is_two_day": 0,
                "station": "Пекарь",
                "is_on_demand": 0,
                "scope": "base",
                "bakery_id": None,
                "valid_from": valid_from,
                "is_active": 1,
                "loaded_at": timestamp,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--valid-from", type=date.fromisoformat, default=date.today())
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    rows = build_rows(args.valid_from)
    print(rows.to_string(index=False))
    if not args.apply:
        print("[dry-run] No rows written. Pass --apply to insert after review.")
        return

    client = create_client(args.env_file)
    suffix = get_table_suffix_from_env_file(args.env_file)
    target = table_name("baking_sku_meta", suffix=suffix)
    client.insert_df(target, rows)
    print(f"Inserted {len(rows)} rows into {target}.")


if __name__ == "__main__":
    main()

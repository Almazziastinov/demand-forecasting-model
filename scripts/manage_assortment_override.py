"""Create a temporary, audited emergency assortment override."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipelines.forecast_publish.assortment_override_store import (
    TABLE_BASE,
    append_override,
    build_override_row,
)
from pipelines.forecast_publish.load_forecast_run import create_client
from pipelines.forecast_publish.table_names import (
    get_table_suffix_from_env_file,
    table_name,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--bakery-id", type=int, required=True)
    parser.add_argument("--product-id", required=True)
    parser.add_argument(
        "--action",
        choices=["force_include", "force_exclude"],
        required=True,
    )
    parser.add_argument("--valid-from", required=True)
    parser.add_argument("--valid-to", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--created-by", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    row = build_override_row(
        bakery_id=args.bakery_id,
        product_id=args.product_id,
        action=args.action,
        valid_from=args.valid_from,
        valid_to=args.valid_to,
        reason=args.reason,
        created_by=args.created_by,
    )
    print(row.drop(columns="override_id").to_string(index=False))
    if not args.apply:
        print("DRY RUN: pass --apply to append the override")
        return
    client = create_client(args.env_file)
    suffix = get_table_suffix_from_env_file(args.env_file)
    override_id = append_override(
        client,
        table=table_name(TABLE_BASE, suffix=suffix),
        row=row,
    )
    print(f"APPENDED override_id={override_id}")


if __name__ == "__main__":
    main()

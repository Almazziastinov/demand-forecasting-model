"""Synchronize baking SKU metadata from the reviewed production-plan template.

The command is dry-run by default. It updates/inserts template-owned base rows
without deactivating newer business-confirmed SKUs that are absent from the
template. Product-name aliases are explicit and auditable; fuzzy matches are
never written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "apps"))

from baking_plan.templates import normalize_name, parse_comments_sheet  # noqa: E402
from pipelines.forecast_publish.load_forecast_run import (  # noqa: E402
    DEFAULT_ENV_PATH,
    create_client,
)
from pipelines.forecast_publish.table_names import (  # noqa: E402
    get_table_suffix_from_env_file,
    table_name,
)


PRODUCT_ID_ALIASES = {
    normalize_name("Пирог с киви"): 11613,
    normalize_name("ЖарПицца с салями"): 11251,
    normalize_name("Жар ролл с ветчиной"): 11567,
    normalize_name("Жар ролл с крабом"): 11568,
    normalize_name("Роллы Вулкан в тубусе"): 11566,
    normalize_name("Роллы Филадельфия в тубусе"): 11565,
    normalize_name("Кексовый манго"): 11575,
    normalize_name("Пицца Маргарита П (целая)"): 10625,
    normalize_name("Пицца с салями П"): 10628,
    normalize_name("Пицца с салями кусок"): 5106,
    normalize_name("Пицца Мясная"): 10627,
}

IGNORED_TEMPLATE_ROWS = {
    normalize_name("Основа чиабатта покупная"): "group heading, not an SKU",
}

ALLOWED_UNRESOLVED_ROWS = {
    normalize_name("Мексиканский ролл"): "not present in dim_products or active forecast",
}


def build_sync_rows(
    template_path: Path,
    products: pd.DataFrame,
    *,
    valid_from: pd.Timestamp,
    loaded_at: pd.Timestamp,
) -> tuple[pd.DataFrame, list[str]]:
    """Build deterministic base-scope rows and return allowed unresolved names."""
    workbook = openpyxl.load_workbook(template_path, data_only=True)
    template_meta = parse_comments_sheet(workbook)

    product_rows = products.copy()
    product_rows["product_id"] = pd.to_numeric(
        product_rows["product_id"], errors="raise"
    ).astype("int64")
    by_normalized_name: dict[str, list[dict]] = {}
    for row in product_rows.to_dict("records"):
        by_normalized_name.setdefault(
            normalize_name(str(row["product_name"])), []
        ).append(row)

    by_id = {
        int(row["product_id"]): row for row in product_rows.to_dict("records")
    }
    output = []
    unresolved = []
    for normalized_name, meta in template_meta.items():
        if normalized_name in IGNORED_TEMPLATE_ROWS:
            continue

        alias_id = PRODUCT_ID_ALIASES.get(normalized_name)
        if alias_id is not None:
            product = by_id.get(alias_id)
            if product is None:
                raise RuntimeError(
                    f"Configured alias product_id={alias_id} is absent: {meta.sku_name}"
                )
        else:
            candidates = by_normalized_name.get(normalized_name, [])
            if len(candidates) > 1:
                raise RuntimeError(
                    f"Ambiguous normalized product name: {meta.sku_name}"
                )
            product = candidates[0] if candidates else None

        if product is None:
            if normalized_name not in ALLOWED_UNRESOLVED_ROWS:
                raise RuntimeError(f"Unresolved template SKU: {meta.sku_name}")
            unresolved.append(meta.sku_name)
            continue

        output.append(
            {
                "product_id": f"{int(product['product_id']):09d}",
                "product_name": str(product["product_name"]),
                "dough_group": meta.dough_group,
                "dough_group_source": "reviewed_template_20260804",
                "kratnost": int(meta.kratnost),
                "is_two_day": int(meta.is_two_day),
                "station": meta.station or "",
                "is_on_demand": int(meta.is_on_demand),
                "scope": "base",
                "bakery_id": pd.NA,
                "valid_from": valid_from.date(),
                "is_active": 1,
                "loaded_at": loaded_at,
                "comment": (
                    "synchronized from Шаблон плана выпекания для ИИ (1).xlsx; "
                    "reviewed 2026-08-31"
                ),
            }
        )

    rows = pd.DataFrame(output).sort_values("product_id").reset_index(drop=True)
    if rows["product_id"].duplicated().any():
        duplicates = rows.loc[rows["product_id"].duplicated(False), "product_id"]
        raise RuntimeError(f"Duplicate product ids after matching: {duplicates.tolist()}")
    return rows, unresolved


def build_deactivation_rows(
    current: pd.DataFrame,
    *,
    replacement_valid_from: pd.Timestamp,
    loaded_at: pd.Timestamp,
) -> pd.DataFrame:
    """Close older active base versions superseded by the reviewed template."""
    if current.empty:
        return current.copy()
    rows = current.copy()
    rows["valid_from"] = pd.to_datetime(rows["valid_from"], errors="raise")
    rows = rows[
        rows["valid_from"].dt.date < replacement_valid_from.date()
    ].copy()
    if rows.empty:
        return rows
    rows["valid_from"] = rows["valid_from"].dt.date
    rows["is_active"] = 0
    rows["loaded_at"] = loaded_at
    rows["comment"] = (
        "superseded by Шаблон плана выпекания для ИИ (1).xlsx on 2026-08-31"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--template-path", type=Path, required=True)
    parser.add_argument("--valid-from", type=pd.Timestamp, default=pd.Timestamp.now())
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    client = create_client(args.env_file)
    products = client.query_df(
        "select toInt64OrZero(toString(product_id)) as product_id, "
        "any(product_name) as product_name from dim_products group by product_id"
    )
    loaded_at = pd.Timestamp.now()
    rows, unresolved = build_sync_rows(
        args.template_path,
        products,
        valid_from=args.valid_from,
        loaded_at=loaded_at,
    )
    print(rows.to_string(index=False))
    print(f"rows={len(rows)} unresolved={unresolved}")
    if not args.apply:
        print("[dry-run] No rows written. Pass --apply after reviewing the output.")
        return

    suffix = get_table_suffix_from_env_file(args.env_file)
    target = table_name("baking_sku_meta", suffix=suffix)
    current = client.query_df(
        f"select * from {target} final "
        "where is_active=1 and scope='base' and product_id in %(product_ids)s",
        parameters={"product_ids": rows["product_id"].tolist()},
    )
    deactivations = build_deactivation_rows(
        current,
        replacement_valid_from=args.valid_from,
        loaded_at=loaded_at,
    )
    if not deactivations.empty:
        client.insert_df(target, deactivations)
        print(f"Deactivated {len(deactivations)} superseded base rows in {target}.")
    client.insert_df(target, rows)
    print(f"Inserted {len(rows)} reviewed rows into {target}.")


if __name__ == "__main__":
    main()

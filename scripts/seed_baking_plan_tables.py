"""One-time seed for the new baking-plan ClickHouse tables.

Source: the old baking-plan Excel ("комментарии" sheet), copied to
`apps/baking_plan/assets/template.xlsx`. This script is a bridge, not a
recurring pipeline step — once `baking_sku_meta` / `baking_dough_group_mapping`
are populated and reviewed, new products should flow in via
`baking_dough_group_mapping` (material -> dough group) joined against
`dim_recipes`, not by re-running this script.

What it does
------------
1. Parse "комментарии" -> {sku_name: SkuMeta} (station, dough_group text,
   kratnost, is_two_day, is_on_demand).
2. Match each SKU to a `dim_products.product_id` by exact normalized name.
   Unmatched rows are printed and skipped (need manual product_id lookup).
3. For matched products, look up `dim_recipes` materials (is_deleted='Нет')
   whose name contains "Тест" (dough materials). Bootstrap
   `baking_dough_group_mapping`: material_id -> the Excel dough_group label
   used by product(s) on that material. Conflicting labels for the same
   material are reported and skipped (not written) rather than guessed.
4. Write `baking_sku_meta`: dough_group comes from the mapping when the
   product's material resolved cleanly (source='recipe_derived'), otherwise
   falls back to the Excel text directly (source='manual_override').
5. Seed `baking_capacity_config` (one default row: 2 bakers, 2 ovens, 6 trays
   per oven batch, 30 bake minutes) and `baking_category_molding_minutes`
   (Пироги сытные/сладкие = 4 min/unit, default = 1 min/unit) from the
   constants confirmed 2026-07-09 — see project memory
   baking_plan_business_rules.

Usage
-----
.venv\\Scripts\\python.exe scripts\\seed_baking_plan_tables.py --dry-run
.venv\\Scripts\\python.exe scripts\\seed_baking_plan_tables.py --env-file .env.dev
.venv\\Scripts\\python.exe scripts\\seed_baking_plan_tables.py
"""

from __future__ import annotations

# ruff: noqa: E501
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "apps"))

import openpyxl  # noqa: E402

from baking_plan.templates import (  # noqa: E402
    BASE_TEMPLATE_PATH,
    normalize_name,
    parse_comments_sheet,
)
from pipelines.forecast_publish.load_forecast_run import (  # noqa: E402
    DEFAULT_ENV_PATH,
    create_client,
)
from pipelines.forecast_publish.table_names import (  # noqa: E402
    get_table_suffix_from_env_file,
    table_name,
)

DOUGH_MATERIAL_MARKER = "тест"
CAPACITY_DEFAULTS = {
    "bakers_count": 2,
    "ovens_count": 2,
    "trays_per_oven_batch": 6,
    "bake_minutes": 30,
}
MOLDING_MINUTES_DEFAULTS = [
    ("Пироги сытные", 4),
    ("Пироги сладкие", 4),
    ("", 1),
]
SOURCE_COMMENT = "seeded from apps/baking_plan/assets/template.xlsx (old Excel one-time seed) 2026-07-09"


def _match_products(sku_meta: dict, products: pd.DataFrame) -> tuple[dict, list[str]]:
    """Return ({normalized_name: product_id}, [unmatched sku_name])."""
    by_norm: dict[str, str] = {}
    for _, row in products.iterrows():
        by_norm.setdefault(normalize_name(row["product_name"]), row["product_id"])

    matched: dict[str, str] = {}
    unmatched: list[str] = []
    for norm_name, meta in sku_meta.items():
        product_id = by_norm.get(norm_name)
        if product_id:
            matched[norm_name] = product_id
        else:
            unmatched.append(meta.sku_name)
    return matched, unmatched


def _build_dough_group_mapping(
    sku_meta: dict,
    matched: dict,
    recipes: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Return (mapping_df, {product_id: material_id}) for products with a clean single-group dough material."""
    dough = recipes[recipes["material_name"].str.contains(DOUGH_MATERIAL_MARKER, case=False, na=False)]
    dough = dough.drop_duplicates(subset=["product_id", "material_id"])

    group_by_norm = {norm: sku_meta[norm].dough_group for norm in matched}
    product_to_group = {matched[norm]: group for norm, group in group_by_norm.items()}

    labels_by_material: dict[str, dict[str, set]] = {}
    for _, row in dough.iterrows():
        pid = row["product_id"]
        group = product_to_group.get(pid)
        if group is None:
            continue
        labels_by_material.setdefault(row["material_id"], {}).setdefault(group, set()).add(pid)
        labels_by_material[row["material_id"]].setdefault("__name__", set()).add(row["material_name"])

    rows = []
    product_material: dict[str, str] = {}
    for material_id, groups in labels_by_material.items():
        material_name = next(iter(groups.pop("__name__")))
        if len(groups) != 1:
            print(f"  CONFLICT material_id={material_id} ({material_name}): {[k for k in groups]} -- skipped")
            continue
        (group_label, pids), = groups.items()
        rows.append(
            {
                "material_id": material_id,
                "material_name": material_name,
                "dough_group": group_label,
                "is_active": 1,
                "loaded_at": pd.Timestamp.now(),
                "comment": SOURCE_COMMENT,
            }
        )
        for pid in pids:
            product_material[pid] = material_id

    mapping_df = pd.DataFrame(rows)
    return mapping_df, product_material


def _build_sku_meta(
    sku_meta: dict,
    matched: dict,
    products: pd.DataFrame,
    mapping_df: pd.DataFrame,
    product_material: dict,
) -> pd.DataFrame:
    name_by_id = dict(zip(products["product_id"], products["product_name"]))
    group_by_material = dict(zip(mapping_df["material_id"], mapping_df["dough_group"])) if not mapping_df.empty else {}

    rows = []
    now = pd.Timestamp.now()
    today = pd.Timestamp.now().date()
    for norm_name, product_id in matched.items():
        meta = sku_meta[norm_name]
        material_id = product_material.get(product_id)
        if material_id and group_by_material.get(material_id) == meta.dough_group:
            dough_group, source = meta.dough_group, "recipe_derived"
        else:
            dough_group, source = meta.dough_group, "manual_override"
        rows.append(
            {
                "product_id": product_id,
                "product_name": name_by_id.get(product_id, meta.sku_name),
                "dough_group": dough_group,
                "dough_group_source": source,
                "kratnost": meta.kratnost,
                "is_two_day": int(meta.is_two_day),
                "station": meta.station or "",
                "is_on_demand": int(meta.is_on_demand),
                "scope": "base",
                "bakery_id": pd.NA,
                "valid_from": today,
                "is_active": 1,
                "loaded_at": now,
                "comment": SOURCE_COMMENT,
            }
        )
    return pd.DataFrame(rows)


def _build_capacity_row() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "bakery_id": pd.NA,
                **CAPACITY_DEFAULTS,
                "valid_from": pd.Timestamp.now().date(),
                "is_active": 1,
                "loaded_at": pd.Timestamp.now(),
                "comment": "initial global constant, confirmed 2026-07-09",
            }
        ]
    )


def _build_molding_minutes_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "category_name": category,
                "minutes_per_unit": minutes,
                "is_active": 1,
                "loaded_at": pd.Timestamp.now(),
                "comment": "confirmed 2026-07-09",
            }
            for category, minutes in MOLDING_MINUTES_DEFAULTS
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=DEFAULT_ENV_PATH, type=Path)
    parser.add_argument("--template-path", default=BASE_TEMPLATE_PATH, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    client = create_client(args.env_file)
    suffix = get_table_suffix_from_env_file(args.env_file)
    sku_meta_table = table_name("baking_sku_meta", suffix=suffix)
    mapping_table = table_name("baking_dough_group_mapping", suffix=suffix)
    capacity_table = table_name("baking_capacity_config", suffix=suffix)
    molding_table = table_name("baking_category_molding_minutes", suffix=suffix)

    print(f"template         : {args.template_path}")
    print(f"target tables    : {sku_meta_table}, {mapping_table}, {capacity_table}, {molding_table}")

    wb = openpyxl.load_workbook(args.template_path, data_only=True)
    sku_meta = parse_comments_sheet(wb)
    print(f"parsed SKU rows  : {len(sku_meta)}")

    products = client.query_df("SELECT product_id, product_name FROM dim_products")
    matched, unmatched = _match_products(sku_meta, products)
    print(f"matched products : {len(matched)} / {len(sku_meta)}")
    if unmatched:
        print("unmatched (need manual product_id):")
        for name in unmatched:
            print(f"  - {name}")

    pids = list(matched.values())
    recipes = client.query_df(
        "SELECT product_id, material_id, material_name FROM dim_recipes "
        "WHERE product_id IN %(pids)s AND is_deleted = %(nd)s",
        parameters={"pids": pids, "nd": "Нет"},
    )
    print("Bootstrapping baking_dough_group_mapping from dim_recipes...")
    mapping_df, product_material = _build_dough_group_mapping(sku_meta, matched, recipes)
    print(f"mapping rows     : {len(mapping_df)}")

    sku_meta_df = _build_sku_meta(sku_meta, matched, products, mapping_df, product_material)
    recipe_derived = (sku_meta_df["dough_group_source"] == "recipe_derived").sum()
    print(f"sku_meta rows    : {len(sku_meta_df)} (recipe_derived={recipe_derived}, manual_override={len(sku_meta_df) - recipe_derived})")

    capacity_df = _build_capacity_row()
    molding_df = _build_molding_minutes_rows()

    if args.dry_run:
        print("\n--dry-run: not writing to ClickHouse.")
        return

    client.insert_df(mapping_table, mapping_df)
    client.insert_df(sku_meta_table, sku_meta_df)
    client.insert_df(capacity_table, capacity_df)
    client.insert_df(molding_table, molding_df)
    print("\nInsert complete.")


if __name__ == "__main__":
    main()

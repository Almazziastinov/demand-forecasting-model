"""Generate MILP baking plan .xlsx files for pilot bakeries.

Runs the MILP allocator and writes the result using the original MILP renderer
(renders from scratch, no template mutation).

Usage
-----
.venv\Scripts\python.exe scripts\export_milp_baking_plan.py --date 2026-07-21
.venv\Scripts\python.exe scripts\export_milp_baking_plan.py --date 2026-07-21 --bakery-ids 21 22
.venv\Scripts\python.exe scripts\export_milp_baking_plan.py --date 2026-07-21 --out-dir output/milp_plans
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date as date_type
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "apps"))
sys.path.insert(0, str(ROOT / "apps" / "forecast_embedded"))

PILOT_BAKERY_IDS = [16, 20, 21, 22, 28, 80, 89, 107, 221, 222, 257]


def _load_env(env_file: str) -> None:
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _resolve_run_and_city(client, table_name_fn, bakery_id: int) -> tuple[str, str, str]:
    run_df = client.query_df(
        f"select run_id from {table_name_fn('forecast_runs_embedded')} where status = 'active' limit 1"
    )
    if run_df.empty:
        raise RuntimeError("No active run")
    run_id = run_df.iloc[0]["run_id"]
    df = client.query_df(
        "select any(city) as city, any(bakery_name) as name from dim_bakeries where toInt64OrNull(bakery_id) = %(bid)s",
        parameters={"bid": bakery_id},
    )
    city = df.iloc[0]["city"] if not df.empty else None
    name = df.iloc[0]["name"] if not df.empty else str(bakery_id)
    return run_id, city, name


def _export_bakery(
    bakery_id: int,
    forecast_date: str,
    out_dir: Path,
) -> Path:
    import openpyxl

    from app.db import get_client
    from app.table_names import table_name

    from apps.baking_plan import templates
    from apps.baking_plan.algorithms.milp import allocate_milp_detailed
    from apps.baking_plan.capacity import CapacityConfig, get_capacity_config, get_molding_minutes_map
    from apps.baking_plan.demand import load_revenue_bucket_input
    from apps.baking_plan.demand_milp import build_sku_demand
    from apps.baking_plan.rendering_milp import MANDATORY_ASSORTMENT, render_workbook

    client = get_client()
    run_id, city, bakery_name = _resolve_run_and_city(client, table_name, bakery_id)

    skus, skipped = build_sku_demand(
        run_id=run_id,
        forecast_date=forecast_date,
        bakery_id=bakery_id,
        city=city,
    )
    if skipped:
        print(f"  [skip] {len(skipped)} SKU bez baking_sku_meta: {skipped}")

    # Resolve mandatory product_ids from MANDATORY_ASSORTMENT constant
    from apps.baking_plan.allocation import normalize_sku_name
    norm_mandatory = {normalize_sku_name(n) for n in MANDATORY_ASSORTMENT}
    mandatory_ids = frozenset(
        sku.product_id for sku in skus
        if normalize_sku_name(sku.product_name) in norm_mandatory
    )

    # Load template to get windows
    revenue = load_revenue_bucket_input(bakery_id)
    bucket = templates.revenue_bucket(revenue)
    template_path = templates.template_path_for_bakery(bakery_id)
    workbook = openpyxl.load_workbook(template_path)
    selected_sheet_name = templates.select_sheet_name(bucket, workbook.sheetnames)
    all_windows = templates.parse_windows(workbook, selected_sheet_name)

    # Deduplicate windows by label (keep first = hourly group)
    seen_labels: set[str] = set()
    milp_windows = []
    for w in all_windows:
        if w.label not in seen_labels:
            milp_windows.append(w)
            seen_labels.add(w.label)

    # Capacity
    try:
        capacity = get_capacity_config(bakery_id)
    except RuntimeError:
        capacity = CapacityConfig(bakers_count=2, ovens_count=2, trays_per_oven_batch=6, bake_minutes=30)
    molding_map = get_molding_minutes_map()

    # Run MILP
    regular, defrost_out, two_day_out, shortfall_by_sku, defrost_shortfall_by_sku = allocate_milp_detailed(
        skus=skus,
        windows=milp_windows,
        capacity=capacity,
        molding_minutes_map=molding_map,
        mandatory_first_window_ids=mandatory_ids,
    )

    wb = render_workbook(
        bakery_name=bakery_name,
        forecast_date=forecast_date,
        windows=milp_windows,
        skus=skus,
        regular_alloc=regular,
        defrost_alloc=defrost_out,
        two_day_alloc=two_day_out,
        shortfall_by_sku=shortfall_by_sku,
        defrost_shortfall_by_sku=defrost_shortfall_by_sku,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = bakery_name.replace(" ", "_").replace("/", "-")[:40]
    out_path = out_dir / f"{bakery_id}_{safe_name}_{forecast_date}.xlsx"
    buffer = BytesIO()
    wb.save(buffer)
    out_path.write_bytes(buffer.getvalue())
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--date", default=str(date_type.today()))
    parser.add_argument("--bakery-ids", type=int, nargs="+", default=PILOT_BAKERY_IDS)
    parser.add_argument("--out-dir", default="output/milp_plans")
    args = parser.parse_args()

    if args.env_file and Path(args.env_file).exists():
        _load_env(args.env_file)

    out_dir = Path(args.out_dir) / args.date
    print(f"Data: {args.date}  |  bakeries: {args.bakery_ids}")
    print(f"Output: {out_dir.resolve()}")

    for bid in args.bakery_ids:
        print(f"\nBakery {bid}...", end=" ", flush=True)
        try:
            out_path = _export_bakery(bid, args.date, out_dir)
            print(f"OK -> {out_path.name}")
        except Exception as exc:
            print(f"ERROR -- {exc}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

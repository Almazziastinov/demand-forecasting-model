"""Generate MILP baking plans for pilot bakeries and print to console.

Usage
-----
.venv\Scripts\python.exe scripts\run_milp_baking_plan.py --date 2026-07-21
.venv\Scripts\python.exe scripts\run_milp_baking_plan.py --date 2026-07-21 --bakery-ids 16 21 22
.venv\Scripts\python.exe scripts\run_milp_baking_plan.py --date 2026-07-21 --max-windows 4
.venv\Scripts\python.exe scripts\run_milp_baking_plan.py --date 2026-07-21 --mandatory-skus "Вак-бэлиш" "Кыстыбый П"
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date as date_type
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
        raise RuntimeError("No active run in forecast_runs_embedded")
    run_id = run_df.iloc[0]["run_id"]

    city_df = client.query_df(
        "select any(city) as city, any(bakery_name) as name from dim_bakeries where toInt64OrNull(bakery_id) = %(bid)s",
        parameters={"bid": bakery_id},
    )
    city = city_df.iloc[0]["city"] if not city_df.empty else None
    name = city_df.iloc[0]["name"] if not city_df.empty else str(bakery_id)
    return run_id, city, name


def _plan_bakery(
    bakery_id: int,
    forecast_date: str,
    mandatory_sku_names: set[str],
    max_active_windows: int | None,
    debug_sku: str | None = None,
) -> None:
    from app.db import get_client
    from app.table_names import table_name

    from apps.baking_plan.algorithms.milp import allocate_milp_detailed
    from apps.baking_plan.capacity import CapacityConfig, get_capacity_config, get_molding_minutes_map
    from apps.baking_plan.demand_milp import build_sku_demand
    from apps.baking_plan.templates import parse_windows, template_path_for_bakery

    import openpyxl

    client = get_client()
    run_id, city, bakery_name = _resolve_run_and_city(client, table_name, bakery_id)

    skus, skipped = build_sku_demand(
        run_id=run_id,
        forecast_date=forecast_date,
        bakery_id=bakery_id,
        city=city,
    )

    # Resolve mandatory SKU product_ids from names
    mandatory_ids: frozenset[str] = frozenset()
    if mandatory_sku_names:
        from apps.baking_plan.allocation import normalize_sku_name
        norm_mandatory = {normalize_sku_name(n) for n in mandatory_sku_names}
        mandatory_ids = frozenset(
            sku.product_id for sku in skus
            if normalize_sku_name(sku.product_name) in norm_mandatory
        )
        resolved_names = {sku.product_name for sku in skus if sku.product_id in mandatory_ids}
        unresolved = mandatory_sku_names - resolved_names
        if unresolved:
            print(f"  [warn] mandatory SKUs not found in assortment: {unresolved}")

    template_path = template_path_for_bakery(bakery_id)
    workbook = openpyxl.load_workbook(template_path)
    from apps.baking_plan.demand import load_revenue_bucket_input
    from apps.baking_plan.templates import revenue_bucket, select_sheet_name
    revenue = load_revenue_bucket_input(bakery_id)
    bucket = revenue_bucket(revenue)
    selected_sheet = select_sheet_name(bucket, workbook.sheetnames)
    windows = parse_windows(workbook, selected_sheet)
    # Use only the hourly (non-hidden) windows: first occurrence of each time label
    seen_labels: set[str] = set()
    unique_windows = []
    for w in windows:
        if w.label not in seen_labels:
            unique_windows.append(w)
            seen_labels.add(w.label)
    windows = unique_windows

    try:
        capacity = get_capacity_config(bakery_id)
    except RuntimeError:
        capacity = CapacityConfig(bakers_count=2, ovens_count=2, trays_per_oven_batch=6, bake_minutes=30)

    molding_map = get_molding_minutes_map()

    if debug_sku:
        from apps.baking_plan.algorithms.common import window_demand as _window_demand
        debug_norm = debug_sku.strip().lower()
        for sku in skus:
            if sku.product_name.strip().lower() != debug_norm:
                continue
            print(f"\n{'='*60}")
            print(f"DEBUG: {sku.product_name}  (id={sku.product_id}, kratnost={sku.kratnost})")
            print(f"  Почасовой прогноз (hourly_qty):")
            for h in sorted(sku.hourly_qty):
                print(f"    {h:02d}:00  {sku.hourly_qty[h]:.1f}")
            print(f"  Итого за день: {sum(sku.hourly_qty.values()):.1f}")
            wd = _window_demand(sku, windows)
            import numpy as np
            cum = list(np.cumsum(wd))
            print(f"\n  Распределение по окнам (window_demand -> MILP input):")
            print(f"  {'Окно':<20} {'Спрос окна':>12} {'Накопленный':>12}")
            for w, window in enumerate(windows):
                print(f"  {window.label:<20} {wd[w]:>12.1f} {cum[w]:>12.1f}")
            print(f"{'='*60}\n")
            break
        else:
            print(f"\n[debug] SKU '{debug_sku}' not found in plan for bakery {bakery_id}\n")

    # --- Debug: hourly profile + per-window demand for specific SKU ---
    if debug_sku:
        from apps.baking_plan.algorithms.common import window_demand
        import re
        norm_debug = re.sub(r"\s+", " ", debug_sku.strip().lower())
        for sku in skus:
            if re.sub(r"\s+", " ", sku.product_name.strip().lower()) != norm_debug:
                continue
            print(f"\n  === DEBUG: {sku.product_name} (kratnost={sku.kratnost}, station={sku.station}) ===")
            print(f"  Почасовой прогноз (hourly_qty):")
            for h in sorted(sku.hourly_qty):
                print(f"    {h:02d}:00  {sku.hourly_qty[h]:.2f}")
            print(f"  Итого за день: {sum(sku.hourly_qty.values()):.2f}")
            wd = window_demand(sku, windows)
            print(f"\n  Распределение по окнам (window_demand -> MILP coverage targets):")
            cumulative = 0.0
            for w, (window, d) in enumerate(zip(windows, wd)):
                cumulative += d
                print(f"    [{w}] {window.label}  demand={d:.2f}  cumulative={cumulative:.2f}")
            break
        else:
            print(f"\n  [debug] SKU '{debug_sku}' не найден в списке (всего {len(skus)} SKU)")

    regular, defrost_out, two_day_out, shortfall_by_sku, _ = allocate_milp_detailed(
        skus=skus,
        windows=windows,
        capacity=capacity,
        molding_minutes_map=molding_map,
        mandatory_first_window_ids=mandatory_ids,
        max_active_windows=max_active_windows,
    )

    # --- Print plan ---
    print(f"\n{'='*70}")
    print(f"Пекарня {bakery_id} — {bakery_name}  |  шаблон: {selected_sheet}  |  run: {run_id[:40]}...")
    print(f"  SKU в плане: {len(skus)}, пропущено (нет meta): {len(skipped)}")
    cap = capacity
    print(f"  Capacity: {cap.bakers_count} пек, {cap.ovens_count} печи x {cap.trays_per_oven_batch} прот, цикл {cap.bake_minutes} мин")
    if mandatory_ids:
        print(f"  Обязательный ассортимент (первое окно): {len(mandatory_ids)} SKU")
    if max_active_windows:
        print(f"  Макс. активных окон: {max_active_windows}")

    window_labels = [w.label for w in windows]
    col_w = 14
    header = f"  {'SKU':<35} {'Кат':<18} " + " ".join(f"{lbl[:col_w]:>{col_w}}" for lbl in window_labels) + f"  {'Итого':>6}  {'Нехватка':>8}"
    print(f"\n{header}")
    print("  " + "-" * (len(header) - 2))

    # Group by category
    from collections import defaultdict
    by_category: dict[str, list] = defaultdict(list)
    for sku in sorted(skus, key=lambda s: s.product_name):
        by_category[sku.category_name].append(sku)

    total_shortfall = 0.0
    active_windows: set[str] = set()

    for category in sorted(by_category):
        print(f"\n  [{category}]")
        for sku in by_category[category]:
            pid = sku.product_id
            row_vals: dict[str, float] = {}
            for label in window_labels:
                qty = regular.get((pid, label), 0.0) + defrost_out.get((pid, label), 0.0) + two_day_out.get((pid, label), 0.0)
                if qty > 0:
                    row_vals[label] = qty
                    active_windows.add(label)

            total = sum(row_vals.values())
            shortfall = shortfall_by_sku.get(pid, 0.0)
            total_shortfall += shortfall

            if total == 0 and shortfall == 0:
                continue  # skip unscheduled SKUs with no shortfall

            flag = " *" if pid in mandatory_ids else ""
            name_str = (sku.product_name[:33] + "..") if len(sku.product_name) > 35 else sku.product_name
            cat_str = (sku.category_name[:16] + "..") if len(sku.category_name) > 18 else sku.category_name

            cells = []
            for lbl in window_labels:
                v = row_vals.get(lbl, 0.0)
                cells.append(f"{int(v):>{col_w}}" if v > 0 else f"{'—':>{col_w}}")

            shortfall_str = f"{shortfall:>8.0f}" if shortfall > 0 else f"{'':>8}"
            print(f"  {name_str:<35}{flag} {cat_str:<18} " + " ".join(cells) + f"  {int(total):>6}  {shortfall_str}")

    print(f"\n  Активных окон: {len(active_windows)} из {len(windows)}")
    print(f"  Суммарная нехватка: {total_shortfall:.0f} шт")
    if skipped:
        print(f"  Пропущено SKU (нет baking_sku_meta): {skipped}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--date", default=str(date_type.today()))
    parser.add_argument("--bakery-ids", type=int, nargs="+", default=PILOT_BAKERY_IDS)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--mandatory-skus", nargs="+", default=[])
    parser.add_argument("--debug-sku", type=str, default=None, help="Print hourly profile for this SKU name")
    args = parser.parse_args()

    if args.env_file and Path(args.env_file).exists():
        _load_env(args.env_file)

    mandatory_sku_names = set(args.mandatory_skus)

    print(f"Дата: {args.date}  |  пекарни: {args.bakery_ids}")
    if args.max_windows:
        print(f"Макс. окон: {args.max_windows}")
    if mandatory_sku_names:
        print(f"Обязательный ассортимент: {mandatory_sku_names}")

    for bid in args.bakery_ids:
        try:
            _plan_bakery(bid, args.date, mandatory_sku_names, args.max_windows, debug_sku=args.debug_sku)
        except Exception as exc:
            print(f"\nПекарня {bid}: ОШИБКА — {exc}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

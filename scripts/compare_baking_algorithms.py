"""Compare the greedy and MILP baking-window allocators on real data.

Loads real ClickHouse forecast/assortment data for one (bakery, date), runs
both `apps/baking_plan/algorithms/greedy.py` and `.../milp.py` on the same
input, and prints a comparison report: runtime, total weighted shortfall,
dough-group window-cohesion spread, and per-SKU differences.

Usage
-----
.venv\\Scripts\\python.exe scripts\\compare_baking_algorithms.py --bakery-id 157 --date 2026-07-10
.venv\\Scripts\\python.exe scripts\\compare_baking_algorithms.py --env-file .env.dev --bakery-id 157 --date 2026-07-10
"""

from __future__ import annotations

# ruff: noqa: E501
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps"))
sys.path.insert(0, str(ROOT / "apps" / "forecast_embedded"))


def _resolve_active_run(client, table_name) -> str:
    df = client.query_df(f"select run_id from {table_name('forecast_runs_embedded')} where status = 'active' limit 1")
    if df.empty:
        raise RuntimeError("No active forecast run found (forecast_runs_embedded.status = 'active')")
    return df.iloc[0]["run_id"]


def _resolve_city(client, bakery_id: int) -> str:
    df = client.query_df(
        "select any(city) as city from dim_bakeries where toInt64OrNull(bakery_id) = %(bakery_id)s",
        parameters={"bakery_id": bakery_id},
    )
    if df.empty or not df.iloc[0]["city"]:
        raise RuntimeError(f"No city found in dim_bakeries for bakery_id={bakery_id}")
    return df.iloc[0]["city"]


def _sku_total_demand(sku, defrost_demand_fn, two_day_demand_fn) -> float:
    if sku.is_two_day:
        return two_day_demand_fn(sku)
    return sum(sku.hourly_qty.values()) + defrost_demand_fn(sku)


def _shortfall_metric(skus, allocation, defrost_demand_fn, two_day_demand_fn) -> float:
    produced: dict[str, float] = {}
    for (pid, _label), qty in allocation.items():
        produced[pid] = produced.get(pid, 0.0) + qty
    total = 0.0
    for sku in skus:
        demand = _sku_total_demand(sku, defrost_demand_fn, two_day_demand_fn)
        shortfall = max(0.0, demand - produced.get(sku.product_id, 0.0))
        total += shortfall * sku.avg_daily_sales
    return total


def _cohesion_metric(skus, allocation) -> dict[str, float]:
    sku_by_id = {sku.product_id: sku for sku in skus}
    windows_by_group: dict[str, set[str]] = {}
    for (pid, label) in allocation:
        group = sku_by_id[pid].dough_group
        windows_by_group.setdefault(group, set()).add(label)
    spreads = [len(labels) for labels in windows_by_group.values()]
    return {
        "groups_with_production": len(spreads),
        "avg_windows_per_group": (sum(spreads) / len(spreads)) if spreads else 0.0,
        "max_windows_in_a_group": max(spreads) if spreads else 0,
    }


def _has_shortfall(skus, allocation, defrost_demand_fn, two_day_demand_fn) -> bool:
    produced: dict[str, float] = {}
    for (pid, _label), qty in allocation.items():
        produced[pid] = produced.get(pid, 0.0) + qty
    for sku in skus:
        demand = _sku_total_demand(sku, defrost_demand_fn, two_day_demand_fn)
        if demand - produced.get(sku.product_id, 0.0) > 1e-6:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--bakery-id", type=int, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    os.environ["ENV_FILE"] = str(args.env_file)

    import openpyxl

    from baking_plan._clickhouse import get_client, table_name
    from baking_plan.algorithms.common import defrost_demand, two_day_demand
    from baking_plan.algorithms.greedy import allocate_greedy
    from baking_plan.algorithms.milp import allocate_milp
    from baking_plan import capacity as capacity_module
    from baking_plan.capacity import get_capacity_config, get_molding_minutes_map
    from baking_plan.demand import build_sku_demand
    from baking_plan.templates import BASE_TEMPLATE_PATH, parse_windows

    client = get_client()
    run_id = args.run_id or _resolve_active_run(client, table_name)
    city = _resolve_city(client, args.bakery_id)

    print(f"run_id      : {run_id}")
    print(f"bakery_id   : {args.bakery_id}  city: {city}  date: {args.date}")

    skus, skipped = build_sku_demand(
        run_id=run_id, forecast_date=args.date, bakery_id=args.bakery_id, city=city
    )
    print(f"SKUs        : {len(skus)}  (skipped for missing baking_sku_meta: {len(skipped)})")

    workbook = openpyxl.load_workbook(BASE_TEMPLATE_PATH, data_only=True)
    windows = parse_windows(workbook)
    capacity_config = get_capacity_config(args.bakery_id)
    molding_map = get_molding_minutes_map()

    t0 = time.time()
    greedy_result = allocate_greedy(skus, windows, capacity_config, molding_map)
    greedy_time = time.time() - t0

    t0 = time.time()
    milp_result = allocate_milp(
        skus,
        windows,
        capacity_config,
        molding_map,
        core_unit_cap=capacity_module.daily_core_unit_cap(capacity_config, peak=False),
    )
    if _has_shortfall(skus, milp_result, defrost_demand, two_day_demand):
        milp_result = allocate_milp(
            skus,
            windows,
            capacity_config,
            capacity_module.MOLDING_MINUTES_FLOOR,
            core_unit_cap=capacity_module.daily_core_unit_cap(capacity_config, peak=True),
        )
    milp_time = time.time() - t0

    greedy_shortfall = _shortfall_metric(skus, greedy_result, defrost_demand, two_day_demand)
    milp_shortfall = _shortfall_metric(skus, milp_result, defrost_demand, two_day_demand)
    greedy_cohesion = _cohesion_metric(skus, greedy_result)
    milp_cohesion = _cohesion_metric(skus, milp_result)

    print()
    header = f"{'metric':<28} {'greedy':>12} {'milp':>12}"
    print(header)
    print("-" * len(header))
    print(f"{'runtime (s)':<28} {greedy_time:>12.3f} {milp_time:>12.3f}")
    print(f"{'scheduled cells':<28} {len(greedy_result):>12} {len(milp_result):>12}")
    print(f"{'total qty baked':<28} {sum(greedy_result.values()):>12.1f} {sum(milp_result.values()):>12.1f}")
    print(f"{'weighted shortfall':<28} {greedy_shortfall:>12.2f} {milp_shortfall:>12.2f}")
    print(
        f"{'avg windows/dough-group':<28} "
        f"{greedy_cohesion['avg_windows_per_group']:>12.2f} "
        f"{milp_cohesion['avg_windows_per_group']:>12.2f}"
    )
    print(
        f"{'max windows in a group':<28} "
        f"{greedy_cohesion['max_windows_in_a_group']:>12} "
        f"{milp_cohesion['max_windows_in_a_group']:>12}"
    )

    print()
    print("Per-SKU differences (total qty baked, greedy vs milp):")
    diffs = 0
    for sku in skus:
        g_total = sum(qty for (pid, _label), qty in greedy_result.items() if pid == sku.product_id)
        m_total = sum(qty for (pid, _label), qty in milp_result.items() if pid == sku.product_id)
        if abs(g_total - m_total) > 1e-6:
            diffs += 1
            print(f"  {sku.product_name:<35} greedy={g_total:>7.1f}  milp={m_total:>7.1f}")
    if not diffs:
        print("  (no differences)")


if __name__ == "__main__":
    main()

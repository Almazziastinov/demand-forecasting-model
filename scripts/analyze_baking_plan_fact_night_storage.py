"""Validate baking-plan capacity on actual sales with night-storage scenarios.

This is a diagnostic script, not production export logic. It answers the
question: if we replace forecast with actual sales for a closed day, does the
current baking-plan configuration cover every SKU once documented overnight
defrost/night-storage allowances are accounted for?

Usage:
    .venv\\Scripts\\python.exe scripts\\analyze_baking_plan_fact_night_storage.py --bakery-id 16 --date 2026-07-06
    .venv\\Scripts\\python.exe scripts\\analyze_baking_plan_fact_night_storage.py --bakery-id 16 --start-date 2026-07-06 --end-date 2026-07-10
"""

from __future__ import annotations

# ruff: noqa: E501
import argparse
import json
import math
import os
import sys
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps"))
sys.path.insert(0, str(ROOT / "apps" / "forecast_embedded"))


# From the 15.05.2026 freezer/refrigerator night-storage PDFs. Names are mapped
# to the current SKU names observed in ClickHouse/baking_sku_meta.
FREEZER_DIRECT_UNITS = {
    "Вак-бэлиш": 20,
    "Шафран творог": 10,
    "Пицца с колбасой": 10,
    "Жар пицца с курицей": 10,
    "Сосиска в тесте": 20,
    "Сосиска под шубой": 10,
    "Киш курица": 10,
    "Киш грибы курица": 10,
}

FRIDGE_DIRECT_UNITS = {
    "Вак-бэлиш": 20,
    "Шафран творог": 10,
    "Кыстыбый П": 100,
    "Яблочный": 4,
    "Пирожок яблоко": 10,
    "Пицца с колбасой": 10,
    "Жар пицца с курицей": 10,
    "Сосиска в тесте": 30,
    "Сосиска под шубой": 20,
}

# Prep-only rows: formed blanks / dough balls reduce daytime work, but do not
# mean the SKU is already baked. This is an intentionally conservative first
# diagnostic assumption: prepared items still need 1 minute/unit of daytime
# finishing/assembly instead of the default 2/4 minute pie molding time.
PREP_LABOR_MINUTES_BY_NAME = {
    "Жар Киш курица": 1.0,
    "Жар Киш грибы курица": 1.0,
    "Сметанник": 1.0,
    "Сметанник маковый": 1.0,
    "Сметанник мини": 1.0,
}

EARLY_STORAGE_HOURS = range(6, 12)


def _date_range(start: str, end: str) -> list[str]:
    current = date.fromisoformat(start)
    last = date.fromisoformat(end)
    result = []
    while current <= last:
        result.append(current.isoformat())
        current += timedelta(days=1)
    return result


def _resolve_active_run(client, table_name) -> str:
    df = client.query_df(
        f"select run_id from {table_name('forecast_runs_embedded')} where status = 'active' order by generated_at desc limit 1"
    )
    if df.empty:
        raise RuntimeError("No active forecast run found")
    return str(df.iloc[0]["run_id"])


def _resolve_city(client, bakery_id: int, day: str) -> str:
    df = client.query_df(
        """
        select any(city) as city
        from mart_sales_60d
        where toInt32OrNull(bakery_id) = %(bakery_id)s
          and check_date = %(day)s
        """,
        parameters={"bakery_id": bakery_id, "day": day},
    )
    if df.empty or not df.iloc[0]["city"]:
        raise RuntimeError(f"No city found for bakery_id={bakery_id}, date={day}")
    return str(df.iloc[0]["city"])


def _load_fact_hourly(client, bakery_id: int, day: str, product_ids: list[str]) -> dict[str, dict[int, float]]:
    from baking_plan._clickhouse import records

    if not product_ids:
        return {}
    df = client.query_df(
        """
        select product_id, toHour(check_datetime) as hour, sum(quantity) as qty
        from mart_sales_60d
        where toInt32OrNull(bakery_id) = %(bakery_id)s
          and check_date = %(day)s
          and product_id in %(product_ids)s
          and quantity > 0
        group by product_id, hour
        """,
        parameters={"bakery_id": bakery_id, "day": day, "product_ids": product_ids},
    )
    result: dict[str, dict[int, float]] = {}
    for row in records(df):
        result.setdefault(str(row["product_id"]).zfill(9), {})[int(row["hour"])] = float(row["qty"])
    return result


def _subtract_early_units(hourly_qty: dict[int, float], units: float) -> dict[int, float]:
    remaining = float(units)
    adjusted = dict(hourly_qty)
    for hour in EARLY_STORAGE_HOURS:
        if remaining <= 1e-9:
            break
        qty = adjusted.get(hour, 0.0)
        if qty <= 0:
            continue
        take = min(qty, remaining)
        adjusted[hour] = qty - take
        remaining -= take
    return {hour: qty for hour, qty in adjusted.items() if qty > 1e-9}


def _clone_fact_skus(base_skus, fact_hourly, direct_units_by_name: dict[str, int]):
    from baking_plan.demand import SkuDemand

    rows = []
    for sku in base_skus:
        hourly = dict(fact_hourly.get(sku.product_id, {}))
        direct_units = direct_units_by_name.get(sku.product_name, 0)
        if direct_units:
            hourly = _subtract_early_units(hourly, direct_units)
        rows.append(
            SkuDemand(
                product_id=sku.product_id,
                product_name=sku.product_name,
                category_name=sku.category_name,
                dough_group=sku.dough_group,
                kratnost=sku.kratnost,
                station=sku.station,
                is_two_day=False,
                is_on_demand=sku.is_on_demand,
                hourly_qty=hourly,
                next_day_hourly_qty={},
                avg_daily_sales=max(sku.avg_daily_sales, 1.0),
            )
        )
    return rows


def _solve_strict(
    *,
    skus,
    windows,
    capacity_config,
    molding_map,
    core_unit_cap: int | None,
    resolve_molding: Callable | None = None,
):
    from baking_plan.algorithms import milp

    old_tail = milp.ROUNDING_TAIL_MAX_FRACTION
    old_resolve = milp.resolve_molding_minutes_for_sku
    milp.ROUNDING_TAIL_MAX_FRACTION = 0.0
    if resolve_molding is not None:
        milp.resolve_molding_minutes_for_sku = resolve_molding
    try:
        regular, defrost, two_day, _shortfall, _defrost_shortfall = milp.allocate_milp_detailed(
            skus,
            windows,
            capacity_config,
            molding_map,
            core_unit_cap=core_unit_cap,
        )
    finally:
        milp.ROUNDING_TAIL_MAX_FRACTION = old_tail
        milp.resolve_molding_minutes_for_sku = old_resolve

    produced: dict[str, float] = {}
    for allocation in (regular, defrost, two_day):
        for (pid, _label), qty in allocation.items():
            produced[pid] = produced.get(pid, 0.0) + qty
    misses = []
    for sku in skus:
        fact = sum(sku.hourly_qty.values())
        if fact <= 1e-9:
            continue
        prod = produced.get(sku.product_id, 0.0)
        miss = max(0.0, fact - prod)
        if miss > 1e-6:
            misses.append(
                {
                    "product_id": sku.product_id,
                    "product_name": sku.product_name,
                    "category_name": sku.category_name,
                    "fact_qty": fact,
                    "produced_qty": prod,
                    "missing_qty": miss,
                }
            )
    return produced, misses


def _rounded_minimum(skus, resolve_molding) -> dict[str, float]:
    from baking_plan import capacity

    core_units = 0.0
    helper_units = 0.0
    core_minutes = 0.0
    helper_minutes = 0.0
    for sku in skus:
        fact = sum(sku.hourly_qty.values())
        if fact <= 1e-9:
            continue
        kratnost = capacity.effective_kratnost(sku)
        rounded = math.ceil(fact / kratnost - 1e-9) * kratnost
        minutes = rounded * resolve_molding(sku)
        if capacity.is_core_baking_category(sku.category_name):
            core_units += rounded
            core_minutes += minutes
        else:
            helper_units += rounded
            helper_minutes += minutes
    return {
        "core_rounded_units": core_units,
        "helper_rounded_units": helper_units,
        "core_minutes": core_minutes,
        "helper_minutes": helper_minutes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--bakery-id", type=int, required=True)
    parser.add_argument("--date")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--run-id")
    args = parser.parse_args()

    if args.date:
        dates = [args.date]
    elif args.start_date and args.end_date:
        dates = _date_range(args.start_date, args.end_date)
    else:
        raise SystemExit("Pass either --date or --start-date + --end-date")

    os.environ["ENV_FILE"] = str(args.env_file)

    import openpyxl

    from baking_plan import capacity, demand
    from baking_plan._clickhouse import get_client, table_name
    from baking_plan.templates import BASE_TEMPLATE_PATH, parse_windows

    client = get_client()
    run_id = args.run_id or _resolve_active_run(client, table_name)
    workbook = openpyxl.load_workbook(BASE_TEMPLATE_PATH, data_only=True)
    windows = parse_windows(workbook)
    capacity_config = capacity.get_capacity_config(args.bakery_id)
    peak_cap = capacity.daily_core_unit_cap(capacity_config, peak=True)

    def normal_molding(sku):
        return capacity.resolve_molding_minutes_for_sku(sku, capacity.MOLDING_MINUTES_FLOOR)

    def prep_molding(sku, _minutes_map):
        if sku.product_name in PREP_LABOR_MINUTES_BY_NAME:
            return PREP_LABOR_MINUTES_BY_NAME[sku.product_name]
        return capacity.resolve_molding_minutes_for_sku(sku, capacity.MOLDING_MINUTES_FLOOR)

    scenarios = {
        "baseline_fact": {},
        "freezer_direct": FREEZER_DIRECT_UNITS,
        "freezer_plus_fridge_direct": {**FREEZER_DIRECT_UNITS, **FRIDGE_DIRECT_UNITS},
    }

    all_rows = []
    for day in dates:
        city = _resolve_city(client, args.bakery_id, day)
        base_skus, skipped = demand.build_sku_demand(
            run_id=run_id, forecast_date=day, bakery_id=args.bakery_id, city=city
        )
        product_ids = [sku.product_id for sku in base_skus]
        fact_hourly = _load_fact_hourly(client, args.bakery_id, day, product_ids)

        for scenario_name, direct_units in scenarios.items():
            fact_skus = _clone_fact_skus(base_skus, fact_hourly, direct_units)
            produced, misses = _solve_strict(
                skus=fact_skus,
                windows=windows,
                capacity_config=capacity_config,
                molding_map=capacity.MOLDING_MINUTES_FLOOR,
                core_unit_cap=peak_cap,
            )
            rounded = _rounded_minimum(fact_skus, normal_molding)
            all_rows.append(
                {
                    "date": day,
                    "scenario": scenario_name,
                    "fact_qty": sum(sum(sku.hourly_qty.values()) for sku in fact_skus),
                    "produced_qty": sum(produced.values()),
                    "short_skus": len(misses),
                    "missing_qty": sum(row["missing_qty"] for row in misses),
                    "core_rounded_units": rounded["core_rounded_units"],
                    "core_minutes": rounded["core_minutes"],
                    "top_misses": sorted(misses, key=lambda row: row["missing_qty"], reverse=True)[:8],
                    "skipped_meta": skipped,
                }
            )

        direct_skus = _clone_fact_skus(base_skus, fact_hourly, {**FREEZER_DIRECT_UNITS, **FRIDGE_DIRECT_UNITS})
        produced, misses = _solve_strict(
            skus=direct_skus,
            windows=windows,
            capacity_config=capacity_config,
            molding_map=capacity.MOLDING_MINUTES_FLOOR,
            core_unit_cap=peak_cap,
            resolve_molding=prep_molding,
        )
        rounded = _rounded_minimum(direct_skus, lambda sku: prep_molding(sku, capacity.MOLDING_MINUTES_FLOOR))
        all_rows.append(
            {
                "date": day,
                "scenario": "freezer_fridge_direct_plus_prep_labor",
                "fact_qty": sum(sum(sku.hourly_qty.values()) for sku in direct_skus),
                "produced_qty": sum(produced.values()),
                "short_skus": len(misses),
                "missing_qty": sum(row["missing_qty"] for row in misses),
                "core_rounded_units": rounded["core_rounded_units"],
                "core_minutes": rounded["core_minutes"],
                "top_misses": sorted(misses, key=lambda row: row["missing_qty"], reverse=True)[:8],
                "skipped_meta": skipped,
            }
        )

    print(json.dumps(all_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

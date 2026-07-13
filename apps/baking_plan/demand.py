"""Assemble the per-(bakery, date) SKU demand dataset the algorithms consume.

Pipeline: assortment (city+bakery scope) -> filter to bakeable categories ->
join `baking_sku_meta` (bakery override, fallback base; missing rows are
skipped and reported) -> hourly forecast (+ next-day for двухдневка SKUs and
DEFROST_SKU_NAMES members, needed for two_day_demand/defrost_demand in
algorithms/common.py) -> credit yesterday's already-baked defrost stock back
out of today's own early hours (DEFROST_SKU_NAMES members only, see
`_load_yesterday_defrost_offset`) -> average sales (slot-priority ranking
input).
"""

from __future__ import annotations

# ruff: noqa: E501
import logging
from dataclasses import dataclass, field
from datetime import date as date_type
from datetime import timedelta

from ._clickhouse import get_client, records, table_name
from .assortment import get_bakeable_products
from .constants import DEFROST_HOURS, DEFROST_SKU_NAMES

logger = logging.getLogger(__name__)

SKU_META_TABLE = table_name("baking_sku_meta")
SKU_FORECAST_HOUR_TABLE = table_name("sku_forecast_hour_embedded")
SKU_FORECAST_HOUR_SNAPSHOTS_TABLE = table_name("sku_forecast_hour_snapshots")
SALES_TABLE = "mart_sales_60d"

BAKEABLE_CATEGORIES = {
    "Пироги сытные",
    "Пироги сладкие",
    "Выпечка сытная",
    "Выпечка сладкая",
    "Фастфуд",
}


@dataclass
class SkuDemand:
    product_id: str
    product_name: str
    category_name: str
    dough_group: str
    kratnost: int
    station: str | None
    is_two_day: bool
    is_on_demand: bool
    hourly_qty: dict[int, float] = field(default_factory=dict)
    next_day_hourly_qty: dict[int, float] = field(default_factory=dict)
    avg_daily_sales: float = 0.0


def _load_sku_meta(product_ids: list[str], bakery_id: int) -> dict[str, dict]:
    if not product_ids:
        return {}
    client = get_client()
    df = client.query_df(
        f"""
        select product_id, dough_group, kratnost, station, is_two_day, is_on_demand, scope
        from {SKU_META_TABLE} final
        where is_active = 1
          and product_id in %(pids)s
          and (scope = 'base' or (scope = 'bakery' and bakery_id = %(bakery_id)s))
        """,
        parameters={"pids": product_ids, "bakery_id": bakery_id},
    )
    by_product: dict[str, dict] = {}
    for row in records(df):
        existing = by_product.get(row["product_id"])
        if existing is None or row["scope"] == "bakery":
            by_product[row["product_id"]] = row
    return by_product


def _load_hourly(
    run_id: str, forecast_date: str, bakery_id: int, product_ids: list[str]
) -> dict[str, dict[int, float]]:
    if not product_ids:
        return {}
    client = get_client()
    pid_by_int = {int(pid): pid for pid in product_ids}
    df = client.query_df(
        f"""
        select product_id, hour, sum(forecast_qty) as qty
        from {SKU_FORECAST_HOUR_TABLE}
        where run_id = %(run_id)s
          and forecast_date = %(forecast_date)s
          and bakery_id = %(bakery_id)s
          and product_id in %(pids)s
        group by product_id, hour
        """,
        parameters={
            "run_id": run_id,
            "forecast_date": forecast_date,
            "bakery_id": bakery_id,
            "pids": list(pid_by_int.keys()),
        },
    )
    result: dict[str, dict[int, float]] = {}
    for row in records(df):
        pid = pid_by_int.get(int(row["product_id"]))
        if pid is None:
            continue
        result.setdefault(pid, {})[int(row["hour"])] = float(row["qty"])
    return result


def _load_yesterday_defrost_offset(
    bakery_id: int, forecast_date: str, product_ids: list[str]
) -> dict[str, dict[int, float]]:
    """Per-(product, hour) forecast_qty from the `lead_days = 1` snapshot for
    `forecast_date` — i.e. what the forecast expected to sell in today's
    early hours as of ~1 day ago. `DEFROST_SKU_NAMES` members get an extra
    overnight batch baked the *night before*, sized to cover exactly this
    (see `algorithms/common.py:defrost_demand`, which uses the same
    `DEFROST_HOURS`) — crediting it back out of today's own regular-window
    demand avoids treating that already-baked stock as if it still needed
    fresh production today. Verified 2026-07-11 that `lead_days = 1` values
    track closely (within a few percent) with directly querying the prior
    day's own run, so this is not a guess at a run-naming convention.
    """
    if not product_ids:
        return {}
    client = get_client()
    pid_by_int = {int(pid): pid for pid in product_ids}
    df = client.query_df(
        f"""
        select product_id, hour, sum(forecast_qty) as qty
        from {SKU_FORECAST_HOUR_SNAPSHOTS_TABLE}
        where forecast_date = %(forecast_date)s
          and lead_days = 1
          and bakery_id = %(bakery_id)s
          and product_id in %(pids)s
          and hour in %(hours)s
        group by product_id, hour
        """,
        parameters={
            "forecast_date": forecast_date,
            "bakery_id": bakery_id,
            "pids": list(pid_by_int.keys()),
            "hours": list(DEFROST_HOURS),
        },
    )
    result: dict[str, dict[int, float]] = {}
    for row in records(df):
        pid = pid_by_int.get(int(row["product_id"]))
        if pid is None:
            continue
        result.setdefault(pid, {})[int(row["hour"])] = float(row["qty"])
    return result


def _apply_defrost_offset(
    hourly: dict[str, dict[int, float]], offset: dict[str, dict[int, float]]
) -> None:
    """Mutate `hourly` in place, subtracting `offset` per (product, hour).

    Clamped at 0: an overnight batch that turned out larger than today's
    (possibly revised) forecast just means surplus shelf stock, not a
    negative demand.
    """
    for pid, hour_offsets in offset.items():
        today_hourly = hourly.setdefault(pid, {})
        for hour, offset_qty in hour_offsets.items():
            today_hourly[hour] = max(0.0, today_hourly.get(hour, 0.0) - offset_qty)


def _load_avg_daily_sales(bakery_id: int, product_ids: list[str]) -> dict[str, float]:
    if not product_ids:
        return {}
    client = get_client()
    df = client.query_df(
        f"""
        select product_id, sum(quantity) / countDistinct(check_date) as avg_qty
        from {SALES_TABLE}
        where toInt32OrNull(bakery_id) = %(bakery_id)s
          and product_id in %(pids)s
          and quantity > 0
        group by product_id
        """,
        parameters={"bakery_id": bakery_id, "pids": product_ids},
    )
    return {row["product_id"]: float(row["avg_qty"]) for row in records(df)}


def build_sku_demand(
    *, run_id: str, forecast_date: str, bakery_id: int, city: str
) -> tuple[list[SkuDemand], list[str]]:
    """Return (demand rows, product_ids skipped for missing baking_sku_meta)."""
    assortment = get_bakeable_products(city, forecast_date, bakery_id=bakery_id)
    candidates = [row for row in assortment if row["category_name"] in BAKEABLE_CATEGORIES]
    if not candidates:
        return [], []

    product_ids = [row["product_id"] for row in candidates]
    sku_meta = _load_sku_meta(product_ids, bakery_id)

    skipped = [pid for pid in product_ids if pid not in sku_meta]
    if skipped:
        logger.warning(
            "build_sku_demand: %d bakeable products have no baking_sku_meta row, skipping: %s",
            len(skipped),
            skipped,
        )

    matched_ids = [pid for pid in product_ids if pid in sku_meta]
    hourly = _load_hourly(run_id, forecast_date, bakery_id, matched_ids)

    # Next-day hourly is needed for both двухдневка SKUs (two_day_demand:
    # full next-day total) and DEFROST_SKU_NAMES members (defrost_demand:
    # next-day early hours only) — see algorithms/common.py. product_name
    # comes from the assortment rows, not baking_sku_meta (which doesn't
    # carry it).
    name_by_id = {row["product_id"]: row["product_name"] for row in candidates}
    next_day_ids = [
        pid
        for pid in matched_ids
        if int(sku_meta[pid]["is_two_day"]) or name_by_id[pid] in DEFROST_SKU_NAMES
    ]
    next_day_date = (date_type.fromisoformat(forecast_date) + timedelta(days=1)).isoformat()
    next_day_hourly = _load_hourly(run_id, next_day_date, bakery_id, next_day_ids)

    # Credit yesterday's overnight defrost batch back out of today's own
    # early-hour demand — only DEFROST_SKU_NAMES members ever got that extra
    # batch, so only they need the offset.
    defrost_ids = [pid for pid in matched_ids if name_by_id[pid] in DEFROST_SKU_NAMES]
    yesterday_defrost_offset = _load_yesterday_defrost_offset(bakery_id, forecast_date, defrost_ids)
    _apply_defrost_offset(hourly, yesterday_defrost_offset)

    avg_sales = _load_avg_daily_sales(bakery_id, matched_ids)

    rows: list[SkuDemand] = []
    for row in candidates:
        pid = row["product_id"]
        meta = sku_meta.get(pid)
        if meta is None:
            continue
        rows.append(
            SkuDemand(
                product_id=pid,
                product_name=row["product_name"],
                category_name=row["category_name"],
                dough_group=meta["dough_group"],
                kratnost=int(meta["kratnost"]),
                station=meta["station"] or None,
                is_two_day=bool(int(meta["is_two_day"])),
                is_on_demand=bool(int(meta["is_on_demand"])),
                hourly_qty=hourly.get(pid, {}),
                next_day_hourly_qty=next_day_hourly.get(pid, {}),
                avg_daily_sales=avg_sales.get(pid, 0.0),
            )
        )
    return rows, skipped

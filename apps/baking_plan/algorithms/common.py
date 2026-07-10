"""Shared window-demand math used by both the greedy and MILP allocators.

A SKU's production in window `w` is meant to cover sales from the end of the
*previous* window up to the end of window `w` itself, proportionally
rescaled so the windows' total always equals the SKU's full-day demand
(`sum(hourly_qty.values())`), not just the hours the window list happens to
span. We don't have (and, per 2026-07-10 discussion, don't need) exact
bakery operating hours: rather than only covering [start_hour, end_hour) of
the last window and leaving later-hour sales invisible to the algorithm —
found 2026-07-10 when a bakery's real sales ran until 21:00 against a
window list (from the reference file) that stops at 16:00, and dumping the
gap entirely onto the last window blew its capacity 17x over — the whole
day's volume is spread across the *existing* windows in proportion to their
share of the (windowed-hours-only) demand curve. A window that already had
a bigger natural share of demand absorbs a bigger share of the "invisible"
tail too, rather than all of it landing on the last window specifically.

Дефрост and двухдневка are separate mechanics, not variants of the same
thing (confirmed against the reference file — the SKUs that show
"(ночная дефр)" there share no members with the SKUs flagged `is_two_day`
in "комментарии"):

- **Дефрост**: a SKU baked with normal same-day window logic *also* gets an
  extra overnight batch sized to next-day's early hours (`DEFROST_HOURS`),
  on top of its regular production. Which SKUs need this isn't derivable
  from any existing flag/table yet (open question — see
  `baking_plan_business_rules` memory) — `DEFROST_SKU_NAMES` below is a
  hardcoded placeholder matching the reference file's example.
- **Двухдневка** (`is_two_day`): the SKU has *zero* regular-window
  production for today (its stock for today was already baked in
  yesterday's last window) — instead, today's last window bakes the
  *entire* next day's forecast in one batch.
"""

from __future__ import annotations

from ..constants import DEFROST_HOURS, DEFROST_SKU_NAMES
from ..demand import SkuDemand
from ..templates import Window

__all__ = [
    "DEFROST_HOURS",
    "DEFROST_SKU_NAMES",
    "window_demand",
    "defrost_demand",
    "two_day_demand",
]


def window_demand(sku: SkuDemand, windows: list[Window]) -> list[float]:
    """Return demand[w] for each window, in the same order as `windows`.

    `is_two_day` SKUs have no regular-window demand — see module docstring.
    Each window's raw (hours-covered) share is rescaled so the windows sum
    to the SKU's full-day demand. If the SKU has zero demand in every
    windowed hour (all its demand falls entirely outside the window list's
    span), there's no curve shape to distribute by — falls back to an even
    split across windows rather than dropping the demand.
    """
    if sku.is_two_day or not windows:
        return [0.0] * len(windows)

    raw: list[float] = []
    prev_end = windows[0].start_hour
    for window in windows:
        hours = range(prev_end, window.end_hour)
        raw.append(sum(sku.hourly_qty.get(h, 0.0) for h in hours))
        prev_end = window.end_hour

    total_day = sum(sku.hourly_qty.values())
    total_windowed = sum(raw)
    if total_windowed <= 0:
        return [total_day / len(windows)] * len(windows) if total_day > 0 else raw

    scale = total_day / total_windowed
    return [r * scale for r in raw]


def defrost_demand(sku: SkuDemand) -> float:
    """Extra overnight batch (next-day early hours) for `DEFROST_SKU_NAMES` members."""
    if sku.product_name not in DEFROST_SKU_NAMES:
        return 0.0
    return sum(sku.next_day_hourly_qty.get(h, 0.0) for h in DEFROST_HOURS)


def two_day_demand(sku: SkuDemand) -> float:
    """Full next-day total demand for `is_two_day` SKUs, baked entirely in
    today's last window (replaces all of tomorrow's regular-window production)."""
    if not sku.is_two_day:
        return 0.0
    return sum(sku.next_day_hourly_qty.values())

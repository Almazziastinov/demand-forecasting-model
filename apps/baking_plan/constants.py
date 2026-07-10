"""Shared constants with no dependency on other `baking_plan` modules.

Kept separate so both `demand.py` (needs to know which product_ids require
next-day hourly data fetched) and `algorithms/common.py` (needs the demand
math) can import without a circular dependency.
"""

from __future__ import annotations

DEFROST_HOURS = range(6, 12)

# SKUs that get an extra overnight-defrost batch (next-day early hours) on
# top of their normal same-day window production — independent of
# `is_two_day` (see algorithms/common.py module docstring). No real
# source-of-truth table exists yet (open question — see
# baking_plan_business_rules memory, 2026-07-10 clarification); this list
# matches the SKUs marked "(ночная дефр)" in the reference file
# (apps/baking_plan/assets/template.xlsx).
DEFROST_SKU_NAMES = {
    "Вак-бэлиш",
    "Жар пицца с курицей",
    "Пицца с колбасой",
    "Сосиска в тесте",
    "Сосиска под шубой",
}

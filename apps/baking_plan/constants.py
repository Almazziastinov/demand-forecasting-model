"""Shared constants with no dependency on other `baking_plan` modules.

Kept separate so both `demand.py` (needs to know which product_ids require
next-day hourly data fetched) and `algorithms/common.py` (needs the demand
math) can import without a circular dependency.
"""

from __future__ import annotations

DEFROST_HOURS = range(6, 12)

# Direct overnight stock from the freezer/refrigerator admission PDFs
# (15.05.2026). These quantities cover next-day early demand and are credited
# back out of that next day's regular production. When the same SKU appears in
# both PDFs, use the refrigerator quantity: it is the less disruptive normal
# night-storage route and is larger for the SKUs where the two differ.
NIGHT_STORAGE_DIRECT_UNITS_BY_SKU = {
    "Вак-бэлиш": 20,
    "Шафран творог": 10,
    "Кыстыбый П": 100,
    "Яблочный": 4,
    "Пирожок яблоко": 10,
    "Пицца с колбасой": 10,
    "Жар пицца с курицей": 10,
    "Сосиска в тесте": 30,
    "Сосиска под шубой": 20,
    "Киш курица": 10,
    "Киш грибы курица": 10,
}

# Prep-only night-storage rows: formed blanks / dough balls reduce daytime
# work, but they are not finished stock. The SKU still needs to be scheduled
# in a baking window; only the daytime labor minutes are reduced.
NIGHT_PREP_LABOR_MINUTES_BY_SKU = {
    "Жар Киш курица": 1.0,
    "Жар Киш грибы курица": 1.0,
    "Сметанник": 1.0,
    "Сметанник маковый": 1.0,
    "Сметанник мини": 1.0,
}

# Backwards-compatible alias used by older tests/imports. Semantically this is
# now "direct overnight stock", not every night-storage/prep allowance.
DEFROST_SKU_NAMES = set(NIGHT_STORAGE_DIRECT_UNITS_BY_SKU)

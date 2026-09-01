"""Automatic seven-day assortment with audited emergency overrides.

The automatic result is authoritative.  Overrides are deliberately narrow,
effective-dated exceptions for incident recovery; they are not a second
assortment source.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd


class OverrideAction(StrEnum):
    FORCE_INCLUDE = "force_include"
    FORCE_EXCLUDE = "force_exclude"


@dataclass(frozen=True)
class AssortmentPolicy:
    window_days: int = 7
    min_sales_qty: float = 0.0


PAIR_KEYS = ["bakery_id", "product_id"]


def build_automatic_assortment(
    sales: pd.DataFrame,
    *,
    as_of_date: str | pd.Timestamp,
    policy: AssortmentPolicy = AssortmentPolicy(),
) -> pd.DataFrame:
    """Return bakery/SKU pairs with positive sales in the prior rolling window."""
    required = {"date", "bakery_id", "product_id", "sold_qty"}
    missing = sorted(required.difference(sales.columns))
    if missing:
        raise KeyError("Missing required columns: " + ", ".join(missing))

    as_of = pd.Timestamp(as_of_date).normalize()
    start = as_of - pd.Timedelta(days=policy.window_days)
    work = sales.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work["sold_qty"] = pd.to_numeric(work["sold_qty"], errors="coerce").fillna(0.0)
    work = work[
        work["date"].ge(start)
        & work["date"].lt(as_of)
        & work["sold_qty"].gt(policy.min_sales_qty)
    ]
    if work.empty:
        return pd.DataFrame(columns=[*PAIR_KEYS, "source"])
    result = work[PAIR_KEYS].drop_duplicates().copy()
    result["source"] = "recent_sales_7d"
    return result.sort_values(PAIR_KEYS).reset_index(drop=True)


def apply_emergency_overrides(
    automatic: pd.DataFrame,
    overrides: pd.DataFrame,
    *,
    effective_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Apply active force-include/exclude records to an automatic assortment."""
    if overrides.empty:
        return automatic.copy()
    required = {
        "bakery_id",
        "product_id",
        "action",
        "valid_from",
        "valid_to",
        "reason",
        "created_by",
    }
    missing = sorted(required.difference(overrides.columns))
    if missing:
        raise KeyError("Missing override columns: " + ", ".join(missing))

    date = pd.Timestamp(effective_date).normalize()
    active = overrides.copy()
    active["valid_from"] = pd.to_datetime(
        active["valid_from"], errors="raise"
    ).dt.normalize()
    active["valid_to"] = pd.to_datetime(
        active["valid_to"], errors="raise"
    ).dt.normalize()
    if active["valid_to"].isna().any():
        raise ValueError("Emergency overrides require valid_to")
    active = active[active["valid_from"].le(date) & active["valid_to"].ge(date)]
    if active.duplicated(PAIR_KEYS, keep=False).any():
        raise ValueError("Multiple active overrides for the same bakery/product")
    valid_actions = {action.value for action in OverrideAction}
    unknown = sorted(set(active["action"].astype(str)) - valid_actions)
    if unknown:
        raise ValueError(f"Unknown override actions: {unknown}")

    result = automatic.copy()
    for column in [*PAIR_KEYS, "source"]:
        if column not in result:
            result[column] = pd.Series(dtype="object")
    excludes = active[active["action"].eq(OverrideAction.FORCE_EXCLUDE.value)]
    if not excludes.empty:
        result = result.merge(
            excludes[PAIR_KEYS].assign(_exclude=1),
            on=PAIR_KEYS,
            how="left",
        )
        result = result[result["_exclude"].isna()].drop(columns="_exclude")
    includes = active[active["action"].eq(OverrideAction.FORCE_INCLUDE.value)]
    if not includes.empty:
        additions = includes[PAIR_KEYS].copy()
        additions["source"] = "emergency_force_include"
        result = pd.concat([result, additions], ignore_index=True)
    return (
        result.drop_duplicates(PAIR_KEYS, keep="last")
        .sort_values(PAIR_KEYS)
        .reset_index(drop=True)
    )


def diagnose_baking_meta_gaps(
    assortment: pd.DataFrame,
    baking_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Return effective assortment pairs that cannot enter a baking plan."""
    required = {"product_id", "scope", "bakery_id", "is_active"}
    missing = sorted(required.difference(baking_meta.columns))
    if missing:
        raise KeyError("Missing baking meta columns: " + ", ".join(missing))
    meta = baking_meta[baking_meta["is_active"].astype(bool)].copy()
    base_ids = set(meta.loc[meta["scope"].eq("base"), "product_id"])
    bakery_keys = set(
        map(
            tuple,
            meta.loc[meta["scope"].eq("bakery"), PAIR_KEYS].values.tolist(),
        )
    )
    gaps = assortment[
        ~assortment.apply(
            lambda row: row["product_id"] in base_ids
            or (row["bakery_id"], row["product_id"]) in bakery_keys,
            axis=1,
        )
    ].copy()
    gaps["reason"] = "missing_baking_sku_meta"
    return gaps

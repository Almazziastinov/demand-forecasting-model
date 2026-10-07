"""Conservative, retrospective weekly-dip targets at bakery/category/SKU levels.

The output is a research hypothesis about latent demand, not an inventory or
stockout estimate. Callers must pass only history available at their origin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.audit_sales_cleaning import KNOWN_HOLIDAYS_BY_YEAR


KEYS = ["date", "bakery_id", "product_id"]


def _triplet(frame: pd.DataFrame, groups: list[str], value: str) -> pd.DataFrame:
    """Attach values at exact adjacent weekdays, leaving absent days missing."""
    base = frame.copy()
    for label, offset in (("previous", 7), ("next", -7)):
        lookup = frame[[*groups, "date", value]].copy()
        lookup["date"] = lookup["date"] + pd.Timedelta(days=offset)
        base = base.merge(
            lookup.rename(columns={value: f"{label}_{value}"}),
            on=[*groups, "date"],
            how="left",
            validate="one_to_one",
        )
    base[f"reference_{value}"] = (base[f"previous_{value}"] + base[f"next_{value}"]) / 2
    return base


def _holiday_buffer(dates: pd.Series) -> pd.Series:
    years = set(dates.dt.year.unique())
    holidays = {
        pd.Timestamp(day) + pd.Timedelta(days=offset)
        for year in years
        for day in KNOWN_HOLIDAYS_BY_YEAR.get(int(year), [])
        for offset in (-1, 0, 1)
    }
    return dates.isin(holidays)


def hierarchical_weekly_bridge(
    history: pd.DataFrame, categories: pd.DataFrame
) -> pd.DataFrame:
    """Propose bakery-level dip uplift, with and without category safeguards.

    Right-hand neighbors are at date+7 and thus must be within the supplied
    history cutoff. A bakery dip is rejected on known holidays and on a
    simultaneous network-wide dip. A SKU never exceeds its neighboring-week
    reference, and aggregate uplift cannot exceed the bakery gap.
    """
    required = {*KEYS, "observed_sales_qty", "release_qty"}
    missing = sorted(required - set(history.columns))
    if missing:
        raise ValueError(f"Missing history columns: {missing}")
    if categories.duplicated("product_id").any():
        raise ValueError("Category mapping must have one row per product")
    work = history[[*KEYS, "observed_sales_qty", "release_qty"]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="raise").dt.normalize()
    if work.duplicated(KEYS).any():
        raise ValueError("Duplicate SKU-day in history")
    for column in ("observed_sales_qty", "release_qty"):
        work[column] = pd.to_numeric(work[column], errors="raise")
        if not np.isfinite(work[column]).all() or work[column].lt(0).any():
            raise ValueError(f"Invalid {column}")
    work = work.merge(
        categories[["product_id", "category_name"]],
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    bakery = _triplet(
        work.groupby(["date", "bakery_id"], as_index=False)["observed_sales_qty"].sum(),
        ["bakery_id"],
        "observed_sales_qty",
    ).rename(
        columns={
            "observed_sales_qty": "bakery_sales",
            "previous_observed_sales_qty": "bakery_previous",
            "next_observed_sales_qty": "bakery_next",
            "reference_observed_sales_qty": "bakery_reference",
        }
    )
    network = _triplet(
        bakery.groupby("date", as_index=False)["bakery_sales"].sum(),
        [],
        "bakery_sales",
    )
    bakery = bakery.merge(
        network[["date", "bakery_sales", "reference_bakery_sales"]].rename(
            columns={
                "bakery_sales": "network_sales",
                "reference_bakery_sales": "network_reference",
            }
        ),
        on="date",
        validate="many_to_one",
    )
    bakery["bakery_gap"] = (bakery["bakery_reference"] - bakery["bakery_sales"]).clip(
        lower=0
    )
    bakery["bakery_dip_pass"] = (
        bakery["bakery_previous"].ge(100)
        & bakery["bakery_next"].ge(100)
        & bakery["bakery_previous"]
        .sub(bakery["bakery_next"])
        .abs()
        .le(bakery["bakery_reference"] * 0.2)
        & bakery["bakery_sales"].gt(0)
        & bakery["bakery_sales"].le(bakery["bakery_reference"] * 0.8)
        & bakery["network_sales"].ge(bakery["network_reference"] * 0.9)
        & ~_holiday_buffer(bakery["date"])
    )

    category = _triplet(
        work.dropna(subset=["category_name"])
        .groupby(["date", "bakery_id", "category_name"], as_index=False)[
            "observed_sales_qty"
        ]
        .sum(),
        ["bakery_id", "category_name"],
        "observed_sales_qty",
    ).rename(
        columns={
            "observed_sales_qty": "category_sales",
            "previous_observed_sales_qty": "category_previous",
            "next_observed_sales_qty": "category_next",
            "reference_observed_sales_qty": "category_reference",
        }
    )
    category = category.merge(
        bakery[["date", "bakery_id", "bakery_previous", "bakery_next"]],
        on=["date", "bakery_id"],
        validate="many_to_one",
    )
    category["previous_share"] = (
        category["category_previous"] / category["bakery_previous"]
    )
    category["next_share"] = category["category_next"] / category["bakery_next"]
    category["category_gap"] = (
        category["category_reference"] - category["category_sales"]
    ).clip(lower=0)
    category["category_pass"] = (
        category["category_previous"].ge(20)
        & category["category_next"].ge(20)
        & category["previous_share"]
        .sub(category["next_share"])
        .abs()
        .le((category["previous_share"] + category["next_share"]) * 0.15)
        & category["category_sales"].le(category["category_reference"] * 0.85)
    )

    sku = _triplet(work, ["bakery_id", "product_id"], "observed_sales_qty").rename(
        columns={
            "previous_observed_sales_qty": "sku_previous",
            "next_observed_sales_qty": "sku_next",
            "reference_observed_sales_qty": "sku_reference",
        }
    )
    sku = sku.merge(
        bakery[["date", "bakery_id", "bakery_gap", "bakery_dip_pass"]],
        on=["date", "bakery_id"],
        validate="many_to_one",
    ).merge(
        category[
            [
                "date",
                "bakery_id",
                "category_name",
                "category_gap",
                "category_pass",
                "category_previous",
                "category_next",
            ]
        ],
        on=["date", "bakery_id", "category_name"],
        how="left",
        validate="many_to_one",
    )
    sku["sku_gap"] = (sku["sku_reference"] - sku["observed_sales_qty"]).clip(lower=0)
    previous_share = sku["sku_previous"] / sku["category_previous"]
    next_share = sku["sku_next"] / sku["category_next"]
    sku["sku_pass"] = (
        sku["sku_previous"].ge(3)
        & sku["sku_next"].ge(3)
        & sku["observed_sales_qty"].gt(0)
        & sku["release_qty"].gt(0)
        & previous_share.sub(next_share).abs().le((previous_share + next_share) * 0.2)
        & sku["sku_gap"].gt(0)
    )
    sku["eligible_gap"] = sku["sku_gap"].where(
        sku["bakery_dip_pass"] & sku["sku_pass"], 0.0
    )
    bakery_eligible = sku.groupby(["date", "bakery_id"])["eligible_gap"].transform(
        "sum"
    )
    bakery_scale = (sku["bakery_gap"] / bakery_eligible).clip(upper=1).fillna(0)
    sku["bakery_only_uplift"] = sku["eligible_gap"] * bakery_scale

    sku["category_eligible_gap"] = sku["eligible_gap"].where(sku["category_pass"], 0.0)
    cat_eligible = sku.groupby(["date", "bakery_id", "category_name"], dropna=False)[
        "category_eligible_gap"
    ].transform("sum")
    cat_scale = (sku["category_gap"] / cat_eligible).clip(upper=1).fillna(0)
    sku["category_capped_gap"] = sku["category_eligible_gap"] * cat_scale
    bakery_capped = sku.groupby(["date", "bakery_id"])["category_capped_gap"].transform(
        "sum"
    )
    bakery_scale = (sku["bakery_gap"] / bakery_capped).clip(upper=1).fillna(0)
    sku["category_guarded_uplift"] = sku["category_capped_gap"] * bakery_scale
    return (
        sku[
            [
                *KEYS,
                "category_name",
                "bakery_gap",
                "bakery_dip_pass",
                "category_pass",
                "sku_pass",
                "bakery_only_uplift",
                "category_guarded_uplift",
            ]
        ]
        .sort_values(KEYS)
        .reset_index(drop=True)
    )

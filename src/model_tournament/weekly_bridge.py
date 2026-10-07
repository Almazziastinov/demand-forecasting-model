"""Retrospective same-weekday interpolation for isolated sales dips.

This is a research target proposal, not observed lost demand. A right-hand
neighbor may be used only when that week has ended before the forecast origin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


KEYS = ["bakery_id", "product_id"]
PANEL_KEYS = ["date", *KEYS]


def weekly_bridge_dips(history: pd.DataFrame) -> pd.DataFrame:
    """Propose 120,80,130 -> 120,125,130 only for a stable weekly triplet.

    Caller must first restrict ``history`` to dates available at the forecast
    origin. Missing exact adjacent weeks are never inferred from row position.
    """
    required = {"date", *KEYS, "observed_sales_qty"}
    missing = sorted(required - set(history.columns))
    if missing:
        raise ValueError(f"History lacks weekly-bridge columns: {missing}")
    work = history[[*PANEL_KEYS, "observed_sales_qty"]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="raise").dt.normalize()
    if work.duplicated(PANEL_KEYS).any():
        raise ValueError("Duplicate SKU-days in weekly-bridge history")
    work["observed_sales_qty"] = pd.to_numeric(
        work["observed_sales_qty"], errors="raise"
    )
    if (~np.isfinite(work["observed_sales_qty"])).any() or work[
        "observed_sales_qty"
    ].lt(0).any():
        raise ValueError("Invalid weekly-bridge sales")
    work = work.sort_values([*KEYS, "date"]).reset_index(drop=True)
    grouped = work.groupby(KEYS, sort=False)
    prev_date = grouped["date"].shift(7)
    next_date = grouped["date"].shift(-7)
    work["previous_week_sales"] = grouped["observed_sales_qty"].shift(7)
    work["next_week_sales"] = grouped["observed_sales_qty"].shift(-7)
    exact_neighbors = work["date"].sub(prev_date).dt.days.eq(7) & next_date.sub(
        work["date"]
    ).dt.days.eq(7)
    work["bridge_reference"] = (
        work["previous_week_sales"] + work["next_week_sales"]
    ) / 2
    reference = work["bridge_reference"]
    stable_neighbors = (
        work["previous_week_sales"].sub(work["next_week_sales"]).abs()
        <= 0.2 * reference
    )
    work["isolated_weekly_dip"] = (
        exact_neighbors
        & stable_neighbors
        & work["previous_week_sales"].ge(10)
        & work["next_week_sales"].ge(10)
        & work["observed_sales_qty"].gt(0)
        & work["observed_sales_qty"].le(0.75 * reference)
    )

    # If the whole bakery or the same SKU across peers also collapsed, a
    # local SKU interpolation is weaker evidence than an isolated SKU dip.
    bakery = (
        work.groupby(["date", "bakery_id"], as_index=False)["observed_sales_qty"]
        .sum()
        .rename(columns={"observed_sales_qty": "bakery_sales"})
        .sort_values(["bakery_id", "date"])
    )
    bakery_group = bakery.groupby("bakery_id", sort=False)
    bakery["bakery_previous"] = bakery_group["bakery_sales"].shift(7)
    bakery["bakery_next"] = bakery_group["bakery_sales"].shift(-7)
    bakery["bakery_prev_date"] = bakery_group["date"].shift(7)
    bakery["bakery_next_date"] = bakery_group["date"].shift(-7)
    product = (
        work.groupby(["date", "product_id"], as_index=False)["observed_sales_qty"]
        .sum()
        .rename(columns={"observed_sales_qty": "network_product_sales"})
        .sort_values(["product_id", "date"])
    )
    product_group = product.groupby("product_id", sort=False)
    product["product_previous"] = product_group["network_product_sales"].shift(7)
    product["product_next"] = product_group["network_product_sales"].shift(-7)
    product["product_prev_date"] = product_group["date"].shift(7)
    product["product_next_date"] = product_group["date"].shift(-7)
    work = work.merge(bakery, on=["date", "bakery_id"], validate="many_to_one")
    work = work.merge(product, on=["date", "product_id"], validate="many_to_one")
    bakery_reference = (work["bakery_previous"] + work["bakery_next"]) / 2
    bakery_valid = (
        work["date"].sub(work["bakery_prev_date"]).dt.days.eq(7)
        & work["bakery_next_date"].sub(work["date"]).dt.days.eq(7)
        & bakery_reference.gt(0)
    )
    work["bakery_context_ratio"] = work["bakery_sales"] / bakery_reference
    peer_current = work["network_product_sales"] - work["observed_sales_qty"]
    peer_previous = work["product_previous"] - work["previous_week_sales"]
    peer_next = work["product_next"] - work["next_week_sales"]
    peer_reference = (peer_previous + peer_next) / 2
    peer_valid = (
        work["date"].sub(work["product_prev_date"]).dt.days.eq(7)
        & work["product_next_date"].sub(work["date"]).dt.days.eq(7)
        & peer_reference.ge(20)
    )
    work["peer_product_context_ratio"] = peer_current / peer_reference
    work["context_guard_pass"] = (
        work["isolated_weekly_dip"]
        & bakery_valid
        & peer_valid
        & work["bakery_context_ratio"].ge(0.85)
        & work["peer_product_context_ratio"].ge(0.85)
    )
    uplift = (reference - work["observed_sales_qty"]).clip(lower=0).fillna(0)
    work["pure_weekly_bridge_uplift"] = uplift.where(work["isolated_weekly_dip"], 0.0)
    work["context_weekly_bridge_uplift"] = uplift.where(work["context_guard_pass"], 0.0)
    return work.sort_values(PANEL_KEYS).reset_index(drop=True)

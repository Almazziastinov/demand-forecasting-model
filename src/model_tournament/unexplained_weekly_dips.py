"""Auditable research policy for smoothing unexplained isolated low sales.

This policy never changes observed sales. It proposes a separate training
target only after both adjacent weekdays are in the available history.
"""

from __future__ import annotations

import pandas as pd


def annotate_unexplained_dips(
    bridge: pd.DataFrame,
    event_dates: set[pd.Timestamp],
    release_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Separate an observed dip, a known-calendar candidate, and a target.

    A calendar hit is *not* proof of causation. Such dates are withheld from
    automatic smoothing pending a repeated-effect audit. Weather is not used
    here because historical as-of forecast snapshots have not been validated.
    """
    required = {
        "date",
        "bakery_id",
        "product_id",
        "observed_sales_qty",
        "bridge_reference",
        "isolated_weekly_dip",
    }
    missing = sorted(required - set(bridge.columns))
    if missing:
        raise ValueError(f"Weekly-bridge panel lacks columns: {missing}")
    work = bridge[
        [
            "date",
            "bakery_id",
            "product_id",
            "observed_sales_qty",
            "bridge_reference",
            "isolated_weekly_dip",
        ]
    ].copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    if work.duplicated(["date", "bakery_id", "product_id"]).any():
        raise ValueError("Duplicate SKU-day in weekly-bridge panel")
    work["release_shortfall_candidate"] = False
    if release_history is not None:
        release = release_history[
            ["date", "bakery_id", "product_id", "release_qty"]
        ].copy()
        release["date"] = pd.to_datetime(release["date"]).dt.normalize()
        if release.duplicated(["date", "bakery_id", "product_id"]).any():
            raise ValueError("Duplicate SKU-day in release history")
        work = work.merge(
            release,
            on=["date", "bakery_id", "product_id"],
            how="left",
            validate="one_to_one",
        )
        for label, offset in (("previous_release", 7), ("next_release", -7)):
            neighbor = release.copy()
            neighbor["date"] += pd.Timedelta(days=offset)
            work = work.merge(
                neighbor.rename(columns={"release_qty": label}),
                on=["date", "bakery_id", "product_id"],
                how="left",
                validate="one_to_one",
            )
        release_reference = (work["previous_release"] + work["next_release"]) / 2
        work["release_shortfall_candidate"] = (
            work["isolated_weekly_dip"]
            & work["previous_release"].ge(10)
            & work["next_release"].ge(10)
            & work["previous_release"]
            .sub(work["next_release"])
            .abs()
            .le(0.2 * release_reference)
            & work["release_qty"].gt(0)
            & work["release_qty"].le(0.75 * release_reference)
        )
    work["calendar_candidate"] = work["date"].isin(event_dates)
    work["regularization_candidate"] = (
        work["isolated_weekly_dip"] & ~work["calendar_candidate"]
    )
    work["unexplained_dip"] = (
        work["regularization_candidate"] & ~work["release_shortfall_candidate"]
    )
    work["reason_code"] = "no_weekly_dip"
    work.loc[work["isolated_weekly_dip"], "reason_code"] = "unexplained_weekly_dip"
    work.loc[
        work["isolated_weekly_dip"] & work["release_shortfall_candidate"],
        "reason_code",
    ] = "release_shortfall_candidate"
    work.loc[
        work["isolated_weekly_dip"] & work["calendar_candidate"], "reason_code"
    ] = "calendar_candidate_not_smoothed"
    gap = (work["bridge_reference"] - work["observed_sales_qty"]).clip(lower=0)
    work["proposed_uplift"] = gap.where(work["regularization_candidate"], 0.0).fillna(
        0.0
    )
    work["release_uplift"] = work["proposed_uplift"].where(
        work["release_shortfall_candidate"], 0.0
    )
    work["unverified_uplift"] = work["proposed_uplift"].where(
        work["unexplained_dip"], 0.0
    )
    work["target_half"] = work["observed_sales_qty"] + 0.5 * work["proposed_uplift"]
    work["target_full"] = work["observed_sales_qty"] + work["proposed_uplift"]
    return work

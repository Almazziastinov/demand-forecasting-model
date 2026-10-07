"""Point-in-time SKU scope checks for the local tournament."""

from __future__ import annotations

import pandas as pd


SCOPE_KEYS = ["date", "bakery_id", "product_id"]


def calendar_prior_release_56(
    flows: pd.DataFrame, *, targets: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Sum releases in the 56 calendar days before each requested SKU-day.

    The source must contain at most one row per SKU and calendar date.
    Requested dates need not be present in the flow panel: zero-activity
    forecast days must not be selected or excluded based on same-day facts.
    """
    missing = sorted(set([*SCOPE_KEYS, "release_qty"]) - set(flows.columns))
    if missing:
        raise ValueError(f"Flow panel is missing columns: {missing}")
    history = flows[[*SCOPE_KEYS, "release_qty"]].copy()
    history["date"] = pd.to_datetime(history["date"], errors="raise").dt.normalize()
    if history.duplicated(SCOPE_KEYS).any():
        raise ValueError("Flow panel contains duplicate SKU-day keys")
    history["release_qty"] = pd.to_numeric(history["release_qty"], errors="raise")
    if history["release_qty"].isna().any():
        raise ValueError("Flow panel contains missing release quantities")
    target_keys = (
        history[SCOPE_KEYS].copy()
        if targets is None
        else targets[SCOPE_KEYS].copy()
    )
    target_keys["date"] = pd.to_datetime(
        target_keys["date"], errors="raise"
    ).dt.normalize()
    if target_keys.duplicated(SCOPE_KEYS).any():
        raise ValueError("Requested scope contains duplicate SKU-day keys")
    calendar = pd.concat([history[SCOPE_KEYS], target_keys], ignore_index=True)
    calendar = calendar.drop_duplicates(SCOPE_KEYS)
    work = calendar.merge(history, on=SCOPE_KEYS, how="left", validate="one_to_one")
    work["release_qty"] = work["release_qty"].fillna(0.0)
    work = work.sort_values(["bakery_id", "product_id", "date"]).reset_index(drop=True)
    indexed = work.set_index("date")
    groups = indexed.groupby([indexed["bakery_id"], indexed["product_id"]], sort=False)
    prior = (
        groups["release_qty"]
        .rolling("56D", closed="left")
        .sum()
        .rename("prior_release_56_calendar")
        .reset_index()
    )
    work = work.merge(prior, on=SCOPE_KEYS, how="left", validate="one_to_one")
    return target_keys.merge(
        work[[*SCOPE_KEYS, "prior_release_56_calendar"]],
        on=SCOPE_KEYS,
        how="left",
        validate="one_to_one",
    )


def snapshot_timing_audit(
    predictions: pd.DataFrame, snapshots: pd.DataFrame
) -> pd.DataFrame:
    """Count selected snapshot rows created after 08:00 Moscow target day.

    A record before the cutoff is not itself proof of point-in-time feature
    provenance. A record after the cutoff cannot substantiate that scope.
    """
    required_predictions = {*SCOPE_KEYS, "scope_source"}
    required_snapshots = {"forecast_date", "bakery_id", "product_id", "generated_at"}
    missing_predictions = sorted(required_predictions - set(predictions.columns))
    missing_snapshots = sorted(required_snapshots - set(snapshots.columns))
    if missing_predictions or missing_snapshots:
        raise ValueError(
            "Missing snapshot timing columns: "
            f"predictions={missing_predictions}, snapshots={missing_snapshots}"
        )
    selected = predictions.loc[
        predictions["scope_source"].eq("lead1_snapshot"), SCOPE_KEYS
    ].copy()
    selected["date"] = pd.to_datetime(selected["date"], errors="raise").dt.normalize()
    if selected.duplicated(SCOPE_KEYS).any():
        raise ValueError("Selected snapshot scope contains duplicate SKU-day keys")
    source = snapshots[
        ["forecast_date", "bakery_id", "product_id", "generated_at"]
    ].rename(columns={"forecast_date": "date"}).copy()
    source["date"] = pd.to_datetime(source["date"], errors="raise").dt.normalize()
    if source.duplicated(SCOPE_KEYS).any():
        raise ValueError("Snapshot file contains duplicate SKU-day keys")
    source["generated_at"] = pd.to_datetime(source["generated_at"], utc=True)
    selected = selected.merge(source, on=SCOPE_KEYS, how="left", validate="one_to_one")
    selected["month"] = selected["date"].dt.to_period("M").astype(str)
    deadline = selected["date"].dt.tz_localize("Europe/Moscow") + pd.Timedelta(
        hours=8
    )
    selected["missing_snapshot_record"] = selected["generated_at"].isna()
    selected["record_after_08_msk"] = selected["generated_at"].gt(
        deadline.dt.tz_convert("UTC")
    )
    return (
        selected.groupby("month")
        .agg(
            selected_rows=("date", "size"),
            missing_snapshot_record=("missing_snapshot_record", "sum"),
            record_after_08_msk=("record_after_08_msk", "sum"),
        )
        .reset_index()
    )

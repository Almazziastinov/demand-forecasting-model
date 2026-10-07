"""Past-only volatility screen for retrospective weekly-dip regularization."""

from __future__ import annotations

import numpy as np
import pandas as pd


KEYS = ["date", "bakery_id", "product_id"]


def _row_median(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Median over finite entries in each row of a narrow numeric matrix."""
    ordered = np.sort(np.where(np.isfinite(values), values, np.inf), axis=1)
    low = np.clip((counts - 1) // 2, 0, values.shape[1] - 1)
    high = np.clip(counts // 2, 0, values.shape[1] - 1)
    selected_low = np.take_along_axis(ordered, low[:, None], axis=1).ravel()
    selected_high = np.take_along_axis(ordered, high[:, None], axis=1).ravel()
    result = 0.5 * (selected_low + selected_high)
    result[counts == 0] = np.nan
    return result


def attach_volatility_guard(
    policy: pd.DataFrame,
    *,
    prior_weeks: int = 8,
    minimum_prior_weeks: int = 6,
    z_threshold: float = 2.5,
    minimum_reference: float = 30.0,
    regime_tolerance: float = 0.25,
) -> pd.DataFrame:
    """Keep a dip only if large relative to its own past weekday variation.

    The prior-week matrix uses exact dates ``date - 7*k`` only. Neither the
    current sale nor any future value enters the volatility estimate. The
    neighboring-week bridge itself must already be origin-truncated by caller.
    """
    if not 2 <= minimum_prior_weeks <= prior_weeks:
        raise ValueError("Invalid prior-week window")
    if z_threshold <= 0 or minimum_reference <= 0 or not 0 < regime_tolerance < 1:
        raise ValueError("Invalid volatility-guard threshold")
    required = {
        *KEYS,
        "observed_sales_qty",
        "bridge_reference",
        "regularization_candidate",
        "release_uplift",
        "unverified_uplift",
    }
    missing = sorted(required - set(policy.columns))
    if missing:
        raise ValueError(f"Policy lacks volatility columns: {missing}")
    work = policy.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    if work.duplicated(KEYS).any():
        raise ValueError("Duplicate SKU-day in volatility history")
    source = work.set_index(["bakery_id", "product_id", "date"])["observed_sales_qty"]
    values = []
    for week in range(1, prior_weeks + 1):
        lookup = pd.MultiIndex.from_arrays(
            [
                work["bakery_id"],
                work["product_id"],
                work["date"] - pd.Timedelta(days=7 * week),
            ],
            names=["bakery_id", "product_id", "date"],
        )
        values.append(source.reindex(lookup).to_numpy(dtype=float))
    matrix = np.column_stack(values)
    count = np.isfinite(matrix).sum(axis=1)
    median = _row_median(matrix, count)
    absolute_deviation = np.abs(matrix - median[:, None])
    mad = _row_median(absolute_deviation, count)
    reference = work["bridge_reference"].to_numpy(dtype=float)
    sales = work["observed_sales_qty"].to_numpy(dtype=float)
    scale = np.maximum(1.4826 * mad, np.sqrt(np.maximum(reference, 1.0)))
    gap = np.maximum(reference - sales, 0.0)
    score = np.divide(
        gap,
        scale,
        out=np.full(len(work), np.nan),
        where=np.isfinite(scale) & (scale > 0),
    )
    stable_regime = (
        np.isfinite(median)
        & (median > 0)
        & (reference >= median * (1 - regime_tolerance))
        & (reference <= median * (1 + regime_tolerance))
    )
    work["prior_weekday_count"] = count
    work["prior_weekday_median"] = median
    work["prior_weekday_mad"] = mad
    work["volatility_scale"] = scale
    work["dip_z_score"] = score
    work["stable_prior_regime"] = stable_regime
    work["volatility_guard_pass"] = (
        work["regularization_candidate"]
        & (count >= minimum_prior_weeks)
        & (reference >= minimum_reference)
        & stable_regime
        & (score >= z_threshold)
    )
    work["guarded_release_uplift"] = work["release_uplift"].where(
        work["volatility_guard_pass"], 0.0
    )
    work["guarded_unverified_uplift"] = work["unverified_uplift"].where(
        work["volatility_guard_pass"], 0.0
    )
    work["guarded_total_uplift"] = (
        work["guarded_release_uplift"] + work["guarded_unverified_uplift"]
    )
    return work


def attach_supply_context(guarded: pd.DataFrame, flows: pd.DataFrame) -> pd.DataFrame:
    """Tag a weekly dip when observed supply also fell; this is not stock truth."""
    required = {
        *KEYS,
        "release_qty",
        "incoming_move_qty",
        "outgoing_move_qty",
    }
    missing = sorted(required - set(flows.columns))
    if missing:
        raise ValueError(f"Flows lack supply columns: {missing}")
    supply = flows[
        [*KEYS, "release_qty", "incoming_move_qty", "outgoing_move_qty"]
    ].copy()
    supply["date"] = pd.to_datetime(supply["date"]).dt.normalize()
    if supply.duplicated(KEYS).any():
        raise ValueError("Duplicate SKU-day in supply flows")
    supply["supply_qty"] = (
        supply["release_qty"]
        + supply["incoming_move_qty"]
        - supply["outgoing_move_qty"]
    ).clip(lower=0)
    supply = supply[[*KEYS, "supply_qty"]]
    work = guarded.merge(supply, on=KEYS, how="left", validate="one_to_one")
    for label, offset in (("previous_supply", 7), ("next_supply", -7)):
        neighbor = supply.copy()
        neighbor["date"] += pd.Timedelta(days=offset)
        work = work.merge(
            neighbor.rename(columns={"supply_qty": label}),
            on=KEYS,
            how="left",
            validate="one_to_one",
        )
    reference = (work["previous_supply"] + work["next_supply"]) / 2
    work["supply_shortfall_candidate"] = (
        work["regularization_candidate"]
        & work["previous_supply"].ge(10)
        & work["next_supply"].ge(10)
        & work["previous_supply"].sub(work["next_supply"]).abs().le(0.2 * reference)
        & work["supply_qty"].gt(0)
        & work["supply_qty"].le(0.75 * reference)
    )
    work["guarded_supply_uplift"] = work["guarded_total_uplift"].where(
        work["supply_shortfall_candidate"], 0.0
    )
    work["guarded_no_supply_signal_uplift"] = work["guarded_total_uplift"].where(
        ~work["supply_shortfall_candidate"], 0.0
    )
    work["reason_code_refined"] = work["reason_code"]
    work.loc[work["supply_shortfall_candidate"], "reason_code_refined"] = (
        "supply_shortfall_candidate"
    )
    work.loc[
        work["regularization_candidate"] & ~work["supply_shortfall_candidate"],
        "reason_code_refined",
    ] = "no_verified_supply_or_calendar_signal"
    return work

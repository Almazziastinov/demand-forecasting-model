"""Research-only causal construction of a demand-oriented daily target.

This is a candidate label, not ground truth. It keeps raw sales and positive
restoration/negative outlier adjustment separately for later audit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.model_tournament.demand_target_contract import (
    validate_demand_target_contract,
)
from src.model_tournament.fixed_origin import KEYS, validate_flows


SOURCE_COLUMNS = [
    "date", "bakery_id", "product_id", "observed_sales_qty", "release_qty",
    "incoming_move_qty", "outgoing_move_qty", "written_off_qty", "last_sale_hour",
]


def select_origin_pairs(
    flows: pd.DataFrame, origins: list[pd.Timestamp]
) -> pd.DataFrame:
    """Freeze the union of positive-release scopes without future facts."""
    if not origins:
        raise ValueError("At least one origin is required")
    parts = []
    for origin in origins:
        history = flows.loc[
            flows["date"].between(origin - pd.Timedelta(days=55), origin)
            & flows["release_qty"].gt(0),
            KEYS,
        ]
        parts.append(history)
    pairs = pd.concat(parts, ignore_index=True).drop_duplicates().sort_values(KEYS)
    if pairs.empty:
        raise ValueError("No positive-release pairs at the chosen origins")
    return pairs.reset_index(drop=True)


def dense_candidate_panel(
    flows: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Materialize daily zero rows only after a pair's first local release."""
    if end < start:
        raise ValueError("End precedes start")
    flows = validate_flows(flows)
    missing = sorted(set(SOURCE_COLUMNS) - set(flows.columns))
    if missing:
        raise ValueError(f"Source panel lacks demand evidence: {missing}")
    bakery_dates = flows.loc[
        flows["date"].between(start, end), ["bakery_id", "date"]
    ].drop_duplicates()
    coverage = bakery_dates.groupby("bakery_id")["date"].agg(["min", "max", "count"])
    expected = (coverage["max"] - coverage["min"]).dt.days + 1
    if coverage["count"].ne(expected).any():
        raise ValueError(
            "Source has an interior missing bakery-day; do not impute it as zero"
        )
    pairs = pairs[KEYS].drop_duplicates().copy()
    release = flows.loc[
        flows["date"].between(start, end) & flows["release_qty"].gt(0),
        ["date", *KEYS],
    ]
    first = release.groupby(KEYS, as_index=False)["date"].min()
    first = pairs.merge(first, on=KEYS, how="inner", validate="one_to_one")
    if first.empty:
        raise ValueError("No pair has a release in the candidate period")
    dates = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    grid = first[KEYS].merge(dates, how="cross")
    grid = grid.merge(
        first.rename(columns={"date": "first_release_date"}),
        on=KEYS, how="left", validate="many_to_one",
    )
    grid = grid.loc[grid["date"] >= grid["first_release_date"]].copy()
    evidence = flows.loc[
        flows["date"].between(start, end), SOURCE_COLUMNS
    ]
    bakery_end = (
        evidence.groupby(["date", "bakery_id"], as_index=False)["last_sale_hour"]
        .max().rename(columns={"last_sale_hour": "bakery_last_sale_hour"})
    )
    work = grid.merge(evidence, on=["date", *KEYS], how="left", validate="one_to_one")
    work = work.merge(
        bakery_end, on=["date", "bakery_id"], how="left", validate="many_to_one"
    )
    for column in (
        "observed_sales_qty", "release_qty", "incoming_move_qty",
        "outgoing_move_qty", "written_off_qty",
    ):
        work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0.0)
        if (work[column] < 0).any() or not np.isfinite(work[column]).all():
            raise ValueError(f"Source has invalid {column}")
    return work.sort_values([*KEYS, "date"]).reset_index(drop=True)


def _prior_weekday_stat(
    work: pd.DataFrame, values: pd.Series, *, stat: str
) -> pd.Series:
    keys = [work["bakery_id"], work["product_id"], work["dow"]]
    shifted = values.groupby(keys, sort=False).shift(1)
    rolling = shifted.groupby(keys, sort=False).rolling(8, min_periods=1)
    if stat == "median":
        result = rolling.median()
    elif stat == "count":
        result = rolling.count()
    else:
        raise ValueError(f"Unsupported statistic: {stat}")
    return result.reset_index(level=[0, 1, 2], drop=True).sort_index()


def adjust_demand_panel(dense: pd.DataFrame) -> pd.DataFrame:
    """Use only earlier same weekdays plus same-day peer evidence.

    Restorations require sell-through, early cessation, a previous uncensored
    reference, and a lower current sale. Isolated positive spikes are reduced
    softly only when peer bakeries do not share the surge.
    """
    work = dense.sort_values([*KEYS, "date"]).reset_index(drop=True).copy()
    if work.duplicated(["date", *KEYS]).any():
        raise ValueError("Dense candidate panel has duplicate SKU-day keys")
    work["dow"] = work["date"].dt.dayofweek
    sales = work["observed_sales_qty"]
    available = (
        work["release_qty"] + work["incoming_move_qty"]
        - work["outgoing_move_qty"]
    ).clip(lower=0.0)
    gap = work["bakery_last_sale_hour"] - work["last_sale_hour"]
    broad_stockout = (
        sales.ge(3.0) & available.gt(0.0)
        & (sales + work["written_off_qty"]).ge(0.95 * available)
        & work["last_sale_hour"].ge(10.0) & gap.ge(2.0)
    )
    clean_sales = sales.where(~broad_stockout)
    clean_last_hour = work["last_sale_hour"].where(
        ~broad_stockout & sales.gt(0)
    )
    work["reference_sales"] = _prior_weekday_stat(
        work, clean_sales, stat="median"
    )
    work["reference_days"] = _prior_weekday_stat(
        work, clean_sales, stat="count"
    ).fillna(0).astype(int)
    work["reference_last_sale_hour"] = _prior_weekday_stat(
        work, clean_last_hour, stat="median"
    )
    restore = (
        broad_stockout & work["reference_days"].ge(3)
        & work["reference_sales"].gt(1.15 * sales)
        & work["reference_last_sale_hour"].ge(work["last_sale_hour"] + 1.5)
    )
    work["restoration_uplift_qty"] = np.minimum.reduce(
        [
            (work["reference_sales"] - sales).clip(lower=0).fillna(0).to_numpy(),
            (0.5 * sales).to_numpy(),
            np.full(len(work), 15.0),
        ]
    )
    work.loc[~restore, "restoration_uplift_qty"] = 0.0
    work["demand_after_restoration"] = sales + work["restoration_uplift_qty"]

    peer_ratio = (sales + 1.0) / (work["reference_sales"] + 1.0)
    peer_ratio = peer_ratio.where(work["reference_days"].ge(3))
    peer_keys = [work["date"], work["product_id"]]
    work["peer_median_ratio"] = peer_ratio.groupby(peer_keys).transform("median")
    work["peer_count"] = peer_ratio.groupby(peer_keys).transform("count")
    isolated_spike = (
        ~broad_stockout & work["reference_days"].ge(3)
        & sales.ge(20.0)
        & sales.gt(
            np.maximum(
                2.5 * work["reference_sales"], work["reference_sales"] + 12.0
            )
        )
        & work["peer_count"].ge(5)
        & work["peer_median_ratio"].lt(1.25)
    )
    soft_cap = np.maximum(
        1.5 * work["reference_sales"], work["reference_sales"] + 5.0
    )
    work["outlier_reduction_qty"] = np.minimum.reduce(
        [
            (sales - soft_cap).clip(lower=0).fillna(0).to_numpy(),
            (0.2 * sales).to_numpy(),
            np.full(len(work), 20.0),
        ]
    )
    work.loc[~isolated_spike, "outlier_reduction_qty"] = 0.0
    work["reconstructed_demand_qty"] = (
        work["demand_after_restoration"] - work["outlier_reduction_qty"]
    )
    work["broad_stockout_signal"] = broad_stockout
    work["is_restored"] = restore
    work["is_reduced_spike"] = isolated_spike
    validate_demand_target_contract(
        work,
        sales_column="observed_sales_qty",
        demand_column="reconstructed_demand_qty",
        uplift_column="restoration_uplift_qty",
        reduction_column="outlier_reduction_qty",
        require_global_uplift=True,
    )
    return work

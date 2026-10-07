"""Strict fixed-origin research panels for 14-day SKU forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.model_tournament.baselines import WEEKDAY_WEIGHTS


KEYS = ["bakery_id", "product_id"]
PANEL_KEYS = ["date", *KEYS]


def validate_flows(flows: pd.DataFrame) -> pd.DataFrame:
    """Normalize a frozen daily fact panel without inventing target-day rows."""
    required = {"date", *KEYS, "observed_sales_qty", "release_qty"}
    missing = sorted(required - set(flows.columns))
    if missing:
        raise ValueError(f"Flow panel is missing columns: {missing}")
    work = flows.copy()
    work["date"] = pd.to_datetime(work["date"], errors="raise").dt.normalize()
    if work.duplicated(PANEL_KEYS).any():
        raise ValueError("Flow panel contains duplicate SKU-day keys")
    for column in ("observed_sales_qty", "release_qty"):
        work[column] = pd.to_numeric(work[column], errors="raise")
        if not np.isfinite(work[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Flow panel has non-finite {column}")
        if (work[column] < 0).any():
            raise ValueError(f"Flow panel has negative {column}")
    return work.sort_values(PANEL_KEYS).reset_index(drop=True)


def _same_weekday_formula(rows: pd.DataFrame, history: pd.DataFrame) -> np.ndarray:
    matrix = []
    for week in range(8, 0, -1):
        lookup = rows[KEYS].copy()
        lookup["source_date"] = rows["date"] - pd.Timedelta(days=7 * week)
        found = lookup.merge(
            history[[*PANEL_KEYS, "observed_sales_qty"]].rename(
                columns={"date": "source_date"}
            ),
            on=[*KEYS, "source_date"],
            how="left",
            validate="many_to_one",
            sort=False,
        )
        matrix.append(found["observed_sales_qty"].to_numpy(dtype=float))
    values = np.column_stack(matrix)
    predictions = np.zeros(len(rows), dtype=float)
    for index, row in enumerate(values):
        tail = row[np.isfinite(row)][-len(WEEKDAY_WEIGHTS) :]
        if len(tail):
            weights = WEEKDAY_WEIGHTS[-len(tail) :]
            predictions[index] = float(np.dot(tail, weights) / weights.sum())
    return predictions


def build_fixed_origin_panel(
    flows: pd.DataFrame,
    origin: str | pd.Timestamp,
    *,
    horizon_days: int = 14,
) -> pd.DataFrame:
    """Freeze pair scope and all forecast features at the origin date.

    Truth for future dates is merged only after the scope and features exist.
    The eligible scope is positive release in origin-55..origin, with one row
    for every eligible pair and future date, including zero-activity dates.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be positive")
    as_of = pd.Timestamp(origin).normalize()
    history = flows[
        flows["date"].between(as_of - pd.Timedelta(days=55), as_of)
    ].copy()
    if history.empty:
        raise ValueError("No historical flows at the forecast origin")
    scope_history = history
    eligible = (
        scope_history.groupby(KEYS, as_index=False)["release_qty"]
        .sum()
        .loc[lambda frame: frame["release_qty"].gt(0), KEYS]
    )
    if eligible.empty:
        raise ValueError("No pairs had positive production in the prior 56 days")
    dates = pd.DataFrame(
        {"date": pd.date_range(as_of + pd.Timedelta(days=1), periods=horizon_days)}
    )
    rows = eligible.merge(dates, how="cross")
    rows["forecast_origin"] = as_of
    rows["lead_days"] = (rows["date"] - as_of).dt.days
    rows["day_of_week"] = rows["date"].dt.dayofweek
    rows["month"] = rows["date"].dt.month
    rows["day_of_month"] = rows["date"].dt.day

    origin_stats = eligible.copy()
    for days in (1, 7, 14):
        recent = history[
            history["date"].between(as_of - pd.Timedelta(days=days - 1), as_of)
        ]
        sums = recent.groupby(KEYS)["observed_sales_qty"].sum()
        index = pd.MultiIndex.from_frame(origin_stats[KEYS])
        origin_stats[f"origin_mean{days}"] = (
            sums.reindex(index, fill_value=0).to_numpy() / days
        )
    rows = rows.merge(origin_stats, on=KEYS, how="left", validate="many_to_one")
    rows["weighted_weekday_sales"] = _same_weekday_formula(rows, scope_history)

    truth = flows[flows["date"].between(dates["date"].min(), dates["date"].max())]
    rows = rows.merge(
        truth[[*PANEL_KEYS, "observed_sales_qty"]].rename(
            columns={"observed_sales_qty": "actual"}
        ),
        on=PANEL_KEYS,
        how="left",
        validate="one_to_one",
    )
    rows["actual"] = rows["actual"].fillna(0.0)
    return rows.sort_values(["date", *KEYS]).reset_index(drop=True)

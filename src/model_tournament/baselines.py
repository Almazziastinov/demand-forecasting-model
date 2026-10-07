"""Deterministic baselines with an explicit fact availability cutoff."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


PredictionFunction = Callable[
    [pd.DataFrame, str, list[str], pd.Series], pd.Series
]
WEEKDAY_WEIGHTS = np.asarray([1.0, 1.15, 1.35, 1.65, 2.0])


def _lookup(
    frame: pd.DataFrame,
    target_col: str,
    key_cols: list[str],
    source_dates: pd.Series,
) -> pd.Series:
    """Read an exact historical calendar date for each evaluation row."""
    history = frame[["date", *key_cols, target_col]].rename(
        columns={"date": "_source_date", target_col: "_prediction"}
    )
    lookup = frame[key_cols].copy()
    lookup["_source_date"] = source_dates.to_numpy()
    joined = lookup.merge(
        history,
        on=["_source_date", *key_cols],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    return joined["_prediction"].reset_index(drop=True)


def _fixed_lag(
    frame: pd.DataFrame,
    target_col: str,
    key_cols: list[str],
    cutoff: pd.Series,
    days: int,
) -> pd.Series:
    source_dates = frame["date"] - pd.Timedelta(days=days)
    prediction = _lookup(frame, target_col, key_cols, source_dates)
    return prediction.where(source_dates.le(cutoff).to_numpy())


def zero(
    frame: pd.DataFrame,
    target_col: str,
    key_cols: list[str],
    cutoff: pd.Series,
) -> pd.Series:
    del target_col, key_cols, cutoff
    return pd.Series(0.0, index=frame.index, dtype=float)


def lag1(
    frame: pd.DataFrame,
    target_col: str,
    key_cols: list[str],
    cutoff: pd.Series,
) -> pd.Series:
    """Last observable calendar day as of the cutoff."""
    return _lookup(frame, target_col, key_cols, cutoff)


def lag7(
    frame: pd.DataFrame,
    target_col: str,
    key_cols: list[str],
    cutoff: pd.Series,
) -> pd.Series:
    """The target date minus seven days, only when already observable."""
    return _fixed_lag(frame, target_col, key_cols, cutoff, 7)


def _rolling_mean(
    frame: pd.DataFrame,
    target_col: str,
    key_cols: list[str],
    cutoff: pd.Series,
    days: int,
) -> pd.Series:
    values = [
        _lookup(frame, target_col, key_cols, cutoff - pd.Timedelta(days=offset))
        for offset in range(days)
    ]
    return pd.concat(values, axis=1).mean(axis=1, skipna=False)


def mean7(
    frame: pd.DataFrame,
    target_col: str,
    key_cols: list[str],
    cutoff: pd.Series,
) -> pd.Series:
    return _rolling_mean(frame, target_col, key_cols, cutoff, 7)


def mean14(
    frame: pd.DataFrame,
    target_col: str,
    key_cols: list[str],
    cutoff: pd.Series,
) -> pd.Series:
    return _rolling_mean(frame, target_col, key_cols, cutoff, 14)


def same_weekday_mean2(
    frame: pd.DataFrame,
    target_col: str,
    key_cols: list[str],
    cutoff: pd.Series,
) -> pd.Series:
    values = [
        _fixed_lag(frame, target_col, key_cols, cutoff, days)
        for days in (7, 14)
    ]
    return pd.concat(values, axis=1).mean(axis=1, skipna=False)


def weighted_weekday_formula_v1(
    frame: pd.DataFrame,
    target_col: str,
    key_cols: list[str],
    cutoff: pd.Series,
) -> pd.Series:
    """Mirror the production weighting on the supplied panel target.

    Missing history is skipped within the eight exact same-weekday dates.
    Production emits zero when the pair has no eligible history.
    """
    values = [
        _fixed_lag(frame, target_col, key_cols, cutoff, 7 * week)
        for week in range(8, 0, -1)
    ]
    matrix = pd.concat(values, axis=1).to_numpy(dtype=float)
    result = np.zeros(len(frame), dtype=float)
    for row_number, row in enumerate(matrix):
        tail = row[np.isfinite(row)][-len(WEEKDAY_WEIGHTS) :]
        if len(tail):
            weights = WEEKDAY_WEIGHTS[-len(tail) :]
            result[row_number] = float(np.dot(tail, weights) / weights.sum())
    return pd.Series(result, index=frame.index)


BASELINES: dict[str, PredictionFunction] = {
    "zero": zero,
    "lag1": lag1,
    "lag7": lag7,
    "mean7": mean7,
    "mean14": mean14,
    "same_weekday_mean2": same_weekday_mean2,
    "weighted_weekday_formula_v1": weighted_weekday_formula_v1,
}


def available_baselines() -> list[str]:
    return sorted(BASELINES)


def predict_baseline(
    name: str,
    frame: pd.DataFrame,
    *,
    target_col: str,
    key_cols: list[str],
    lead_days: int = 1,
    history_cutoff: pd.Timestamp | None = None,
) -> pd.Series:
    """Predict using facts available by each row's forecast cutoff.

    ``lead_days=1`` means a target for day D may use facts through D-1.
    A fixed ``history_cutoff`` further limits every row to one forecast origin.
    """
    if lead_days < 1:
        raise ValueError("lead_days must be at least 1")
    try:
        function = BASELINES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown baseline {name!r}; available: {', '.join(available_baselines())}"
        ) from exc
    cutoff = frame["date"] - pd.Timedelta(days=lead_days)
    if history_cutoff is not None:
        cutoff = cutoff.clip(upper=pd.Timestamp(history_cutoff).normalize())
    prediction = function(frame, target_col, key_cols, cutoff)
    return pd.to_numeric(prediction, errors="coerce").clip(lower=0.0)

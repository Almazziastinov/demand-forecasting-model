"""Comparable descriptive diagnostics for stages of a daily target pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.model_tournament.fixed_origin import KEYS


STAGES = {
    "raw_sales": "observed_sales_qty",
    "after_restoration": "demand_after_restoration",
    "after_outlier_reduction": "reconstructed_demand_qty",
}


def _acf(values: np.ndarray, lag: int) -> float:
    if len(values) <= lag:
        return float("nan")
    centered = values - values.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 1e-12:
        return float("nan")
    return float(np.dot(centered[:-lag], centered[lag:]) / denominator)


def _pacf7(values: np.ndarray) -> float:
    acf = np.array([1.0, *[_acf(values, lag) for lag in range(1, 8)]])
    if not np.isfinite(acf).all():
        return float("nan")
    matrix = acf[np.abs(np.subtract.outer(np.arange(7), np.arange(7)))]
    try:
        coefficients = np.linalg.solve(matrix + np.eye(7) * 1e-8, acf[1:])
    except np.linalg.LinAlgError:
        return float("nan")
    return float(coefficients[-1])


def _stage_metrics(values: np.ndarray, dates: np.ndarray) -> dict[str, float]:
    mean = float(values.mean())
    variance = float(values.var())
    days = pd.DatetimeIndex(dates).dayofweek.to_numpy()
    weekday_means = np.array(
        [
            values[days == day].mean() if np.any(days == day) else mean
            for day in range(7)
        ]
    )
    seasonal_fit = weekday_means[days]
    residual = values - seasonal_fit
    first, second = np.array_split(values, 2)
    tail = values[56:]
    weekday_lag = values[49:-7] if len(values) > 56 else np.array([])
    # The prequential baseline is exactly the prior same-weekday observation.
    r2 = float("nan")
    if len(tail) and len(weekday_lag) == len(tail) and tail.var() > 1e-12:
        squared_error = np.square(tail - weekday_lag).sum()
        total_variance = np.square(tail - tail.mean()).sum()
        r2 = float(1 - squared_error / total_variance)
    centered = values - mean
    frequencies = np.fft.rfftfreq(len(values), d=1.0)
    power = np.abs(np.fft.rfft(centered)) ** 2
    valid = (frequencies >= 1 / 35) & (frequencies <= 1 / 2)
    dominant_period = float("nan")
    if valid.any() and power[valid].max() > 1e-12:
        strongest = np.flatnonzero(valid)[power[valid].argmax()]
        dominant_period = float(1 / frequencies[strongest])
    return {
        "mean": mean,
        "cv": float(np.sqrt(variance) / mean) if mean > 1e-12 else float("nan"),
        "acf1": _acf(values, 1),
        "acf7": _acf(values, 7),
        "acf14": _acf(values, 14),
        "pacf7": _pacf7(values),
        "weekly_seasonality_strength": (
            float(max(0.0, 1 - residual.var() / variance))
            if variance > 1e-12 else float("nan")
        ),
        "dominant_period_days": dominant_period,
        "mean_second_to_first": (
            float(second.mean() / first.mean())
            if first.mean() > 1e-12 else float("nan")
        ),
        "variance_second_to_first": (
            float(second.var() / first.var()) if first.var() > 1e-12 else float("nan")
        ),
        "prior_weekday_r2": r2,
    }


def stage_diagnostics(panel: pd.DataFrame, *, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Describe eligible training series only; holdout days never affect diagnostics."""
    history = panel.loc[panel["date"].le(cutoff)].sort_values([*KEYS, "date"])
    rows = []
    for (bakery_id, product_id), group in history.groupby(KEYS, sort=False):
        if len(group) < 70 or group["observed_sales_qty"].sum() <= 0:
            continue
        dates = group["date"].to_numpy()
        for stage, column in STAGES.items():
            values = group[column].to_numpy(dtype=float)
            rows.append(
                {
                    "bakery_id": bakery_id,
                    "product_id": product_id,
                    "stage": stage,
                    "n_days": len(group),
                    **_stage_metrics(values, dates),
                }
            )
    return pd.DataFrame(rows)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import warnings

from src.experiments_v2.benchmark_common import BAKERY_COL, DATE_COL, DEMAND_TARGET, PRODUCT_COL
from statsmodels.tsa.stattools import adfuller, kpss


@dataclass(frozen=True)
class StabilityFeatureSet:
    acf_lag1: float
    acf_lag7: float
    trend_abs_60: float
    mean_shift_norm_30_60: float
    variance_shift_norm_7_30: float
    seasonal_strength_7: float
    zero_run_max: float
    zero_run_ratio: float


@dataclass(frozen=True)
class StationarityFeatureSet:
    adf_pvalue: float
    adf_stat: float
    kpss_pvalue: float
    kpss_stat: float
    stationary_adf: int
    stationary_kpss: int


def _safe_autocorr(values: np.ndarray, lag: int) -> float:
    if len(values) <= lag + 1:
        return 0.0
    s = pd.Series(values)
    val = s.autocorr(lag=lag)
    if pd.isna(val):
        return 0.0
    return float(np.clip(val, -1.0, 1.0))


def _max_zero_run(values: np.ndarray) -> int:
    max_run = 0
    current = 0
    for value in values:
        if value == 0:
            current += 1
            if current > max_run:
                max_run = current
        else:
            current = 0
    return max_run


def _split_windows(values: np.ndarray, recent_size: int, prior_size: int) -> tuple[np.ndarray, np.ndarray]:
    if len(values) < 4:
        return values, values

    recent = values[-recent_size:]
    prior = values[-(recent_size + prior_size) : -recent_size]

    if len(prior) >= 3 and len(recent) >= 3:
        return recent, prior

    half = len(values) // 2
    recent = values[half:]
    prior = values[:half]
    if len(prior) == 0:
        return values, values
    return recent, prior


def _normalized_level_shift(values: np.ndarray, recent_size: int = 30, prior_size: int = 30) -> float:
    recent, prior = _split_windows(values, recent_size, prior_size)
    recent_mean = float(np.mean(recent)) if len(recent) else 0.0
    prior_mean = float(np.mean(prior)) if len(prior) else 0.0
    denom = abs(prior_mean) + 1.0
    return float((recent_mean - prior_mean) / denom)


def _normalized_volatility_shift(values: np.ndarray, recent_size: int = 7, prior_size: int = 30) -> float:
    recent, prior = _split_windows(values, recent_size, prior_size)
    recent_std = float(np.std(recent, ddof=0)) if len(recent) else 0.0
    prior_std = float(np.std(prior, ddof=0)) if len(prior) else 0.0
    denom = prior_std + 1.0
    return float((recent_std - prior_std) / denom)


def _trend_abs_60(values: np.ndarray) -> float:
    if len(values) < 5:
        return 0.0
    recent = values[-60:]
    x = np.arange(len(recent), dtype=float)
    y = recent.astype(float)
    if np.nanstd(y) < 1e-8:
        return 0.0
    slope = np.polyfit(x, y, 1)[0]
    scale = abs(float(np.nanmean(y))) + 1.0
    return float(abs(slope) / scale)


def _seasonal_strength_7(group: pd.DataFrame, value_col: str, dow_col: str) -> float:
    if dow_col not in group.columns or len(group) < 8:
        return 0.0

    work = group[[dow_col, value_col]].dropna().copy()
    if work.empty:
        return 0.0

    overall_std = float(work[value_col].std(ddof=0))
    if overall_std < 1e-8:
        return 0.0

    dow_means = work.groupby(dow_col)[value_col].mean()
    return float(np.clip(dow_means.std(ddof=0) / overall_std, 0.0, 5.0))


def _safe_adf(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 8 or np.nanstd(values) < 1e-8:
        return 1.0, 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            stat, pvalue, *_ = adfuller(values, autolag="AIC")
            return float(pvalue), float(stat)
        except Exception:
            return 1.0, 0.0


def _safe_kpss(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 8 or np.nanstd(values) < 1e-8:
        return 0.1, 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            stat, pvalue, *_ = kpss(values, regression="c", nlags="auto")
            return float(pvalue), float(stat)
        except Exception:
            return 0.1, 0.0


def build_sku_stability_features(
    df: pd.DataFrame,
    bakery_col: str = BAKERY_COL,
    product_col: str = PRODUCT_COL,
    date_col: str = DATE_COL,
    value_col: str = DEMAND_TARGET,
    dow_col: str = "ДеньНедели",
) -> pd.DataFrame:
    """
    Build simple stability and stationarity-proxy features per bakery-product SKU.

    The goal is not a formal stationarity verdict. Instead, these are lightweight
    regime descriptors that can be used for routing and for grouping problematic SKU.
    """

    cols = [bakery_col, product_col, date_col, value_col]
    if dow_col in df.columns:
        cols.append(dow_col)

    work = df[cols].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[bakery_col, product_col, date_col]).sort_values([bakery_col, product_col, date_col])

    rows: list[dict[str, object]] = []
    for (bakery, product), group in work.groupby([bakery_col, product_col], sort=False):
        values = pd.to_numeric(group[value_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        n = len(values)
        zero_run_max = _max_zero_run(values)

        row = {
            bakery_col: bakery,
            product_col: product,
            "acf_lag1": _safe_autocorr(values, 1),
            "acf_lag7": _safe_autocorr(values, 7),
            "trend_abs_60": _trend_abs_60(values),
            "mean_shift_norm_30_60": abs(_normalized_level_shift(values, recent_size=30, prior_size=30)),
            "variance_shift_norm_7_30": abs(_normalized_volatility_shift(values, recent_size=7, prior_size=30)),
            "seasonal_strength_7": _seasonal_strength_7(group, value_col=value_col, dow_col=dow_col),
            "zero_run_max": float(zero_run_max),
            "zero_run_ratio": float(zero_run_max / max(n, 1)),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def build_sku_stationarity_features(
    df: pd.DataFrame,
    bakery_col: str = BAKERY_COL,
    product_col: str = PRODUCT_COL,
    date_col: str = DATE_COL,
    value_col: str = DEMAND_TARGET,
) -> pd.DataFrame:
    """
    Build formal stationarity-test proxies per bakery-product SKU.

    ADF and KPSS are not treated as hard labels here. They are diagnostic
    features that can be used together with stability proxies.
    """

    work = df[[bakery_col, product_col, date_col, value_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[bakery_col, product_col, date_col]).sort_values([bakery_col, product_col, date_col])

    rows: list[dict[str, object]] = []
    for (bakery, product), group in work.groupby([bakery_col, product_col], sort=False):
        values = pd.to_numeric(group[value_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        adf_pvalue, adf_stat = _safe_adf(values)
        kpss_pvalue, kpss_stat = _safe_kpss(values)

        rows.append(
            {
                bakery_col: bakery,
                product_col: product,
                "adf_pvalue": adf_pvalue,
                "adf_stat": adf_stat,
                "kpss_pvalue": kpss_pvalue,
                "kpss_stat": kpss_stat,
                "stationary_adf": int(adf_pvalue < 0.05),
                "stationary_kpss": int(kpss_pvalue > 0.05),
            }
        )

    return pd.DataFrame(rows)

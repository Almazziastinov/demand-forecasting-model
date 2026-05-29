from __future__ import annotations

import numpy as np
import pandas as pd


def add_robust_sales_outlier_flags(
    df: pd.DataFrame,
    *,
    value_col: str,
    entity_cols: list[str],
    seasonal_cols: list[str] | None = None,
    min_seasonal_rows: int = 8,
    robust_z_threshold: float = 3.5,
    high_ratio_threshold: float = 2.0,
) -> pd.DataFrame:
    """Add robust high-outlier flags using entity/seasonal medians.

    This function marks suspicious observations but does not mutate the target.
    It first estimates an entity + seasonal baseline, then falls back to the
    entity-level baseline when the seasonal bucket is thin.
    """
    work = df.copy()
    seasonal_cols = seasonal_cols or []
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce").fillna(0.0)

    entity_stats = _robust_stats(
        work, value_col=value_col, group_cols=entity_cols, prefix="_entity"
    )
    work = work.merge(entity_stats, on=entity_cols, how="left")

    if seasonal_cols:
        seasonal_keys = entity_cols + seasonal_cols
        seasonal_stats = _robust_stats(
            work, value_col=value_col, group_cols=seasonal_keys, prefix="_seasonal"
        )
        work = work.merge(seasonal_stats, on=seasonal_keys, how="left")
        use_seasonal = work["_seasonal_count"].fillna(0) >= min_seasonal_rows
        work["expected_base_qty"] = np.where(
            use_seasonal, work["_seasonal_median"], work["_entity_median"]
        )
        work["expected_base_scale"] = np.where(
            use_seasonal, work["_seasonal_scale"], work["_entity_scale"]
        )
        work["expected_base_rows"] = np.where(
            use_seasonal, work["_seasonal_count"], work["_entity_count"]
        )
        work["expected_base_source"] = np.where(use_seasonal, "seasonal", "entity")
    else:
        work["expected_base_qty"] = work["_entity_median"]
        work["expected_base_scale"] = work["_entity_scale"]
        work["expected_base_rows"] = work["_entity_count"]
        work["expected_base_source"] = "entity"

    scale = work["expected_base_scale"].replace(0, np.nan)
    work["robust_sales_z"] = (
        (work[value_col] - work["expected_base_qty"]) / scale
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    ratio_denominator = work["expected_base_qty"].replace(0, np.nan)
    work["sales_to_expected_ratio"] = (work[value_col] / ratio_denominator).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    work["sales_high_outlier_flag"] = (
        (work[value_col] > work["expected_base_qty"])
        & (
            (work["robust_sales_z"] >= robust_z_threshold)
            | (work["sales_to_expected_ratio"] >= high_ratio_threshold)
        )
    ).astype(int)
    work["sales_low_outlier_flag"] = (
        (work[value_col] < work["expected_base_qty"])
        & (work["robust_sales_z"] <= -robust_z_threshold)
    ).astype(int)
    return work.drop(
        columns=[
            c
            for c in work.columns
            if c.startswith("_entity_") or c.startswith("_seasonal_")
        ]
    )


def add_base_training_policy_flags(
    df: pd.DataFrame,
    *,
    contextual_flag_cols: list[str] | None = None,
    missing_flag_col: str = "sales_missing_flag",
    imputed_sample_weight: float = 0.25,
) -> pd.DataFrame:
    """Separate baseline-training treatment from contextual correction signals."""
    work = df.copy()
    contextual_flag_cols = contextual_flag_cols or []
    contextual_mask = pd.Series(False, index=work.index)
    for col in contextual_flag_cols:
        if col in work.columns:
            contextual_mask = contextual_mask | (
                pd.to_numeric(work[col], errors="coerce").fillna(0) == 1
            )

    high_outlier = (
        pd.to_numeric(work.get("sales_high_outlier_flag", 0), errors="coerce").fillna(0)
        == 1
    )
    low_outlier = (
        pd.to_numeric(work.get("sales_low_outlier_flag", 0), errors="coerce").fillna(0)
        == 1
    )

    work["contextual_high_outlier_flag"] = (high_outlier & contextual_mask).astype(int)
    work["unexplained_high_outlier_flag"] = (high_outlier & ~contextual_mask).astype(
        int
    )
    work["base_model_downweight_flag"] = (high_outlier | low_outlier).astype(int)
    work["correction_candidate_flag"] = (high_outlier & contextual_mask).astype(int)
    work["base_model_sample_weight"] = np.select(
        [
            work["unexplained_high_outlier_flag"] == 1,
            work["contextual_high_outlier_flag"] == 1,
            low_outlier,
        ],
        [
            0.35,
            0.60,
            0.75,
        ],
        default=1.0,
    )
    if missing_flag_col in work.columns:
        missing_mask = (
            pd.to_numeric(work[missing_flag_col], errors="coerce").fillna(0).astype(int)
            == 1
        )
        work.loc[missing_mask, "base_model_sample_weight"] = np.minimum(
            work.loc[missing_mask, "base_model_sample_weight"],
            imputed_sample_weight,
        )
    return work


def add_capped_base_target(
    df: pd.DataFrame,
    *,
    value_col: str,
    capped_col: str | None = None,
    upper_multiplier: float = 1.5,
    lower_multiplier: float = 0.4,
    cap_contextual_high_outliers: bool = False,
) -> pd.DataFrame:
    """Create a base-training target with residual unexplained outliers capped.

    The cap uses the robust expected baseline from
    `add_robust_sales_outlier_flags`. By default, high outliers with event or
    other context are not capped; they are left to the correction layer and only
    downweighted for the base model.
    """
    work = df.copy()
    capped_col = capped_col or f"{value_col}_base_capped"
    observed = pd.to_numeric(work[value_col], errors="coerce").fillna(0.0)
    expected = pd.to_numeric(
        work.get("expected_base_qty", observed), errors="coerce"
    ).fillna(observed)
    upper_cap = (expected * upper_multiplier).clip(lower=0.0)
    lower_cap = (expected * lower_multiplier).clip(lower=0.0)

    high_flag = (
        pd.to_numeric(work.get("sales_high_outlier_flag", 0), errors="coerce")
        .fillna(0)
        .astype(int)
        == 1
    )
    low_flag = (
        pd.to_numeric(work.get("sales_low_outlier_flag", 0), errors="coerce")
        .fillna(0)
        .astype(int)
        == 1
    )
    contextual_high = (
        pd.to_numeric(work.get("contextual_high_outlier_flag", 0), errors="coerce")
        .fillna(0)
        .astype(int)
        == 1
    )
    unexplained_high = (
        pd.to_numeric(work.get("unexplained_high_outlier_flag", 0), errors="coerce")
        .fillna(0)
        .astype(int)
        == 1
    )

    high_cap_mask = unexplained_high | (
        high_flag & (~contextual_high | cap_contextual_high_outliers)
    )
    capped = observed.copy()
    capped = np.where(high_cap_mask, np.minimum(capped, upper_cap), capped)
    capped = np.where(low_flag, np.maximum(capped, lower_cap), capped)

    work[capped_col] = np.asarray(capped, dtype=float)
    work["base_target_cap_upper"] = upper_cap
    work["base_target_cap_lower"] = lower_cap
    work["base_target_capped_flag"] = (
        np.abs(work[capped_col] - observed) > 1e-9
    ).astype(int)
    work["base_target_cap_delta"] = work[capped_col] - observed
    return work


def add_quantile_capped_base_target(
    df: pd.DataFrame,
    *,
    value_col: str,
    entity_cols: list[str],
    seasonal_cols: list[str] | None = None,
    capped_col: str | None = None,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    min_seasonal_rows: int = 8,
    cap_contextual_high_outliers: bool = False,
) -> pd.DataFrame:
    """Create a base target clipped by entity/weekday quantiles.

    The primary cap is estimated by `entity_cols + seasonal_cols`, usually
    `bakery_id + dow`. If the seasonal bucket is thin, the cap falls back to the
    entity-level quantiles. Contextual high outliers are preserved by default.
    """
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("Expected 0 <= lower_quantile < upper_quantile <= 1")

    work = df.copy()
    seasonal_cols = seasonal_cols or []
    capped_col = capped_col or f"{value_col}_base_quantile_capped"
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce").fillna(0.0)

    entity_caps = _quantile_caps(
        work,
        value_col=value_col,
        group_cols=entity_cols,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        prefix="_entity_q",
    )
    work = work.merge(entity_caps, on=entity_cols, how="left")

    if seasonal_cols:
        seasonal_keys = entity_cols + seasonal_cols
        seasonal_caps = _quantile_caps(
            work,
            value_col=value_col,
            group_cols=seasonal_keys,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
            prefix="_seasonal_q",
        )
        work = work.merge(seasonal_caps, on=seasonal_keys, how="left")
        use_seasonal = work["_seasonal_q_count"].fillna(0) >= min_seasonal_rows
        work["quantile_cap_lower"] = np.where(
            use_seasonal, work["_seasonal_q_lower"], work["_entity_q_lower"]
        )
        work["quantile_cap_upper"] = np.where(
            use_seasonal, work["_seasonal_q_upper"], work["_entity_q_upper"]
        )
        work["quantile_cap_rows"] = np.where(
            use_seasonal, work["_seasonal_q_count"], work["_entity_q_count"]
        )
        work["quantile_cap_source"] = np.where(use_seasonal, "seasonal", "entity")
    else:
        work["quantile_cap_lower"] = work["_entity_q_lower"]
        work["quantile_cap_upper"] = work["_entity_q_upper"]
        work["quantile_cap_rows"] = work["_entity_q_count"]
        work["quantile_cap_source"] = "entity"

    if "contextual_high_outlier_flag" in work.columns:
        contextual_high = (
            pd.to_numeric(work["contextual_high_outlier_flag"], errors="coerce")
            .fillna(0)
            .astype(int)
            == 1
        )
    else:
        contextual_high = pd.Series(False, index=work.index)
    high_preserve_mask = contextual_high & (not cap_contextual_high_outliers)

    capped = work[value_col].clip(
        lower=work["quantile_cap_lower"], upper=work["quantile_cap_upper"]
    )
    capped = np.where(high_preserve_mask, work[value_col], capped)

    work[capped_col] = np.asarray(capped, dtype=float)
    work["quantile_base_target_capped_flag"] = (
        np.abs(work[capped_col] - work[value_col]) > 1e-9
    ).astype(int)
    work["quantile_base_target_cap_delta"] = work[capped_col] - work[value_col]
    return work.drop(
        columns=[
            c
            for c in work.columns
            if c.startswith("_entity_q_") or c.startswith("_seasonal_q_")
        ]
    )


def add_rolling_median_capped_base_target(
    df: pd.DataFrame,
    *,
    value_col: str,
    entity_cols: list[str],
    date_col: str = "date",
    seasonal_cols: list[str] | None = None,
    capped_col: str | None = None,
    window: int = 8,
    min_periods: int = 4,
    upper_multiplier: float = 1.6,
    lower_multiplier: float = 0.5,
    cap_contextual_high_outliers: bool = False,
) -> pd.DataFrame:
    """Create a base target capped against trailing rolling median.

    The baseline is computed inside `entity_cols + seasonal_cols`, typically
    `bakery_id + dow`, sorted by date and shifted by one row. This prevents
    future information from defining today's cap and lets the cap follow trend.
    """
    if window < 1:
        raise ValueError("window must be positive")
    if min_periods < 1:
        raise ValueError("min_periods must be positive")
    if lower_multiplier < 0 or upper_multiplier <= 0:
        raise ValueError("multipliers must be positive")
    if lower_multiplier >= upper_multiplier:
        raise ValueError("lower_multiplier must be < upper_multiplier")

    work = df.copy()
    seasonal_cols = seasonal_cols or []
    capped_col = capped_col or f"{value_col}_base_rolling_capped"
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce").fillna(0.0)
    work["_original_order"] = range(len(work))

    group_cols = entity_cols + seasonal_cols
    work = work.sort_values(group_cols + [date_col]).reset_index(drop=True)
    grouped = work.groupby(group_cols, dropna=False)[value_col]

    work["rolling_base_median"] = grouped.transform(
        lambda s: s.shift(1).rolling(window=window, min_periods=min_periods).median()
    )
    work["rolling_base_rows"] = grouped.transform(
        lambda s: s.shift(1).rolling(window=window, min_periods=1).count()
    )
    fallback = grouped.transform(
        lambda s: s.shift(1).expanding(min_periods=min_periods).median()
    )
    work["rolling_base_median"] = work["rolling_base_median"].fillna(fallback)
    work["rolling_cap_source"] = "rolling_same_bucket"
    work.loc[work["rolling_base_median"].isna(), "rolling_cap_source"] = (
        "uncapped_insufficient_history"
    )

    work["rolling_cap_lower"] = (work["rolling_base_median"] * lower_multiplier).clip(
        lower=0.0
    )
    work["rolling_cap_upper"] = (work["rolling_base_median"] * upper_multiplier).clip(
        lower=0.0
    )

    contextual_high = _bool_series(work, "contextual_high_outlier_flag")
    high_preserve_mask = contextual_high & (not cap_contextual_high_outliers)
    capped = work[value_col].astype(float).copy()
    has_caps = work["rolling_base_median"].notna()
    clipped = capped.clip(
        lower=work["rolling_cap_lower"], upper=work["rolling_cap_upper"]
    )
    capped = clipped.where(has_caps & ~high_preserve_mask, capped)

    work[capped_col] = capped.astype(float)
    work["rolling_base_target_capped_flag"] = (
        (work[capped_col] - work[value_col]).abs() > 1e-9
    ).astype(int)
    work["rolling_base_target_cap_delta"] = work[capped_col] - work[value_col]
    work = work.sort_values("_original_order").drop(columns=["_original_order"])
    return work.reset_index(drop=True)


def add_rolling_quantile_capped_base_target(
    df: pd.DataFrame,
    *,
    value_col: str,
    entity_cols: list[str],
    date_col: str = "date",
    seasonal_cols: list[str] | None = None,
    capped_col: str | None = None,
    window: int = 26,
    min_periods: int = 8,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    cap_contextual_high_outliers: bool = False,
) -> pd.DataFrame:
    """Create a base target clipped by trailing rolling quantiles.

    Unlike `add_quantile_capped_base_target`, this version uses only the
    trailing history of each `entity + seasonal` bucket. The cap moves with
    trend and slow seasonality, while a long window (default 26 same-dow
    observations ~ 6 months) keeps the quantile estimate stable.

    Sorted by date, shifted by one row so today's value never defines its own
    cap. Falls back to expanding quantile when the rolling window is thin.
    """
    if window < 1:
        raise ValueError("window must be positive")
    if min_periods < 1:
        raise ValueError("min_periods must be positive")
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("Expected 0 <= lower_quantile < upper_quantile <= 1")

    work = df.copy()
    seasonal_cols = seasonal_cols or []
    capped_col = capped_col or f"{value_col}_base_rolling_quantile_capped"
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce").fillna(0.0)
    work["_original_order"] = range(len(work))

    group_cols = entity_cols + seasonal_cols
    work = work.sort_values(group_cols + [date_col]).reset_index(drop=True)
    grouped = work.groupby(group_cols, dropna=False)[value_col]

    work["rolling_q_lower"] = grouped.transform(
        lambda s: s.shift(1)
        .rolling(window=window, min_periods=min_periods)
        .quantile(lower_quantile)
    )
    work["rolling_q_upper"] = grouped.transform(
        lambda s: s.shift(1)
        .rolling(window=window, min_periods=min_periods)
        .quantile(upper_quantile)
    )
    work["rolling_q_rows"] = grouped.transform(
        lambda s: s.shift(1).rolling(window=window, min_periods=1).count()
    )
    fallback_lower = grouped.transform(
        lambda s: s.shift(1).expanding(min_periods=min_periods).quantile(lower_quantile)
    )
    fallback_upper = grouped.transform(
        lambda s: s.shift(1).expanding(min_periods=min_periods).quantile(upper_quantile)
    )
    work["rolling_q_lower"] = work["rolling_q_lower"].fillna(fallback_lower)
    work["rolling_q_upper"] = work["rolling_q_upper"].fillna(fallback_upper)

    work["rolling_q_cap_source"] = "rolling_same_bucket"
    work.loc[work["rolling_q_upper"].isna(), "rolling_q_cap_source"] = (
        "uncapped_insufficient_history"
    )

    has_caps = work["rolling_q_lower"].notna() & work["rolling_q_upper"].notna()
    contextual_high = _bool_series(work, "contextual_high_outlier_flag")
    high_preserve_mask = contextual_high & (not cap_contextual_high_outliers)

    capped = work[value_col].astype(float).copy()
    clipped = capped.clip(
        lower=work["rolling_q_lower"].clip(lower=0.0),
        upper=work["rolling_q_upper"].clip(lower=0.0),
    )
    capped = clipped.where(has_caps & ~high_preserve_mask, capped)

    work[capped_col] = capped.astype(float)
    work["rolling_quantile_base_target_capped_flag"] = (
        (work[capped_col] - work[value_col]).abs() > 1e-9
    ).astype(int)
    work["rolling_quantile_base_target_cap_delta"] = (
        work[capped_col] - work[value_col]
    )
    work = work.sort_values("_original_order").drop(columns=["_original_order"])
    return work.reset_index(drop=True)


def _bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int) == 1


def _robust_stats(
    df: pd.DataFrame,
    *,
    value_col: str,
    group_cols: list[str],
    prefix: str,
) -> pd.DataFrame:
    grouped = df.groupby(group_cols, dropna=False)[value_col]
    stats = grouped.agg(["count", "median"]).reset_index()
    stats = stats.rename(
        columns={"count": f"{prefix}_count", "median": f"{prefix}_median"}
    )

    abs_dev = df[group_cols + [value_col]].merge(
        stats[group_cols + [f"{prefix}_median"]], on=group_cols, how="left"
    )
    abs_dev["_abs_dev"] = (abs_dev[value_col] - abs_dev[f"{prefix}_median"]).abs()
    mad = (
        abs_dev.groupby(group_cols, dropna=False)["_abs_dev"]
        .median()
        .rename(f"{prefix}_mad")
        .reset_index()
    )
    stats = stats.merge(mad, on=group_cols, how="left")

    stats[f"{prefix}_scale"] = 1.4826 * stats[f"{prefix}_mad"]
    fallback_scale = stats[f"{prefix}_median"].abs().clip(lower=1.0) * 0.10
    stats[f"{prefix}_scale"] = (
        stats[f"{prefix}_scale"].replace(0, np.nan).fillna(fallback_scale)
    )
    return stats


def _quantile_caps(
    df: pd.DataFrame,
    *,
    value_col: str,
    group_cols: list[str],
    lower_quantile: float,
    upper_quantile: float,
    prefix: str,
) -> pd.DataFrame:
    grouped = df.groupby(group_cols, dropna=False)[value_col]
    caps = grouped.quantile([lower_quantile, upper_quantile]).unstack()
    caps = caps.rename(
        columns={lower_quantile: f"{prefix}_lower", upper_quantile: f"{prefix}_upper"}
    )
    counts = grouped.size().rename(f"{prefix}_count")
    result = caps.join(counts).reset_index()
    return result

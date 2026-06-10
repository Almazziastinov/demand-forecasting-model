"""Evaluate post-allocation SKU share correction variants on holdout data.

The script starts from the production-style SKU holdout compare file built by
scripts/build_prod_holdout_sku_backtest.py. It does not retrain the bakery
model. Each variant changes only how the bakery-day forecast is distributed
across SKU for a bakery-day, then renormalizes to preserve the original
bakery-day forecast total.

Recent assortment statistics are queried from mart_sales_60d for the period
before the holdout window, so the experiment does not use holdout facts to build
the correction.
"""

from __future__ import annotations

# ruff: noqa: E402,E501

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipelines.forecast_publish.load_forecast_run import DEFAULT_ENV_PATH
from pipelines.forecast_publish.load_forecast_run import create_client


DEFAULT_COMPARE_PATH = (
    REPO_ROOT / "reports" / "prod_holdout_sku_backtest" / "prod_holdout_sku_compare.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "prod_holdout_sku_backtest_variants"
DEFAULT_START_DATE = "2026-05-02"
DEFAULT_END_DATE = "2026-05-31"
DEFAULT_RECENT_DAYS = 30
SALES_LINE_TABLE = "mart_sales_60d"
ECLAIR_PATTERN = "эклер"
SERVICE_CATEGORY_PATTERN = "прочие|заказ"


def _is_eclair(work: pd.DataFrame) -> pd.Series:
    return (
        work["product_name"]
        .fillna("")
        .astype(str)
        .str.casefold()
        .str.contains(ECLAIR_PATTERN, regex=False)
    )


def _is_service(work: pd.DataFrame) -> pd.Series:
    return (
        work["category_name"]
        .fillna("")
        .astype(str)
        .str.casefold()
        .str.contains(SERVICE_CATEGORY_PATTERN, regex=True)
    )


def _prod_share(base: pd.Series, base_total: pd.Series) -> np.ndarray:
    return np.where(base_total > 0, base / base_total, 0.0)


def _core_runner_mask(work: pd.DataFrame, *, min_share: float = 0.01) -> pd.Series:
    return (
        (work["recent_days_sold"] >= 20)
        & (work["recent_share"] >= min_share)
        & ~_is_service(work)
    )


def _topn_runner_mask(work: pd.DataFrame, *, top_n: int = 20) -> pd.Series:
    ranks = work.groupby(["date", "bakery_id"])["recent_qty"].rank(method="first", ascending=False)
    return (
        (work["recent_days_sold"] >= 20)
        & ((work["recent_share"] >= 0.005) | (ranks <= top_n))
        & ~_is_service(work)
    )


def _city_top_runner_mask(work: pd.DataFrame, *, max_rank: int = 5) -> pd.Series:
    return (
        (work["recent_days_sold"] >= 10)
        & (work["city_recent_rank"] <= max_rank)
        & (work["city_recent_share"] >= 0.015)
        & ~_is_service(work)
    )


def _adaptive_recent_raw(work: pd.DataFrame, base: pd.Series, base_total: pd.Series) -> pd.Series:
    recent_share_qty = work["recent_share"] * base_total
    recent_share = pd.to_numeric(work["recent_share"], errors="coerce").fillna(0.0)
    prod_share = _prod_share(base, base_total)
    core = _core_runner_mask(work)
    profile_too_high = prod_share > recent_share * 2.0
    profile_too_low = prod_share < recent_share * 0.5

    raw = 0.3 * base + 0.7 * recent_share_qty
    raw = pd.Series(np.where(core, 0.5 * base + 0.5 * recent_share_qty, raw), index=work.index)
    raw = pd.Series(np.where(core & profile_too_high, 0.2 * base + 0.8 * recent_share_qty, raw), index=work.index)
    raw = pd.Series(np.where(core & profile_too_low, 0.3 * base + 0.7 * recent_share_qty, raw), index=work.index)

    eclair_recent_cap = recent_share_qty * 1.3
    eclair_recent_heavy = 0.2 * base + 0.8 * recent_share_qty
    raw = pd.Series(
        np.where(_is_eclair(work), np.minimum(eclair_recent_cap, eclair_recent_heavy), raw),
        index=work.index,
    )
    return raw


def _apply_category_preserving_raw(
    work: pd.DataFrame,
    base: pd.Series,
    base_total: pd.Series,
) -> pd.Series:
    raw = _adaptive_recent_raw(work, base, base_total)
    temp = work[["date", "bakery_id", "category_name"]].copy()
    temp["_base"] = base
    temp["_raw"] = raw
    temp["_recent_qty"] = work["recent_qty"]
    temp["_bakery_total"] = base_total

    category = (
        temp.groupby(["date", "bakery_id", "category_name"], as_index=False, dropna=False)
        .agg(
            category_base=("_base", "sum"),
            category_raw=("_raw", "sum"),
            category_recent_qty=("_recent_qty", "sum"),
            bakery_total=("_bakery_total", "max"),
        )
    )
    bakery_recent = (
        temp.groupby(["date", "bakery_id"], as_index=False)["_recent_qty"]
        .sum()
        .rename(columns={"_recent_qty": "bakery_recent_qty"})
    )
    category = category.merge(bakery_recent, on=["date", "bakery_id"], how="left")
    category["category_base_share"] = np.where(
        category["bakery_total"] > 0,
        category["category_base"] / category["bakery_total"],
        0.0,
    )
    category["category_recent_share"] = np.where(
        category["bakery_recent_qty"] > 0,
        category["category_recent_qty"] / category["bakery_recent_qty"],
        category["category_base_share"],
    )
    category["category_target"] = category["bakery_total"] * (
        0.3 * category["category_base_share"] + 0.7 * category["category_recent_share"]
    )
    category["category_factor"] = np.where(
        category["category_raw"] > 0,
        category["category_target"] / category["category_raw"],
        1.0,
    )
    temp = temp.merge(
        category[["date", "bakery_id", "category_name", "category_factor"]],
        on=["date", "bakery_id", "category_name"],
        how="left",
    )
    return raw * temp["category_factor"].fillna(1.0)


def _apply_runner_residual(
    work: pd.DataFrame,
    base: pd.Series,
    base_total: pd.Series,
) -> pd.Series:
    raw = _adaptive_recent_raw(work, base, base_total)
    recent_share_qty = work["recent_share"] * base_total
    prod_share = _prod_share(base, base_total)
    recent_share = pd.to_numeric(work["recent_share"], errors="coerce").fillna(0.0)
    runner = _topn_runner_mask(work)
    over_profile = (work["recent_days_sold"] >= 20) & (prod_share > recent_share * 2.0)

    floor = pd.Series(0.0, index=work.index)
    floor = pd.Series(np.where(runner, recent_share_qty * 0.9, floor), index=work.index)
    cap = pd.Series(np.inf, index=work.index)
    cap = pd.Series(np.where(_is_eclair(work), recent_share_qty * 1.3, cap), index=work.index)
    cap = pd.Series(np.where(over_profile & ~runner, recent_share_qty * 1.3, cap), index=work.index)

    adjusted = raw.clip(lower=floor)
    adjusted = np.minimum(adjusted, cap)
    adjusted = pd.Series(adjusted, index=work.index)

    fixed = ((floor > 0) & (adjusted <= floor + 1e-9)) | (
        np.isfinite(cap) & (adjusted >= cap - 1e-9)
    )
    temp = work[["date", "bakery_id"]].copy()
    temp["_adjusted"] = adjusted
    temp["_base_total"] = base_total
    temp["_is_free"] = ~fixed
    temp["_free_raw"] = np.where(temp["_is_free"], adjusted, 0.0)
    grouped = (
        temp.groupby(["date", "bakery_id"], as_index=False)
        .agg(
            adjusted_sum=("_adjusted", "sum"),
            free_sum=("_free_raw", "sum"),
            base_total=("_base_total", "max"),
        )
    )
    temp = temp.merge(grouped, on=["date", "bakery_id"], how="left")
    residual = (temp["base_total"] - temp["adjusted_sum"]).clip(lower=0.0)
    free_factor = np.where(temp["free_sum"] > 0, residual / temp["free_sum"], 1.0)
    return pd.Series(
        np.where(temp["_is_free"], adjusted * free_factor, adjusted),
        index=work.index,
    )


def _with_recent_model(work: pd.DataFrame, mode: str) -> pd.DataFrame:
    modeled = work.copy()
    overall = pd.to_numeric(
        modeled["recent_share_daily_winsor"],
        errors="coerce",
    ).fillna(modeled["recent_share"])
    weekpart = pd.to_numeric(
        modeled["recent_share_weekpart_winsor"],
        errors="coerce",
    ).fillna(overall)
    dow = pd.to_numeric(
        modeled["recent_share_dow_winsor"],
        errors="coerce",
    ).fillna(weekpart)

    if mode == "weekpart":
        week_obs = modeled["recent_weekpart_obs"].clip(lower=0, upper=12) / 12.0
        alpha = np.minimum(0.6, 0.6 * week_obs)
        modeled["recent_share"] = alpha * weekpart + (1.0 - alpha) * overall
    elif mode == "dow_shrink":
        dow_obs = modeled["recent_dow_obs"].clip(lower=0, upper=5) / 5.0
        week_obs = modeled["recent_weekpart_obs"].clip(lower=0, upper=12) / 12.0
        dow_weight = np.minimum(0.35, 0.35 * dow_obs)
        week_weight = np.minimum(0.45, 0.45 * week_obs)
        total = dow_weight + week_weight
        overflow = total > 0.8
        dow_weight = np.where(overflow, dow_weight / total * 0.8, dow_weight)
        week_weight = np.where(overflow, week_weight / total * 0.8, week_weight)
        overall_weight = 1.0 - dow_weight - week_weight
        modeled["recent_share"] = dow_weight * dow + week_weight * weekpart + overall_weight * overall
    elif mode == "robust_weekpart":
        raw_overall = pd.to_numeric(
            modeled["recent_share_daily_mean"],
            errors="coerce",
        ).fillna(modeled["recent_share"])
        raw_weekpart = pd.to_numeric(
            modeled["recent_share_weekpart"],
            errors="coerce",
        ).fillna(raw_overall)
        robust_overall = 0.5 * overall + 0.5 * raw_overall
        robust_weekpart = 0.5 * weekpart + 0.5 * raw_weekpart
        week_obs = modeled["recent_weekpart_obs"].clip(lower=0, upper=12) / 12.0
        alpha = np.minimum(0.7, 0.7 * week_obs)
        modeled["recent_share"] = alpha * robust_weekpart + (1.0 - alpha) * robust_overall
    else:
        raise ValueError(f"Unknown recent model: {mode}")

    modeled["recent_share"] = pd.to_numeric(
        modeled["recent_share"],
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)
    return modeled


def _attach_city_recent_prior(work: pd.DataFrame) -> pd.DataFrame:
    city_product = (
        work[["city", "bakery_id", "product_id", "recent_qty"]]
        .drop_duplicates(["city", "bakery_id", "product_id"])
        .groupby(["city", "product_id"], as_index=False, dropna=False)["recent_qty"]
        .sum()
    )
    city_total = (
        city_product.groupby("city", as_index=False, dropna=False)["recent_qty"]
        .sum()
        .rename(columns={"recent_qty": "city_recent_total_qty"})
    )
    city_product = city_product.merge(city_total, on="city", how="left", validate="many_to_one")
    city_product["city_recent_share"] = np.where(
        city_product["city_recent_total_qty"] > 0,
        city_product["recent_qty"] / city_product["city_recent_total_qty"],
        0.0,
    )
    city_product["city_recent_rank"] = city_product.groupby("city", dropna=False)[
        "city_recent_share"
    ].rank(method="first", ascending=False)
    return work.merge(
        city_product[
            [
                "city",
                "product_id",
                "city_recent_share",
                "city_recent_rank",
                "city_recent_total_qty",
            ]
        ],
        on=["city", "product_id"],
        how="left",
        validate="many_to_one",
    )


def _apply_city_prior_guard_raw(
    work: pd.DataFrame,
    base: pd.Series,
    base_total: pd.Series,
    *,
    strength: str,
) -> pd.Series:
    raw = _apply_variant(work, "runner_recent_heavy")["forecast_variant"]
    city_share = pd.to_numeric(work["city_recent_share"], errors="coerce").fillna(0.0)
    city_rank = pd.to_numeric(work["city_recent_rank"], errors="coerce").fillna(999.0)
    recent_share = pd.to_numeric(work["recent_share"], errors="coerce").fillna(0.0)
    city_runner = _city_top_runner_mask(work)

    if strength == "soft":
        rank_floor = np.select(
            [city_rank <= 1, city_rank <= 3, city_rank <= 5],
            [0.75, 0.65, 0.55],
            default=0.0,
        )
        local_weight = 0.30
    elif strength == "strong":
        rank_floor = np.select(
            [city_rank <= 1, city_rank <= 3, city_rank <= 5],
            [0.90, 0.80, 0.65],
            default=0.0,
        )
        local_weight = 0.20
    else:
        raise ValueError(f"Unknown city prior strength: {strength}")

    prior_share = np.maximum(
        recent_share,
        local_weight * recent_share + (1.0 - local_weight) * city_share * rank_floor,
    )
    guard_qty = prior_share * base_total
    guarded = np.where(city_runner, np.maximum(raw, guard_qty), raw)
    return pd.Series(guarded, index=work.index)


def _query_recent_stats(
    *,
    env_file: str | Path,
    recent_start: str,
    recent_end: str,
    table: str,
) -> pd.DataFrame:
    client = create_client(env_file)
    query = f"""
        select
            toInt64OrNull(toString(bakery_id)) as bakery_id,
            toInt64OrNull(toString(product_id)) as product_id,
            any(product_name) as recent_product_name,
            any(category_name) as recent_category_name,
            sum(toFloat64(quantity)) as recent_qty,
            uniqExact(check_date) as recent_days_sold
        from {table}
        where check_date between %(recent_start)s and %(recent_end)s
          and toInt64OrNull(toString(bakery_id)) is not null
          and toInt64OrNull(toString(product_id)) is not null
          and toFloat64(quantity) > 0
        group by bakery_id, product_id
    """
    stats = client.query_df(
        query,
        parameters={"recent_start": recent_start, "recent_end": recent_end},
    )
    bakery_totals = (
        stats.groupby("bakery_id", as_index=False)["recent_qty"]
        .sum()
        .rename(columns={"recent_qty": "bakery_recent_qty"})
    )
    stats = stats.merge(bakery_totals, on="bakery_id", how="left", validate="many_to_one")
    stats["recent_share"] = np.where(
        stats["bakery_recent_qty"] > 0,
        stats["recent_qty"] / stats["bakery_recent_qty"],
        0.0,
    )
    return stats


def _winsor_mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return 0.0
    if len(values) < 4:
        return float(values.mean())
    lower = values.quantile(0.10)
    upper = values.quantile(0.90)
    return float(values.clip(lower=lower, upper=upper).mean())


def _query_recent_daily_share_stats(
    *,
    env_file: str | Path,
    recent_start: str,
    recent_end: str,
    table: str,
) -> pd.DataFrame:
    client = create_client(env_file)
    query = f"""
        select
            check_date,
            toInt64OrNull(toString(bakery_id)) as bakery_id,
            toInt64OrNull(toString(product_id)) as product_id,
            sum(toFloat64(quantity)) as recent_day_qty
        from {table}
        where check_date between %(recent_start)s and %(recent_end)s
          and toInt64OrNull(toString(bakery_id)) is not null
          and toInt64OrNull(toString(product_id)) is not null
          and toFloat64(quantity) > 0
        group by check_date, bakery_id, product_id
    """
    daily = client.query_df(
        query,
        parameters={"recent_start": recent_start, "recent_end": recent_end},
    )
    if daily.empty:
        return pd.DataFrame(columns=["bakery_id", "product_id"])

    daily["check_date"] = pd.to_datetime(daily["check_date"], errors="coerce")
    daily = daily.dropna(subset=["check_date"])
    daily["recent_day_qty"] = pd.to_numeric(
        daily["recent_day_qty"],
        errors="coerce",
    ).fillna(0.0)

    bakery_days = (
        daily.groupby(["check_date", "bakery_id"], as_index=False)["recent_day_qty"]
        .sum()
        .rename(columns={"recent_day_qty": "bakery_recent_day_qty"})
    )
    pairs = daily[["bakery_id", "product_id"]].drop_duplicates()
    grid = bakery_days.merge(pairs, on="bakery_id", how="inner")
    daily = grid.merge(
        daily[["check_date", "bakery_id", "product_id", "recent_day_qty"]],
        on=["check_date", "bakery_id", "product_id"],
        how="left",
    )
    daily["recent_day_qty"] = pd.to_numeric(
        daily["recent_day_qty"],
        errors="coerce",
    ).fillna(0.0)
    daily["daily_share"] = np.where(
        daily["bakery_recent_day_qty"] > 0,
        daily["recent_day_qty"] / daily["bakery_recent_day_qty"],
        0.0,
    )
    daily["dow"] = daily["check_date"].dt.dayofweek
    daily["is_weekend"] = daily["dow"].isin([5, 6]).astype("int64")

    overall = (
        daily.groupby(["bakery_id", "product_id"], as_index=False)
        .agg(
            recent_share_daily_mean=("daily_share", "mean"),
            recent_share_daily_winsor=("daily_share", _winsor_mean),
            recent_daily_obs=("daily_share", "size"),
        )
    )
    weekpart = (
        daily.groupby(["bakery_id", "product_id", "is_weekend"], as_index=False)
        .agg(
            recent_share_weekpart=("daily_share", "mean"),
            recent_share_weekpart_winsor=("daily_share", _winsor_mean),
            recent_weekpart_obs=("daily_share", "size"),
        )
    )
    dow = (
        daily.groupby(["bakery_id", "product_id", "dow"], as_index=False)
        .agg(
            recent_share_dow=("daily_share", "mean"),
            recent_share_dow_winsor=("daily_share", _winsor_mean),
            recent_dow_obs=("daily_share", "size"),
        )
    )
    stats = overall.merge(weekpart, on=["bakery_id", "product_id"], how="left")
    stats = stats.merge(dow, on=["bakery_id", "product_id"], how="left")
    return stats


def _attach_recent(
    df: pd.DataFrame,
    recent: pd.DataFrame,
    recent_daily: pd.DataFrame | None = None,
) -> pd.DataFrame:
    work = df.merge(
        recent,
        on=["bakery_id", "product_id"],
        how="left",
        validate="many_to_one",
    )
    work["recent_qty"] = pd.to_numeric(work["recent_qty"], errors="coerce").fillna(0.0)
    work["recent_days_sold"] = (
        pd.to_numeric(work["recent_days_sold"], errors="coerce").fillna(0).astype("int64")
    )
    work["recent_share"] = pd.to_numeric(work["recent_share"], errors="coerce").fillna(0.0)
    work["_date_for_recent"] = pd.to_datetime(work["date"], errors="coerce")
    work["dow"] = work["_date_for_recent"].dt.dayofweek
    work["is_weekend"] = work["dow"].isin([5, 6]).astype("int64")
    if recent_daily is not None and not recent_daily.empty:
        daily_cols = [
            "bakery_id",
            "product_id",
            "is_weekend",
            "dow",
            "recent_share_daily_mean",
            "recent_share_daily_winsor",
            "recent_daily_obs",
            "recent_share_weekpart",
            "recent_share_weekpart_winsor",
            "recent_weekpart_obs",
            "recent_share_dow",
            "recent_share_dow_winsor",
            "recent_dow_obs",
        ]
        work = work.merge(
            recent_daily[daily_cols],
            on=["bakery_id", "product_id", "is_weekend", "dow"],
            how="left",
            validate="many_to_many",
        )
    for col in [
        "recent_share_daily_mean",
        "recent_share_daily_winsor",
        "recent_share_weekpart",
        "recent_share_weekpart_winsor",
        "recent_share_dow",
        "recent_share_dow_winsor",
    ]:
        if col not in work.columns:
            work[col] = work["recent_share"]
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(work["recent_share"])
    for col in ["recent_daily_obs", "recent_weekpart_obs", "recent_dow_obs"]:
        if col not in work.columns:
            work[col] = 0
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0).astype("int64")
    work = _attach_city_recent_prior(work)
    for col in ["city_recent_share", "city_recent_rank", "city_recent_total_qty"]:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
    return work


def _renormalize_preserving_bakery_day(
    work: pd.DataFrame,
    raw_col: str,
    out_col: str,
) -> pd.DataFrame:
    raw_sum = (
        work.groupby(["date", "bakery_id"], as_index=False)[raw_col]
        .sum()
        .rename(columns={raw_col: "_raw_variant_sum"})
    )
    work = work.merge(raw_sum, on=["date", "bakery_id"], how="left", validate="many_to_one")
    base_sum = pd.to_numeric(work["bakery_forecast_qty"], errors="coerce").fillna(0.0)
    raw_sum_series = pd.to_numeric(work["_raw_variant_sum"], errors="coerce").fillna(0.0)
    fallback = pd.to_numeric(work["forecast_qty"], errors="coerce").fillna(0.0)
    work[out_col] = np.where(
        raw_sum_series > 0,
        pd.to_numeric(work[raw_col], errors="coerce").fillna(0.0) / raw_sum_series * base_sum,
        fallback,
    )
    return work.drop(columns=["_raw_variant_sum"])


def _apply_variant(df: pd.DataFrame, name: str) -> pd.DataFrame:
    work = df.copy()
    base = pd.to_numeric(work["forecast_qty"], errors="coerce").fillna(0.0)
    base_total = pd.to_numeric(work["bakery_forecast_qty"], errors="coerce").fillna(0.0)
    recent_share_qty = work["recent_share"] * base_total

    if name == "baseline":
        work["forecast_variant"] = base
        return work

    if name == "dead_0d":
        active = work["recent_days_sold"] > 0
        work["_raw_variant"] = np.where(active, base, 0.0)
    elif name == "active_3d":
        active = (work["recent_days_sold"] >= 3) | (work["recent_qty"] >= 10)
        work["_raw_variant"] = np.where(active, base, 0.0)
    elif name == "blend_recent_50":
        active = work["recent_days_sold"] > 0
        blended = 0.5 * base + 0.5 * recent_share_qty
        work["_raw_variant"] = np.where(active, blended, 0.0)
    elif name == "core_recent_70":
        active = work["recent_days_sold"] > 0
        core = (work["recent_days_sold"] >= 20) & (work["recent_share"] >= 0.01)
        blended_core = 0.3 * base + 0.7 * recent_share_qty
        blended_regular = 0.7 * base + 0.3 * recent_share_qty
        work["_raw_variant"] = np.where(core, blended_core, blended_regular)
        work["_raw_variant"] = np.where(active, work["_raw_variant"], 0.0)
    elif name == "adaptive_recent":
        active = work["recent_days_sold"] > 0
        work["_raw_variant"] = _adaptive_recent_raw(work, base, base_total)
        work["_raw_variant"] = np.where(active, work["_raw_variant"], 0.0)
    elif name == "runner_floor_90":
        active = work["recent_days_sold"] > 0
        runner = _core_runner_mask(work, min_share=0.005)
        raw = _adaptive_recent_raw(work, base, base_total)
        raw = np.where(runner, np.maximum(raw, recent_share_qty * 0.9), raw)
        work["_raw_variant"] = np.where(active, raw, 0.0)
    elif name == "runner_recent_heavy":
        active = work["recent_days_sold"] > 0
        runner = _core_runner_mask(work, min_share=0.005)
        raw = _adaptive_recent_raw(work, base, base_total)
        runner_raw = np.maximum(0.15 * base + 0.85 * recent_share_qty, recent_share_qty * 0.9)
        raw = np.where(runner, runner_raw, raw)
        work["_raw_variant"] = np.where(active, raw, 0.0)
    elif name == "runner_recent_heavy_weekpart":
        return _apply_variant(_with_recent_model(work, "weekpart"), "runner_recent_heavy")
    elif name == "runner_city_prior_soft":
        active = work["recent_days_sold"] > 0
        raw = _apply_city_prior_guard_raw(work, base, base_total, strength="soft")
        work["_raw_variant"] = np.where(active, raw, 0.0)
    elif name == "runner_city_prior_strong":
        active = work["recent_days_sold"] > 0
        raw = _apply_city_prior_guard_raw(work, base, base_total, strength="strong")
        work["_raw_variant"] = np.where(active, raw, 0.0)
    elif name == "runner_city_prior_soft_weekpart":
        return _apply_variant(_with_recent_model(work, "weekpart"), "runner_city_prior_soft")
    elif name == "runner_city_prior_strong_weekpart":
        return _apply_variant(_with_recent_model(work, "weekpart"), "runner_city_prior_strong")
    elif name == "runner_recent_heavy_dow_shrink":
        return _apply_variant(_with_recent_model(work, "dow_shrink"), "runner_recent_heavy")
    elif name == "runner_recent_heavy_robust_weekpart":
        return _apply_variant(_with_recent_model(work, "robust_weekpart"), "runner_recent_heavy")
    elif name == "category_preserving":
        active = work["recent_days_sold"] > 0
        raw = _apply_category_preserving_raw(work, base, base_total)
        work["_raw_variant"] = np.where(active, raw, 0.0)
    elif name == "topn_runner_protection":
        active = work["recent_days_sold"] > 0
        runner = _topn_runner_mask(work)
        raw = _adaptive_recent_raw(work, base, base_total)
        runner_raw = np.maximum(0.15 * base + 0.85 * recent_share_qty, recent_share_qty * 0.9)
        raw = np.where(runner, runner_raw, raw)
        work["_raw_variant"] = np.where(active, raw, 0.0)
    elif name == "anti_cannibalization_cap":
        active = work["recent_days_sold"] > 0
        raw = _adaptive_recent_raw(work, base, base_total)
        prod_share = _prod_share(base, base_total)
        recent_share = pd.to_numeric(work["recent_share"], errors="coerce").fillna(0.0)
        over_profile = (work["recent_days_sold"] >= 20) & (prod_share > recent_share * 2.0)
        cap = recent_share_qty * 1.3
        raw = np.where(over_profile, np.minimum(raw, cap), raw)
        work["_raw_variant"] = np.where(active, raw, 0.0)
    elif name == "runner_residual_redistribution":
        active = work["recent_days_sold"] > 0
        raw = _apply_runner_residual(work, base, base_total)
        work["_raw_variant"] = np.where(active, raw, 0.0)
    else:
        raise ValueError(f"Unknown variant: {name}")

    work = _renormalize_preserving_bakery_day(work, "_raw_variant", "forecast_variant")
    return work.drop(columns=["_raw_variant"])


def _score_variant(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, dict]:
    work = df.copy()
    fact = pd.to_numeric(work["fact_qty"], errors="coerce").fillna(0.0)
    forecast = pd.to_numeric(work["forecast_variant"], errors="coerce").fillna(0.0)

    bakery_totals = (
        work.assign(forecast_variant=forecast, fact_qty=fact)
        .groupby(["date", "bakery_id"], as_index=False)
        .agg(
            bakery_fact_qty=("fact_qty", "sum"),
            bakery_forecast_variant_qty=("forecast_variant", "sum"),
        )
    )
    work = work.drop(columns=["bakery_fact_qty"], errors="ignore").merge(
        bakery_totals,
        on=["date", "bakery_id"],
        how="left",
        validate="many_to_one",
    )
    scale = np.where(
        work["bakery_fact_qty"] > 0,
        work["bakery_forecast_variant_qty"] / work["bakery_fact_qty"],
        1.0,
    )
    scaled_fact = fact * scale
    err_raw = forecast - fact
    err_scaled = forecast - scaled_fact

    work["forecast_variant"] = forecast
    work["fact_scaled_to_variant_total"] = scaled_fact
    work["err_raw_fact"] = err_raw
    work["abs_err_raw_fact"] = np.abs(err_raw)
    work["err_scaled_fact"] = err_scaled
    work["abs_err_scaled_fact"] = np.abs(err_scaled)
    work["cell_type_variant"] = np.select(
        [
            (forecast > 0) & (fact > 0),
            (forecast > 0) & (fact <= 0),
            (forecast <= 0) & (fact > 0),
        ],
        ["both_positive", "forecast_only_fact_zero", "fact_only_forecast_zero"],
        default="both_zero",
    )

    pair_totals = (
        work.groupby(["bakery_id", "product_id"], as_index=False)
        .agg(
            pair_window_fact_qty=("fact_qty", "sum"),
            pair_window_forecast_variant_qty=("forecast_variant", "sum"),
        )
    )
    work = work.drop(
        columns=["pair_window_fact_qty", "pair_window_forecast_variant_qty"],
        errors="ignore",
    ).merge(pair_totals, on=["bakery_id", "product_id"], how="left", validate="many_to_one")
    work["dead_pair_window_variant"] = (
        (work["pair_window_fact_qty"] <= 0)
        & (work["pair_window_forecast_variant_qty"] > 0)
    )

    fact_sum = float(fact.sum())
    scaled_fact_sum = float(scaled_fact.sum())
    dead_sum = float(work.loc[work["dead_pair_window_variant"], "forecast_variant"].sum())
    summary = {
        "variant": name,
        "rows": int(len(work)),
        "fact_sum": fact_sum,
        "forecast_sum": float(forecast.sum()),
        "wmape_raw_fact_pct": float(work["abs_err_raw_fact"].sum() / fact_sum * 100) if fact_sum else None,
        "wmape_scaled_fact_pct": float(work["abs_err_scaled_fact"].sum() / scaled_fact_sum * 100)
        if scaled_fact_sum
        else None,
        "bias_sum": float(err_raw.sum()),
        "dead_pair_forecast_qty": dead_sum,
        "dead_pair_forecast_share_pct": float(dead_sum / forecast.sum() * 100) if forecast.sum() else 0.0,
        "forecast_only_fact_zero_qty": float(
            work.loc[work["cell_type_variant"] == "forecast_only_fact_zero", "forecast_variant"].sum()
        ),
        "fact_only_forecast_zero_qty": float(
            work.loc[work["cell_type_variant"] == "fact_only_forecast_zero", "fact_qty"].sum()
        ),
    }
    return work, summary


def _write_variant_artifacts(scored: pd.DataFrame, name: str, out_dir: Path) -> None:
    by_pair = (
        scored.groupby(
            ["bakery_id", "bakery_name", "city", "product_id", "product_name", "category_name"],
            as_index=False,
            dropna=False,
        )
        .agg(
            fact_qty=("fact_qty", "sum"),
            forecast_variant=("forecast_variant", "sum"),
            abs_err_raw_fact=("abs_err_raw_fact", "sum"),
            abs_err_scaled_fact=("abs_err_scaled_fact", "sum"),
            recent_qty=("recent_qty", "max"),
            recent_days_sold=("recent_days_sold", "max"),
            city_recent_share=("city_recent_share", "max"),
            city_recent_rank=("city_recent_rank", "min"),
        )
    )
    by_pair["bias_qty"] = by_pair["forecast_variant"] - by_pair["fact_qty"]
    by_pair["dead_pair_window"] = (
        (by_pair["fact_qty"] <= 0) & (by_pair["forecast_variant"] > 0)
    )
    by_pair.to_csv(out_dir / f"{name}_by_bakery_sku.csv", index=False, encoding="utf-8-sig")
    by_pair[by_pair["fact_qty"] > 0].sort_values("bias_qty").head(100).to_csv(
        out_dir / f"{name}_top_underforecast.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_pair[by_pair["dead_pair_window"]].sort_values(
        "forecast_variant",
        ascending=False,
    ).head(100).to_csv(
        out_dir / f"{name}_top_dead_pair_forecast.csv",
        index=False,
        encoding="utf-8-sig",
    )


def evaluate(args: argparse.Namespace) -> dict:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    holdout_start = pd.Timestamp(args.start_date)
    recent_end = holdout_start - pd.Timedelta(days=1)
    recent_start = holdout_start - pd.Timedelta(days=args.recent_days)

    base = pd.read_csv(args.compare_path, parse_dates=["date"])
    recent_path = out_dir / "recent_assortment_stats.csv"
    recent_daily_path = out_dir / "recent_daily_share_stats.csv"
    if args.reuse_recent_stats and recent_path.exists() and recent_daily_path.exists():
        recent = pd.read_csv(recent_path)
        recent_daily = pd.read_csv(recent_daily_path)
    else:
        recent = _query_recent_stats(
            env_file=args.env_file,
            recent_start=str(recent_start.date()),
            recent_end=str(recent_end.date()),
            table=args.sales_table,
        )
        recent_daily = _query_recent_daily_share_stats(
            env_file=args.env_file,
            recent_start=str(recent_start.date()),
            recent_end=str(recent_end.date()),
            table=args.sales_table,
        )
    recent.to_csv(out_dir / "recent_assortment_stats.csv", index=False, encoding="utf-8-sig")
    recent_daily.to_csv(
        out_dir / "recent_daily_share_stats.csv",
        index=False,
        encoding="utf-8-sig",
    )
    base = _attach_recent(base, recent, recent_daily)

    variants = [
        "baseline",
        "dead_0d",
        "active_3d",
        "blend_recent_50",
        "core_recent_70",
        "adaptive_recent",
        "runner_floor_90",
        "runner_recent_heavy",
        "runner_recent_heavy_weekpart",
        "runner_city_prior_soft",
        "runner_city_prior_strong",
        "runner_city_prior_soft_weekpart",
        "runner_city_prior_strong_weekpart",
        "runner_recent_heavy_dow_shrink",
        "runner_recent_heavy_robust_weekpart",
        "category_preserving",
        "topn_runner_protection",
        "anti_cannibalization_cap",
        "runner_residual_redistribution",
    ]
    summaries = []
    for name in variants:
        variant_df = _apply_variant(base, name)
        scored, summary = _score_variant(variant_df, name)
        summaries.append(summary)
        if name in {
            "baseline",
            "blend_recent_50",
            "core_recent_70",
            "adaptive_recent",
            "runner_floor_90",
            "runner_recent_heavy",
            "runner_recent_heavy_weekpart",
            "runner_city_prior_soft",
            "runner_city_prior_strong",
            "runner_city_prior_soft_weekpart",
            "runner_city_prior_strong_weekpart",
            "runner_recent_heavy_dow_shrink",
            "runner_recent_heavy_robust_weekpart",
            "category_preserving",
            "topn_runner_protection",
            "anti_cannibalization_cap",
            "runner_residual_redistribution",
        }:
            slim = scored[
                [
                    "date",
                    "bakery_id",
                    "bakery_name",
                    "city",
                    "product_id",
                    "product_name",
                    "category_name",
                    "fact_qty",
                    "forecast_variant",
                    "fact_scaled_to_variant_total",
                    "err_raw_fact",
                    "err_scaled_fact",
                    "cell_type_variant",
                    "recent_qty",
                    "recent_days_sold",
                    "recent_share",
                    "city_recent_share",
                    "city_recent_rank",
                ]
            ].sort_values(["bakery_id", "product_id", "date"])
            slim.to_csv(out_dir / f"{name}_compare.csv", index=False, encoding="utf-8-sig")
        _write_variant_artifacts(scored, name, out_dir)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "variant_summary.csv", index=False, encoding="utf-8-sig")

    result = {
        "holdout_window": {"start": args.start_date, "end": args.end_date},
        "recent_window": {
            "start": str(recent_start.date()),
            "end": str(recent_end.date()),
            "days": args.recent_days,
        },
        "compare_path": str(Path(args.compare_path)),
        "output_dir": str(out_dir),
        "variants": summaries,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate SKU allocation variants")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH))
    parser.add_argument("--compare-path", default=str(DEFAULT_COMPARE_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sales-table", default=SALES_LINE_TABLE)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--recent-days", type=int, default=DEFAULT_RECENT_DAYS)
    parser.add_argument(
        "--reuse-recent-stats",
        action="store_true",
        help="Reuse recent_assortment_stats.csv and recent_daily_share_stats.csv from output-dir.",
    )
    return parser


def main() -> None:
    result = evaluate(build_parser().parse_args())
    summary = pd.DataFrame(result["variants"])
    print("=" * 72)
    print("SKU ALLOCATION VARIANT EVALUATION")
    print("=" * 72)
    print(f"recent window: {result['recent_window']['start']} .. {result['recent_window']['end']}")
    print(f"output_dir: {result['output_dir']}")
    print(
        summary[
            [
                "variant",
                "wmape_raw_fact_pct",
                "wmape_scaled_fact_pct",
                "dead_pair_forecast_share_pct",
                "forecast_only_fact_zero_qty",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

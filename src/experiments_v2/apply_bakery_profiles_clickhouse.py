"""
Apply bakery forecasts to SKU forecasts using ClickHouse-stored profiles.

This is the production-friendly allocation path: the large SKU hour profile is
streamed from ClickHouse instead of being stored as a local CSV on the app VM.
"""

from __future__ import annotations

# ruff: noqa: E402,E501

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipelines.forecast_publish.load_forecast_run import DEFAULT_ENV_PATH
from pipelines.forecast_publish.load_forecast_run import create_client
from pipelines.forecast_publish.sku_hour_profile_store import PROFILE_TABLE
from pipelines.forecast_publish.sku_hour_profile_store import UPLIFT_MULTIPLIER_TABLE
from src.experiments_v2.apply_bakery_profiles import BAKERY_HOUR_FORECAST_COL
from src.experiments_v2.apply_bakery_profiles import BAKERY_ID_COL
from src.experiments_v2.apply_bakery_profiles import BAKERY_FORECAST_COL
from src.experiments_v2.apply_bakery_profiles import CATEGORY_COL
from src.experiments_v2.apply_bakery_profiles import CITY_COL
from src.experiments_v2.apply_bakery_profiles import DAILY_OUTPUT_NAME
from src.experiments_v2.apply_bakery_profiles import DATE_COL
from src.experiments_v2.apply_bakery_profiles import DOW_COL
from src.experiments_v2.apply_bakery_profiles import DEFAULT_BAKERY_HOUR_PROFILE_PATH
from src.experiments_v2.apply_bakery_profiles import HOUR_COL
from src.experiments_v2.apply_bakery_profiles import HOURLY_OUTPUT_NAME
from src.experiments_v2.apply_bakery_profiles import HOURLY_OUTPUT_COLS
from src.experiments_v2.apply_bakery_profiles import MIN_FALLBACK_N_DAYS
from src.experiments_v2.apply_bakery_profiles import MIN_TIER1_N_DAYS
from src.experiments_v2.apply_bakery_profiles import PRODUCT_ID_COL
from src.experiments_v2.apply_bakery_profiles import PRODUCT_NAME_COL
from src.experiments_v2.apply_bakery_profiles import SKU_DAY_FORECAST_COL
from src.experiments_v2.apply_bakery_profiles import SKU_HOUR_FORECAST_COL
from src.experiments_v2.apply_bakery_profiles import SKU_PROFILE_CHUNK_SIZE
from src.experiments_v2.apply_bakery_profiles import SKU_SHARE_COL
from src.experiments_v2.apply_bakery_profiles import SUMMARY_OUTPUT_NAME
from src.experiments_v2.apply_bakery_profiles import allocate_bakery_to_hour
from src.experiments_v2.apply_bakery_profiles import build_summary_from_daily
from src.experiments_v2.apply_bakery_profiles import load_bakery_day_forecast
from src.experiments_v2.apply_bakery_profiles import load_bakery_hour_profile


DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed"
SKU_UPLIFT_MULTIPLIER_COL = "sku_uplift_multiplier"
STOCKOUT_CORRECTION_TABLE = "sku_hour_stockout_correction_embedded"
SALES_LINE_TABLE = "mart_sales_60d"
RAW_SALES_LINE_TABLE = "Svezhar.fct_check_lines"
SALES_EVENT_HEX = "D09FD180D0BED0B4D0B0D0B6D0B0"
ASSORTMENT_TABLE = "assortment_city_products"
RECENT_CORRECTION_MODES = (
    "none",
    "dead_0d",
    "blend_recent_50",
    "core_recent_70",
    "runner_city_prior_soft_weekpart",
)
DEFAULT_RECENT_UPWARD_CAP_CATEGORY_PATTERN = (
    "\u043f\u0438\u0440\u043e\u0433\u0438 \u0441\u044b\u0442\u043d\u044b\u0435"
    "|"
    "\u043f\u0438\u0440\u043e\u0433\u0438 \u0441\u043b\u0430\u0434\u043a\u0438\u0435"
)
DEFAULT_RECENT_UPWARD_CAP_MULTIPLIER = 1.0
DEFAULT_RECENT_ABSOLUTE_CAP_MULTIPLIER = 1.0
ECLAIR_PATTERN = "эклер"
SERVICE_CATEGORY_PATTERN = "прочие|заказ"


def _write_hourly_chunk(df: pd.DataFrame, path: Path, *, header: bool) -> None:
    df.to_csv(path, mode="a", index=False, encoding="utf-8-sig", header=header)


def load_active_assortment_pairs(
    client,
    *,
    assortment_table: str,
    effective_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    if "bakeable_products" in assortment_table:
        if effective_date is None:
            raise ValueError("effective_date is required for bakeable assortment")
        return client.query_df(
            f"""
            select
                b.city,
                toInt64OrNull(replaceRegexpOne(toString(b.product_id), '^0+', '')) as product_id_int,
                if(ifNull(b.scope, 'city') = 'bakery', b.bakery_id, null) as bakery_id,
                b.valid_from
            from {assortment_table} as b final
            inner join (
                select city, max(valid_from) as latest_valid_from
                from {assortment_table} final
                where valid_from <= toDate(%(effective_date)s)
                group by city
            ) as latest
              on b.city = latest.city and b.valid_from = latest.latest_valid_from
            where b.is_active = 1
              and b.is_bakeable = 1
              and (b.valid_to is null or b.valid_to >= toDate(%(effective_date)s))
              and toInt64OrNull(replaceRegexpOne(toString(b.product_id), '^0+', '')) is not null
            group by b.city, product_id_int, bakery_id, b.valid_from
            """,
            parameters={"effective_date": str(pd.Timestamp(effective_date).date())},
        ).rename(columns={"product_id_int": PRODUCT_ID_COL})
    if effective_date is None:
        raise ValueError("effective_date is required for allocation assortment")
    return client.query_df(
        f"""
        select
            a.city,
            toInt64OrNull(replaceRegexpOne(toString(a.product_id), '^0+', '')) as product_id_int,
            a.valid_from
        from {assortment_table} as a final
        inner join (
            select city, max(valid_from) as latest_valid_from
            from {assortment_table} final
            where valid_from <= toDate(%(effective_date)s)
            group by city
        ) as latest
          on a.city = latest.city and a.valid_from = latest.latest_valid_from
        where a.is_active = 1
          and (a.valid_to is null or a.valid_to >= toDate(%(effective_date)s))
          and toInt64OrNull(replaceRegexpOne(toString(a.product_id), '^0+', '')) is not null
        group by a.city, product_id_int, a.valid_from
        """,
        parameters={"effective_date": str(pd.Timestamp(effective_date).date())},
    ).rename(
        columns={"product_id_int": PRODUCT_ID_COL}
    )


def validate_assortment_freshness(
    allowed_pairs: pd.DataFrame,
    *,
    expected_cities: set[str],
    effective_date: str | pd.Timestamp,
    max_age_days: int,
) -> None:
    if max_age_days < 0 or allowed_pairs.empty or "valid_from" not in allowed_pairs:
        return
    latest = allowed_pairs.groupby(CITY_COL)["valid_from"].max()
    missing = sorted(expected_cities - set(latest.index.astype(str)))
    if missing:
        raise RuntimeError(f"Assortment has no active batch for cities: {missing}")
    ages = (
        pd.Timestamp(effective_date).normalize()
        - pd.to_datetime(latest.loc[sorted(expected_cities)]).dt.normalize()
    ).dt.days
    stale = ages[ages > max_age_days]
    if not stale.empty:
        details = ", ".join(f"{city}={int(age)}d" for city, age in stale.items())
        raise RuntimeError(
            f"Assortment is stale for forecast date {pd.Timestamp(effective_date).date()}: "
            f"{details}; max_age_days={max_age_days}"
        )


def find_recent_sales_missing_from_assortment(
    recent: pd.DataFrame,
    allowed_pairs: pd.DataFrame,
    *,
    bakery_city_lookup: pd.DataFrame,
    min_days_sold: int = 2,
    min_qty: float = 2.0,
) -> pd.DataFrame:
    """Return established, recently sold bakery/SKU pairs absent from allowlist."""
    if recent.empty:
        return recent.copy()
    work = recent.copy().drop(columns=[CITY_COL], errors="ignore").merge(
        bakery_city_lookup[[BAKERY_ID_COL, CITY_COL]].drop_duplicates(BAKERY_ID_COL),
        on=BAKERY_ID_COL,
        how="left",
        validate="many_to_one",
    )
    work = work[
        work["recent_days_sold"].ge(min_days_sold)
        & work["recent_qty"].ge(min_qty)
        & work[CITY_COL].notna()
    ].copy()
    if work.empty:
        return work

    allowed = allowed_pairs.copy()
    keys = [CITY_COL, PRODUCT_ID_COL]
    if BAKERY_ID_COL in allowed.columns:
        city_allowed = allowed[allowed[BAKERY_ID_COL].isna()][keys]
        bakery_allowed = allowed[allowed[BAKERY_ID_COL].notna()][
            [*keys, BAKERY_ID_COL]
        ]
        city_keys = set(map(tuple, city_allowed.values.tolist()))
        bakery_keys = set(map(tuple, bakery_allowed.values.tolist()))
        present = [
            (row.city, row.product_id) in city_keys
            or (row.city, row.product_id, row.bakery_id) in bakery_keys
            for row in work.itertuples()
        ]
        return work.loc[~pd.Series(present, index=work.index)].copy()
    checked = work.merge(allowed[keys].drop_duplicates(), on=keys, how="left", indicator=True)
    return checked[checked["_merge"].eq("left_only")].drop(columns="_merge")


def _build_bakery_city_lookup(bakery_forecast: pd.DataFrame) -> pd.DataFrame:
    if CITY_COL not in bakery_forecast.columns:
        return pd.DataFrame(columns=[BAKERY_ID_COL, CITY_COL])
    return (
        bakery_forecast[[BAKERY_ID_COL, CITY_COL]]
        .dropna(subset=[BAKERY_ID_COL, CITY_COL])
        .drop_duplicates([BAKERY_ID_COL])
    )


def filter_by_active_assortment(
    df: pd.DataFrame,
    *,
    allowed_pairs: pd.DataFrame,
    bakery_city_lookup: pd.DataFrame,
    forecast_col: str,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    if df.empty or allowed_pairs.empty:
        return df, {"rows_removed": 0, "forecast_removed": 0.0}

    original_cols = list(df.columns)
    work = df.copy()
    if CITY_COL not in work.columns and not bakery_city_lookup.empty:
        work = work.merge(bakery_city_lookup, on=BAKERY_ID_COL, how="left")
    if CITY_COL not in work.columns:
        return df, {"rows_removed": 0, "forecast_removed": 0.0}

    work["_assortment_product_id"] = pd.to_numeric(
        work[PRODUCT_ID_COL],
        errors="coerce",
    ).astype("Int64")
    allowed = allowed_pairs.rename(columns={PRODUCT_ID_COL: "_assortment_product_id"})
    allowed["_assortment_product_id"] = pd.to_numeric(
        allowed["_assortment_product_id"],
        errors="coerce",
    ).astype("Int64")
    if BAKERY_ID_COL in allowed.columns:
        city_allowed = allowed[allowed[BAKERY_ID_COL].isna()][
            [CITY_COL, "_assortment_product_id"]
        ].drop_duplicates()
        bakery_allowed = allowed[allowed[BAKERY_ID_COL].notna()][
            [CITY_COL, BAKERY_ID_COL, "_assortment_product_id"]
        ].drop_duplicates()
        work = work.merge(
            city_allowed.assign(_in_city_assortment=1),
            on=[CITY_COL, "_assortment_product_id"],
            how="left",
        ).merge(
            bakery_allowed.assign(_in_bakery_assortment=1),
            on=[CITY_COL, BAKERY_ID_COL, "_assortment_product_id"],
            how="left",
        )
        work["_in_active_assortment"] = work["_in_city_assortment"].fillna(0) + work[
            "_in_bakery_assortment"
        ].fillna(0)
    else:
        work = work.merge(
            allowed.assign(_in_active_assortment=1),
            on=[CITY_COL, "_assortment_product_id"],
            how="left",
        )
    scoped_cities = set(allowed[CITY_COL].dropna().astype(str))
    city_is_scoped = work[CITY_COL].astype(str).isin(scoped_cities)
    keep_mask = (~city_is_scoped) | work["_in_active_assortment"].fillna(0).astype(
        int
    ).eq(1)
    removed = work.loc[~keep_mask]
    forecast_removed = float(
        pd.to_numeric(removed.get(forecast_col, 0.0), errors="coerce")
        .fillna(0.0)
        .sum()
    )
    filtered = work.loc[keep_mask, original_cols].copy()
    return filtered, {
        "rows_removed": int(len(removed)),
        "forecast_removed": forecast_removed,
    }


def _merge_filter_stats(
    total: dict[str, float | int],
    stats: dict[str, float | int],
) -> None:
    total["rows_removed"] = int(total.get("rows_removed", 0)) + int(
        stats.get("rows_removed", 0)
    )
    total["forecast_removed"] = float(total.get("forecast_removed", 0.0)) + float(
        stats.get("forecast_removed", 0.0)
    )


def renormalize_hourly_to_bakery_forecast(
    sku_hourly: pd.DataFrame,
    bakery_hourly: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    if sku_hourly.empty:
        return sku_hourly, {"groups_scaled": 0, "groups_without_sku": 0}

    keys = [DATE_COL, BAKERY_ID_COL, HOUR_COL]
    original_cols = list(sku_hourly.columns)
    work = sku_hourly.copy()
    targets = bakery_hourly[[*keys, BAKERY_HOUR_FORECAST_COL]].copy()
    work[DATE_COL] = pd.to_datetime(work[DATE_COL])
    targets[DATE_COL] = pd.to_datetime(targets[DATE_COL])
    targets = targets.groupby(keys, as_index=False)[BAKERY_HOUR_FORECAST_COL].sum()
    totals = (
        work.groupby(keys, as_index=False)[SKU_HOUR_FORECAST_COL]
        .sum()
        .rename(columns={SKU_HOUR_FORECAST_COL: "_sku_hour_sum"})
    )
    scales = targets.merge(totals, on=keys, how="left")
    positive = scales["_sku_hour_sum"].fillna(0.0).gt(0.0)
    scales["_assortment_scale"] = 1.0
    scales.loc[positive, "_assortment_scale"] = (
        scales.loc[positive, BAKERY_HOUR_FORECAST_COL]
        / scales.loc[positive, "_sku_hour_sum"]
    )
    work = work.merge(scales[[*keys, "_assortment_scale"]], on=keys, how="left")
    work[SKU_HOUR_FORECAST_COL] = (
        work[SKU_HOUR_FORECAST_COL]
        * work["_assortment_scale"].fillna(1.0)
    )
    return work[original_cols], {
        "groups_scaled": int(positive.sum()),
        "groups_without_sku": int((~positive).sum()),
    }


def compensate_for_assortment_exclusion(
    pre_filter: pd.DataFrame,
    post_filter: pd.DataFrame,
    *,
    group_keys: list[str],
    forecast_col: str,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Redistribute demand dropped by assortment filtering back onto the
    remaining (in-assortment) rows of each group, without touching whatever
    the raw SKU-hour uplift already did to those rows.

    Unlike `renormalize_hourly_to_bakery_forecast` (which rescales to the
    flat bakery-hour forecast and would silently cancel the uplift's
    deliberate inflation), the scale here is
    `sum(pre_filter) / sum(post_filter)` per group — purely the fraction
    lost to assortment exclusion, since both sides of that ratio already
    carry the same uplift effect. Only used when
    `use_raw_uplift_multiplier=True`, where assortment-filtered demand
    would otherwise vanish instead of being redistributed (see
    `docs/ops/DECISIONS.md`, 2026-07-14 entry).
    """
    if pre_filter.empty:
        return post_filter, {"groups_scaled": 0, "groups_without_remaining_rows": 0}

    pre_totals = (
        pre_filter.groupby(group_keys, as_index=False)[forecast_col]
        .sum()
        .rename(columns={forecast_col: "_pre_total"})
    )
    post_totals = (
        post_filter.groupby(group_keys, as_index=False)[forecast_col]
        .sum()
        .rename(columns={forecast_col: "_post_total"})
    )
    scales = pre_totals.merge(post_totals, on=group_keys, how="left")
    scales["_post_total"] = scales["_post_total"].fillna(0.0)
    has_remaining = scales["_post_total"].gt(0.0)
    scales["_compensation_scale"] = 1.0
    scales.loc[has_remaining, "_compensation_scale"] = (
        scales.loc[has_remaining, "_pre_total"] / scales.loc[has_remaining, "_post_total"]
    )
    work = post_filter.merge(
        scales[[*group_keys, "_compensation_scale"]], on=group_keys, how="left"
    )
    work[forecast_col] = work[forecast_col] * work["_compensation_scale"].fillna(1.0)
    work = work.drop(columns=["_compensation_scale"])
    return work, {
        "groups_scaled": int(has_remaining.sum()),
        "groups_without_remaining_rows": int((~has_remaining).sum()),
    }


def cap_sku_uplift_to_bakery_forecast(
    hourly: pd.DataFrame,
    bakery_forecast: pd.DataFrame,
    *,
    max_ratio: float,
    forecast_col: str,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Scale SKU-hour forecasts down when their per-(date, bakery) sum exceeds
    max_ratio × bakery-day total.  Only ever scales down (clip upper=1.0).

    Preserves relative SKU distribution within each bakery-day — all rows for
    the same (date, bakery) receive the same multiplicative factor, so the
    shape of the hour/SKU breakdown is unchanged.  Used after assortment
    compensation under ``use_raw_uplift_multiplier=True`` to soft-cap the
    bakery-level bias introduced by the mean-share floor uplift.
    """
    group_keys = [DATE_COL, BAKERY_ID_COL]
    sku_sums = (
        hourly.groupby(group_keys, as_index=False)[SKU_HOUR_FORECAST_COL]
        .sum()
        .rename(columns={SKU_HOUR_FORECAST_COL: "_sku_day_sum"})
    )
    bak_day = (
        bakery_forecast[[DATE_COL, BAKERY_ID_COL, forecast_col]]
        .copy()
        .rename(columns={forecast_col: "_bak_day"})
    )
    bak_day[DATE_COL] = pd.to_datetime(bak_day[DATE_COL])
    sku_sums[DATE_COL] = pd.to_datetime(sku_sums[DATE_COL])

    ratio_df = sku_sums.merge(bak_day, on=group_keys, how="left")
    ratio_df["_ratio"] = ratio_df["_sku_day_sum"] / ratio_df["_bak_day"].replace(0.0, float("nan"))
    ratio_df["_scale"] = (max_ratio / ratio_df["_ratio"]).clip(upper=1.0).fillna(1.0)

    n_capped = int((ratio_df["_scale"] < 1.0).sum())
    avg_scale = (
        float(ratio_df.loc[ratio_df["_scale"] < 1.0, "_scale"].mean())
        if n_capped > 0
        else 1.0
    )

    work = hourly.merge(ratio_df[group_keys + ["_scale"]], on=group_keys, how="left")
    work[SKU_HOUR_FORECAST_COL] = work[SKU_HOUR_FORECAST_COL] * work["_scale"].fillna(1.0)
    work = work.drop(columns=["_scale"])
    return work, {
        "max_ratio": max_ratio,
        "bakery_days_capped": n_capped,
        "bakery_days_total": int(len(ratio_df)),
        "avg_scale_when_capped": round(avg_scale, 4),
    }


def cap_sku_uplift_per_sku(
    hourly: pd.DataFrame,
    recent_stats: pd.DataFrame,
    *,
    max_ratio: float,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Cap each per-(date, bakery, product) SKU-hour sum at max_ratio times
    its rolling daily mean from recent sales.  Only ever scales down.

    recent_stats must have columns: bakery_id, product_id, recent_qty,
    recent_days_sold  (from load_recent_assortment_stats).  SKUs with no
    recent history are left uncapped.
    """
    sku_keys = [DATE_COL, BAKERY_ID_COL, PRODUCT_ID_COL]

    rolling = recent_stats[
        [BAKERY_ID_COL, PRODUCT_ID_COL, "recent_qty", "recent_days_sold"]
    ].copy()
    rolling["_rolling_mean"] = (
        rolling["recent_qty"] / rolling["recent_days_sold"].replace(0, float("nan"))
    ).fillna(0.0)

    sku_sums = (
        hourly.groupby(sku_keys, as_index=False)[SKU_HOUR_FORECAST_COL]
        .sum()
        .rename(columns={SKU_HOUR_FORECAST_COL: "_fc_sku_day"})
    )
    sku_sums[DATE_COL] = pd.to_datetime(sku_sums[DATE_COL])

    sku_sums = sku_sums.merge(
        rolling[[BAKERY_ID_COL, PRODUCT_ID_COL, "_rolling_mean"]],
        on=[BAKERY_ID_COL, PRODUCT_ID_COL],
        how="left",
    )

    # SKUs not in recent_stats get NaN rolling_mean → no cap → scale=1.0
    sku_sums["_scale"] = np.where(
        (sku_sums["_rolling_mean"].fillna(0.0) > 0) & (sku_sums["_fc_sku_day"] > 0),
        ((sku_sums["_rolling_mean"] * max_ratio) / sku_sums["_fc_sku_day"]).clip(upper=1.0),
        1.0,
    )

    n_capped = int((sku_sums["_scale"] < 1.0).sum())
    avg_scale = (
        float(sku_sums.loc[sku_sums["_scale"] < 1.0, "_scale"].mean())
        if n_capped > 0
        else 1.0
    )

    work = hourly.copy()
    work[DATE_COL] = pd.to_datetime(work[DATE_COL])
    work = work.merge(sku_sums[sku_keys + ["_scale"]], on=sku_keys, how="left")
    work[SKU_HOUR_FORECAST_COL] = work[SKU_HOUR_FORECAST_COL] * work["_scale"].fillna(1.0)
    work = work.drop(columns=["_scale"])
    return work, {
        "max_ratio": max_ratio,
        "sku_days_capped": n_capped,
        "sku_days_total": int(len(sku_sums)),
        "avg_scale_when_capped": round(avg_scale, 4),
    }


def load_stockout_correction(
    client,
    *,
    table: str = STOCKOUT_CORRECTION_TABLE,
    profile_version: str,
) -> pd.DataFrame:
    """Load per-(bakery_id, product_id, hour) correction factors from ClickHouse.

    Returns a DataFrame with columns [bakery_id, product_id, hour, correction_factor].
    Returns an empty DataFrame if the table does not exist or has no rows for the version.
    """
    try:
        df = client.query_df(
            f"""
            SELECT bakery_id, product_id, hour,
                   argMax(correction_factor, generated_at) AS correction_factor
            FROM {table}
            WHERE profile_version = %(version)s
            GROUP BY bakery_id, product_id, hour
            """,
            parameters={"version": profile_version},
        )
    except Exception:
        return pd.DataFrame(columns=["bakery_id", "product_id", "hour", "correction_factor"])
    return df


def apply_stockout_hour_correction(
    hourly: pd.DataFrame,
    correction: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Multiply SKU-hour forecasts by pre-computed per-(bakery, product, hour) factors.

    Only rows where correction_factor > 1.0 are lifted; never scales down. The
    correction is applied in-place on a copy so the caller's DataFrame is not
    mutated.
    """
    if correction.empty or hourly.empty:
        return hourly, {"rows_corrected": 0, "avg_factor_when_applied": 1.0}

    work = hourly.copy()
    work["_bakery_id_int"] = pd.to_numeric(work[BAKERY_ID_COL], errors="coerce")
    work["_product_id_int"] = pd.to_numeric(work[PRODUCT_ID_COL], errors="coerce")

    cf = correction[correction["correction_factor"] > 1.0].copy()
    cf["bakery_id"]  = cf["bakery_id"].astype("int64")
    cf["product_id"] = cf["product_id"].astype("int64")
    cf["hour"]       = cf["hour"].astype("int16")

    work = work.merge(
        cf.rename(columns={"bakery_id": "_bakery_id_int", "product_id": "_product_id_int"}),
        on=["_bakery_id_int", "_product_id_int", HOUR_COL],
        how="left",
    )
    applied_mask = work["correction_factor"].notna() & (work["correction_factor"] > 1.0)
    n_corrected = int(applied_mask.sum())
    avg_factor = (
        float(work.loc[applied_mask, "correction_factor"].mean()) if n_corrected > 0 else 1.0
    )

    work[SKU_HOUR_FORECAST_COL] = np.where(
        applied_mask,
        work[SKU_HOUR_FORECAST_COL] * work["correction_factor"].fillna(1.0),
        work[SKU_HOUR_FORECAST_COL],
    )
    work = work.drop(columns=["_bakery_id_int", "_product_id_int", "correction_factor"])
    return work, {
        "rows_corrected": n_corrected,
        "rows_total": len(work),
        "avg_factor_when_applied": round(avg_factor, 4),
    }


def build_hierarchical_haircut_coefficients(
    history: pd.DataFrame,
    *,
    target_ratio: float,
    min_coefficient: float,
    pair_prior_days: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build downward-only bakery and bakery-SKU correction coefficients."""
    if target_ratio <= 0:
        raise ValueError("target_ratio must be positive")
    if not 0 < min_coefficient <= 1:
        raise ValueError("min_coefficient must be in (0, 1]")
    if pair_prior_days <= 0:
        raise ValueError("pair_prior_days must be positive")
    keys = [BAKERY_ID_COL, PRODUCT_ID_COL]
    work = history.copy()
    work["forecast_qty"] = pd.to_numeric(work["forecast_qty"], errors="coerce").fillna(0.0)
    work["actual_qty"] = pd.to_numeric(work["actual_qty"], errors="coerce").fillna(0.0)

    bakery = (
        work.groupby(BAKERY_ID_COL, as_index=False)
        .agg(history_forecast=("forecast_qty", "sum"), history_actual=("actual_qty", "sum"))
    )
    bakery["_raw_bakery_coefficient"] = np.where(
        bakery["history_forecast"] > 0,
        target_ratio * bakery["history_actual"] / bakery["history_forecast"],
        1.0,
    )
    bakery["protect_from_haircut"] = bakery["_raw_bakery_coefficient"] >= 1.0
    bakery["bakery_coefficient"] = bakery["_raw_bakery_coefficient"].clip(
        lower=min_coefficient,
        upper=1.0,
    )

    pair = (
        work.groupby(keys, as_index=False)
        .agg(
            history_forecast=("forecast_qty", "sum"),
            history_actual=("actual_qty", "sum"),
            history_days=(DATE_COL, "nunique"),
        )
    )
    pair["pair_coefficient"] = np.where(
        pair["history_forecast"] > 0,
        target_ratio * pair["history_actual"] / pair["history_forecast"],
        1.0,
    )
    pair["pair_coefficient"] = pair["pair_coefficient"].clip(
        lower=min_coefficient,
        upper=1.0,
    )
    pair = pair.merge(
        bakery[[BAKERY_ID_COL, "bakery_coefficient", "protect_from_haircut"]],
        on=BAKERY_ID_COL,
        how="left",
        validate="many_to_one",
    )
    weight = pair["history_days"] / (pair["history_days"] + pair_prior_days)
    pair["hierarchical_coefficient"] = (
        weight * pair["pair_coefficient"]
        + (1.0 - weight) * pair["bakery_coefficient"]
    ).clip(lower=min_coefficient, upper=1.0)
    pair.loc[pair["protect_from_haircut"], "hierarchical_coefficient"] = 1.0
    return bakery, pair


def load_hierarchical_haircut_history(
    client,
    *,
    forecast_start: pd.Timestamp,
    history_days: int,
) -> pd.DataFrame:
    """Load latest lead-1 raw-uplift forecasts and UI-equivalent actuals."""
    history_end = forecast_start - pd.Timedelta(days=1)
    history_start = forecast_start - pd.Timedelta(days=history_days)
    params = {
        "history_start": str(history_start.date()),
        "history_end": str(history_end.date()),
    }
    forecast = client.query_df(
        """
        select
            forecast_date as date,
            bakery_id,
            product_id,
            argMax(forecast_qty, generated_at) as forecast_qty
        from sku_forecast_day_snapshots
        where lead_days = 1
          and forecast_date between %(history_start)s and %(history_end)s
          and source_run_id like '%%_raw_uplift_%%'
        group by date, bakery_id, product_id
        """,
        parameters=params,
    )
    actual = client.query_df(
        f"""
        select
            check_date as date,
            toInt64(bakery_id) as bakery_id,
            toInt64(product_id) as product_id,
            sum(toFloat64(quantity)) as actual_qty
        from (
            select distinct
                check_datetime,
                check_date,
                bakery_id,
                product_id,
                quantity,
                price,
                line_amount,
                cash_event_type
            from {SALES_LINE_TABLE}
            where hex(cash_event_type) = '{SALES_EVENT_HEX}'
              and check_date between %(history_start)s and %(history_end)s
        )
        group by date, bakery_id, product_id
        """,
        parameters=params,
    )
    keys = [DATE_COL, BAKERY_ID_COL, PRODUCT_ID_COL]
    history = forecast.merge(actual, on=keys, how="outer")
    history["forecast_qty"] = history["forecast_qty"].fillna(0.0)
    history["actual_qty"] = history["actual_qty"].fillna(0.0)
    return history


def apply_hierarchical_haircut(
    hourly: pd.DataFrame,
    bakery_coefficients: pd.DataFrame,
    pair_coefficients: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Apply fixed per-bakery/SKU coefficients while preserving hourly shape."""
    work = hourly.copy()
    before = float(work[SKU_HOUR_FORECAST_COL].sum())
    work = work.merge(
        pair_coefficients[[BAKERY_ID_COL, PRODUCT_ID_COL, "hierarchical_coefficient"]],
        on=[BAKERY_ID_COL, PRODUCT_ID_COL],
        how="left",
    ).merge(
        bakery_coefficients[[BAKERY_ID_COL, "bakery_coefficient", "protect_from_haircut"]],
        on=BAKERY_ID_COL,
        how="left",
    )
    coefficient = work["hierarchical_coefficient"].fillna(work["bakery_coefficient"]).fillna(1.0)
    coefficient = np.where(work["protect_from_haircut"].fillna(True), 1.0, coefficient)
    work["_haircut_coefficient"] = coefficient
    work[SKU_HOUR_FORECAST_COL] = work[SKU_HOUR_FORECAST_COL] * work["_haircut_coefficient"]
    after = float(work[SKU_HOUR_FORECAST_COL].sum())
    rows_scaled = int((work["_haircut_coefficient"] < 1.0).sum())
    work = work.drop(
        columns=[
            "hierarchical_coefficient",
            "bakery_coefficient",
            "protect_from_haircut",
            "_haircut_coefficient",
        ]
    )
    return work, {
        "rows_scaled": rows_scaled,
        "rows_total": int(len(work)),
        "forecast_before": round(before, 6),
        "forecast_after": round(after, 6),
        "overall_coefficient": round(after / before, 6) if before else 1.0,
    }


def fill_missing_bakery_hours(
    sku_hourly: pd.DataFrame,
    bakery_hourly: pd.DataFrame,
    *,
    bakery_city_lookup: pd.DataFrame | None = None,
    allowed_pairs: pd.DataFrame | None = None,
    recent_product_weights: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if sku_hourly.empty:
        return sku_hourly, {"groups_filled": 0, "groups_unfilled": 0}

    keys = [DATE_COL, BAKERY_ID_COL, HOUR_COL]
    work = sku_hourly.copy()
    targets = bakery_hourly[
        [DATE_COL, DOW_COL, BAKERY_ID_COL, HOUR_COL, BAKERY_HOUR_FORECAST_COL]
    ].copy()
    work[DATE_COL] = pd.to_datetime(work[DATE_COL])
    targets[DATE_COL] = pd.to_datetime(targets[DATE_COL])
    current = work.groupby(keys, as_index=False)[SKU_HOUR_FORECAST_COL].sum()
    missing = targets.merge(current, on=keys, how="left")
    missing = missing[
        missing[SKU_HOUR_FORECAST_COL].fillna(0.0).le(0.0)
        & missing[BAKERY_HOUR_FORECAST_COL].gt(0.0)
    ].copy()
    if missing.empty:
        return work, {"groups_filled": 0, "groups_unfilled": 0}

    daily_weights = (
        work.groupby(
            [DATE_COL, DOW_COL, BAKERY_ID_COL, PRODUCT_ID_COL],
            as_index=False,
        )[SKU_HOUR_FORECAST_COL]
        .sum()
        .rename(columns={SKU_HOUR_FORECAST_COL: "_product_day_total"})
    )
    daily_weights["_bakery_day_total"] = daily_weights.groupby(
        [DATE_COL, BAKERY_ID_COL]
    )["_product_day_total"].transform("sum")
    daily_weights = daily_weights[daily_weights["_bakery_day_total"].gt(0.0)].copy()
    daily_weights["_product_day_share"] = (
        daily_weights["_product_day_total"] / daily_weights["_bakery_day_total"]
    )
    added = missing.merge(
        daily_weights,
        on=[DATE_COL, DOW_COL, BAKERY_ID_COL],
        how="left",
    )
    filled_mask = added["_product_day_share"].notna()
    added = added.loc[filled_mask].copy()
    if not added.empty:
        added[SKU_HOUR_FORECAST_COL] = (
            added[BAKERY_HOUR_FORECAST_COL] * added["_product_day_share"]
        )
        added["source"] = "assortment_hour_gap_fallback"
        added = added[[*HOURLY_OUTPUT_COLS, "source"]]
    result = work if added.empty else pd.concat([work, added], ignore_index=True)

    filled_keys = added[keys].drop_duplicates()
    remaining = missing.merge(
        filled_keys.assign(_filled=1),
        on=keys,
        how="left",
    )
    remaining = remaining[remaining["_filled"].isna()].drop(columns=["_filled"])

    city_added = pd.DataFrame(columns=[*HOURLY_OUTPUT_COLS, "source"])
    if (
        not remaining.empty
        and bakery_city_lookup is not None
        and not bakery_city_lookup.empty
    ):
        city_work = result.merge(bakery_city_lookup, on=BAKERY_ID_COL, how="left")
        city_keys = [DATE_COL, DOW_COL, CITY_COL, HOUR_COL]
        city_weights = city_work.groupby(
            [*city_keys, PRODUCT_ID_COL],
            as_index=False,
        )[SKU_HOUR_FORECAST_COL].sum()
        city_weights["_city_hour_total"] = city_weights.groupby(city_keys)[
            SKU_HOUR_FORECAST_COL
        ].transform("sum")
        city_weights = city_weights[city_weights["_city_hour_total"].gt(0.0)]
        city_weights["_city_product_share"] = (
            city_weights[SKU_HOUR_FORECAST_COL]
            / city_weights["_city_hour_total"]
        )
        city_added = remaining.merge(
            bakery_city_lookup,
            on=BAKERY_ID_COL,
            how="left",
        ).merge(
            city_weights[[*city_keys, PRODUCT_ID_COL, "_city_product_share"]],
            on=city_keys,
            how="left",
        )
        city_added = city_added[city_added["_city_product_share"].notna()].copy()
        if not city_added.empty:
            city_added[SKU_HOUR_FORECAST_COL] = (
                city_added[BAKERY_HOUR_FORECAST_COL]
                * city_added["_city_product_share"]
            )
            city_added["source"] = "assortment_city_hour_fallback"
            city_added = city_added[[*HOURLY_OUTPUT_COLS, "source"]]
            result = pd.concat([result, city_added], ignore_index=True)

    filled_key_parts = [frame for frame in [filled_keys, city_added[keys]] if not frame.empty]
    all_filled_keys = (
        pd.concat(filled_key_parts, ignore_index=True).drop_duplicates()
        if filled_key_parts
        else pd.DataFrame(columns=keys)
    )
    remaining = missing.merge(
        all_filled_keys.assign(_filled=1),
        on=keys,
        how="left",
    )
    remaining = remaining[remaining["_filled"].isna()].drop(columns=["_filled"])

    recent_added = pd.DataFrame(columns=[*HOURLY_OUTPUT_COLS, "source"])
    if (
        not remaining.empty
        and recent_product_weights is not None
        and not recent_product_weights.empty
    ):
        weight_col = "recent_share"
        recent_weights = recent_product_weights[
            [BAKERY_ID_COL, PRODUCT_ID_COL, weight_col]
        ].copy()
        recent_weights[weight_col] = pd.to_numeric(
            recent_weights[weight_col], errors="coerce"
        ).fillna(0.0)
        recent_weights = recent_weights[recent_weights[weight_col].gt(0.0)]
        recent_added = remaining.merge(
            recent_weights,
            on=BAKERY_ID_COL,
            how="inner",
        )
        if not recent_added.empty:
            recent_added, _ = filter_by_active_assortment(
                recent_added,
                allowed_pairs=(
                    allowed_pairs
                    if allowed_pairs is not None
                    else pd.DataFrame(columns=[CITY_COL, PRODUCT_ID_COL])
                ),
                bakery_city_lookup=(
                    bakery_city_lookup
                    if bakery_city_lookup is not None
                    else pd.DataFrame(columns=[BAKERY_ID_COL, CITY_COL])
                ),
                forecast_col=weight_col,
            )
            recent_added["_recent_weight_sum"] = recent_added.groupby(keys)[
                weight_col
            ].transform("sum")
            recent_added = recent_added[recent_added["_recent_weight_sum"].gt(0.0)]
            recent_added[SKU_HOUR_FORECAST_COL] = (
                recent_added[BAKERY_HOUR_FORECAST_COL]
                * recent_added[weight_col]
                / recent_added["_recent_weight_sum"]
            )
            recent_added["source"] = "assortment_recent_bakery_fallback"
            recent_added = recent_added[[*HOURLY_OUTPUT_COLS, "source"]]
            result = pd.concat([result, recent_added], ignore_index=True)

    if not recent_added.empty:
        recent_filled_keys = recent_added[keys].drop_duplicates()
        all_filled_keys = (
            recent_filled_keys
            if all_filled_keys.empty
            else pd.concat(
                [all_filled_keys, recent_filled_keys], ignore_index=True
            ).drop_duplicates()
        )
    remaining = missing.merge(
        all_filled_keys.assign(_filled=1),
        on=keys,
        how="left",
    )
    remaining = remaining[remaining["_filled"].isna()].drop(columns=["_filled"])

    network_added = pd.DataFrame(columns=[*HOURLY_OUTPUT_COLS, "source"])
    if not remaining.empty and not result.empty:
        network_keys = [DATE_COL, DOW_COL, HOUR_COL]
        network_weights = result.groupby(
            [*network_keys, PRODUCT_ID_COL], as_index=False
        )[SKU_HOUR_FORECAST_COL].sum()
        network_weights["_network_hour_total"] = network_weights.groupby(
            network_keys
        )[SKU_HOUR_FORECAST_COL].transform("sum")
        network_weights = network_weights[
            network_weights["_network_hour_total"].gt(0.0)
        ].copy()
        network_weights["_network_product_share"] = (
            network_weights[SKU_HOUR_FORECAST_COL]
            / network_weights["_network_hour_total"]
        )
        network_added = remaining.merge(
            network_weights[
                [*network_keys, PRODUCT_ID_COL, "_network_product_share"]
            ],
            on=network_keys,
            how="left",
        )
        network_added = network_added[
            network_added["_network_product_share"].notna()
        ].copy()
        if not network_added.empty:
            network_added, _ = filter_by_active_assortment(
                network_added,
                allowed_pairs=(
                    allowed_pairs
                    if allowed_pairs is not None
                    else pd.DataFrame(columns=[CITY_COL, PRODUCT_ID_COL])
                ),
                bakery_city_lookup=(
                    bakery_city_lookup
                    if bakery_city_lookup is not None
                    else pd.DataFrame(columns=[BAKERY_ID_COL, CITY_COL])
                ),
                forecast_col="_network_product_share",
            )
            network_added["_network_weight_sum"] = network_added.groupby(keys)[
                "_network_product_share"
            ].transform("sum")
            network_added = network_added[
                network_added["_network_weight_sum"].gt(0.0)
            ]
            network_added[SKU_HOUR_FORECAST_COL] = (
                network_added[BAKERY_HOUR_FORECAST_COL]
                * network_added["_network_product_share"]
                / network_added["_network_weight_sum"]
            )
            network_added["source"] = "assortment_network_hour_fallback"
            network_added = network_added[[*HOURLY_OUTPUT_COLS, "source"]]
            result = pd.concat([result, network_added], ignore_index=True)

    if not network_added.empty:
        network_filled_keys = network_added[keys].drop_duplicates()
        all_filled_keys = (
            network_filled_keys
            if all_filled_keys.empty
            else pd.concat(
                [all_filled_keys, network_filled_keys], ignore_index=True
            ).drop_duplicates()
        )
    remaining = missing.merge(
        all_filled_keys.assign(_filled=1),
        on=keys,
        how="left",
    )
    remaining = remaining[remaining["_filled"].isna()].drop(columns=["_filled"])

    uniform_added = pd.DataFrame(columns=[*HOURLY_OUTPUT_COLS, "source"])
    if (
        not remaining.empty
        and bakery_city_lookup is not None
        and allowed_pairs is not None
        and not allowed_pairs.empty
    ):
        uniform_added = remaining.merge(
            bakery_city_lookup,
            on=BAKERY_ID_COL,
            how="left",
        ).merge(allowed_pairs, on=CITY_COL, how="left")
        uniform_added = uniform_added.dropna(subset=[PRODUCT_ID_COL]).copy()
        uniform_added["_allowed_count"] = uniform_added.groupby(keys)[
            PRODUCT_ID_COL
        ].transform("count")
        uniform_added[SKU_HOUR_FORECAST_COL] = (
            uniform_added[BAKERY_HOUR_FORECAST_COL]
            / uniform_added["_allowed_count"]
        )
        uniform_added["source"] = "assortment_uniform_fallback"
        uniform_added = uniform_added[[*HOURLY_OUTPUT_COLS, "source"]]
        result = pd.concat([result, uniform_added], ignore_index=True)

    if not uniform_added.empty:
        all_filled_keys = pd.concat(
            [all_filled_keys, uniform_added[keys]],
            ignore_index=True,
        ).drop_duplicates()
    total_filled = int(all_filled_keys.shape[0])
    return result, {
        "groups_filled": total_filled,
        "groups_unfilled": int(len(missing) - total_filled),
    }


def load_profile_lookup_frames(
    client,
    *,
    profile_table: str,
    bakery_ids: list | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Optional bakery filter keeps queries fast when only a subset is needed.
    # Without it the full-table GROUP BY can hit LB connection timeouts (~70s).
    bak_filter = ""
    if bakery_ids:
        ids_str = ", ".join(str(int(b)) for b in bakery_ids)
        bak_filter = f"and toInt64(bakery_id) in ({ids_str})"

    tier1_sums = client.query_df(
        f"""
        select
            bakery_id,
            dow,
            hour,
            sum(mean_sku_share_in_hour_norm) as tier1_share_sum
        from {profile_table}
        where n_days >= {MIN_TIER1_N_DAYS}
          {bak_filter}
        group by bakery_id, dow, hour
        """
    )
    fallback = client.query_df(
        f"""
        select
            bakery_id,
            hour,
            product_id,
            avg(mean_sku_share_in_hour_norm) as mean_sku_share_in_hour_norm
        from {profile_table}
        where n_days >= {MIN_FALLBACK_N_DAYS}
          {bak_filter}
        group by bakery_id, hour, product_id
        """
    )
    fallback_sums = (
        fallback.groupby([BAKERY_ID_COL, HOUR_COL], as_index=False)[SKU_SHARE_COL]
        .sum()
        .rename(columns={SKU_SHARE_COL: "profile_sum"})
    )
    fallback = fallback.merge(fallback_sums, on=[BAKERY_ID_COL, HOUR_COL], how="left")
    fallback[SKU_SHARE_COL] = np.where(
        fallback["profile_sum"] > 0,
        fallback[SKU_SHARE_COL] / fallback["profile_sum"],
        0.0,
    )
    fallback = fallback.drop(columns=["profile_sum"])

    thin_triples = client.query_df(
        f"""
        select distinct bakery_id, dow, hour, 1 as is_thin
        from {profile_table}
        where n_days < {MIN_TIER1_N_DAYS}
          {bak_filter}
        """
    )
    return tier1_sums, fallback, thin_triples


def load_uplift_multipliers(
    client,
    *,
    uplift_table: str,
    profile_version: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    where = ""
    if profile_version:
        safe_version = profile_version.replace("'", "''")
        where = f"where profile_version = '{safe_version}'"

    exact = client.query_df(
        f"""
        select bakery_id, dow, hour, argMax(sku_uplift_multiplier, generated_at) as sku_uplift_multiplier
        from {uplift_table}
        {where}
        {"and" if where else "where"} dow >= 0
        group by bakery_id, dow, hour
        """
    )
    fallback = client.query_df(
        f"""
        select bakery_id, hour, argMax(sku_uplift_multiplier, generated_at) as sku_uplift_multiplier
        from {uplift_table}
        {where}
        {"and" if where else "where"} dow = -1
        group by bakery_id, hour
        """
    )
    return exact, fallback


def apply_multipliers(
    shares: pd.DataFrame,
    multipliers: pd.DataFrame,
    *,
    keys: list[str],
) -> pd.DataFrame:
    if shares.empty or multipliers.empty:
        return shares
    work = shares.merge(multipliers, on=keys, how="left", validate="many_to_one")
    multiplier = pd.to_numeric(
        work[SKU_UPLIFT_MULTIPLIER_COL],
        errors="coerce",
    ).fillna(1.0)
    work[SKU_SHARE_COL] = pd.to_numeric(work[SKU_SHARE_COL], errors="coerce").fillna(0.0) * multiplier
    return work.drop(columns=[SKU_UPLIFT_MULTIPLIER_COL])


def _update_source_stats(stats: dict[str, dict[str, float | int]], df: pd.DataFrame) -> None:
    if df.empty or "source" not in df.columns:
        return
    grouped = (
        df.groupby("source", as_index=False)
        .agg(
            rows=(SKU_HOUR_FORECAST_COL, "size"),
            forecast_sum=(SKU_HOUR_FORECAST_COL, "sum"),
        )
    )
    for row in grouped.to_dict("records"):
        source = str(row["source"])
        if source not in stats:
            stats[source] = {"rows": 0, "forecast_sum": 0.0}
        stats[source]["rows"] = int(stats[source]["rows"]) + int(row["rows"])
        stats[source]["forecast_sum"] = float(stats[source]["forecast_sum"]) + float(
            row["forecast_sum"]
        )


def _finalize_source_stats(stats: dict[str, dict[str, float | int]]) -> list[dict]:
    total_rows = sum(int(v["rows"]) for v in stats.values())
    total_forecast = sum(float(v["forecast_sum"]) for v in stats.values())
    result = []
    for source, values in sorted(stats.items()):
        rows = int(values["rows"])
        forecast_sum = float(values["forecast_sum"])
        result.append(
            {
                "source": source,
                "rows": rows,
                "row_share": round(rows / total_rows, 6) if total_rows else 0.0,
                "forecast_sum": round(forecast_sum, 6),
                "forecast_share": round(forecast_sum / total_forecast, 6)
                if total_forecast
                else 0.0,
            }
        )
    return result


def stream_profile_chunks(
    client,
    *,
    profile_table: str,
    chunk_size: int,
    bakery_ids: list | None = None,
):
    bak_filter = ""
    if bakery_ids:
        ids_str = ", ".join(str(int(b)) for b in bakery_ids)
        bak_filter = f"and toInt64(bakery_id) in ({ids_str})"
    query = f"""
        select
            bakery_id,
            dow,
            hour,
            product_id,
            n_days,
            mean_sku_share_in_hour_norm
        from {profile_table}
        where n_days >= {MIN_TIER1_N_DAYS}
          {bak_filter}
        order by bakery_id, dow, hour, product_id
    """
    with client.query_df_stream(query, settings={"max_block_size": chunk_size}) as stream:
        for block in stream:
            if not block.empty:
                yield block


def _recent_sales_source_sql(sales_table: str) -> str:
    if sales_table == RAW_SALES_LINE_TABLE:
        return f"""
        (
            select distinct
                fcl.check_datetime as check_datetime,
                fcl.check_date as check_date,
                fcl.bakery_id as bakery_id,
                fcl.product_id as product_id,
                fcl.quantity as quantity,
                db.city as city,
                dp.product_name as product_name,
                dp.category_name as category_name
            from {RAW_SALES_LINE_TABLE} as fcl
            any left join Svezhar.dim_bakeries as db
                on db.bakery_id = fcl.bakery_id
            any left join Svezhar.dim_products as dp
                on dp.product_id = fcl.product_id
            where hex(fcl.cash_event_type) = '{SALES_EVENT_HEX}'
              and fcl.check_date between %(recent_start)s and %(recent_end)s
        )
        """
    return sales_table


def load_recent_assortment_stats(
    client,
    *,
    forecast_start: pd.Timestamp,
    recent_days: int,
    sales_table: str,
    bakery_ids: list[int] | None = None,
) -> pd.DataFrame:
    recent_end = forecast_start - pd.Timedelta(days=1)
    recent_start = forecast_start - pd.Timedelta(days=recent_days)
    source_sql = _recent_sales_source_sql(sales_table)
    bakery_clause = (
        "  and toInt64OrNull(toString(bakery_id)) in %(bakery_ids)s\n"
        if bakery_ids else ""
    )
    params: dict = {
        "recent_start": str(recent_start.date()),
        "recent_end": str(recent_end.date()),
    }
    if bakery_ids:
        params["bakery_ids"] = list(bakery_ids)
    stats = client.query_df(
        f"""
        select
            toInt64OrNull(toString(bakery_id)) as bakery_id,
            toInt64OrNull(toString(product_id)) as product_id,
            any(city) as city,
            any(product_name) as product_name,
            any(category_name) as category_name,
            sum(toFloat64(quantity)) as recent_qty,
            uniqExact(check_date) as recent_days_sold
        from {source_sql}
        where check_date between %(recent_start)s and %(recent_end)s
          and toInt64OrNull(toString(bakery_id)) is not null
          and toInt64OrNull(toString(product_id)) is not null
          and toFloat64(quantity) > 0
{bakery_clause}        group by bakery_id, product_id
        """,
        parameters=params,
    )
    if stats.empty:
        stats["bakery_recent_qty"] = pd.Series(dtype=float)
        stats["recent_share"] = pd.Series(dtype=float)
        return stats
    totals = (
        stats.groupby(BAKERY_ID_COL, as_index=False)["recent_qty"]
        .sum()
        .rename(columns={"recent_qty": "bakery_recent_qty"})
    )
    stats = stats.merge(totals, on=BAKERY_ID_COL, how="left", validate="many_to_one")
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


def load_recent_daily_share_stats(
    client,
    *,
    forecast_start: pd.Timestamp,
    recent_days: int,
    sales_table: str,
    bakery_ids: list[int] | None = None,
) -> pd.DataFrame:
    recent_end = forecast_start - pd.Timedelta(days=1)
    recent_start = forecast_start - pd.Timedelta(days=recent_days)
    source_sql = _recent_sales_source_sql(sales_table)
    bakery_clause = (
        "  and toInt64OrNull(toString(bakery_id)) in %(bakery_ids)s\n"
        if bakery_ids else ""
    )
    params: dict = {
        "recent_start": str(recent_start.date()),
        "recent_end": str(recent_end.date()),
    }
    if bakery_ids:
        params["bakery_ids"] = list(bakery_ids)
    daily = client.query_df(
        f"""
        select
            check_date,
            toInt64OrNull(toString(bakery_id)) as bakery_id,
            toInt64OrNull(toString(product_id)) as product_id,
            sum(toFloat64(quantity)) as recent_day_qty
        from {source_sql}
        where check_date between %(recent_start)s and %(recent_end)s
          and toInt64OrNull(toString(bakery_id)) is not null
          and toInt64OrNull(toString(product_id)) is not null
          and toFloat64(quantity) > 0
{bakery_clause}        group by check_date, bakery_id, product_id
        """,
        parameters=params,
    )
    if daily.empty:
        return pd.DataFrame(columns=[BAKERY_ID_COL, PRODUCT_ID_COL])

    daily["check_date"] = pd.to_datetime(daily["check_date"], errors="coerce")
    daily = daily.dropna(subset=["check_date"])
    daily["recent_day_qty"] = pd.to_numeric(
        daily["recent_day_qty"],
        errors="coerce",
    ).fillna(0.0)
    bakery_days = (
        daily.groupby(["check_date", BAKERY_ID_COL], as_index=False)["recent_day_qty"]
        .sum()
        .rename(columns={"recent_day_qty": "bakery_recent_day_qty"})
    )
    pairs = daily[[BAKERY_ID_COL, PRODUCT_ID_COL]].drop_duplicates()
    grid = bakery_days.merge(pairs, on=BAKERY_ID_COL, how="inner")
    daily = grid.merge(
        daily[["check_date", BAKERY_ID_COL, PRODUCT_ID_COL, "recent_day_qty"]],
        on=["check_date", BAKERY_ID_COL, PRODUCT_ID_COL],
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
    daily[DOW_COL] = daily["check_date"].dt.dayofweek
    daily["is_weekend"] = daily[DOW_COL].isin([5, 6]).astype("int64")

    overall = (
        daily.groupby([BAKERY_ID_COL, PRODUCT_ID_COL], as_index=False)
        .agg(
            recent_share_daily_winsor=("daily_share", _winsor_mean),
            recent_daily_obs=("daily_share", "size"),
        )
    )
    weekpart = (
        daily.groupby([BAKERY_ID_COL, PRODUCT_ID_COL, "is_weekend"], as_index=False)
        .agg(
            recent_share_weekpart_winsor=("daily_share", _winsor_mean),
            recent_weekpart_obs=("daily_share", "size"),
        )
    )
    dow = (
        daily.groupby([BAKERY_ID_COL, PRODUCT_ID_COL, DOW_COL], as_index=False)
        .agg(
            recent_share_dow_winsor=("daily_share", _winsor_mean),
            recent_dow_obs=("daily_share", "size"),
            recent_dow_avg_qty=("recent_day_qty", "mean"),
        )
    )
    return overall.merge(weekpart, on=[BAKERY_ID_COL, PRODUCT_ID_COL], how="left").merge(
        dow,
        on=[BAKERY_ID_COL, PRODUCT_ID_COL],
        how="left",
    )


def _attach_city_recent_prior(
    candidates: pd.DataFrame,
    recent: pd.DataFrame,
) -> pd.DataFrame:
    if CITY_COL not in recent.columns:
        recent = recent.copy()
        recent[CITY_COL] = ""
    city_product = (
        recent[[CITY_COL, BAKERY_ID_COL, PRODUCT_ID_COL, "recent_qty"]]
        .drop_duplicates([CITY_COL, BAKERY_ID_COL, PRODUCT_ID_COL])
        .groupby([CITY_COL, PRODUCT_ID_COL], as_index=False, dropna=False)["recent_qty"]
        .sum()
    )
    city_total = (
        city_product.groupby(CITY_COL, as_index=False, dropna=False)["recent_qty"]
        .sum()
        .rename(columns={"recent_qty": "city_recent_total_qty"})
    )
    city_product = city_product.merge(city_total, on=CITY_COL, how="left", validate="many_to_one")
    city_product["city_recent_share"] = np.where(
        city_product["city_recent_total_qty"] > 0,
        city_product["recent_qty"] / city_product["city_recent_total_qty"],
        0.0,
    )
    city_product["city_recent_rank"] = city_product.groupby(CITY_COL, dropna=False)[
        "city_recent_share"
    ].rank(method="first", ascending=False)
    return candidates.merge(
        city_product[
            [
                CITY_COL,
                PRODUCT_ID_COL,
                "city_recent_share",
                "city_recent_rank",
            ]
        ],
        on=[CITY_COL, PRODUCT_ID_COL],
        how="left",
        validate="many_to_one",
    )


def _contains_pattern(series: pd.Series, pattern: str, *, regex: bool) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.casefold()
        .str.contains(pattern, regex=regex)
    )


def _apply_category_upward_cap(
    candidates: pd.DataFrame,
    *,
    category_pattern: str | None,
    max_multiplier: float,
    recent_window_days: int | None = None,
    recent_absolute_cap_multiplier: float = DEFAULT_RECENT_ABSOLUTE_CAP_MULTIPLIER,
    recent_daily: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if not category_pattern:
        return candidates
    if max_multiplier < 0:
        raise ValueError("max_multiplier must be non-negative")
    if candidates.empty or CATEGORY_COL not in candidates.columns:
        return candidates

    work = candidates.copy()
    protected = _contains_pattern(work[CATEGORY_COL], category_pattern, regex=True)
    if not protected.any():
        return work

    corrected = pd.to_numeric(work["corrected_daily_forecast"], errors="coerce").fillna(0.0)
    base = pd.to_numeric(work["base_daily_forecast"], errors="coerce").fillna(0.0)
    cap = (base * max_multiplier).clip(lower=0.0)

    # DOW-aware recent avg cap: prefer recent_dow_avg_qty from recent_daily if available
    if (
        recent_daily is not None
        and not recent_daily.empty
        and "recent_dow_avg_qty" in recent_daily.columns
        and DOW_COL in work.columns
    ):
        dow_avg = recent_daily[[BAKERY_ID_COL, PRODUCT_ID_COL, DOW_COL, "recent_dow_avg_qty"]].drop_duplicates(
            subset=[BAKERY_ID_COL, PRODUCT_ID_COL, DOW_COL]
        ).copy()
        dow_avg["recent_dow_avg_qty"] = pd.to_numeric(
            dow_avg["recent_dow_avg_qty"], errors="coerce"
        ).fillna(0.0)
        work = work.merge(dow_avg, on=[BAKERY_ID_COL, PRODUCT_ID_COL, DOW_COL], how="left")
        # recompute after merge so all series share the same (potentially expanded) index
        corrected = pd.to_numeric(work["corrected_daily_forecast"], errors="coerce").fillna(0.0)
        base = pd.to_numeric(work["base_daily_forecast"], errors="coerce").fillna(0.0)
        cap = (base * max_multiplier).clip(lower=0.0)
        protected = _contains_pattern(work[CATEGORY_COL], category_pattern, regex=True)
        recent_avg_cap = (
            pd.to_numeric(work["recent_dow_avg_qty"], errors="coerce").fillna(0.0)
            * recent_absolute_cap_multiplier
        ).clip(lower=0.0)
        has_recent_cap = recent_avg_cap.gt(0.0)
        # For base=0 rows (product not in profile), base*multiplier=0 wrongly clamps
        # the cap to 0. Use recent_avg_cap directly when base=0, same as the elif branch.
        cap = pd.Series(
            np.where(
                has_recent_cap & base.gt(0),
                np.minimum(cap, recent_avg_cap),
                np.where(has_recent_cap, recent_avg_cap, cap),
            ),
            index=work.index,
        )
        work = work.drop(columns=["recent_dow_avg_qty"], errors="ignore")
    elif recent_window_days and recent_window_days > 0 and "recent_qty" in work.columns:
        # fallback: flat avg over the whole window
        recent_avg_cap = (
            pd.to_numeric(work["recent_qty"], errors="coerce")
            / float(recent_window_days)
            * recent_absolute_cap_multiplier
        ).clip(lower=0.0)
        has_recent_cap = recent_avg_cap.notna()
        cap = np.where(
            has_recent_cap & base.gt(0.0),
            np.minimum(cap, recent_avg_cap),
            cap,
        )
        cap = np.where(
            has_recent_cap & base.le(0.0),
            recent_avg_cap,
            cap,
        )
        cap = pd.Series(cap, index=work.index)
    work["_capped_corrected_daily_forecast"] = corrected
    work.loc[protected, "_capped_corrected_daily_forecast"] = np.minimum(
        corrected.loc[protected],
        cap.loc[protected],
    )

    keys = [DATE_COL, BAKERY_ID_COL]
    protected_totals = (
        work.loc[protected]
        .groupby(keys, as_index=False)["_capped_corrected_daily_forecast"]
        .sum()
        .rename(columns={"_capped_corrected_daily_forecast": "_protected_total"})
    )
    nonprotected_totals = (
        work.loc[~protected]
        .groupby(keys, as_index=False)["corrected_daily_forecast"]
        .sum()
        .rename(columns={"corrected_daily_forecast": "_nonprotected_total"})
    )
    group_targets = (
        work[keys + [BAKERY_FORECAST_COL]]
        .drop_duplicates(keys)
        .merge(protected_totals, on=keys, how="left")
        .merge(nonprotected_totals, on=keys, how="left")
    )
    group_targets[["_protected_total", "_nonprotected_total"]] = group_targets[
        ["_protected_total", "_nonprotected_total"]
    ].fillna(0.0)
    group_targets["_nonprotected_target"] = (
        pd.to_numeric(group_targets[BAKERY_FORECAST_COL], errors="coerce").fillna(0.0)
        - group_targets["_protected_total"]
    ).clip(lower=0.0)
    # Cap scale at 1.0: when pies are capped down, do NOT redistribute the
    # freed budget upward to other SKUs — just let the bakery total decrease.
    # Scaling non-pies up inflates high-volume SKUs that were already correct.
    group_targets["_nonprotected_scale"] = np.where(
        group_targets["_nonprotected_total"] > 0,
        np.minimum(
            1.0,
            group_targets["_nonprotected_target"] / group_targets["_nonprotected_total"],
        ),
        1.0,
    )
    work = work.merge(
        group_targets[keys + ["_nonprotected_scale"]],
        on=keys,
        how="left",
        validate="many_to_one",
    )
    work.loc[~protected, "_capped_corrected_daily_forecast"] = (
        corrected.loc[~protected]
        * work.loc[~protected, "_nonprotected_scale"].fillna(1.0)
    )
    work["corrected_daily_forecast"] = work["_capped_corrected_daily_forecast"]
    return work.drop(
        columns=["_capped_corrected_daily_forecast", "_nonprotected_scale"],
        errors="ignore",
    )


def _build_recent_correction_targets(
    hourly: pd.DataFrame,
    recent: pd.DataFrame,
    *,
    mode: str,
    recent_daily: pd.DataFrame | None = None,
    category_upward_cap_pattern: str | None = DEFAULT_RECENT_UPWARD_CAP_CATEGORY_PATTERN,
    category_upward_cap_multiplier: float = DEFAULT_RECENT_UPWARD_CAP_MULTIPLIER,
    category_recent_absolute_cap_days: int | None = None,
    category_recent_absolute_cap_multiplier: float = DEFAULT_RECENT_ABSOLUTE_CAP_MULTIPLIER,
) -> pd.DataFrame:
    base_daily = (
        hourly.groupby([DATE_COL, DOW_COL, BAKERY_ID_COL, PRODUCT_ID_COL], as_index=False)
        .agg(base_daily_forecast=(SKU_HOUR_FORECAST_COL, "sum"))
    )
    bakery_day = (
        base_daily.groupby([DATE_COL, BAKERY_ID_COL], as_index=False)["base_daily_forecast"]
        .sum()
        .rename(columns={"base_daily_forecast": BAKERY_FORECAST_COL})
    )
    dates_by_bakery = base_daily[[DATE_COL, DOW_COL, BAKERY_ID_COL]].drop_duplicates()
    recent_active = recent.loc[
        pd.to_numeric(recent["recent_days_sold"], errors="coerce").fillna(0) > 0,
        [BAKERY_ID_COL, PRODUCT_ID_COL],
    ].drop_duplicates()
    recent_grid = dates_by_bakery.merge(recent_active, on=BAKERY_ID_COL, how="inner")
    candidates = pd.concat(
        [
            base_daily[[DATE_COL, DOW_COL, BAKERY_ID_COL, PRODUCT_ID_COL]],
            recent_grid[[DATE_COL, DOW_COL, BAKERY_ID_COL, PRODUCT_ID_COL]],
        ],
        ignore_index=True,
    ).drop_duplicates([DATE_COL, BAKERY_ID_COL, PRODUCT_ID_COL])
    candidates = candidates.merge(
        base_daily,
        on=[DATE_COL, DOW_COL, BAKERY_ID_COL, PRODUCT_ID_COL],
        how="left",
        validate="one_to_one",
    )
    candidates["base_daily_forecast"] = (
        pd.to_numeric(candidates["base_daily_forecast"], errors="coerce").fillna(0.0)
    )
    candidates = candidates.merge(bakery_day, on=[DATE_COL, BAKERY_ID_COL], how="left", validate="many_to_one")
    candidates = candidates.merge(
        recent[
            [
                col
                for col in [
                    BAKERY_ID_COL,
                    PRODUCT_ID_COL,
                    CITY_COL,
                    PRODUCT_NAME_COL,
                    CATEGORY_COL,
                    "recent_qty",
                    "recent_days_sold",
                    "recent_share",
                ]
                if col in recent.columns
            ]
        ],
        on=[BAKERY_ID_COL, PRODUCT_ID_COL],
        how="left",
        validate="many_to_one",
    )
    candidates[["recent_qty", "recent_days_sold", "recent_share"]] = candidates[
        ["recent_qty", "recent_days_sold", "recent_share"]
    ].fillna(0.0)
    for col in [CITY_COL, PRODUCT_NAME_COL, CATEGORY_COL]:
        if col not in candidates.columns:
            candidates[col] = ""
        candidates[col] = candidates[col].fillna("")
    candidates["prod_share"] = np.where(
        candidates[BAKERY_FORECAST_COL] > 0,
        candidates["base_daily_forecast"] / candidates[BAKERY_FORECAST_COL],
        0.0,
    )
    active = candidates["recent_days_sold"] > 0
    if mode == "dead_0d":
        candidates["raw_share"] = np.where(active, candidates["prod_share"], 0.0)
    elif mode == "blend_recent_50":
        candidates["raw_share"] = np.where(
            active,
            0.5 * candidates["prod_share"] + 0.5 * candidates["recent_share"],
            0.0,
        )
    elif mode == "core_recent_70":
        core = (candidates["recent_days_sold"] >= 20) & (candidates["recent_share"] >= 0.01)
        regular_share = 0.7 * candidates["prod_share"] + 0.3 * candidates["recent_share"]
        core_share = 0.3 * candidates["prod_share"] + 0.7 * candidates["recent_share"]
        candidates["raw_share"] = np.where(core, core_share, regular_share)
        candidates["raw_share"] = np.where(active, candidates["raw_share"], 0.0)
    elif mode == "runner_city_prior_soft_weekpart":
        candidates["is_weekend"] = candidates[DOW_COL].isin([5, 6]).astype("int64")
        if recent_daily is not None and not recent_daily.empty:
            daily_cols = [
                BAKERY_ID_COL,
                PRODUCT_ID_COL,
                "is_weekend",
                DOW_COL,
                "recent_share_daily_winsor",
                "recent_share_weekpart_winsor",
                "recent_weekpart_obs",
            ]
            candidates = candidates.merge(
                recent_daily[[col for col in daily_cols if col in recent_daily.columns]],
                on=[BAKERY_ID_COL, PRODUCT_ID_COL, "is_weekend", DOW_COL],
                how="left",
                validate="many_to_many",
            )
        for col in ["recent_share_daily_winsor", "recent_share_weekpart_winsor"]:
            if col not in candidates.columns:
                candidates[col] = candidates["recent_share"]
            candidates[col] = pd.to_numeric(candidates[col], errors="coerce").fillna(
                candidates["recent_share"]
            )
        if "recent_weekpart_obs" not in candidates.columns:
            candidates["recent_weekpart_obs"] = 0
        candidates["recent_weekpart_obs"] = (
            pd.to_numeric(candidates["recent_weekpart_obs"], errors="coerce")
            .fillna(0)
            .clip(lower=0, upper=12)
        )
        weekpart_alpha = np.minimum(0.6, 0.6 * candidates["recent_weekpart_obs"] / 12.0)
        modeled_recent_share = (
            weekpart_alpha * candidates["recent_share_weekpart_winsor"]
            + (1.0 - weekpart_alpha) * candidates["recent_share_daily_winsor"]
        )
        modeled_recent_share = pd.to_numeric(modeled_recent_share, errors="coerce").fillna(
            candidates["recent_share"]
        ).clip(lower=0.0)
        # Winsorized share collapses to 0 for intermittent products (most days = 0 share,
        # 90th pct = 0 → _winsor_mean clips everything to 0). Fall back to flat recent_share
        # so products with real but infrequent sales still get a non-zero correction signal.
        flat_recent = pd.to_numeric(candidates["recent_share"], errors="coerce").fillna(0.0)
        modeled_recent_share = pd.Series(
            np.where((modeled_recent_share == 0) & flat_recent.gt(0), flat_recent, modeled_recent_share),
            index=candidates.index,
        )

        is_service = _contains_pattern(candidates[CATEGORY_COL], SERVICE_CATEGORY_PATTERN, regex=True)
        is_eclair = _contains_pattern(candidates[PRODUCT_NAME_COL], ECLAIR_PATTERN, regex=False)
        core = (
            (candidates["recent_days_sold"] >= 20)
            & (modeled_recent_share >= 0.01)
            & ~is_service
        )
        profile_too_high = candidates["prod_share"] > modeled_recent_share * 2.0
        profile_too_low = candidates["prod_share"] < modeled_recent_share * 0.5

        raw_share = 0.3 * candidates["prod_share"] + 0.7 * modeled_recent_share
        raw_share = np.where(
            core,
            0.5 * candidates["prod_share"] + 0.5 * modeled_recent_share,
            raw_share,
        )
        raw_share = np.where(
            core & profile_too_high,
            0.2 * candidates["prod_share"] + 0.8 * modeled_recent_share,
            raw_share,
        )
        raw_share = np.where(
            core & profile_too_low,
            0.3 * candidates["prod_share"] + 0.7 * modeled_recent_share,
            raw_share,
        )
        raw_share = np.where(
            is_eclair,
            np.minimum(modeled_recent_share * 1.3, 0.2 * candidates["prod_share"] + 0.8 * modeled_recent_share),
            raw_share,
        )

        runner = (
            (candidates["recent_days_sold"] >= 20)
            & (modeled_recent_share >= 0.005)
            & ~is_service
        )
        runner_share = np.maximum(
            0.15 * candidates["prod_share"] + 0.85 * modeled_recent_share,
            modeled_recent_share * 0.9,
        )
        raw_share = np.where(runner, runner_share, raw_share)

        candidates = _attach_city_recent_prior(candidates, recent)
        city_share = pd.to_numeric(candidates["city_recent_share"], errors="coerce").fillna(0.0)
        city_rank = pd.to_numeric(candidates["city_recent_rank"], errors="coerce").fillna(999.0)
        rank_floor = np.select(
            [city_rank <= 1, city_rank <= 3, city_rank <= 5],
            [0.75, 0.65, 0.55],
            default=0.0,
        )
        city_runner = (
            (candidates["recent_days_sold"] >= 10)
            & (city_rank <= 5)
            & (city_share >= 0.015)
            & ~is_service
        )
        prior_share = np.maximum(
            modeled_recent_share,
            0.30 * modeled_recent_share + 0.70 * city_share * rank_floor,
        )
        raw_share = np.where(city_runner, np.maximum(raw_share, prior_share), raw_share)
        candidates["raw_share"] = np.where(active, raw_share, 0.0)
    else:
        raise ValueError(f"Unsupported recent correction mode: {mode}")

    raw_sum = (
        candidates.groupby([DATE_COL, BAKERY_ID_COL], as_index=False)["raw_share"]
        .sum()
        .rename(columns={"raw_share": "raw_share_sum"})
    )
    candidates = candidates.merge(raw_sum, on=[DATE_COL, BAKERY_ID_COL], how="left", validate="many_to_one")
    candidates["corrected_daily_forecast"] = np.where(
        candidates["raw_share_sum"] > 0,
        candidates["raw_share"] / candidates["raw_share_sum"] * candidates[BAKERY_FORECAST_COL],
        candidates["base_daily_forecast"],
    )
    candidates = _apply_category_upward_cap(
        candidates,
        category_pattern=category_upward_cap_pattern,
        max_multiplier=category_upward_cap_multiplier,
        recent_window_days=category_recent_absolute_cap_days,
        recent_absolute_cap_multiplier=category_recent_absolute_cap_multiplier,
        recent_daily=recent_daily,
    )
    return candidates


def apply_recent_sku_hour_correction(
    *,
    hourly_path: Path,
    daily_path: Path,
    client,
    mode: str,
    recent_days: int,
    sales_table: str,
    category_upward_cap_pattern: str | None = DEFAULT_RECENT_UPWARD_CAP_CATEGORY_PATTERN,
    category_upward_cap_multiplier: float = DEFAULT_RECENT_UPWARD_CAP_MULTIPLIER,
    category_recent_absolute_cap_days: int | None = None,
    category_recent_absolute_cap_multiplier: float = DEFAULT_RECENT_ABSOLUTE_CAP_MULTIPLIER,
    bakery_ids: list[int] | None = None,
) -> tuple[pd.DataFrame, int, list[dict], pd.DataFrame]:
    hourly = pd.read_csv(hourly_path, encoding="utf-8-sig", parse_dates=[DATE_COL])
    hourly[DATE_COL] = pd.to_datetime(hourly[DATE_COL], errors="coerce")
    forecast_start = hourly[DATE_COL].min()
    if pd.isna(forecast_start):
        return (
            pd.read_csv(daily_path, encoding="utf-8-sig"),
            len(hourly),
            [],
            pd.DataFrame(),
        )

    recent = load_recent_assortment_stats(
        client,
        forecast_start=forecast_start,
        recent_days=recent_days,
        sales_table=sales_table,
        bakery_ids=bakery_ids,
    )
    if recent.empty:
        return (
            pd.read_csv(daily_path, encoding="utf-8-sig"),
            len(hourly),
            [],
            recent,
        )

    recent_daily = pd.DataFrame()
    if mode == "runner_city_prior_soft_weekpart" or category_upward_cap_pattern:
        recent_daily = load_recent_daily_share_stats(
            client,
            forecast_start=forecast_start,
            recent_days=recent_days,
            sales_table=sales_table,
            bakery_ids=bakery_ids,
        )

    targets = _build_recent_correction_targets(
        hourly,
        recent,
        mode=mode,
        recent_daily=recent_daily,
        category_upward_cap_pattern=category_upward_cap_pattern,
        category_upward_cap_multiplier=category_upward_cap_multiplier,
        category_recent_absolute_cap_days=category_recent_absolute_cap_days,
        category_recent_absolute_cap_multiplier=category_recent_absolute_cap_multiplier,
    )
    bakery_hour = (
        hourly.groupby([DATE_COL, DOW_COL, BAKERY_ID_COL, HOUR_COL], as_index=False)
        .agg(bakery_hour_forecast=(SKU_HOUR_FORECAST_COL, "sum"))
    )
    targets["daily_multiplier"] = np.where(
        targets["base_daily_forecast"] > 0,
        targets["corrected_daily_forecast"] / targets["base_daily_forecast"],
        np.nan,
    )
    base = hourly.merge(
        targets[[DATE_COL, BAKERY_ID_COL, PRODUCT_ID_COL, "daily_multiplier"]],
        on=[DATE_COL, BAKERY_ID_COL, PRODUCT_ID_COL],
        how="left",
        validate="many_to_one",
    )
    base["raw_hour_forecast"] = (
        pd.to_numeric(base[SKU_HOUR_FORECAST_COL], errors="coerce").fillna(0.0)
        * pd.to_numeric(base["daily_multiplier"], errors="coerce").fillna(0.0)
    )
    base["source"] = base["source"].fillna("profile") + f"+recent_{mode}"
    base = base[[DATE_COL, DOW_COL, BAKERY_ID_COL, HOUR_COL, PRODUCT_ID_COL, "raw_hour_forecast", "source"]]

    new_daily = targets[
        (targets["base_daily_forecast"] <= 0)
        & (targets["corrected_daily_forecast"] > 0)
    ].copy()
    new_rows = pd.DataFrame()
    if len(new_daily):
        new_rows = new_daily.merge(
            bakery_hour,
            on=[DATE_COL, DOW_COL, BAKERY_ID_COL],
            how="inner",
            validate="many_to_many",
        )
        new_rows["raw_hour_forecast"] = np.where(
            new_rows[BAKERY_FORECAST_COL] > 0,
            new_rows["corrected_daily_forecast"]
            * new_rows["bakery_hour_forecast"]
            / new_rows[BAKERY_FORECAST_COL],
            0.0,
        )
        new_rows["source"] = f"recent_{mode}_new"
        new_rows = new_rows[
            [DATE_COL, DOW_COL, BAKERY_ID_COL, HOUR_COL, PRODUCT_ID_COL, "raw_hour_forecast", "source"]
        ]

    combined = pd.concat([base, new_rows], ignore_index=True)
    raw_hour_sum = (
        combined.groupby([DATE_COL, BAKERY_ID_COL, HOUR_COL], as_index=False)["raw_hour_forecast"]
        .sum()
        .rename(columns={"raw_hour_forecast": "raw_hour_sum"})
    )
    combined = combined.merge(raw_hour_sum, on=[DATE_COL, BAKERY_ID_COL, HOUR_COL], how="left", validate="many_to_one")
    combined = combined.merge(
        bakery_hour[[DATE_COL, BAKERY_ID_COL, HOUR_COL, "bakery_hour_forecast"]],
        on=[DATE_COL, BAKERY_ID_COL, HOUR_COL],
        how="left",
        validate="many_to_one",
    )
    combined[SKU_HOUR_FORECAST_COL] = np.where(
        combined["raw_hour_sum"] > 0,
        combined["raw_hour_forecast"] / combined["raw_hour_sum"] * combined["bakery_hour_forecast"],
        0.0,
    )
    corrected_hourly = combined[
        [DATE_COL, DOW_COL, BAKERY_ID_COL, HOUR_COL, PRODUCT_ID_COL, SKU_HOUR_FORECAST_COL, "source"]
    ].sort_values([BAKERY_ID_COL, PRODUCT_ID_COL, DATE_COL, HOUR_COL])
    corrected_hourly[DATE_COL] = corrected_hourly[DATE_COL].dt.date
    corrected_hourly.to_csv(hourly_path, index=False, encoding="utf-8-sig")

    sku_daily = (
        corrected_hourly.groupby([DATE_COL, DOW_COL, BAKERY_ID_COL, PRODUCT_ID_COL], as_index=False)
        .agg(sku_day_forecast=(SKU_HOUR_FORECAST_COL, "sum"))
        .sort_values([BAKERY_ID_COL, PRODUCT_ID_COL, DATE_COL])
        .reset_index(drop=True)
    )
    sku_daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    source_summary = _finalize_source_stats({})
    source_stats: dict[str, dict[str, float | int]] = {}
    _update_source_stats(source_stats, corrected_hourly.rename(columns={"sku_hour_forecast": SKU_HOUR_FORECAST_COL}))
    source_summary = _finalize_source_stats(source_stats)
    return sku_daily, len(corrected_hourly), source_summary, recent


def allocate_from_clickhouse(
    *,
    bakery_forecast_path: str | Path,
    bakery_hour_profile_path: str | Path,
    output_dir: str | Path,
    env_file: str | Path = DEFAULT_ENV_PATH,
    client=None,
    profile_table: str = PROFILE_TABLE,
    uplift_table: str = UPLIFT_MULTIPLIER_TABLE,
    forecast_col: str = BAKERY_FORECAST_COL,
    output_suffix: str = "",
    use_raw_uplift_multiplier: bool = False,
    uplift_profile_version: str | None = None,
    recent_correction_mode: str = "none",
    recent_correction_days: int = 30,
    recent_sales_table: str = SALES_LINE_TABLE,
    chunk_size: int = SKU_PROFILE_CHUNK_SIZE,
    assortment_table: str = ASSORTMENT_TABLE,
    disable_assortment_filter: bool = False,
    disable_assortment_renormalization: bool = False,
    max_uplift_ratio: float | None = None,
    max_sku_uplift_ratio: float | None = None,
    stockout_correction_version: str | None = None,
    hierarchical_haircut_target_ratio: float | None = None,
    hierarchical_haircut_history_days: int = 7,
    hierarchical_haircut_pair_prior_days: float = 7.0,
    hierarchical_haircut_min_coefficient: float = 0.85,
    recent_category_upward_cap_pattern: str | None = DEFAULT_RECENT_UPWARD_CAP_CATEGORY_PATTERN,
    recent_category_upward_cap_multiplier: float = DEFAULT_RECENT_UPWARD_CAP_MULTIPLIER,
    recent_category_absolute_cap_days: int | None = None,
    recent_category_absolute_cap_multiplier: float = DEFAULT_RECENT_ABSOLUTE_CAP_MULTIPLIER,
    assortment_max_age_days: int = -1,
    assortment_guard_recent_days: int = 7,
    assortment_guard_min_days_sold: int = 2,
    assortment_guard_min_qty: float = 2.0,
    disable_assortment_coverage_guard: bool = False,
    bakery_ids: list[int] | None = None,
) -> dict[str, Path]:
    if recent_correction_mode not in RECENT_CORRECTION_MODES:
        raise ValueError(
            f"recent_correction_mode must be one of {RECENT_CORRECTION_MODES}"
        )
    if client is None:
        client = create_client(env_file)
    bakery_forecast = load_bakery_day_forecast(
        bakery_forecast_path,
        forecast_col=forecast_col,
    )
    bakery_hour_profile = load_bakery_hour_profile(bakery_hour_profile_path)
    hourly_forecast = allocate_bakery_to_hour(bakery_forecast, bakery_hour_profile)
    bakery_city_lookup = _build_bakery_city_lookup(bakery_forecast)
    bakery_ids_for_filter = (
        bakery_forecast[BAKERY_ID_COL].dropna().astype(int).unique().tolist()
    )
    forecast_start = pd.to_datetime(bakery_forecast[DATE_COL]).min().normalize()
    allowed_assortment_pairs = (
        pd.DataFrame(columns=[CITY_COL, PRODUCT_ID_COL])
        if disable_assortment_filter
        else load_active_assortment_pairs(
            client,
            assortment_table=assortment_table,
            effective_date=forecast_start,
        )
    )
    if not disable_assortment_filter:
        validate_assortment_freshness(
            allowed_assortment_pairs,
            expected_cities=set(bakery_city_lookup[CITY_COL].dropna().astype(str)),
            effective_date=forecast_start,
            max_age_days=assortment_max_age_days,
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not disable_assortment_filter and not disable_assortment_coverage_guard:
        recent_guard = load_recent_assortment_stats(
            client,
            forecast_start=forecast_start,
            recent_days=assortment_guard_recent_days,
            sales_table=recent_sales_table,
            bakery_ids=bakery_ids_for_filter or bakery_ids,
        )
        missing_guard = find_recent_sales_missing_from_assortment(
            recent_guard,
            allowed_assortment_pairs,
            bakery_city_lookup=bakery_city_lookup,
            min_days_sold=assortment_guard_min_days_sold,
            min_qty=assortment_guard_min_qty,
        )
        guard_path = out_dir / "assortment_coverage_guard.csv"
        missing_guard.to_csv(guard_path, index=False, encoding="utf-8-sig")
        guard_summary_path = out_dir / "assortment_coverage_guard.json"
        guard_summary = {
            "status": "failed" if not missing_guard.empty else "passed",
            "forecast_start": str(forecast_start.date()),
            "recent_days": assortment_guard_recent_days,
            "min_days_sold": assortment_guard_min_days_sold,
            "min_qty": assortment_guard_min_qty,
            "recent_pairs_checked": int(len(recent_guard)),
            "blocking_pairs": int(len(missing_guard)),
            "details_path": str(guard_path),
        }
        guard_summary_path.write_text(
            json.dumps(guard_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if not missing_guard.empty:
            examples = missing_guard[
                [BAKERY_ID_COL, PRODUCT_ID_COL, "recent_days_sold", "recent_qty"]
            ].head(10).to_dict("records")
            raise RuntimeError(
                "Assortment coverage guard found established recently sold SKU "
                f"outside the selected batch: pairs={len(missing_guard)} "
                f"examples={examples}; report={guard_path}"
            )
    assortment_filter_stats: dict[str, float | int] = {
        "rows_removed": 0,
        "forecast_removed": 0.0,
    }
    # Under raw uplift + recent-correction, filtering here (before
    # recent-correction reads this file back in) would drop excluded-product
    # demand before compensate_for_assortment_exclusion ever gets a chance to
    # redistribute it — defer to the single final filter+compensate pass
    # below instead. Without recent-correction there is no later pass, so
    # filtering must still happen here.
    defer_assortment_filter = use_raw_uplift_multiplier and recent_correction_mode != "none"

    suffix = f"_{output_suffix}" if output_suffix else ""
    hourly_path = out_dir / HOURLY_OUTPUT_NAME.replace(".csv", f"{suffix}.csv")
    daily_path = out_dir / DAILY_OUTPUT_NAME.replace(".csv", f"{suffix}.csv")
    summary_path = out_dir / SUMMARY_OUTPUT_NAME.replace(".json", f"{suffix}.json")
    if hourly_path.exists():
        hourly_path.unlink()

    hourly_cols = [DATE_COL, DOW_COL, HOUR_COL, BAKERY_ID_COL, BAKERY_HOUR_FORECAST_COL]
    if CITY_COL in hourly_forecast.columns:
        hourly_cols.append(CITY_COL)
    hourly_lookup = hourly_forecast[hourly_cols].copy()
    hourly_lookup["_row_id"] = np.arange(len(hourly_lookup))

    # Pass bakery_ids so profile queries are filtered — avoids full-table scans
    # that hit load-balancer connection timeouts on large remote CH clusters.
    bakery_ids_for_filter = [str(value) for value in bakery_ids_for_filter]
    tier1_sums, fallback, thin_triples = load_profile_lookup_frames(
        client,
        profile_table=profile_table,
        bakery_ids=bakery_ids_for_filter,
    )
    exact_keys = tier1_sums[[BAKERY_ID_COL, DOW_COL, HOUR_COL]].drop_duplicates()
    exact_keys["has_exact"] = 1

    exact_multipliers = pd.DataFrame()
    fallback_multipliers = pd.DataFrame()
    if use_raw_uplift_multiplier:
        exact_multipliers, fallback_multipliers = load_uplift_multipliers(
            client,
            uplift_table=uplift_table,
            profile_version=uplift_profile_version,
        )

    stockout_correction_df = pd.DataFrame()
    if stockout_correction_version:
        stockout_correction_df = load_stockout_correction(
            client,
            table=STOCKOUT_CORRECTION_TABLE,
            profile_version=stockout_correction_version,
        )

    daily_parts: list[pd.DataFrame] = []
    source_stats: dict[str, dict[str, float | int]] = {}
    sku_hour_rows = 0
    products_seen: set[str] = set()
    wrote_header = False

    for i, sku_chunk in enumerate(
        stream_profile_chunks(
            client,
            profile_table=profile_table,
            chunk_size=chunk_size,
            bakery_ids=bakery_ids_for_filter,
        ),
        start=1,
    ):
        sku_chunk = sku_chunk.merge(
            tier1_sums,
            on=[BAKERY_ID_COL, DOW_COL, HOUR_COL],
            how="left",
            validate="many_to_one",
        )
        sku_chunk[SKU_SHARE_COL] = (
            pd.to_numeric(sku_chunk[SKU_SHARE_COL], errors="coerce").fillna(0.0)
            / pd.to_numeric(sku_chunk["tier1_share_sum"], errors="coerce").replace(0, np.nan)
        ).fillna(0.0)
        sku_chunk = sku_chunk.drop(columns=["tier1_share_sum"])
        if use_raw_uplift_multiplier:
            sku_chunk = apply_multipliers(
                sku_chunk,
                exact_multipliers,
                keys=[BAKERY_ID_COL, DOW_COL, HOUR_COL],
            )

        merged = hourly_lookup.merge(
            sku_chunk,
            on=[BAKERY_ID_COL, DOW_COL, HOUR_COL],
            how="inner",
            validate="many_to_many",
            sort=False,
        )
        merged["source"] = "exact"
        merged[SKU_HOUR_FORECAST_COL] = (
            merged[BAKERY_HOUR_FORECAST_COL] * merged[SKU_SHARE_COL]
        )
        if defer_assortment_filter:
            # Raw uplift + recent-correction: filtering here would drop
            # excluded-product demand before compensate_for_assortment_exclusion
            # ever sees it (recent-correction reads this file back in as its
            # own input) — defer to the single final pass instead.
            filter_stats = {"rows_removed": 0, "forecast_removed": 0.0}
        else:
            merged, filter_stats = filter_by_active_assortment(
                merged,
                allowed_pairs=allowed_assortment_pairs,
                bakery_city_lookup=bakery_city_lookup,
                forecast_col=SKU_HOUR_FORECAST_COL,
            )
        _merge_filter_stats(assortment_filter_stats, filter_stats)
        merged = merged[[*HOURLY_OUTPUT_COLS, "_row_id", "source"]]
        _write_hourly_chunk(merged, hourly_path, header=not wrote_header)
        wrote_header = True
        sku_hour_rows += len(merged)
        _update_source_stats(source_stats, merged)
        products_seen.update(merged[PRODUCT_ID_COL].dropna().astype(str).unique().tolist())
        daily_parts.append(
            merged.groupby(
                [DATE_COL, DOW_COL, BAKERY_ID_COL, PRODUCT_ID_COL],
                as_index=False,
                sort=False,
            ).agg(sku_day_forecast=(SKU_HOUR_FORECAST_COL, "sum"))
        )
        if i % 10 == 0:
            print(f"processed clickhouse profile chunks: {i}", flush=True)

    unmatched = hourly_lookup.merge(
        exact_keys,
        on=[BAKERY_ID_COL, DOW_COL, HOUR_COL],
        how="left",
    )
    unmatched = unmatched[unmatched["has_exact"].isna()].drop(columns=["has_exact"])
    if len(unmatched):
        unmatched = unmatched.merge(
            thin_triples,
            on=[BAKERY_ID_COL, DOW_COL, HOUR_COL],
            how="left",
        )
        fallback_merged = unmatched.merge(
            fallback,
            on=[BAKERY_ID_COL, HOUR_COL],
            how="inner",
            validate="many_to_many",
            sort=False,
        )
        if use_raw_uplift_multiplier:
            fallback_merged = apply_multipliers(
                fallback_merged,
                fallback_multipliers,
                keys=[BAKERY_ID_COL, HOUR_COL],
            )
        fallback_merged[SKU_HOUR_FORECAST_COL] = (
            fallback_merged[BAKERY_HOUR_FORECAST_COL] * fallback_merged[SKU_SHARE_COL]
        )
        fallback_merged["source"] = np.where(
            fallback_merged["is_thin"].fillna(0).astype(int) == 1,
            "bakery_hour_fallback_thin",
            "bakery_hour_fallback",
        )
        if defer_assortment_filter:
            filter_stats = {"rows_removed": 0, "forecast_removed": 0.0}
        else:
            fallback_merged, filter_stats = filter_by_active_assortment(
                fallback_merged,
                allowed_pairs=allowed_assortment_pairs,
                bakery_city_lookup=bakery_city_lookup,
                forecast_col=SKU_HOUR_FORECAST_COL,
            )
        _merge_filter_stats(assortment_filter_stats, filter_stats)
        fallback_merged = fallback_merged[[*HOURLY_OUTPUT_COLS, "_row_id", "source"]]
        _write_hourly_chunk(fallback_merged, hourly_path, header=not wrote_header)
        sku_hour_rows += len(fallback_merged)
        _update_source_stats(source_stats, fallback_merged)
        products_seen.update(
            fallback_merged[PRODUCT_ID_COL].dropna().astype(str).unique().tolist()
        )
        daily_parts.append(
            fallback_merged.groupby(
                [DATE_COL, DOW_COL, BAKERY_ID_COL, PRODUCT_ID_COL],
                as_index=False,
                sort=False,
            ).agg(sku_day_forecast=(SKU_HOUR_FORECAST_COL, "sum"))
        )

    sku_daily = pd.concat(daily_parts, ignore_index=True)
    sku_daily = (
        sku_daily.groupby(
            [DATE_COL, DOW_COL, BAKERY_ID_COL, PRODUCT_ID_COL],
            as_index=False,
            sort=False,
        )
        .agg(sku_day_forecast=(SKU_DAY_FORECAST_COL, "sum"))
        .sort_values([BAKERY_ID_COL, PRODUCT_ID_COL, DATE_COL])
        .reset_index(drop=True)
    )
    sku_daily.to_csv(daily_path, index=False, encoding="utf-8-sig")

    hourly_final = pd.read_csv(hourly_path, encoding="utf-8-sig")
    hourly_final.drop(columns=["_row_id"], errors="ignore").to_csv(
        hourly_path,
        index=False,
        encoding="utf-8-sig",
    )

    source_summary = _finalize_source_stats(source_stats)
    uplift_cap_stats: dict[str, float | int] = {}
    sku_uplift_cap_stats: dict[str, float | int] = {}
    stockout_correction_stats: dict[str, float | int] = {}
    hierarchical_haircut_stats: dict[str, float | int] = {}
    if recent_correction_mode != "none":
        sku_daily, sku_hour_rows, source_summary, recent_product_weights = (
            apply_recent_sku_hour_correction(
            hourly_path=hourly_path,
            daily_path=daily_path,
            client=client,
            mode=recent_correction_mode,
            recent_days=recent_correction_days,
            sales_table=recent_sales_table,
            category_upward_cap_pattern=recent_category_upward_cap_pattern,
            category_upward_cap_multiplier=recent_category_upward_cap_multiplier,
            category_recent_absolute_cap_days=(
                recent_category_absolute_cap_days or recent_correction_days
            ),
            category_recent_absolute_cap_multiplier=recent_category_absolute_cap_multiplier,
            bakery_ids=bakery_ids,
            )
        )
        hourly_after_recent = pd.read_csv(hourly_path, encoding="utf-8-sig")
        if use_raw_uplift_multiplier and max_sku_uplift_ratio is not None:
            # Cap the complete pre-assortment SKU set. Applying this after
            # assortment compensation would cut away the demand that
            # compensation just redistributed onto the remaining SKUs.
            _forecast_start = pd.to_datetime(bakery_forecast[DATE_COL]).min()
            _recent_for_sku_cap = load_recent_assortment_stats(
                client,
                forecast_start=_forecast_start,
                recent_days=recent_correction_days,
                sales_table=recent_sales_table,
                bakery_ids=bakery_ids,
            )
            hourly_after_recent, sku_uplift_cap_stats = cap_sku_uplift_per_sku(
                hourly_after_recent,
                _recent_for_sku_cap,
                max_ratio=max_sku_uplift_ratio,
            )
        if use_raw_uplift_multiplier and not stockout_correction_df.empty:
            hourly_after_recent, stockout_correction_stats = apply_stockout_hour_correction(
                hourly_after_recent, stockout_correction_df,
            )
        pre_assortment_hourly = hourly_after_recent.copy()
        hourly_after_recent, hour_filter_stats = filter_by_active_assortment(
            hourly_after_recent,
            allowed_pairs=allowed_assortment_pairs,
            bakery_city_lookup=bakery_city_lookup,
            forecast_col=SKU_HOUR_FORECAST_COL,
        )
        _merge_filter_stats(assortment_filter_stats, hour_filter_stats)
        renormalization_stats = {"groups_scaled": 0, "groups_without_sku": 0}
        gap_fill_stats = {"groups_filled": 0, "groups_unfilled": 0}
        assortment_compensation_stats = {"groups_scaled": 0, "groups_without_remaining_rows": 0}
        if not use_raw_uplift_multiplier and not disable_assortment_renormalization:
            hourly_after_recent, gap_fill_stats = fill_missing_bakery_hours(
                hourly_after_recent,
                hourly_forecast,
                bakery_city_lookup=bakery_city_lookup,
                allowed_pairs=allowed_assortment_pairs,
                recent_product_weights=recent_product_weights,
            )
            hourly_after_recent, renormalization_stats = (
                renormalize_hourly_to_bakery_forecast(
                    hourly_after_recent,
                    hourly_forecast,
                )
            )
        elif use_raw_uplift_multiplier and not disable_assortment_renormalization:
            # Raw uplift never renormalizes to the flat bakery-hour total by
            # design (that's the whole point of the uplift), but assortment
            # exclusion still needs *some* compensation — otherwise demand
            # for now-excluded products just vanishes instead of landing on
            # the remaining in-assortment SKUs. See
            # compensate_for_assortment_exclusion's docstring and
            # docs/ops/DECISIONS.md (2026-07-14 entry) for why this is a
            # separate step from renormalize_hourly_to_bakery_forecast.
            hourly_after_recent, assortment_compensation_stats = (
                compensate_for_assortment_exclusion(
                    pre_assortment_hourly,
                    hourly_after_recent,
                    group_keys=[DATE_COL, BAKERY_ID_COL, HOUR_COL],
                    forecast_col=SKU_HOUR_FORECAST_COL,
                )
            )
        if use_raw_uplift_multiplier and max_uplift_ratio is not None:
            # bakery_forecast always has BAKERY_FORECAST_COL regardless of the
            # forecast_col parameter (load_bakery_day_forecast normalises it).
            hourly_after_recent, uplift_cap_stats = cap_sku_uplift_to_bakery_forecast(
                hourly_after_recent,
                bakery_forecast,
                max_ratio=max_uplift_ratio,
                forecast_col=BAKERY_FORECAST_COL,
            )
        if use_raw_uplift_multiplier and hierarchical_haircut_target_ratio is not None:
            _forecast_start = pd.to_datetime(bakery_forecast[DATE_COL]).min()
            _haircut_history = load_hierarchical_haircut_history(
                client,
                forecast_start=_forecast_start,
                history_days=hierarchical_haircut_history_days,
            )
            _bakery_coefficients, _pair_coefficients = (
                build_hierarchical_haircut_coefficients(
                    _haircut_history,
                    target_ratio=hierarchical_haircut_target_ratio,
                    min_coefficient=hierarchical_haircut_min_coefficient,
                    pair_prior_days=hierarchical_haircut_pair_prior_days,
                )
            )
            hourly_after_recent, hierarchical_haircut_stats = apply_hierarchical_haircut(
                hourly_after_recent,
                _bakery_coefficients,
                _pair_coefficients,
            )
            hierarchical_haircut_stats.update(
                {
                    "target_ratio": hierarchical_haircut_target_ratio,
                    "history_days": hierarchical_haircut_history_days,
                    "pair_prior_days": hierarchical_haircut_pair_prior_days,
                    "min_coefficient": hierarchical_haircut_min_coefficient,
                    "bakeries_protected": int(_bakery_coefficients["protect_from_haircut"].sum()),
                    "bakeries_total": int(len(_bakery_coefficients)),
                    "pairs_total": int(len(_pair_coefficients)),
                }
            )
        hourly_after_recent.to_csv(hourly_path, index=False, encoding="utf-8-sig")
        sku_daily = (
            hourly_after_recent.groupby(
                [DATE_COL, DOW_COL, BAKERY_ID_COL, PRODUCT_ID_COL],
                as_index=False,
                sort=False,
            )
            .agg(sku_day_forecast=(SKU_HOUR_FORECAST_COL, "sum"))
            .sort_values([BAKERY_ID_COL, PRODUCT_ID_COL, DATE_COL])
            .reset_index(drop=True)
        )
        sku_daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        sku_hour_rows = int(len(hourly_after_recent))
        products_seen = set(sku_daily[PRODUCT_ID_COL].dropna().astype(str).unique().tolist())
        normalized_source_stats: dict[str, dict[str, float]] = {}
        _update_source_stats(normalized_source_stats, hourly_after_recent)
        source_summary = _finalize_source_stats(normalized_source_stats)

    summary = build_summary_from_daily(
        bakery_forecast,
        hourly_forecast,
        sku_daily,
        sku_hour_rows=sku_hour_rows,
        products=len(products_seen),
        source_stats=source_summary,
    )
    if recent_correction_mode != "none":
        summary["recent_correction"] = {
            "mode": recent_correction_mode,
            "recent_days": recent_correction_days,
            "sales_table": recent_sales_table,
            "category_upward_cap_pattern": recent_category_upward_cap_pattern,
            "category_upward_cap_multiplier": recent_category_upward_cap_multiplier,
            "category_absolute_cap_days": (
                recent_category_absolute_cap_days or recent_correction_days
            ),
            "category_absolute_cap_multiplier": recent_category_absolute_cap_multiplier,
        }
    summary["assortment_filter"] = {
        "enabled": not disable_assortment_filter,
        "renormalization_enabled": not disable_assortment_renormalization,
        "assortment_table": assortment_table,
        "allowed_pairs": int(len(allowed_assortment_pairs)),
        "rows_removed": int(assortment_filter_stats["rows_removed"]),
        "forecast_removed": round(float(assortment_filter_stats["forecast_removed"]), 6),
    }
    if recent_correction_mode != "none" and not use_raw_uplift_multiplier:
        summary["assortment_filter"]["renormalization"] = renormalization_stats
        summary["assortment_filter"]["hour_gap_fallback"] = gap_fill_stats
    if recent_correction_mode != "none" and use_raw_uplift_multiplier:
        summary["assortment_filter"]["exclusion_compensation"] = assortment_compensation_stats
    if use_raw_uplift_multiplier and uplift_cap_stats:
        summary["uplift_cap"] = uplift_cap_stats
    if use_raw_uplift_multiplier and sku_uplift_cap_stats:
        summary["sku_uplift_cap"] = sku_uplift_cap_stats
    if use_raw_uplift_multiplier and stockout_correction_stats:
        summary["stockout_correction"] = stockout_correction_stats
    if use_raw_uplift_multiplier and hierarchical_haircut_stats:
        summary["hierarchical_haircut"] = hierarchical_haircut_stats
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"sku_hourly": hourly_path, "sku_daily": daily_path, "summary": summary_path}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply bakery profiles using ClickHouse-stored SKU profiles"
    )
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH))
    parser.add_argument("--bakery-forecast-path", required=True)
    parser.add_argument(
        "--bakery-hour-profile-path",
        default=str(DEFAULT_BAKERY_HOUR_PROFILE_PATH),
    )
    parser.add_argument("--profile-table", default=PROFILE_TABLE)
    parser.add_argument("--uplift-table", default=UPLIFT_MULTIPLIER_TABLE)
    parser.add_argument("--forecast-col", default=BAKERY_FORECAST_COL)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--use-raw-uplift-multiplier", action="store_true")
    parser.add_argument("--uplift-profile-version", default=None)
    parser.add_argument(
        "--recent-correction-mode",
        choices=RECENT_CORRECTION_MODES,
        default="none",
    )
    parser.add_argument("--recent-correction-days", type=int, default=30)
    parser.add_argument("--recent-sales-table", default=SALES_LINE_TABLE)
    parser.add_argument("--chunk-size", type=int, default=SKU_PROFILE_CHUNK_SIZE)
    parser.add_argument("--assortment-table", default=ASSORTMENT_TABLE)
    parser.add_argument("--disable-assortment-filter", action="store_true")
    parser.add_argument("--disable-assortment-renormalization", action="store_true")
    parser.add_argument(
        "--max-uplift-ratio",
        type=float,
        default=None,
        help=(
            "When --use-raw-uplift-multiplier is set, cap the per-(date, bakery) "
            "SKU-hour sum to at most this multiple of the bakery-day forecast. "
            "E.g. 1.35 limits uplift to +35%%. Default: no cap."
        ),
    )
    parser.add_argument(
        "--max-sku-uplift-ratio",
        type=float,
        default=None,
        help=(
            "When --use-raw-uplift-multiplier is set, cap each per-(date, bakery, "
            "product) SKU-hour sum to at most this multiple of the rolling daily "
            "mean from recent sales. E.g. 1.2 limits per-SKU uplift to +20%%. "
            "Default: no cap."
        ),
    )
    parser.add_argument(
        "--stockout-correction-version",
        default=None,
        help=(
            "Profile version from sku_hour_stockout_correction_embedded. "
            "When set, applies per-(bakery, product, hour) stockout correction "
            "after the SKU cap step. Only active with --use-raw-uplift-multiplier."
        ),
    )
    parser.add_argument("--hierarchical-haircut-target-ratio", type=float, default=None)
    parser.add_argument("--hierarchical-haircut-history-days", type=int, default=7)
    parser.add_argument("--hierarchical-haircut-pair-prior-days", type=float, default=7.0)
    parser.add_argument("--hierarchical-haircut-min-coefficient", type=float, default=0.85)
    parser.add_argument(
        "--recent-category-upward-cap-pattern",
        default=DEFAULT_RECENT_UPWARD_CAP_CATEGORY_PATTERN,
        help=(
            "Regex over category_name for costly categories that recent correction "
            "may not lift above base profile forecast. Empty string disables."
        ),
    )
    parser.add_argument(
        "--recent-category-upward-cap-multiplier",
        type=float,
        default=DEFAULT_RECENT_UPWARD_CAP_MULTIPLIER,
    )
    parser.add_argument(
        "--recent-category-absolute-cap-days",
        type=int,
        default=0,
        help=(
            "Calendar-day window for recent absolute cap. 0 means use "
            "--recent-correction-days."
        ),
    )
    parser.add_argument(
        "--recent-category-absolute-cap-multiplier",
        type=float,
        default=DEFAULT_RECENT_ABSOLUTE_CAP_MULTIPLIER,
    )
    args = parser.parse_args()

    paths = allocate_from_clickhouse(
        bakery_forecast_path=args.bakery_forecast_path,
        bakery_hour_profile_path=args.bakery_hour_profile_path,
        output_dir=args.output_dir,
        env_file=args.env_file,
        profile_table=args.profile_table,
        uplift_table=args.uplift_table,
        forecast_col=args.forecast_col,
        output_suffix=args.output_suffix,
        use_raw_uplift_multiplier=args.use_raw_uplift_multiplier,
        uplift_profile_version=args.uplift_profile_version,
        recent_correction_mode=args.recent_correction_mode,
        recent_correction_days=args.recent_correction_days,
        recent_sales_table=args.recent_sales_table,
        chunk_size=args.chunk_size,
        assortment_table=args.assortment_table,
        disable_assortment_filter=args.disable_assortment_filter,
        disable_assortment_renormalization=args.disable_assortment_renormalization,
        max_uplift_ratio=args.max_uplift_ratio,
        max_sku_uplift_ratio=args.max_sku_uplift_ratio,
        stockout_correction_version=args.stockout_correction_version,
        hierarchical_haircut_target_ratio=args.hierarchical_haircut_target_ratio,
        hierarchical_haircut_history_days=args.hierarchical_haircut_history_days,
        hierarchical_haircut_pair_prior_days=args.hierarchical_haircut_pair_prior_days,
        hierarchical_haircut_min_coefficient=args.hierarchical_haircut_min_coefficient,
        recent_category_upward_cap_pattern=(
            args.recent_category_upward_cap_pattern or None
        ),
        recent_category_upward_cap_multiplier=args.recent_category_upward_cap_multiplier,
        recent_category_absolute_cap_days=(
            args.recent_category_absolute_cap_days or None
        ),
        recent_category_absolute_cap_multiplier=args.recent_category_absolute_cap_multiplier,
    )

    print("=" * 72)
    print("APPLY BAKERY PROFILES FROM CLICKHOUSE")
    print("=" * 72)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

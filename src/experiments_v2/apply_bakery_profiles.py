"""
Apply bakery-driven profiles to daily bakery forecasts.

Pipeline:
  1. bakery_day_forecast
  2. bakery_hour_profile
  3. sku_hour_share_profile

Outputs:
  - sku_hour_forecast.csv
  - sku_day_forecast.csv
  - apply_bakery_profiles_summary.json
"""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATE_COL = "date"
DOW_COL = "dow"
HOUR_COL = "hour"
BAKERY_ID_COL = "bakery_id"
BAKERY_NAME_COL = "bakery_name"
CITY_COL = "city"
PRODUCT_ID_COL = "product_id"
PRODUCT_NAME_COL = "product_name"
CATEGORY_COL = "category_name"

BAKERY_FORECAST_COL = "bakery_day_forecast"
BAKERY_HOUR_SHARE_COL = "mean_hour_share_norm"
BAKERY_HOUR_FORECAST_COL = "bakery_hour_forecast"
SKU_SHARE_COL = "mean_sku_share_in_hour_norm"
DEFAULT_SKU_SHARE_COL = SKU_SHARE_COL
SKU_HOUR_FORECAST_COL = "sku_hour_forecast"
SKU_DAY_FORECAST_COL = "sku_day_forecast"

DEFAULT_BAKERY_FORECAST_PATH = ROOT / "data" / "processed" / "bakery_daily_sales.csv"
DEFAULT_BAKERY_HOUR_PROFILE_PATH = ROOT / "data" / "processed" / "bakery_hour_profile.csv"
DEFAULT_SKU_PROFILE_PATH = ROOT / "data" / "processed" / "sku_hour_share_profile.csv"
DEFAULT_SKU_APPLIED_PROFILE_PATH = (
    ROOT / "data" / "processed" / "sku_hour_share_profile_daily_smoothed.csv"
)
SKU_PROFILE_CHUNK_SIZE = 100_000

# Tier-1 (bakery x sku x dow x hour) profile rows are trusted only when backed
# by at least this many same-dow observations. Below the gate the consumer
# routes to tier-2 (bakery x sku x hour) with a "thin" source tag so audits
# can tell "no tier-1 data" apart from "tier-1 data too thin".
MIN_TIER1_N_DAYS = 8
N_DAYS_COL = "n_days"

HOURLY_OUTPUT_NAME = "sku_hour_forecast.csv"
DAILY_OUTPUT_NAME = "sku_day_forecast.csv"
SUMMARY_OUTPUT_NAME = "apply_bakery_profiles_summary.json"

HOURLY_OUTPUT_COLS = [
    DATE_COL,
    DOW_COL,
    BAKERY_ID_COL,
    HOUR_COL,
    PRODUCT_ID_COL,
    SKU_HOUR_FORECAST_COL,
]

DAILY_OUTPUT_COLS = [
    DATE_COL,
    DOW_COL,
    BAKERY_ID_COL,
    PRODUCT_ID_COL,
    SKU_DAY_FORECAST_COL,
]

APPLIED_UPLIFT_SHARE_COL = "sku_share_in_hour_adj"
APPLIED_UPLIFT_SHARE_NORM_COL = "sku_share_in_hour_adj_norm"
APPLIED_BAKERY_HOUR_SALES_COL = "bakery_hour_sales"
SKU_UPLIFT_MULTIPLIER_COL = "sku_uplift_multiplier"


def load_bakery_day_forecast(path: str | Path, *, forecast_col: str = BAKERY_FORECAST_COL) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if forecast_col not in df.columns:
        if forecast_col == BAKERY_FORECAST_COL and "bakery_sales" in df.columns:
            forecast_col = "bakery_sales"
        else:
            raise KeyError(f"Forecast column '{forecast_col}' not found in {path}")

    required = [DATE_COL, BAKERY_ID_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in bakery forecast file: {missing}")

    work = df.copy()
    work[DATE_COL] = pd.to_datetime(work[DATE_COL], errors="coerce")
    work = work.dropna(subset=[DATE_COL, BAKERY_ID_COL]).copy()
    work[BAKERY_FORECAST_COL] = pd.to_numeric(work[forecast_col], errors="coerce").fillna(0.0)
    work[BAKERY_FORECAST_COL] = work[BAKERY_FORECAST_COL].clip(lower=0.0)
    work[DOW_COL] = work[DATE_COL].dt.dayofweek

    keep_cols = [DATE_COL, DOW_COL, BAKERY_ID_COL, BAKERY_FORECAST_COL]
    for optional in [BAKERY_NAME_COL, CITY_COL]:
        if optional in work.columns:
            keep_cols.append(optional)
    return work[keep_cols].copy()


def load_bakery_hour_profile(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = [BAKERY_ID_COL, DOW_COL, HOUR_COL, BAKERY_HOUR_SHARE_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in bakery hour profile: {missing}")

    keep_cols = [BAKERY_ID_COL, DOW_COL, HOUR_COL, BAKERY_HOUR_SHARE_COL]
    if BAKERY_NAME_COL in df.columns:
        keep_cols.append(BAKERY_NAME_COL)

    work = df[keep_cols].copy()
    work[BAKERY_HOUR_SHARE_COL] = pd.to_numeric(work[BAKERY_HOUR_SHARE_COL], errors="coerce").fillna(0.0)
    work = (
        work.groupby([col for col in keep_cols if col != BAKERY_HOUR_SHARE_COL], as_index=False)
        .agg(mean_hour_share_norm=(BAKERY_HOUR_SHARE_COL, "mean"))
    )
    return work


def build_bakery_hour_profile_fallback(profile: pd.DataFrame) -> pd.DataFrame:
    work = (
        profile.groupby([BAKERY_ID_COL, HOUR_COL], as_index=False)
        .agg(mean_hour_share_norm=(BAKERY_HOUR_SHARE_COL, "mean"))
    )
    totals = work.groupby(BAKERY_ID_COL, as_index=False)["mean_hour_share_norm"].sum().rename(
        columns={"mean_hour_share_norm": "profile_sum"}
    )
    work = work.merge(totals, on=BAKERY_ID_COL, how="left")
    work[BAKERY_HOUR_SHARE_COL] = np.where(
        work["profile_sum"] > 0,
        work[BAKERY_HOUR_SHARE_COL] / work["profile_sum"],
        0.0,
    )
    return work.drop(columns=["profile_sum"])


def load_sku_hour_profile(
    path: str | Path,
    *,
    sku_share_col: str = DEFAULT_SKU_SHARE_COL,
) -> pd.DataFrame:
    wanted = {
        BAKERY_ID_COL,
        PRODUCT_ID_COL,
        DOW_COL,
        HOUR_COL,
        sku_share_col,
        N_DAYS_COL,
    }
    df = pd.read_csv(path, encoding="utf-8-sig", usecols=lambda c: c in wanted)
    return load_sku_hour_profile_frame(df, sku_share_col=sku_share_col)


def build_sku_hour_profile_fallback(
    profile: pd.DataFrame,
    *,
    normalize_sku_shares: bool = True,
) -> pd.DataFrame:
    work = (
        profile.groupby([BAKERY_ID_COL, HOUR_COL, PRODUCT_ID_COL], as_index=False)
        .agg(mean_sku_share_in_hour_norm=(SKU_SHARE_COL, "mean"))
    )
    if not normalize_sku_shares:
        return work

    totals = work.groupby([BAKERY_ID_COL, HOUR_COL], as_index=False)["mean_sku_share_in_hour_norm"].sum().rename(
        columns={"mean_sku_share_in_hour_norm": "profile_sum"}
    )
    work = work.merge(totals, on=[BAKERY_ID_COL, HOUR_COL], how="left")
    work[SKU_SHARE_COL] = np.where(
        work["profile_sum"] > 0,
        work[SKU_SHARE_COL] / work["profile_sum"],
        0.0,
    )
    return work.drop(columns=["profile_sum"])


def build_tier1_share_sums(profile: pd.DataFrame) -> pd.DataFrame:
    tier1 = profile[profile[N_DAYS_COL] >= MIN_TIER1_N_DAYS].copy()
    if tier1.empty:
        return pd.DataFrame(
            columns=[BAKERY_ID_COL, DOW_COL, HOUR_COL, "tier1_share_sum"]
        )
    return (
        tier1.groupby([BAKERY_ID_COL, DOW_COL, HOUR_COL], as_index=False)[
            SKU_SHARE_COL
        ]
        .sum()
        .rename(columns={SKU_SHARE_COL: "tier1_share_sum"})
    )


def normalize_tier1_sku_shares(
    sku_chunk: pd.DataFrame,
    tier1_share_sums: pd.DataFrame,
) -> pd.DataFrame:
    if sku_chunk.empty:
        return sku_chunk.copy()
    work = sku_chunk.merge(
        tier1_share_sums,
        on=[BAKERY_ID_COL, DOW_COL, HOUR_COL],
        how="left",
        validate="many_to_one",
    )
    denominator = pd.to_numeric(
        work["tier1_share_sum"],
        errors="coerce",
    ).replace(0, np.nan)
    work[SKU_SHARE_COL] = (
        pd.to_numeric(work[SKU_SHARE_COL], errors="coerce").fillna(0.0)
        / denominator
    ).fillna(0.0)
    return work.drop(columns=["tier1_share_sum"])


def build_sku_uplift_multipliers(
    applied_path: str | Path,
    *,
    chunk_size: int = 1_000_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    usecols = {
        DATE_COL,
        DOW_COL,
        BAKERY_ID_COL,
        HOUR_COL,
        APPLIED_BAKERY_HOUR_SALES_COL,
        APPLIED_UPLIFT_SHARE_COL,
        APPLIED_UPLIFT_SHARE_NORM_COL,
    }
    parts: list[pd.DataFrame] = []
    reader = pd.read_csv(
        applied_path,
        encoding="utf-8-sig",
        usecols=lambda c: c in usecols,
        chunksize=chunk_size,
    )
    for i, chunk in enumerate(reader, start=1):
        chunk[APPLIED_BAKERY_HOUR_SALES_COL] = pd.to_numeric(
            chunk[APPLIED_BAKERY_HOUR_SALES_COL],
            errors="coerce",
        ).fillna(0.0)
        chunk[APPLIED_UPLIFT_SHARE_COL] = pd.to_numeric(
            chunk[APPLIED_UPLIFT_SHARE_COL],
            errors="coerce",
        ).fillna(0.0)
        chunk[APPLIED_UPLIFT_SHARE_NORM_COL] = pd.to_numeric(
            chunk[APPLIED_UPLIFT_SHARE_NORM_COL],
            errors="coerce",
        ).fillna(0.0)
        hourly = (
            chunk.groupby([DATE_COL, DOW_COL, BAKERY_ID_COL, HOUR_COL], as_index=False)
            .agg(
                bakery_hour_sales=(APPLIED_BAKERY_HOUR_SALES_COL, "max"),
                raw_share_sum=(APPLIED_UPLIFT_SHARE_COL, "sum"),
                norm_share_sum=(APPLIED_UPLIFT_SHARE_NORM_COL, "sum"),
            )
        )
        hourly[SKU_UPLIFT_MULTIPLIER_COL] = (
            hourly["raw_share_sum"] / hourly["norm_share_sum"].replace(0.0, np.nan)
        ).fillna(1.0).clip(lower=1.0)
        hourly["weighted_multiplier"] = (
            hourly[SKU_UPLIFT_MULTIPLIER_COL] * hourly["bakery_hour_sales"]
        )
        parts.append(
            hourly[
                [
                    DOW_COL,
                    BAKERY_ID_COL,
                    HOUR_COL,
                    "bakery_hour_sales",
                    "weighted_multiplier",
                ]
            ]
        )
        if i % 10 == 0:
            print(f"processed uplift multiplier chunks: {i}", flush=True)

    if not parts:
        empty_exact = pd.DataFrame(
            columns=[BAKERY_ID_COL, DOW_COL, HOUR_COL, SKU_UPLIFT_MULTIPLIER_COL]
        )
        empty_fallback = pd.DataFrame(
            columns=[BAKERY_ID_COL, HOUR_COL, SKU_UPLIFT_MULTIPLIER_COL]
        )
        return empty_exact, empty_fallback

    combined = pd.concat(parts, ignore_index=True)
    exact = (
        combined.groupby([BAKERY_ID_COL, DOW_COL, HOUR_COL], as_index=False)
        .agg(
            bakery_hour_sales=("bakery_hour_sales", "sum"),
            weighted_multiplier=("weighted_multiplier", "sum"),
        )
    )
    exact[SKU_UPLIFT_MULTIPLIER_COL] = (
        exact["weighted_multiplier"] / exact["bakery_hour_sales"].replace(0.0, np.nan)
    ).fillna(1.0).clip(lower=1.0)
    exact = exact[[BAKERY_ID_COL, DOW_COL, HOUR_COL, SKU_UPLIFT_MULTIPLIER_COL]]

    fallback = (
        combined.groupby([BAKERY_ID_COL, HOUR_COL], as_index=False)
        .agg(
            bakery_hour_sales=("bakery_hour_sales", "sum"),
            weighted_multiplier=("weighted_multiplier", "sum"),
        )
    )
    fallback[SKU_UPLIFT_MULTIPLIER_COL] = (
        fallback["weighted_multiplier"] / fallback["bakery_hour_sales"].replace(0.0, np.nan)
    ).fillna(1.0).clip(lower=1.0)
    fallback = fallback[[BAKERY_ID_COL, HOUR_COL, SKU_UPLIFT_MULTIPLIER_COL]]
    return exact, fallback


def apply_uplift_multiplier(
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


def allocate_bakery_to_hour(bakery_forecast: pd.DataFrame, bakery_hour_profile: pd.DataFrame) -> pd.DataFrame:
    exact = bakery_forecast.merge(
        bakery_hour_profile,
        on=[BAKERY_ID_COL, DOW_COL],
        how="inner",
        validate="many_to_many",
        suffixes=("", "_profile"),
    )

    fallback_profile = build_bakery_hour_profile_fallback(bakery_hour_profile)
    covered = bakery_hour_profile[[BAKERY_ID_COL, DOW_COL]].drop_duplicates().assign(has_profile=1)
    missing = bakery_forecast.merge(covered, on=[BAKERY_ID_COL, DOW_COL], how="left")
    missing = missing[missing["has_profile"].isna()].drop(columns=["has_profile"])
    fallback = missing.merge(
        fallback_profile,
        on=[BAKERY_ID_COL],
        how="left",
        validate="many_to_many",
    )

    hourly = pd.concat([exact, fallback], ignore_index=True, sort=False)
    if BAKERY_NAME_COL not in hourly.columns and f"{BAKERY_NAME_COL}_profile" in hourly.columns:
        hourly[BAKERY_NAME_COL] = hourly[f"{BAKERY_NAME_COL}_profile"]
    for col in [f"{BAKERY_NAME_COL}_profile"]:
        if col in hourly.columns:
            hourly.drop(columns=[col], inplace=True)

    hourly[BAKERY_HOUR_FORECAST_COL] = hourly[BAKERY_FORECAST_COL] * hourly[BAKERY_HOUR_SHARE_COL]
    return hourly.sort_values([BAKERY_ID_COL, DATE_COL, HOUR_COL]).reset_index(drop=True)


def allocate_hour_to_sku(hourly_forecast: pd.DataFrame, sku_hour_profile: pd.DataFrame) -> pd.DataFrame:
    sku_hourly = hourly_forecast.merge(
        sku_hour_profile,
        on=[BAKERY_ID_COL, DOW_COL, HOUR_COL],
        how="inner",
        validate="many_to_many",
        suffixes=("", "_sku"),
    )

    sku_hourly[SKU_HOUR_FORECAST_COL] = sku_hourly[BAKERY_HOUR_FORECAST_COL] * sku_hourly[SKU_SHARE_COL]
    return sku_hourly[HOURLY_OUTPUT_COLS].sort_values([BAKERY_ID_COL, PRODUCT_ID_COL, DATE_COL, HOUR_COL]).reset_index(drop=True)


def _write_hourly_chunk(df: pd.DataFrame, path: Path, *, header: bool) -> None:
    df.to_csv(
        path,
        mode="a",
        index=False,
        encoding="utf-8-sig",
        header=header,
    )


def _stream_allocate_hour_to_sku(
    hourly_forecast: pd.DataFrame,
    sku_hour_profile_path: str | Path,
    hourly_output_path: Path,
    *,
    chunk_size: int = SKU_PROFILE_CHUNK_SIZE,
    sku_share_col: str = DEFAULT_SKU_SHARE_COL,
    normalize_sku_shares: bool = True,
    uplift_multiplier_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    if hourly_output_path.exists():
        hourly_output_path.unlink()

    hourly_cols = [DATE_COL, DOW_COL, HOUR_COL, BAKERY_ID_COL, BAKERY_HOUR_FORECAST_COL]
    hourly_lookup = hourly_forecast[hourly_cols].copy()
    hourly_lookup["_row_id"] = np.arange(len(hourly_lookup))

    daily_parts: list[pd.DataFrame] = []
    sku_hour_rows = 0
    products_seen: set[str] = set()
    source_stats: dict[str, dict[str, float | int]] = {}
    exact_multipliers = pd.DataFrame()
    fallback_multipliers = pd.DataFrame()
    if uplift_multiplier_path is not None:
        exact_multipliers, fallback_multipliers = build_sku_uplift_multipliers(
            uplift_multiplier_path,
        )

    full_profile = load_sku_hour_profile(
        sku_hour_profile_path,
        sku_share_col=sku_share_col,
    )
    # Tier-2 fallback is built from the FULL profile (we still want the
    # bakery-hour-level average to include the thin same-dow observations).
    sku_fallback = build_sku_hour_profile_fallback(
        full_profile,
        normalize_sku_shares=normalize_sku_shares,
    )
    # Tier-1 is the gated subset; only this set defines "covered" triples.
    tier1_profile = full_profile[full_profile[N_DAYS_COL] >= MIN_TIER1_N_DAYS]
    tier1_share_sums = build_tier1_share_sums(full_profile)
    exact_keys = (
        tier1_profile[[BAKERY_ID_COL, DOW_COL, HOUR_COL]]
        .drop_duplicates()
        .assign(has_exact=1)
    )
    # Raw triples that DID exist in the profile but failed the gate, so we can
    # tag those tier-2 hits as "thin" vs genuinely missing tier-1 data.
    thin_triples = (
        full_profile[full_profile[N_DAYS_COL] < MIN_TIER1_N_DAYS][
            [BAKERY_ID_COL, DOW_COL, HOUR_COL]
        ]
        .drop_duplicates()
        .assign(is_thin=1)
    )

    reader = pd.read_csv(
        sku_hour_profile_path,
        encoding="utf-8-sig",
        usecols=lambda c: c
        in {BAKERY_ID_COL, DOW_COL, HOUR_COL, PRODUCT_ID_COL, sku_share_col, N_DAYS_COL},
        chunksize=chunk_size,
    )
    for i, chunk in enumerate(reader, start=1):
        sku_chunk = load_sku_hour_profile_frame(
            chunk,
            sku_share_col=sku_share_col,
        )
        sku_chunk = sku_chunk[sku_chunk[N_DAYS_COL] >= MIN_TIER1_N_DAYS]
        if normalize_sku_shares:
            sku_chunk = normalize_tier1_sku_shares(sku_chunk, tier1_share_sums)
        if uplift_multiplier_path is not None:
            sku_chunk = apply_uplift_multiplier(
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
        merged[SKU_HOUR_FORECAST_COL] = merged[BAKERY_HOUR_FORECAST_COL] * merged[SKU_SHARE_COL]
        merged = merged[[*HOURLY_OUTPUT_COLS, "_row_id", "source"]]

        _write_hourly_chunk(merged, hourly_output_path, header=(i == 1))
        sku_hour_rows += len(merged)
        _update_source_stats(source_stats, merged)
        products_seen.update(merged[PRODUCT_ID_COL].dropna().astype(str).unique().tolist())

        daily_part = (
            merged.groupby([DATE_COL, DOW_COL, BAKERY_ID_COL, PRODUCT_ID_COL], as_index=False, sort=False)
            .agg(sku_day_forecast=(SKU_HOUR_FORECAST_COL, "sum"))
        )
        daily_parts.append(daily_part)

        if i % 10 == 0:
            print(f"processed sku profile chunks: {i}", flush=True)

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
            sku_fallback,
            on=[BAKERY_ID_COL, HOUR_COL],
            how="inner",
            validate="many_to_many",
            sort=False,
        )
        if uplift_multiplier_path is not None:
            fallback_merged = apply_uplift_multiplier(
                fallback_merged,
                fallback_multipliers,
                keys=[BAKERY_ID_COL, HOUR_COL],
            )
        fallback_merged[SKU_HOUR_FORECAST_COL] = fallback_merged[BAKERY_HOUR_FORECAST_COL] * fallback_merged[SKU_SHARE_COL]
        fallback_merged["source"] = np.where(
            fallback_merged["is_thin"].fillna(0).astype(int) == 1,
            "bakery_hour_fallback_thin",
            "bakery_hour_fallback",
        )
        fallback_merged = fallback_merged[[*HOURLY_OUTPUT_COLS, "_row_id", "source"]].sort_values(
            [BAKERY_ID_COL, PRODUCT_ID_COL, DATE_COL, HOUR_COL]
        ).reset_index(drop=True)
        _write_hourly_chunk(fallback_merged, hourly_output_path, header=False)
        sku_hour_rows += len(fallback_merged)
        _update_source_stats(source_stats, fallback_merged)
        products_seen.update(fallback_merged[PRODUCT_ID_COL].dropna().astype(str).unique().tolist())
        daily_parts.append(
            fallback_merged.groupby([DATE_COL, DOW_COL, BAKERY_ID_COL, PRODUCT_ID_COL], as_index=False, sort=False)
            .agg(sku_day_forecast=(SKU_HOUR_FORECAST_COL, "sum"))
        )

    if daily_parts:
        sku_daily = pd.concat(daily_parts, ignore_index=True)
        sku_daily = (
            sku_daily.groupby([DATE_COL, DOW_COL, BAKERY_ID_COL, PRODUCT_ID_COL], as_index=False, sort=False)
            .agg(sku_day_forecast=(SKU_DAY_FORECAST_COL, "sum"))
            .sort_values([BAKERY_ID_COL, PRODUCT_ID_COL, DATE_COL])
            .reset_index(drop=True)
        )
    else:
        sku_daily = pd.DataFrame(columns=[DATE_COL, DOW_COL, BAKERY_ID_COL, PRODUCT_ID_COL, SKU_DAY_FORECAST_COL])

    hourly_final = pd.read_csv(hourly_output_path, encoding="utf-8-sig")
    drop_cols = [col for col in ["_row_id"] if col in hourly_final.columns]
    if drop_cols:
        hourly_final.drop(columns=drop_cols, inplace=True)
    hourly_final.to_csv(hourly_output_path, index=False, encoding="utf-8-sig")

    stats = {
        "sku_hour_rows": sku_hour_rows,
        "products": len(products_seen),
        "source_stats": _finalize_source_stats(source_stats),
    }
    return sku_daily, stats


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


def load_sku_hour_profile_frame(
    df: pd.DataFrame,
    *,
    sku_share_col: str = DEFAULT_SKU_SHARE_COL,
) -> pd.DataFrame:
    required = [
        BAKERY_ID_COL,
        PRODUCT_ID_COL,
        DOW_COL,
        HOUR_COL,
        sku_share_col,
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in SKU hour-share profile: {missing}")

    keep_cols = [BAKERY_ID_COL, DOW_COL, HOUR_COL, PRODUCT_ID_COL, sku_share_col]
    if N_DAYS_COL in df.columns:
        keep_cols.append(N_DAYS_COL)
    work = df[keep_cols].copy()
    work[sku_share_col] = pd.to_numeric(work[sku_share_col], errors="coerce").fillna(0.0)
    if N_DAYS_COL in work.columns:
        work[N_DAYS_COL] = (
            pd.to_numeric(work[N_DAYS_COL], errors="coerce").fillna(0).astype(int)
        )
    else:
        work[N_DAYS_COL] = 0
    work.rename(columns={sku_share_col: SKU_SHARE_COL}, inplace=True)
    return work


def build_sku_day_forecast(sku_hourly: pd.DataFrame) -> pd.DataFrame:
    sku_daily = (
        sku_hourly.groupby([DATE_COL, DOW_COL, BAKERY_ID_COL, PRODUCT_ID_COL], as_index=False, sort=False)
        .agg(sku_day_forecast=(SKU_HOUR_FORECAST_COL, "sum"))
        .sort_values([BAKERY_ID_COL, PRODUCT_ID_COL, DATE_COL])
        .reset_index(drop=True)
    )
    return sku_daily[DAILY_OUTPUT_COLS]


def build_summary(bakery_forecast: pd.DataFrame, hourly_forecast: pd.DataFrame, sku_hourly: pd.DataFrame) -> dict:
    bakery_total = float(bakery_forecast[BAKERY_FORECAST_COL].sum()) if len(bakery_forecast) else 0.0
    sku_total = float(sku_hourly[SKU_HOUR_FORECAST_COL].sum()) if len(sku_hourly) else 0.0
    return {
        "bakery_day_rows": int(len(bakery_forecast)),
        "bakery_hour_rows": int(len(hourly_forecast)),
        "sku_hour_rows": int(len(sku_hourly)),
        "dates": int(bakery_forecast[DATE_COL].nunique()) if len(bakery_forecast) else 0,
        "bakeries": int(bakery_forecast[BAKERY_ID_COL].nunique()) if len(bakery_forecast) else 0,
        "products": int(sku_hourly[PRODUCT_ID_COL].nunique()) if len(sku_hourly) else 0,
        "bakery_forecast_total": round(bakery_total, 6),
        "sku_forecast_total": round(sku_total, 6),
        "allocation_ratio": round(sku_total / bakery_total, 6) if bakery_total > 0 else 0.0,
        "date_min": None if bakery_forecast.empty else str(bakery_forecast[DATE_COL].min().date()),
        "date_max": None if bakery_forecast.empty else str(bakery_forecast[DATE_COL].max().date()),
    }


def build_summary_from_daily(
    bakery_forecast: pd.DataFrame,
    hourly_forecast: pd.DataFrame,
    sku_daily: pd.DataFrame,
    *,
    sku_hour_rows: int,
    products: int,
    source_stats: list[dict] | None = None,
) -> dict:
    bakery_total = float(bakery_forecast[BAKERY_FORECAST_COL].sum()) if len(bakery_forecast) else 0.0
    sku_total = float(sku_daily[SKU_DAY_FORECAST_COL].sum()) if len(sku_daily) else 0.0
    summary = {
        "bakery_day_rows": int(len(bakery_forecast)),
        "bakery_hour_rows": int(len(hourly_forecast)),
        "sku_hour_rows": int(sku_hour_rows),
        "sku_day_rows": int(len(sku_daily)),
        "dates": int(bakery_forecast[DATE_COL].nunique()) if len(bakery_forecast) else 0,
        "bakeries": int(bakery_forecast[BAKERY_ID_COL].nunique()) if len(bakery_forecast) else 0,
        "products": int(products),
        "bakery_forecast_total": round(bakery_total, 6),
        "sku_forecast_total": round(sku_total, 6),
        "allocation_ratio": round(sku_total / bakery_total, 6) if bakery_total > 0 else 0.0,
        "date_min": None if bakery_forecast.empty else str(bakery_forecast[DATE_COL].min().date()),
        "date_max": None if bakery_forecast.empty else str(bakery_forecast[DATE_COL].max().date()),
    }
    summary["source_summary"] = source_stats or []
    return summary


def save_outputs(
    output_dir: str | Path,
    sku_hourly: pd.DataFrame | None,
    sku_daily: pd.DataFrame,
    summary: dict,
    *,
    output_suffix: str = "",
    prewritten_hourly_path: Path | None = None,
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{output_suffix}" if output_suffix else ""

    hourly_path = out_dir / HOURLY_OUTPUT_NAME.replace(".csv", f"{suffix}.csv")
    daily_path = out_dir / DAILY_OUTPUT_NAME.replace(".csv", f"{suffix}.csv")
    summary_path = out_dir / SUMMARY_OUTPUT_NAME.replace(".json", f"{suffix}.json")

    if prewritten_hourly_path is None:
        if sku_hourly is None:
            raise ValueError("sku_hourly must be provided when prewritten_hourly_path is not set")
        sku_hourly.to_csv(hourly_path, index=False, encoding="utf-8-sig")
    elif prewritten_hourly_path != hourly_path:
        prewritten_hourly_path.replace(hourly_path)
    sku_daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "sku_hourly": hourly_path,
        "sku_daily": daily_path,
        "summary": summary_path,
    }


def apply_profiles(
    bakery_forecast_path: str | Path,
    bakery_hour_profile_path: str | Path,
    sku_hour_profile_path: str | Path,
    output_dir: str | Path,
    *,
    forecast_col: str = BAKERY_FORECAST_COL,
    output_suffix: str = "",
    sku_share_col: str = DEFAULT_SKU_SHARE_COL,
    normalize_sku_shares: bool = True,
    uplift_multiplier_path: str | Path | None = None,
) -> dict[str, Path]:
    bakery_forecast = load_bakery_day_forecast(bakery_forecast_path, forecast_col=forecast_col)
    bakery_hour_profile = load_bakery_hour_profile(bakery_hour_profile_path)

    hourly_forecast = allocate_bakery_to_hour(bakery_forecast, bakery_hour_profile)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{output_suffix}" if output_suffix else ""
    hourly_output_path = out_dir / HOURLY_OUTPUT_NAME.replace(".csv", f"{suffix}.csv")

    sku_daily, stats = _stream_allocate_hour_to_sku(
        hourly_forecast,
        sku_hour_profile_path,
        hourly_output_path,
        sku_share_col=sku_share_col,
        normalize_sku_shares=normalize_sku_shares,
        uplift_multiplier_path=uplift_multiplier_path,
    )
    summary = build_summary_from_daily(
        bakery_forecast,
        hourly_forecast,
        sku_daily,
        sku_hour_rows=stats["sku_hour_rows"],
        products=stats["products"],
        source_stats=stats.get("source_stats", []),
    )
    return save_outputs(
        out_dir,
        None,
        sku_daily,
        summary,
        output_suffix=output_suffix,
        prewritten_hourly_path=hourly_output_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply bakery hour and SKU hour-share profiles to bakery forecasts")
    parser.add_argument("--bakery-forecast-path", default=str(DEFAULT_BAKERY_FORECAST_PATH))
    parser.add_argument("--bakery-hour-profile-path", default=str(DEFAULT_BAKERY_HOUR_PROFILE_PATH))
    parser.add_argument("--sku-hour-profile-path", default=str(DEFAULT_SKU_PROFILE_PATH))
    parser.add_argument("--forecast-col", default=BAKERY_FORECAST_COL)
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--sku-share-col", default=DEFAULT_SKU_SHARE_COL)
    parser.add_argument("--no-normalize-sku-shares", action="store_true")
    parser.add_argument("--uplift-multiplier-path", default=None)
    args = parser.parse_args()

    paths = apply_profiles(
        args.bakery_forecast_path,
        args.bakery_hour_profile_path,
        args.sku_hour_profile_path,
        args.output_dir,
        forecast_col=args.forecast_col,
        output_suffix=args.output_suffix,
        sku_share_col=args.sku_share_col,
        normalize_sku_shares=not args.no_normalize_sku_shares,
        uplift_multiplier_path=args.uplift_multiplier_path,
    )

    print("=" * 72)
    print("APPLY BAKERY PROFILES")
    print("=" * 72)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

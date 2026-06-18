"""
Build SKU hour-share profiles from raw check lines using chunked processing.

Input:
  data/raw/sales_hrs_all.csv (English or legacy Russian raw schema)

Output:
  data/processed/sku_hour_share_profile.csv
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.experiments_v2.raw_snapshot_schema import normalize_snapshot_chunk
from src.experiments_v2.raw_snapshot_schema import parse_snapshot_date
from src.experiments_v2.raw_snapshot_schema import parse_snapshot_datetime
from src.experiments_v2.build_bakery_hour_profile import DEFAULT_DAILY_CLEANING_PATH
from src.experiments_v2.build_bakery_hour_profile import add_profile_weights
from src.experiments_v2.build_bakery_hour_profile import load_daily_profile_weights
from src.experiments_v2.build_bakery_hour_profile import _weighted_mean
from scripts.build_required_assortment_contract import normalize_text
from scripts.compare_required_assortment_to_dim_products import (
    normalize_product_for_dim_lookup,
)

RAW_DATETIME_COL = "check_datetime"
RAW_DATE_COL = "check_date"
RAW_EVENT_COL = "cash_event_type"
RAW_QTY_COL = "quantity"
RAW_BAKERY_ID_COL = "bakery_id"
RAW_BAKERY_NAME_COL = "bakery_name"
RAW_CITY_COL = "city"
RAW_PRODUCT_ID_COL = "product_id"
RAW_PRODUCT_NAME_COL = "product_name"
RAW_CATEGORY_COL = "category_name"

DATE_COL = "date"
DOW_COL = "dow"
HOUR_COL = "hour"
BAKERY_ID_COL = "bakery_id"
BAKERY_NAME_COL = "bakery_name"
CITY_COL = "city"
PRODUCT_ID_COL = "product_id"
PRODUCT_NAME_COL = "product_name"
CATEGORY_COL = "category_name"
SKU_HOUR_SALES_COL = "sku_hour_sales"
BAKERY_HOUR_SALES_COL = "bakery_hour_sales"
SKU_SHARE_COL = "sku_share_in_hour"

SALES_EVENT = "РџСЂРѕРґР°Р¶Р°"
SALES_EVENTS = {"Продажа", SALES_EVENT}
CHUNK_SIZE = 1_000_000

OUTPUT_NAME = "sku_hour_share_profile.csv"
APPLIED_DAILY_OUTPUT_NAME = "sku_hour_share_profile_daily.csv"
SUMMARY_OUTPUT_NAME = "sku_hour_share_profile_summary.json"
DEFAULT_ASSORTMENT_PATH = (
    ROOT / "reports" / "required_assortment" / "assortment_city_products.csv"
)
DEFAULT_PRODUCT_LOOKUP_PATH = (
    ROOT / "reports" / "required_assortment" / "dim_products_lookup.csv"
)

RELIABILITY_HISTORY_SATURATION = 20.0
LOW_RELIABILITY_THRESHOLD = 0.2
RECENT_SHARE_DAYS = 28
RECENT_SHARE_ALPHA = 0.4


def _add_reliability_score(profile: pd.DataFrame) -> pd.DataFrame:
    """Aggregate four sample-quality components into a single 0..1 score.

    Observational metric only. The weighting between history / presence /
    stability / cleanliness is a guess — its real correlation with forecast
    error can be validated only on a production A/B. Do not use as a gate
    for fallback routing or uplift gating; use the raw component columns
    (n_days, zero_share_rate, cv_share, anomaly_share) for decisions and
    keep this score for audit dashboards.
    """
    if profile.empty:
        profile["cv_share"] = pd.Series(dtype=float)
        profile["reliability_score"] = pd.Series(dtype=float)
        return profile

    mean_share = profile["mean_sku_share_in_hour"].replace(0, np.nan)
    cv_share = profile["std_sku_share_in_hour"] / mean_share
    cv_share = cv_share.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    profile["cv_share"] = cv_share

    history_term = np.minimum(profile["n_days"] / RELIABILITY_HISTORY_SATURATION, 1.0)
    presence_term = 1.0 - profile["zero_share_rate"].clip(lower=0.0, upper=1.0)
    stability_term = 1.0 / (1.0 + cv_share)
    cleanliness_term = 1.0 - profile["anomaly_share"].clip(lower=0.0, upper=1.0)
    profile["reliability_score"] = np.clip(
        history_term * presence_term * stability_term * cleanliness_term,
        0.0,
        1.0,
    )
    return profile


def aggregate_sku_hourly_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = normalize_snapshot_chunk(chunk)
    sales = chunk[chunk[RAW_EVENT_COL].isin(SALES_EVENTS)].copy()
    if sales.empty:
        return pd.DataFrame(
            columns=[
                DATE_COL,
                DOW_COL,
                HOUR_COL,
                BAKERY_ID_COL,
                BAKERY_NAME_COL,
                CITY_COL,
                PRODUCT_ID_COL,
                PRODUCT_NAME_COL,
                CATEGORY_COL,
                SKU_HOUR_SALES_COL,
            ]
        )

    sales[DATE_COL] = parse_snapshot_date(sales[RAW_DATE_COL]).dt.normalize()
    sales["_dt"] = parse_snapshot_datetime(sales[RAW_DATETIME_COL])
    sales = sales.dropna(
        subset=[
            DATE_COL,
            "_dt",
            RAW_BAKERY_ID_COL,
            RAW_BAKERY_NAME_COL,
            RAW_CITY_COL,
            RAW_PRODUCT_ID_COL,
            RAW_PRODUCT_NAME_COL,
        ]
    )
    if sales.empty:
        return pd.DataFrame(
            columns=[
                DATE_COL,
                DOW_COL,
                HOUR_COL,
                BAKERY_ID_COL,
                BAKERY_NAME_COL,
                CITY_COL,
                PRODUCT_ID_COL,
                PRODUCT_NAME_COL,
                CATEGORY_COL,
                SKU_HOUR_SALES_COL,
            ]
        )

    sales[RAW_QTY_COL] = (
        pd.to_numeric(sales[RAW_QTY_COL], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    sales[HOUR_COL] = sales["_dt"].dt.hour
    sales[DOW_COL] = sales[DATE_COL].dt.dayofweek
    sales[RAW_CATEGORY_COL] = sales[RAW_CATEGORY_COL].fillna("unknown")

    grouped = (
        sales.groupby(
            [
                DATE_COL,
                DOW_COL,
                HOUR_COL,
                RAW_BAKERY_ID_COL,
                RAW_BAKERY_NAME_COL,
                RAW_CITY_COL,
                RAW_PRODUCT_ID_COL,
                RAW_PRODUCT_NAME_COL,
                RAW_CATEGORY_COL,
            ],
            as_index=False,
        )
        .agg(sku_hour_sales=(RAW_QTY_COL, "sum"))
        .rename(
            columns={
                RAW_BAKERY_ID_COL: BAKERY_ID_COL,
                RAW_BAKERY_NAME_COL: BAKERY_NAME_COL,
                RAW_CITY_COL: CITY_COL,
                RAW_PRODUCT_ID_COL: PRODUCT_ID_COL,
                RAW_PRODUCT_NAME_COL: PRODUCT_NAME_COL,
                RAW_CATEGORY_COL: CATEGORY_COL,
            }
        )
    )
    return grouped


def merge_hourly_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()

    hourly = pd.concat(parts, ignore_index=True)
    hourly = (
        hourly.groupby(
            [
                DATE_COL,
                DOW_COL,
                HOUR_COL,
                BAKERY_ID_COL,
                BAKERY_NAME_COL,
                CITY_COL,
                PRODUCT_ID_COL,
                PRODUCT_NAME_COL,
                CATEGORY_COL,
            ],
            as_index=False,
        )
        .agg(sku_hour_sales=(SKU_HOUR_SALES_COL, "sum"))
        .sort_values([BAKERY_ID_COL, PRODUCT_ID_COL, DATE_COL, HOUR_COL])
        .reset_index(drop=True)
    )
    return hourly


def build_sku_hour_share_profile(
    hourly: pd.DataFrame,
    daily_weights: pd.DataFrame | None = None,
    *,
    recent_days: int = RECENT_SHARE_DAYS,
    recent_alpha: float = RECENT_SHARE_ALPHA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hourly = hourly.copy()
    if CITY_COL not in hourly.columns:
        hourly[CITY_COL] = "unknown"
    bakery_hour = hourly.groupby(
        [DATE_COL, DOW_COL, HOUR_COL, BAKERY_ID_COL, BAKERY_NAME_COL, CITY_COL],
        as_index=False,
    ).agg(bakery_hour_sales=(SKU_HOUR_SALES_COL, "sum"))
    applied = hourly.merge(
        bakery_hour,
        on=[DATE_COL, DOW_COL, HOUR_COL, BAKERY_ID_COL, BAKERY_NAME_COL, CITY_COL],
        how="left",
    )
    applied[SKU_SHARE_COL] = (
        applied[SKU_HOUR_SALES_COL] / applied[BAKERY_HOUR_SALES_COL].replace(0, np.nan)
    ).fillna(0.0)
    applied = add_profile_weights(applied, daily_weights)

    max_date = applied[DATE_COL].max() if len(applied) else pd.NaT
    recent_start = (
        max_date - pd.Timedelta(days=recent_days - 1)
        if pd.notna(max_date)
        else pd.NaT
    )
    profile_parts = []
    group_cols = [
        BAKERY_ID_COL,
        BAKERY_NAME_COL,
        PRODUCT_ID_COL,
        PRODUCT_NAME_COL,
        CATEGORY_COL,
        DOW_COL,
        HOUR_COL,
    ]
    for key, group in applied.groupby(group_cols, dropna=False):
        n_days = int(len(group))
        long_share = _weighted_mean(group[SKU_SHARE_COL], group["profile_weight"])
        recent_group = (
            group[group[DATE_COL] >= recent_start]
            if pd.notna(recent_start)
            else group.iloc[0:0]
        )
        recent_n_days = int(len(recent_group))
        recent_share = (
            _weighted_mean(recent_group[SKU_SHARE_COL], recent_group["profile_weight"])
            if recent_n_days
            else np.nan
        )
        if recent_n_days:
            mean_share = (1.0 - recent_alpha) * long_share + recent_alpha * recent_share
            blend_alpha = recent_alpha
        else:
            mean_share = long_share
            blend_alpha = 0.0
        std_share = float(group[SKU_SHARE_COL].std()) if n_days > 1 else 0.0
        if not np.isfinite(std_share):
            std_share = 0.0
        zero_share_rate = (
            float((group[SKU_HOUR_SALES_COL] <= 0).mean()) if n_days else 0.0
        )
        anomaly_share = (
            float((group["profile_weight"] < 1.0).mean()) if n_days else 0.0
        )
        profile_parts.append(
            {
                BAKERY_ID_COL: key[0],
                BAKERY_NAME_COL: key[1],
                PRODUCT_ID_COL: key[2],
                PRODUCT_NAME_COL: key[3],
                CATEGORY_COL: key[4],
                DOW_COL: key[5],
                HOUR_COL: key[6],
                "n_days": n_days,
                "recent_n_days": recent_n_days,
                "effective_weight": float(group["profile_weight"].sum()),
                "long_sku_share_in_hour": long_share,
                "recent_sku_share_in_hour": recent_share,
                "share_recent_alpha": blend_alpha,
                "mean_sku_share_in_hour": mean_share,
                "median_sku_share_in_hour": float(group[SKU_SHARE_COL].median()),
                "std_sku_share_in_hour": std_share,
                "mean_sku_hour_sales": _weighted_mean(
                    group[SKU_HOUR_SALES_COL], group["profile_weight"]
                ),
                "zero_share_rate": zero_share_rate,
                "anomaly_share": anomaly_share,
            }
        )
    profile = pd.DataFrame(profile_parts)
    profile["std_sku_share_in_hour"] = profile["std_sku_share_in_hour"].fillna(0.0)
    profile = _add_reliability_score(profile)

    totals = (
        profile.groupby([BAKERY_ID_COL, DOW_COL, HOUR_COL])["mean_sku_share_in_hour"]
        .sum()
        .rename("profile_sum")
        .reset_index()
    )
    profile = profile.merge(totals, on=[BAKERY_ID_COL, DOW_COL, HOUR_COL], how="left")
    profile["mean_sku_share_in_hour_norm"] = np.where(
        profile["profile_sum"] > 0,
        profile["mean_sku_share_in_hour"] / profile["profile_sum"],
        0.0,
    )
    profile.drop(columns=["profile_sum"], inplace=True)
    profile = profile.sort_values(
        [BAKERY_ID_COL, DOW_COL, HOUR_COL, PRODUCT_ID_COL]
    ).reset_index(drop=True)
    return profile, applied


def load_assortment_pairs(path: str | Path) -> pd.DataFrame:
    assortment = pd.read_csv(path, dtype={PRODUCT_ID_COL: str})
    required = {CITY_COL, PRODUCT_ID_COL, "is_active"}
    missing = required.difference(assortment.columns)
    if missing:
        raise KeyError(f"Missing columns in assortment file: {sorted(missing)}")

    active = assortment[assortment["is_active"].astype(int).eq(1)].copy()
    active[PRODUCT_ID_COL] = pd.to_numeric(
        active[PRODUCT_ID_COL],
        errors="coerce",
    ).astype("Int64")
    active = active.dropna(subset=[CITY_COL, PRODUCT_ID_COL])
    return active[[CITY_COL, PRODUCT_ID_COL]].drop_duplicates()


def load_bakery_lookup(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["bakery_key", "_lookup_bakery_id", CITY_COL])
    lookup = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=lambda c: c in {BAKERY_ID_COL, BAKERY_NAME_COL, CITY_COL},
    )
    lookup = lookup.dropna(subset=[BAKERY_NAME_COL, BAKERY_ID_COL, CITY_COL]).copy()
    lookup["bakery_key"] = lookup[BAKERY_NAME_COL].map(normalize_text)
    lookup["_lookup_bakery_id"] = pd.to_numeric(
        lookup[BAKERY_ID_COL],
        errors="coerce",
    ).astype("Int64")
    lookup = lookup.dropna(subset=["_lookup_bakery_id"])
    lookup = lookup.drop_duplicates("bakery_key", keep="first")
    return lookup[["bakery_key", "_lookup_bakery_id", CITY_COL]]


def load_product_lookup(path: str | Path) -> pd.DataFrame:
    lookup = pd.read_csv(path, dtype={PRODUCT_ID_COL: str})
    lookup = lookup[lookup["is_inactive_product"].astype(str).str.lower().ne("true")]
    lookup["product_key_lookup"] = lookup[PRODUCT_NAME_COL].map(
        normalize_product_for_dim_lookup
    )
    lookup["_lookup_product_id"] = pd.to_numeric(
        lookup[PRODUCT_ID_COL],
        errors="coerce",
    ).astype("Int64")
    lookup = lookup.dropna(subset=["product_key_lookup", "_lookup_product_id"])
    counts = lookup.groupby("product_key_lookup")[PRODUCT_ID_COL].nunique()
    unique_keys = counts[counts.eq(1)].index
    lookup = lookup[lookup["product_key_lookup"].isin(unique_keys)].copy()
    lookup = lookup.drop_duplicates("product_key_lookup", keep="first")
    return lookup[
        [
            "product_key_lookup",
            "_lookup_product_id",
            PRODUCT_NAME_COL,
            CATEGORY_COL,
        ]
    ].rename(
        columns={
            PRODUCT_NAME_COL: "_lookup_product_name",
            CATEGORY_COL: "_lookup_category_name",
        }
    )


def enrich_hourly_dimensions(
    hourly: pd.DataFrame,
    *,
    bakery_lookup: pd.DataFrame,
    product_lookup: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    work = hourly.copy()
    work["bakery_key"] = work[BAKERY_NAME_COL].map(normalize_text)
    work["product_key_lookup"] = work[PRODUCT_NAME_COL].map(
        normalize_product_for_dim_lookup
    )
    work = work.merge(bakery_lookup, on="bakery_key", how="left")
    work = work.merge(product_lookup, on="product_key_lookup", how="left")

    matched_bakeries = int(work["_lookup_bakery_id"].notna().sum())
    matched_products = int(work["_lookup_product_id"].notna().sum())

    work[BAKERY_ID_COL] = work["_lookup_bakery_id"].astype(object).where(
        work["_lookup_bakery_id"].notna(),
        work[BAKERY_ID_COL].astype(object),
    )
    work[PRODUCT_ID_COL] = work["_lookup_product_id"].astype(object).where(
        work["_lookup_product_id"].notna(),
        work[PRODUCT_ID_COL].astype(object),
    )
    if f"{CITY_COL}_x" in work.columns:
        work[CITY_COL] = work[f"{CITY_COL}_y"].fillna(work[f"{CITY_COL}_x"])
    work[PRODUCT_NAME_COL] = work["_lookup_product_name"].fillna(
        work[PRODUCT_NAME_COL]
    )
    work[CATEGORY_COL] = work["_lookup_category_name"].fillna(work[CATEGORY_COL])

    drop_cols = [
        "bakery_key",
        "product_key_lookup",
        "_lookup_bakery_id",
        "_lookup_product_id",
        "_lookup_product_name",
        "_lookup_category_name",
        f"{CITY_COL}_x",
        f"{CITY_COL}_y",
    ]
    work = work.drop(columns=[col for col in drop_cols if col in work.columns])
    return work, {
        "rows": int(len(work)),
        "matched_bakery_rows": matched_bakeries,
        "matched_product_rows": matched_products,
    }


def filter_hourly_by_assortment(
    hourly: pd.DataFrame,
    assortment_pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    if hourly.empty or assortment_pairs.empty:
        return hourly, {"rows_removed": 0, "sales_removed": 0.0}

    missing = {CITY_COL, PRODUCT_ID_COL}.difference(hourly.columns)
    if missing:
        raise KeyError(
            f"Hourly data cannot be assortment-filtered; missing {sorted(missing)}"
        )

    work = hourly.copy()
    original_cols = list(work.columns)
    work["_assortment_product_id"] = pd.to_numeric(
        work[PRODUCT_ID_COL],
        errors="coerce",
    ).astype("Int64")
    allowed = assortment_pairs.rename(
        columns={PRODUCT_ID_COL: "_assortment_product_id"}
    ).copy()
    allowed["_assortment_product_id"] = pd.to_numeric(
        allowed["_assortment_product_id"],
        errors="coerce",
    ).astype("Int64")
    work = work.merge(
        allowed.assign(_in_active_assortment=1),
        on=[CITY_COL, "_assortment_product_id"],
        how="left",
    )
    keep = work["_in_active_assortment"].fillna(0).astype(int).eq(1)
    removed = work.loc[~keep]
    return work.loc[keep, original_cols].copy(), {
        "rows_removed": int(len(removed)),
        "sales_removed": float(
            pd.to_numeric(removed[SKU_HOUR_SALES_COL], errors="coerce")
            .fillna(0.0)
            .sum()
        ),
    }


def build_from_raw(
    source_path: str | Path,
    *,
    chunk_size: int = CHUNK_SIZE,
    daily_cleaning_path: str | Path | None = DEFAULT_DAILY_CLEANING_PATH,
    assortment_path: str | Path | None = None,
    product_lookup_path: str | Path = DEFAULT_PRODUCT_LOOKUP_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    usecols = [
        RAW_DATETIME_COL,
        RAW_DATE_COL,
        RAW_EVENT_COL,
        RAW_QTY_COL,
        RAW_BAKERY_ID_COL,
        RAW_BAKERY_NAME_COL,
        RAW_CITY_COL,
        RAW_PRODUCT_ID_COL,
        RAW_PRODUCT_NAME_COL,
        RAW_CATEGORY_COL,
        "Дата время чека",
        "Дата продажи",
        "Вид события по кассе",
        "Кол-во",
        "Касса.Торговая точка",
        "Номенклатура",
        "Категория",
    ]
    parts: list[pd.DataFrame] = []
    reader = pd.read_csv(
        source_path,
        encoding="utf-8-sig",
        usecols=lambda c: c in usecols,
        chunksize=chunk_size,
    )
    for i, chunk in enumerate(reader, start=1):
        parts.append(aggregate_sku_hourly_chunk(chunk))
        if i % 5 == 0:
            print(f"processed chunks: {i}", flush=True)

    hourly = merge_hourly_parts(parts)
    if assortment_path:
        bakery_lookup = load_bakery_lookup(daily_cleaning_path)
        product_lookup = load_product_lookup(product_lookup_path)
        hourly, enrich_stats = enrich_hourly_dimensions(
            hourly,
            bakery_lookup=bakery_lookup,
            product_lookup=product_lookup,
        )
        print(
            "dimension lookup matched rows: "
            f"bakery={enrich_stats['matched_bakery_rows']}/{enrich_stats['rows']} "
            f"product={enrich_stats['matched_product_rows']}/{enrich_stats['rows']}",
            flush=True,
        )
        assortment_pairs = load_assortment_pairs(assortment_path)
        hourly, filter_stats = filter_hourly_by_assortment(hourly, assortment_pairs)
        print(
            "assortment filter removed rows: "
            f"{filter_stats['rows_removed']} "
            f"sales: {filter_stats['sales_removed']:.4f}",
            flush=True,
        )
    daily_weights = load_daily_profile_weights(daily_cleaning_path)
    return build_sku_hour_share_profile(hourly, daily_weights=daily_weights)


def build_summary(profile: pd.DataFrame, applied: pd.DataFrame) -> dict:
    return {
        "profile_rows": int(len(profile)),
        "applied_rows": int(len(applied)),
        "dates": int(applied[DATE_COL].nunique()) if len(applied) else 0,
        "bakeries": int(profile[BAKERY_ID_COL].nunique()) if len(profile) else 0,
        "products": int(profile[PRODUCT_ID_COL].nunique()) if len(profile) else 0,
        "mean_norm_share_sum": round(
            float(
                profile.groupby([BAKERY_ID_COL, DOW_COL, HOUR_COL])[
                    "mean_sku_share_in_hour_norm"
                ]
                .sum()
                .mean()
            ),
            6,
        )
        if len(profile)
        else 0.0,
        "mean_profile_weight": round(float(applied["profile_weight"].mean()), 6)
        if len(applied) and "profile_weight" in applied.columns
        else 1.0,
        "mean_reliability_score": round(
            float(profile["reliability_score"].mean()), 6
        )
        if len(profile) and "reliability_score" in profile.columns
        else 0.0,
        "median_reliability_score": round(
            float(profile["reliability_score"].median()), 6
        )
        if len(profile) and "reliability_score" in profile.columns
        else 0.0,
        "low_reliability_share": round(
            float(
                (profile["reliability_score"] < LOW_RELIABILITY_THRESHOLD).mean()
            ),
            6,
        )
        if len(profile) and "reliability_score" in profile.columns
        else 0.0,
        "recent_share_days": RECENT_SHARE_DAYS,
        "recent_share_alpha": RECENT_SHARE_ALPHA,
        "mean_recent_n_days": round(float(profile["recent_n_days"].mean()), 6)
        if len(profile) and "recent_n_days" in profile.columns
        else 0.0,
    }


def save_outputs(
    output_dir: str | Path, profile: pd.DataFrame, applied: pd.DataFrame, summary: dict
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_path = out_dir / OUTPUT_NAME
    applied_path = out_dir / APPLIED_DAILY_OUTPUT_NAME
    summary_path = out_dir / SUMMARY_OUTPUT_NAME
    profile.to_csv(profile_path, index=False, encoding="utf-8-sig")
    applied.to_csv(applied_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"profile": profile_path, "applied": applied_path, "summary": summary_path}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SKU hour-share profile from raw checks"
    )
    parser.add_argument(
        "--source-path", default=str(ROOT / "data" / "raw" / "sales_hrs_all.csv")
    )
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument(
        "--daily-cleaning-path", default=str(DEFAULT_DAILY_CLEANING_PATH)
    )
    parser.add_argument("--assortment-path", default="")
    parser.add_argument(
        "--product-lookup-path",
        default=str(DEFAULT_PRODUCT_LOOKUP_PATH),
    )
    args = parser.parse_args()

    profile, applied = build_from_raw(
        args.source_path,
        chunk_size=args.chunk_size,
        daily_cleaning_path=args.daily_cleaning_path,
        assortment_path=args.assortment_path or None,
        product_lookup_path=args.product_lookup_path,
    )
    summary = build_summary(profile, applied)
    paths = save_outputs(args.output_dir, profile, applied, summary)

    print("=" * 72)
    print("SKU HOUR SHARE PROFILE")
    print("=" * 72)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

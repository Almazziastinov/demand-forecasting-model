"""
Smooth applied SKU hour shares by lifting weak observations to the historical mean.

Input:
  - data/processed/sku_hour_share_profile.csv
  - data/processed/sku_hour_share_profile_daily.csv

Logic:
  1. Join each applied row with historical mean_sku_share_in_hour
  2. Replace sku_share_in_hour with max(sku_share_in_hour, mean_sku_share_in_hour)
  3. Renormalize shares within date x bakery x hour
  4. Rebuild an adjusted profile from adjusted applied rows
"""

from __future__ import annotations

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
PRODUCT_ID_COL = "product_id"
PRODUCT_NAME_COL = "product_name"
CATEGORY_COL = "category_name"
SKU_HOUR_SALES_COL = "sku_hour_sales"
BAKERY_HOUR_SALES_COL = "bakery_hour_sales"
SKU_SHARE_COL = "sku_share_in_hour"
PROFILE_MEAN_COL = "mean_sku_share_in_hour"
ADJUSTED_SHARE_COL = "sku_share_in_hour_adj"
ADJUSTED_SHARE_NORM_COL = "sku_share_in_hour_adj_norm"
ADJUSTED_SALES_COL = "sku_hour_sales_adj"

DEFAULT_PROFILE_PATH = ROOT / "data" / "processed" / "sku_hour_share_profile.csv"
DEFAULT_APPLIED_PATH = ROOT / "data" / "processed" / "sku_hour_share_profile_daily.csv"
CHUNK_SIZE = 1_000_000

OUTPUT_PROFILE_NAME = "sku_hour_share_profile_smoothed.csv"
OUTPUT_APPLIED_NAME = "sku_hour_share_profile_daily_smoothed.csv"
OUTPUT_SUMMARY_NAME = "sku_hour_share_profile_smoothed_summary.json"


def deduplicate_profile_means(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        BAKERY_ID_COL,
        PRODUCT_ID_COL,
        DOW_COL,
        HOUR_COL,
        PROFILE_MEAN_COL,
    ]
    missing = [col for col in keep_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in profile file: {missing}")

    work = df[keep_cols].copy()
    group_cols = [BAKERY_ID_COL, PRODUCT_ID_COL, DOW_COL, HOUR_COL]
    work[PROFILE_MEAN_COL] = pd.to_numeric(work[PROFILE_MEAN_COL], errors="coerce")
    work = (
        work.groupby(group_cols, as_index=False)
        .agg(mean_sku_share_in_hour=(PROFILE_MEAN_COL, "mean"))
    )
    if work.duplicated(group_cols).any():
        raise ValueError("Profile means are not unique after deduplication")
    return work


def load_profile_means(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    return deduplicate_profile_means(df)


def smooth_applied_chunk(chunk: pd.DataFrame, profile_means: pd.DataFrame) -> pd.DataFrame:
    work = chunk.merge(
        profile_means,
        on=[BAKERY_ID_COL, PRODUCT_ID_COL, DOW_COL, HOUR_COL],
        how="left",
        validate="many_to_one",
    )
    work[SKU_SHARE_COL] = pd.to_numeric(work[SKU_SHARE_COL], errors="coerce").fillna(0.0)
    work[PROFILE_MEAN_COL] = pd.to_numeric(work[PROFILE_MEAN_COL], errors="coerce").fillna(work[SKU_SHARE_COL])
    work[BAKERY_HOUR_SALES_COL] = pd.to_numeric(work[BAKERY_HOUR_SALES_COL], errors="coerce").fillna(0.0)

    work[ADJUSTED_SHARE_COL] = np.maximum(work[SKU_SHARE_COL], work[PROFILE_MEAN_COL])
    group_cols = [DATE_COL, BAKERY_ID_COL, HOUR_COL]
    denom = work.groupby(group_cols)[ADJUSTED_SHARE_COL].transform("sum")
    work[ADJUSTED_SHARE_NORM_COL] = np.where(
        denom > 0,
        work[ADJUSTED_SHARE_COL] / denom,
        0.0,
    )
    work[ADJUSTED_SALES_COL] = work[ADJUSTED_SHARE_NORM_COL] * work[BAKERY_HOUR_SALES_COL]
    return work


def build_adjusted_profile(applied_path: str | Path, *, chunk_size: int = CHUNK_SIZE) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    reader = pd.read_csv(applied_path, encoding="utf-8-sig", chunksize=chunk_size)
    for i, chunk in enumerate(reader, start=1):
        chunk[ADJUSTED_SHARE_NORM_COL] = pd.to_numeric(chunk[ADJUSTED_SHARE_NORM_COL], errors="coerce").fillna(0.0)
        chunk[ADJUSTED_SALES_COL] = pd.to_numeric(chunk[ADJUSTED_SALES_COL], errors="coerce").fillna(0.0)
        chunk["share_sum"] = chunk[ADJUSTED_SHARE_NORM_COL]
        chunk["sales_sum"] = chunk[ADJUSTED_SALES_COL]
        chunk["share_sq_sum"] = chunk[ADJUSTED_SHARE_NORM_COL] ** 2
        part = (
            chunk.groupby(
                [
                    BAKERY_ID_COL,
                    BAKERY_NAME_COL,
                    PRODUCT_ID_COL,
                    PRODUCT_NAME_COL,
                    CATEGORY_COL,
                    DOW_COL,
                    HOUR_COL,
                ],
                as_index=False,
            )
            .agg(
                n_days=(ADJUSTED_SHARE_NORM_COL, "size"),
                share_sum=("share_sum", "sum"),
                sales_sum=("sales_sum", "sum"),
                share_sq_sum=("share_sq_sum", "sum"),
            )
        )
        chunks.append(part)
        if i % 5 == 0:
            print(f"profile rebuild chunks: {i}", flush=True)

    profile = pd.concat(chunks, ignore_index=True)
    profile = (
        profile.groupby(
            [
                BAKERY_ID_COL,
                BAKERY_NAME_COL,
                PRODUCT_ID_COL,
                PRODUCT_NAME_COL,
                CATEGORY_COL,
                DOW_COL,
                HOUR_COL,
            ],
            as_index=False,
        )
        .agg(
            n_days=("n_days", "sum"),
            share_sum=("share_sum", "sum"),
            sales_sum=("sales_sum", "sum"),
            share_sq_sum=("share_sq_sum", "sum"),
        )
    )
    profile["mean_sku_share_in_hour"] = np.where(profile["n_days"] > 0, profile["share_sum"] / profile["n_days"], 0.0)
    profile["mean_sku_hour_sales"] = np.where(profile["n_days"] > 0, profile["sales_sum"] / profile["n_days"], 0.0)
    profile["median_sku_share_in_hour"] = profile["mean_sku_share_in_hour"]
    mean_sq = profile["mean_sku_share_in_hour"] ** 2
    profile["std_sku_share_in_hour"] = np.sqrt(
        np.maximum(profile["share_sq_sum"] / profile["n_days"].replace(0, np.nan) - mean_sq, 0.0)
    ).fillna(0.0)
    profile.drop(columns=["share_sum", "sales_sum", "share_sq_sum"], inplace=True)

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
    return profile.sort_values([BAKERY_ID_COL, DOW_COL, HOUR_COL, PRODUCT_ID_COL]).reset_index(drop=True)


def build_summary(profile: pd.DataFrame, applied_rows: int) -> dict:
    return {
        "profile_rows": int(len(profile)),
        "applied_rows": int(applied_rows),
        "bakeries": int(profile[BAKERY_ID_COL].nunique()) if len(profile) else 0,
        "products": int(profile[PRODUCT_ID_COL].nunique()) if len(profile) else 0,
        "mean_norm_share_sum": round(
            float(
                profile.groupby([BAKERY_ID_COL, DOW_COL, HOUR_COL])["mean_sku_share_in_hour_norm"].sum().mean()
            ),
            6,
        )
        if len(profile)
        else 0.0,
    }


def smooth_profile(
    profile_path: str | Path,
    applied_path: str | Path,
    output_dir: str | Path,
    *,
    chunk_size: int = CHUNK_SIZE,
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_applied_path = out_dir / OUTPUT_APPLIED_NAME
    output_profile_path = out_dir / OUTPUT_PROFILE_NAME
    output_summary_path = out_dir / OUTPUT_SUMMARY_NAME

    if output_applied_path.exists():
        output_applied_path.unlink()

    profile_means = load_profile_means(profile_path)
    applied_rows = 0

    reader = pd.read_csv(applied_path, encoding="utf-8-sig", chunksize=chunk_size)
    for i, chunk in enumerate(reader, start=1):
        smoothed = smooth_applied_chunk(chunk, profile_means)
        applied_rows += len(smoothed)
        smoothed.to_csv(
            output_applied_path,
            mode="a",
            index=False,
            encoding="utf-8-sig",
            header=(i == 1),
        )
        if i % 5 == 0:
            print(f"smoothed chunks: {i}", flush=True)

    profile = build_adjusted_profile(output_applied_path, chunk_size=chunk_size)
    profile.to_csv(output_profile_path, index=False, encoding="utf-8-sig")
    summary = build_summary(profile, applied_rows)
    output_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "profile": output_profile_path,
        "applied": output_applied_path,
        "summary": output_summary_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smooth SKU hour-share applied/profile outputs")
    parser.add_argument("--profile-path", default=str(DEFAULT_PROFILE_PATH))
    parser.add_argument("--applied-path", default=str(DEFAULT_APPLIED_PATH))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    paths = smooth_profile(
        args.profile_path,
        args.applied_path,
        args.output_dir,
        chunk_size=args.chunk_size,
    )
    print("=" * 72)
    print("SKU HOUR SHARE PROFILE SMOOTHED")
    print("=" * 72)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

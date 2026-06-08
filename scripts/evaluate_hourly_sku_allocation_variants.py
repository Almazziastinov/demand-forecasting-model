"""Evaluate hour-consistent SKU allocation correction variants.

This extends evaluate_sku_allocation_variants.py from daily-only correction to
hourly correction:

* base SKU-hour forecast keeps its production hour shape;
* dead recent SKU are zeroed;
* recent/core daily share targets become SKU multipliers;
* recent SKU absent from the production profile get the bakery hour shape;
* final SKU-hour rows are renormalized within date x bakery x hour, preserving
  the original bakery-hour forecast totals.
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


DEFAULT_HOURLY_PATH = (
    REPO_ROOT / "data" / "processed" / "sku_hour_forecast_holdout_30d_prod_uplifted_norm.csv"
)
DEFAULT_COMPARE_PATH = (
    REPO_ROOT / "reports" / "prod_holdout_sku_backtest" / "prod_holdout_sku_compare.csv"
)
DEFAULT_RECENT_PATH = (
    REPO_ROOT / "reports" / "prod_holdout_sku_backtest_variants" / "recent_assortment_stats.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "prod_holdout_sku_backtest_hourly_variants"


def _load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hourly = pd.read_csv(args.hourly_path, encoding="utf-8-sig", parse_dates=["date"])
    compare = pd.read_csv(args.compare_path, encoding="utf-8-sig", parse_dates=["date"])
    recent = pd.read_csv(args.recent_path, encoding="utf-8-sig")
    for frame in (hourly, compare):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return hourly, compare, recent


def _build_daily_targets(
    hourly: pd.DataFrame,
    compare: pd.DataFrame,
    recent: pd.DataFrame,
    variant: str,
) -> pd.DataFrame:
    base_daily = (
        hourly.groupby(["date", "dow", "bakery_id", "product_id"], as_index=False)
        .agg(base_daily_forecast=("sku_hour_forecast", "sum"))
    )
    bakery_day = (
        base_daily.groupby(["date", "bakery_id"], as_index=False)["base_daily_forecast"]
        .sum()
        .rename(columns={"base_daily_forecast": "bakery_day_forecast"})
    )
    dates_by_bakery = base_daily[["date", "dow", "bakery_id"]].drop_duplicates()
    recent_active = recent.loc[
        pd.to_numeric(recent["recent_days_sold"], errors="coerce").fillna(0) > 0,
        ["bakery_id", "product_id"],
    ].drop_duplicates()
    recent_grid = dates_by_bakery.merge(recent_active, on="bakery_id", how="inner")

    candidates = pd.concat(
        [
            base_daily[["date", "dow", "bakery_id", "product_id"]],
            recent_grid[["date", "dow", "bakery_id", "product_id"]],
        ],
        ignore_index=True,
    ).drop_duplicates(["date", "bakery_id", "product_id"])
    candidates = candidates.merge(
        base_daily,
        on=["date", "dow", "bakery_id", "product_id"],
        how="left",
        validate="one_to_one",
    )
    candidates["base_daily_forecast"] = (
        pd.to_numeric(candidates["base_daily_forecast"], errors="coerce").fillna(0.0)
    )
    candidates = candidates.merge(bakery_day, on=["date", "bakery_id"], how="left", validate="many_to_one")
    candidates = candidates.merge(
        recent[
            [
                "bakery_id",
                "product_id",
                "recent_qty",
                "recent_days_sold",
                "recent_share",
            ]
        ],
        on=["bakery_id", "product_id"],
        how="left",
        validate="many_to_one",
    )
    candidates[["recent_qty", "recent_days_sold", "recent_share"]] = candidates[
        ["recent_qty", "recent_days_sold", "recent_share"]
    ].fillna(0.0)
    candidates["prod_share"] = np.where(
        candidates["bakery_day_forecast"] > 0,
        candidates["base_daily_forecast"] / candidates["bakery_day_forecast"],
        0.0,
    )

    active = candidates["recent_days_sold"] > 0
    if variant == "dead_0d_hour":
        candidates["raw_share"] = np.where(active, candidates["prod_share"], 0.0)
    elif variant == "blend_recent_50_hour":
        candidates["raw_share"] = np.where(
            active,
            0.5 * candidates["prod_share"] + 0.5 * candidates["recent_share"],
            0.0,
        )
    elif variant == "core_recent_70_hour":
        core = (candidates["recent_days_sold"] >= 20) & (candidates["recent_share"] >= 0.01)
        regular_share = 0.7 * candidates["prod_share"] + 0.3 * candidates["recent_share"]
        core_share = 0.3 * candidates["prod_share"] + 0.7 * candidates["recent_share"]
        candidates["raw_share"] = np.where(core, core_share, regular_share)
        candidates["raw_share"] = np.where(active, candidates["raw_share"], 0.0)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    raw_sum = (
        candidates.groupby(["date", "bakery_id"], as_index=False)["raw_share"]
        .sum()
        .rename(columns={"raw_share": "raw_share_sum"})
    )
    candidates = candidates.merge(raw_sum, on=["date", "bakery_id"], how="left", validate="many_to_one")
    candidates["corrected_daily_forecast"] = np.where(
        candidates["raw_share_sum"] > 0,
        candidates["raw_share"] / candidates["raw_share_sum"] * candidates["bakery_day_forecast"],
        candidates["base_daily_forecast"],
    )

    # Keep metadata available for final daily compare.
    meta_cols = ["bakery_id", "bakery_name", "city", "product_id", "product_name", "category_name"]
    meta = compare[meta_cols].drop_duplicates(["bakery_id", "product_id"])
    return candidates.merge(meta, on=["bakery_id", "product_id"], how="left")


def _build_hourly_variant(
    hourly: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    bakery_hour = (
        hourly.groupby(["date", "dow", "bakery_id", "hour"], as_index=False)
        .agg(bakery_hour_forecast=("sku_hour_forecast", "sum"))
    )
    multipliers = targets.copy()
    multipliers["daily_multiplier"] = np.where(
        multipliers["base_daily_forecast"] > 0,
        multipliers["corrected_daily_forecast"] / multipliers["base_daily_forecast"],
        np.nan,
    )

    base = hourly.merge(
        multipliers[["date", "bakery_id", "product_id", "daily_multiplier"]],
        on=["date", "bakery_id", "product_id"],
        how="left",
        validate="many_to_one",
    )
    base["raw_hour_forecast"] = (
        pd.to_numeric(base["sku_hour_forecast"], errors="coerce").fillna(0.0)
        * pd.to_numeric(base["daily_multiplier"], errors="coerce").fillna(0.0)
    )
    base = base[["date", "dow", "bakery_id", "hour", "product_id", "raw_hour_forecast", "source"]]

    new_daily = multipliers[
        (multipliers["base_daily_forecast"] <= 0)
        & (multipliers["corrected_daily_forecast"] > 0)
    ].copy()
    new_rows = pd.DataFrame()
    if len(new_daily):
        new_rows = new_daily.merge(
            bakery_hour,
            on=["date", "dow", "bakery_id"],
            how="inner",
            validate="many_to_many",
        )
        new_rows["raw_hour_forecast"] = np.where(
            new_rows["bakery_day_forecast"] > 0,
            new_rows["corrected_daily_forecast"]
            * new_rows["bakery_hour_forecast"]
            / new_rows["bakery_day_forecast"],
            0.0,
        )
        new_rows["source"] = "recent_daily_new"
        new_rows = new_rows[["date", "dow", "bakery_id", "hour", "product_id", "raw_hour_forecast", "source"]]

    combined = pd.concat([base, new_rows], ignore_index=True)
    raw_hour_sum = (
        combined.groupby(["date", "bakery_id", "hour"], as_index=False)["raw_hour_forecast"]
        .sum()
        .rename(columns={"raw_hour_forecast": "raw_hour_sum"})
    )
    combined = combined.merge(raw_hour_sum, on=["date", "bakery_id", "hour"], how="left", validate="many_to_one")
    combined = combined.merge(
        bakery_hour[["date", "bakery_id", "hour", "bakery_hour_forecast"]],
        on=["date", "bakery_id", "hour"],
        how="left",
        validate="many_to_one",
    )
    combined["sku_hour_forecast_variant"] = np.where(
        combined["raw_hour_sum"] > 0,
        combined["raw_hour_forecast"] / combined["raw_hour_sum"] * combined["bakery_hour_forecast"],
        0.0,
    )
    return combined[
        [
            "date",
            "dow",
            "bakery_id",
            "hour",
            "product_id",
            "sku_hour_forecast_variant",
            "source",
        ]
    ]


def _score_daily(
    daily_variant: pd.DataFrame,
    compare: pd.DataFrame,
    variant: str,
    out_dir: Path,
) -> dict:
    fact_cols = [
        "date",
        "bakery_id",
        "bakery_name",
        "city",
        "product_id",
        "product_name",
        "category_name",
        "fact_qty",
        "fact_revenue",
    ]
    facts = compare[fact_cols].drop_duplicates(["date", "bakery_id", "product_id"])
    merged = daily_variant.merge(
        facts,
        on=["date", "bakery_id", "product_id"],
        how="outer",
        suffixes=("", "_fact"),
        validate="one_to_one",
    )
    for col in ["bakery_name", "city", "product_name", "category_name"]:
        fact_col = f"{col}_fact"
        if fact_col in merged.columns:
            merged[col] = merged[col].fillna(merged[fact_col])
            merged = merged.drop(columns=[fact_col])
    merged["forecast_variant"] = pd.to_numeric(
        merged["forecast_variant"],
        errors="coerce",
    ).fillna(0.0)
    merged["fact_qty"] = pd.to_numeric(merged["fact_qty"], errors="coerce").fillna(0.0)
    merged["err_raw_fact"] = merged["forecast_variant"] - merged["fact_qty"]
    merged["abs_err_raw_fact"] = merged["err_raw_fact"].abs()

    bakery_totals = (
        merged.groupby(["date", "bakery_id"], as_index=False)
        .agg(
            bakery_fact_qty=("fact_qty", "sum"),
            bakery_forecast_qty=("forecast_variant", "sum"),
        )
    )
    merged = merged.merge(bakery_totals, on=["date", "bakery_id"], how="left", validate="many_to_one")
    scale = np.where(
        merged["bakery_fact_qty"] > 0,
        merged["bakery_forecast_qty"] / merged["bakery_fact_qty"],
        1.0,
    )
    merged["fact_scaled_to_forecast_total"] = merged["fact_qty"] * scale
    merged["err_scaled_fact"] = merged["forecast_variant"] - merged["fact_scaled_to_forecast_total"]
    merged["abs_err_scaled_fact"] = merged["err_scaled_fact"].abs()
    merged["cell_type"] = np.select(
        [
            (merged["forecast_variant"] > 0) & (merged["fact_qty"] > 0),
            (merged["forecast_variant"] > 0) & (merged["fact_qty"] <= 0),
            (merged["forecast_variant"] <= 0) & (merged["fact_qty"] > 0),
        ],
        ["both_positive", "forecast_only_fact_zero", "fact_only_forecast_zero"],
        default="both_zero",
    )
    pair_totals = (
        merged.groupby(["bakery_id", "product_id"], as_index=False)
        .agg(
            pair_fact=("fact_qty", "sum"),
            pair_forecast=("forecast_variant", "sum"),
        )
    )
    merged = merged.merge(pair_totals, on=["bakery_id", "product_id"], how="left", validate="many_to_one")
    merged["dead_pair_window"] = (merged["pair_fact"] <= 0) & (merged["pair_forecast"] > 0)
    merged.to_csv(out_dir / f"{variant}_daily_compare.csv", index=False, encoding="utf-8-sig")

    by_pair = (
        merged.groupby(["bakery_id", "bakery_name", "city", "product_id", "product_name", "category_name"], as_index=False, dropna=False)
        .agg(
            fact_qty=("fact_qty", "sum"),
            forecast_variant=("forecast_variant", "sum"),
            abs_err_raw_fact=("abs_err_raw_fact", "sum"),
            abs_err_scaled_fact=("abs_err_scaled_fact", "sum"),
        )
    )
    by_pair["bias_qty"] = by_pair["forecast_variant"] - by_pair["fact_qty"]
    by_pair.to_csv(out_dir / f"{variant}_by_bakery_sku.csv", index=False, encoding="utf-8-sig")

    fact_sum = float(merged["fact_qty"].sum())
    scaled_fact_sum = float(merged["fact_scaled_to_forecast_total"].sum())
    forecast_sum = float(merged["forecast_variant"].sum())
    dead_sum = float(merged.loc[merged["dead_pair_window"], "forecast_variant"].sum())
    return {
        "variant": variant,
        "rows": int(len(merged)),
        "fact_sum": fact_sum,
        "forecast_sum": forecast_sum,
        "wmape_raw_fact_pct": float(merged["abs_err_raw_fact"].sum() / fact_sum * 100),
        "wmape_scaled_fact_pct": float(
            merged["abs_err_scaled_fact"].sum() / scaled_fact_sum * 100
        ),
        "bias_sum": float(merged["err_raw_fact"].sum()),
        "dead_pair_forecast_qty": dead_sum,
        "dead_pair_forecast_share_pct": float(dead_sum / forecast_sum * 100) if forecast_sum else 0.0,
        "forecast_only_fact_zero_qty": float(
            merged.loc[merged["cell_type"] == "forecast_only_fact_zero", "forecast_variant"].sum()
        ),
        "fact_only_forecast_zero_qty": float(
            merged.loc[merged["cell_type"] == "fact_only_forecast_zero", "fact_qty"].sum()
        ),
    }


def evaluate(args: argparse.Namespace) -> dict:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hourly, compare, recent = _load_inputs(args)
    variants = ["dead_0d_hour", "blend_recent_50_hour", "core_recent_70_hour"]
    summaries = []
    for variant in variants:
        targets = _build_daily_targets(hourly, compare, recent, variant)
        hour_variant = _build_hourly_variant(hourly, targets)
        hour_path = out_dir / f"{variant}_hourly_forecast.csv"
        hour_variant.to_csv(hour_path, index=False, encoding="utf-8-sig")
        daily_variant = (
            hour_variant.groupby(["date", "dow", "bakery_id", "product_id"], as_index=False)
            .agg(forecast_variant=("sku_hour_forecast_variant", "sum"))
        )
        meta = targets[["bakery_id", "bakery_name", "city", "product_id", "product_name", "category_name"]].drop_duplicates(["bakery_id", "product_id"])
        daily_variant = daily_variant.merge(meta, on=["bakery_id", "product_id"], how="left", validate="many_to_one")
        summaries.append(_score_daily(daily_variant, compare, variant, out_dir))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "hourly_variant_summary.csv", index=False, encoding="utf-8-sig")
    result = {"output_dir": str(out_dir), "variants": summaries}
    (out_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate hour-level SKU allocation variants")
    parser.add_argument("--hourly-path", default=str(DEFAULT_HOURLY_PATH))
    parser.add_argument("--compare-path", default=str(DEFAULT_COMPARE_PATH))
    parser.add_argument("--recent-path", default=str(DEFAULT_RECENT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main() -> None:
    result = evaluate(build_parser().parse_args())
    summary = pd.DataFrame(result["variants"])
    print("=" * 72)
    print("HOURLY SKU ALLOCATION VARIANT EVALUATION")
    print("=" * 72)
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

"""Explain root causes for the main SKU allocation failure modes.

The script decomposes production-style SKU forecasts into:

- baseline profile share (before recent correction);
- recent assortment share used by blend_recent_50;
- final blend_recent_50 forecast share;
- holdout fact share.

This is a diagnostic artifact, not production code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
VARIANT_DIR = REPO_ROOT / "reports" / "prod_holdout_sku_backtest_variants"
OUT_DIR = REPO_ROOT / "reports" / "sku_allocation_cause_analysis"

BASELINE_COMPARE = VARIANT_DIR / "baseline_compare.csv"
BLEND_COMPARE = VARIANT_DIR / "blend_recent_50_compare.csv"
PROFILE_PATH = REPO_ROOT / "data" / "processed" / "sku_hour_share_profile_smoothed.csv"

TARGET_BAKERIES = [30, 60, 105]
KEY_PRODUCT_IDS = [34, 36, 1071, 1076, 10340, 10556, 10667, 10760]
ECLAIR_PATTERN = "эклер"


def _safe_div(num: pd.Series | np.ndarray, den: pd.Series | np.ndarray) -> np.ndarray:
    num_arr = np.asarray(num, dtype="float64")
    den_arr = np.asarray(den, dtype="float64")
    return np.divide(
        num_arr,
        den_arr,
        out=np.zeros_like(num_arr, dtype="float64"),
        where=den_arr != 0,
    )


def _load_daily_pair(path: Path, forecast_col: str) -> pd.DataFrame:
    cols = [
        "date",
        "bakery_id",
        "bakery_name",
        "city",
        "product_id",
        "product_name",
        "category_name",
        "fact_qty",
        forecast_col,
        "recent_qty",
        "recent_days_sold",
        "recent_share",
    ]
    return pd.read_csv(path, usecols=cols, parse_dates=["date"])


def _pair_decomposition() -> pd.DataFrame:
    baseline = _load_daily_pair(BASELINE_COMPARE, "forecast_variant").rename(
        columns={"forecast_variant": "baseline_forecast"}
    )
    blend = _load_daily_pair(BLEND_COMPARE, "forecast_variant").rename(
        columns={"forecast_variant": "blend_forecast"}
    )

    keys = ["date", "bakery_id", "product_id"]
    work = blend.merge(
        baseline[keys + ["baseline_forecast"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    work["baseline_forecast"] = pd.to_numeric(
        work["baseline_forecast"],
        errors="coerce",
    ).fillna(0.0)

    pair = (
        work.groupby(
            [
                "bakery_id",
                "bakery_name",
                "city",
                "product_id",
                "product_name",
                "category_name",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            fact_qty=("fact_qty", "sum"),
            baseline_forecast=("baseline_forecast", "sum"),
            blend_forecast=("blend_forecast", "sum"),
            recent_qty=("recent_qty", "max"),
            recent_days_sold=("recent_days_sold", "max"),
            recent_share=("recent_share", "max"),
        )
    )
    totals = (
        pair.groupby("bakery_id", as_index=False)
        .agg(
            bakery_fact_qty=("fact_qty", "sum"),
            bakery_baseline_forecast=("baseline_forecast", "sum"),
            bakery_blend_forecast=("blend_forecast", "sum"),
        )
    )
    pair = pair.merge(totals, on="bakery_id", how="left", validate="many_to_one")
    pair["fact_share"] = _safe_div(pair["fact_qty"], pair["bakery_fact_qty"])
    pair["baseline_share"] = _safe_div(
        pair["baseline_forecast"],
        pair["bakery_baseline_forecast"],
    )
    pair["blend_share"] = _safe_div(
        pair["blend_forecast"],
        pair["bakery_blend_forecast"],
    )
    pair["baseline_minus_fact_share_pp"] = (
        pair["baseline_share"] - pair["fact_share"]
    ) * 100
    pair["recent_minus_fact_share_pp"] = (
        pair["recent_share"] - pair["fact_share"]
    ) * 100
    pair["blend_minus_fact_share_pp"] = (
        pair["blend_share"] - pair["fact_share"]
    ) * 100
    pair["blend_vs_baseline_qty"] = pair["blend_forecast"] - pair["baseline_forecast"]
    pair["blend_bias_qty"] = pair["blend_forecast"] - pair["fact_qty"]
    return pair


def _profile_quality(pair: pd.DataFrame) -> pd.DataFrame:
    profile = pd.read_csv(PROFILE_PATH)
    product_ids = set(pair["product_id"].dropna().astype("int64"))
    profile = profile[
        profile["product_id"].isin(product_ids)
        & profile["bakery_id"].isin(pair["bakery_id"].unique())
    ].copy()
    if profile.empty:
        return pair

    agg = (
        profile.groupby(["bakery_id", "product_id"], as_index=False)
        .agg(
            profile_rows=("product_id", "size"),
            profile_mean_n_days=("n_days", "mean"),
            profile_mean_base_n_days=("base_n_days", "mean"),
            profile_mean_base_recent_n_days=("base_recent_n_days", "mean"),
            profile_mean_reliability=("base_reliability_score", "mean"),
            profile_mean_zero_share_rate=("base_zero_share_rate", "mean"),
            profile_mean_cv_share=("base_cv_share", "mean"),
        )
    )
    return pair.merge(agg, on=["bakery_id", "product_id"], how="left")


def _write_report(
    target_pairs: pd.DataFrame,
    eclairs: pd.DataFrame,
    eclair_by_bakery: pd.DataFrame,
) -> None:
    lines = [
        "# SKU Allocation Cause Analysis",
        "",
        "The production-style forecast is decomposed into baseline profile share, "
        "recent 30-day assortment share, final blend_recent_50 share, and holdout "
        "fact share.",
        "",
        "## Key Bakeries And Products",
        "",
        target_pairs[
            [
                "bakery_id",
                "bakery_name",
                "product_id",
                "product_name",
                "fact_qty",
                "baseline_forecast",
                "blend_forecast",
                "blend_bias_qty",
                "fact_share",
                "baseline_share",
                "recent_share",
                "blend_share",
                "baseline_minus_fact_share_pp",
                "recent_minus_fact_share_pp",
                "blend_minus_fact_share_pp",
                "recent_days_sold",
                "profile_mean_base_n_days",
                "profile_mean_base_recent_n_days",
                "profile_mean_reliability",
            ]
        ].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Eclair Network Summary",
        "",
        eclairs[
            [
                "product_id",
                "product_name",
                "fact_qty",
                "baseline_forecast",
                "blend_forecast",
                "blend_bias_qty",
                "fact_share",
                "baseline_share",
                "recent_share",
                "blend_share",
                "baseline_minus_fact_share_pp",
                "recent_minus_fact_share_pp",
                "blend_minus_fact_share_pp",
                "recent_days_sold",
            ]
        ].head(30).to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Worst Eclair Bakeries",
        "",
        eclair_by_bakery[
            [
                "bakery_id",
                "bakery_name",
                "city",
                "fact_qty",
                "baseline_forecast",
                "blend_forecast",
                "blend_bias_qty",
                "fact_share",
                "baseline_share",
                "recent_share_weighted",
                "blend_share",
                "baseline_minus_fact_share_pp",
                "recent_minus_fact_share_pp",
                "blend_minus_fact_share_pp",
            ]
        ].head(30).to_markdown(index=False, floatfmt=".3f"),
        "",
    ]
    (OUT_DIR / "sku_allocation_cause_analysis.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pair = _profile_quality(_pair_decomposition())

    target_pairs = pair[
        pair["bakery_id"].isin(TARGET_BAKERIES)
        & pair["product_id"].isin(KEY_PRODUCT_IDS)
    ].copy()
    target_pairs = target_pairs.sort_values(
        ["bakery_id", "blend_minus_fact_share_pp"],
        ascending=[True, False],
    )
    target_pairs.to_csv(
        OUT_DIR / "target_bakery_product_share_decomposition.csv",
        index=False,
        encoding="utf-8-sig",
    )

    eclair_rows = pair[
        pair["product_name"].fillna("").str.casefold().str.contains(ECLAIR_PATTERN)
    ].copy()
    eclair_by_product = (
        eclair_rows.groupby(["product_id", "product_name"], as_index=False)
        .agg(
            fact_qty=("fact_qty", "sum"),
            baseline_forecast=("baseline_forecast", "sum"),
            blend_forecast=("blend_forecast", "sum"),
            recent_qty=("recent_qty", "sum"),
            recent_days_sold=("recent_days_sold", "max"),
        )
    )
    totals = {
        "fact_qty": pair["fact_qty"].sum(),
        "baseline_forecast": pair["baseline_forecast"].sum(),
        "blend_forecast": pair["blend_forecast"].sum(),
        "recent_qty": pair["recent_qty"].sum(),
    }
    eclair_by_product["fact_share"] = eclair_by_product["fact_qty"] / totals["fact_qty"]
    eclair_by_product["baseline_share"] = (
        eclair_by_product["baseline_forecast"] / totals["baseline_forecast"]
    )
    eclair_by_product["blend_share"] = (
        eclair_by_product["blend_forecast"] / totals["blend_forecast"]
    )
    eclair_by_product["recent_share"] = (
        eclair_by_product["recent_qty"] / totals["recent_qty"]
    )
    eclair_by_product["blend_bias_qty"] = (
        eclair_by_product["blend_forecast"] - eclair_by_product["fact_qty"]
    )
    for source in ["baseline", "recent", "blend"]:
        eclair_by_product[f"{source}_minus_fact_share_pp"] = (
            eclair_by_product[f"{source}_share"] - eclair_by_product["fact_share"]
        ) * 100
    eclair_by_product = eclair_by_product.sort_values(
        "blend_bias_qty",
        ascending=False,
    )
    eclair_by_product.to_csv(
        OUT_DIR / "eclair_share_decomposition_by_product.csv",
        index=False,
        encoding="utf-8-sig",
    )

    eclair_by_bakery = (
        eclair_rows.groupby(["bakery_id", "bakery_name", "city"], as_index=False)
        .agg(
            fact_qty=("fact_qty", "sum"),
            baseline_forecast=("baseline_forecast", "sum"),
            blend_forecast=("blend_forecast", "sum"),
            recent_qty=("recent_qty", "sum"),
        )
    )
    bakery_totals = (
        pair.groupby(["bakery_id"], as_index=False)
        .agg(
            bakery_fact_qty=("fact_qty", "sum"),
            bakery_baseline_forecast=("baseline_forecast", "sum"),
            bakery_blend_forecast=("blend_forecast", "sum"),
            bakery_recent_qty=("recent_qty", "sum"),
        )
    )
    eclair_by_bakery = eclair_by_bakery.merge(
        bakery_totals,
        on="bakery_id",
        how="left",
    )
    eclair_by_bakery["fact_share"] = _safe_div(
        eclair_by_bakery["fact_qty"],
        eclair_by_bakery["bakery_fact_qty"],
    )
    eclair_by_bakery["baseline_share"] = _safe_div(
        eclair_by_bakery["baseline_forecast"],
        eclair_by_bakery["bakery_baseline_forecast"],
    )
    eclair_by_bakery["blend_share"] = _safe_div(
        eclair_by_bakery["blend_forecast"],
        eclair_by_bakery["bakery_blend_forecast"],
    )
    eclair_by_bakery["recent_share_weighted"] = _safe_div(
        eclair_by_bakery["recent_qty"],
        eclair_by_bakery["bakery_recent_qty"],
    )
    eclair_by_bakery["blend_bias_qty"] = (
        eclair_by_bakery["blend_forecast"] - eclair_by_bakery["fact_qty"]
    )
    eclair_by_bakery["baseline_minus_fact_share_pp"] = (
        eclair_by_bakery["baseline_share"] - eclair_by_bakery["fact_share"]
    ) * 100
    eclair_by_bakery["recent_minus_fact_share_pp"] = (
        eclair_by_bakery["recent_share_weighted"] - eclair_by_bakery["fact_share"]
    ) * 100
    eclair_by_bakery["blend_minus_fact_share_pp"] = (
        eclair_by_bakery["blend_share"] - eclair_by_bakery["fact_share"]
    ) * 100
    eclair_by_bakery = eclair_by_bakery.sort_values(
        "blend_bias_qty",
        ascending=False,
    )
    eclair_by_bakery.to_csv(
        OUT_DIR / "eclair_share_decomposition_by_bakery.csv",
        index=False,
        encoding="utf-8-sig",
    )

    _write_report(target_pairs, eclair_by_product, eclair_by_bakery)
    print(f"Wrote {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

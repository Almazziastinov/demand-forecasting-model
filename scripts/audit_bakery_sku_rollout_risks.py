"""Audit SKU allocation rollout risks across all bakeries.

The audit uses existing holdout artifacts. It is intended for rollout selection:
bakery-day totals are checked, but the main signal is SKU/category allocation
quality inside each bakery.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
VARIANT_DIR = REPORTS_DIR / "prod_holdout_sku_backtest_variants"

BAKERY_DAY_PATH = REPORTS_DIR / "bakery_day_model_bias_by_bakery.csv"
BASELINE_SKU_PATH = VARIANT_DIR / "baseline_by_bakery_sku.csv"
PROD_SKU_PATH = VARIANT_DIR / "blend_recent_50_by_bakery_sku.csv"

OUT_DIR = REPORTS_DIR / "rollout_sku_risk_audit"
OUT_SUMMARY = OUT_DIR / "bakery_sku_risk_summary.csv"
OUT_FLAGS = OUT_DIR / "bakery_sku_risk_flags.csv"
OUT_TOP_SKU = OUT_DIR / "top_problem_bakery_sku.csv"
OUT_TOP_CATEGORY = OUT_DIR / "top_problem_bakery_category.csv"
OUT_REPORT = OUT_DIR / "rollout_sku_risk_audit.md"

SERVICE_CATEGORY_PATTERNS = (
    "прочие",
    "заказ",
)
ECLAIR_PATTERN = "эклер"


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _safe_div(
    numerator: pd.Series | np.ndarray | float,
    denominator: pd.Series | np.ndarray | float,
) -> np.ndarray:
    numerator_arr = np.asarray(numerator, dtype="float64")
    denominator_arr = np.asarray(denominator, dtype="float64")
    return np.divide(
        numerator_arr,
        denominator_arr,
        out=np.zeros_like(numerator_arr, dtype="float64"),
        where=denominator_arr != 0,
    )


def _contains_any(series: pd.Series, patterns: tuple[str, ...]) -> pd.Series:
    text = series.fillna("").astype(str).str.casefold()
    result = pd.Series(False, index=series.index)
    for pattern in patterns:
        result = result | text.str.contains(pattern, regex=False)
    return result


def _prepare_sku(path: Path, forecast_col: str = "forecast_variant") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["fact_qty"] = _num(df["fact_qty"])
    df[forecast_col] = _num(df[forecast_col])
    df["abs_err_scaled_fact"] = _num(df["abs_err_scaled_fact"])
    df["bias_qty"] = df[forecast_col] - df["fact_qty"]
    df["abs_bias_qty"] = df["bias_qty"].abs()
    if "recent_days_sold" not in df.columns:
        df["recent_days_sold"] = 0
    df["recent_days_sold"] = _num(df["recent_days_sold"]).astype("int64")
    if "recent_qty" not in df.columns:
        df["recent_qty"] = 0.0
    df["recent_qty"] = _num(df["recent_qty"])
    return df


def _bakery_share_metrics(df: pd.DataFrame) -> pd.DataFrame:
    totals = (
        df.groupby("bakery_id", as_index=False)
        .agg(
            sku_fact_total=("fact_qty", "sum"),
            sku_forecast_total=("forecast_variant", "sum"),
            sku_abs_err_scaled=("abs_err_scaled_fact", "sum"),
            sku_abs_bias=("abs_bias_qty", "sum"),
            sku_count=("product_id", "nunique"),
        )
    )
    work = df.merge(
        totals[["bakery_id", "sku_fact_total", "sku_forecast_total"]],
        on="bakery_id",
        how="left",
    )
    work["fact_share"] = _safe_div(work["fact_qty"], work["sku_fact_total"])
    work["forecast_share"] = _safe_div(
        work["forecast_variant"],
        work["sku_forecast_total"],
    )
    work["share_abs_diff"] = (work["forecast_share"] - work["fact_share"]).abs()

    share = (
        work.groupby("bakery_id", as_index=False)
        .agg(
            sku_share_distance=("share_abs_diff", lambda x: float(x.sum() / 2.0)),
            max_sku_share_abs_diff=("share_abs_diff", "max"),
        )
    )

    top_fact = (
        work.sort_values(["bakery_id", "fact_qty"], ascending=[True, False])
        .groupby("bakery_id", as_index=False)
        .head(1)[["bakery_id", "product_name", "fact_share"]]
        .rename(
            columns={
                "product_name": "top_fact_sku",
                "fact_share": "top_fact_sku_share",
            }
        )
    )
    top_forecast = (
        work.sort_values(["bakery_id", "forecast_variant"], ascending=[True, False])
        .groupby("bakery_id", as_index=False)
        .head(1)[["bakery_id", "product_name", "forecast_share"]]
        .rename(
            columns={
                "product_name": "top_forecast_sku",
                "forecast_share": "top_forecast_sku_share",
            }
        )
    )
    out = totals.merge(share, on="bakery_id", how="left")
    out = out.merge(top_fact, on="bakery_id", how="left").merge(
        top_forecast,
        on="bakery_id",
        how="left",
    )
    out["top_sku_mismatch"] = out["top_fact_sku"] != out["top_forecast_sku"]
    out["sku_wmape_scaled_pct"] = (
        _safe_div(out["sku_abs_err_scaled"], out["sku_forecast_total"]) * 100
    )
    return out


def _segment_metrics(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["is_service_category"] = _contains_any(
        work["category_name"],
        SERVICE_CATEGORY_PATTERNS,
    )
    work["is_eclair"] = (
        work["product_name"]
        .fillna("")
        .astype(str)
        .str.casefold()
        .str.contains(ECLAIR_PATTERN, regex=False)
    )
    work["is_runner"] = (
        (work["fact_qty"] >= 500)
        & (work["recent_days_sold"] >= 20)
        & ~work["is_service_category"]
    )
    work["is_forecast_only"] = (work["fact_qty"] <= 0) & (work["forecast_variant"] > 0)
    work["is_fact_only"] = (work["fact_qty"] > 0) & (work["forecast_variant"] <= 0)
    work["is_dead_recent_forecast"] = (
        (work["recent_days_sold"] <= 0) & (work["forecast_variant"] > 0)
    )
    work["is_severe_runner"] = work["is_runner"] & (
        work["bias_qty"].abs() >= np.maximum(100.0, work["fact_qty"] * 0.35)
    )

    def agg_segment(mask_col: str, prefix: str) -> pd.DataFrame:
        seg = work[work[mask_col]]
        if seg.empty:
            return work[["bakery_id"]].drop_duplicates().assign(
                **{
                    f"{prefix}_fact_qty": 0.0,
                    f"{prefix}_forecast_qty": 0.0,
                    f"{prefix}_abs_err_scaled": 0.0,
                    f"{prefix}_bias_qty": 0.0,
                    f"{prefix}_sku_count": 0,
                }
            )
        return (
            seg.groupby("bakery_id", as_index=False)
            .agg(
                **{
                    f"{prefix}_fact_qty": ("fact_qty", "sum"),
                    f"{prefix}_forecast_qty": ("forecast_variant", "sum"),
                    f"{prefix}_abs_err_scaled": ("abs_err_scaled_fact", "sum"),
                    f"{prefix}_bias_qty": ("bias_qty", "sum"),
                    f"{prefix}_sku_count": ("product_id", "nunique"),
                }
            )
        )

    result = work[["bakery_id"]].drop_duplicates()
    for mask_col, prefix in [
        ("is_runner", "runner"),
        ("is_eclair", "eclair"),
        ("is_service_category", "service"),
        ("is_forecast_only", "forecast_only"),
        ("is_fact_only", "fact_only"),
        ("is_dead_recent_forecast", "dead_recent_forecast"),
        ("is_severe_runner", "severe_runner"),
    ]:
        result = result.merge(agg_segment(mask_col, prefix), on="bakery_id", how="left")

    for col in result.columns:
        if col != "bakery_id":
            result[col] = result[col].fillna(0)
    return result


def _category_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cat = (
        df.groupby(
            ["bakery_id", "bakery_name", "city", "category_name"],
            as_index=False,
            dropna=False,
        )
        .agg(
            fact_qty=("fact_qty", "sum"),
            forecast_qty=("forecast_variant", "sum"),
            abs_err_scaled=("abs_err_scaled_fact", "sum"),
        )
    )
    cat["bias_qty"] = cat["forecast_qty"] - cat["fact_qty"]
    totals = (
        cat.groupby("bakery_id", as_index=False)
        .agg(total_fact=("fact_qty", "sum"), total_forecast=("forecast_qty", "sum"))
    )
    cat = cat.merge(totals, on="bakery_id", how="left")
    cat["fact_share"] = _safe_div(cat["fact_qty"], cat["total_fact"])
    cat["forecast_share"] = _safe_div(cat["forecast_qty"], cat["total_forecast"])
    cat["share_diff"] = cat["forecast_share"] - cat["fact_share"]
    cat["share_abs_diff"] = cat["share_diff"].abs()
    summary = (
        cat.groupby("bakery_id", as_index=False)
        .agg(
            category_share_distance=("share_abs_diff", lambda x: float(x.sum() / 2.0)),
            max_category_share_abs_diff=("share_abs_diff", "max"),
        )
    )
    worst = (
        cat.sort_values(["bakery_id", "share_abs_diff"], ascending=[True, False])
        .groupby("bakery_id", as_index=False)
        .head(1)[["bakery_id", "category_name", "share_diff", "bias_qty"]]
        .rename(
            columns={
                "category_name": "worst_category",
                "share_diff": "worst_category_share_diff",
                "bias_qty": "worst_category_bias_qty",
            }
        )
    )
    return summary.merge(worst, on="bakery_id", how="left"), cat


def _baseline_metrics() -> pd.DataFrame:
    baseline = _prepare_sku(BASELINE_SKU_PATH)
    summary = _bakery_share_metrics(
        baseline.rename(columns={"forecast_variant": "forecast_variant"})
    )
    return summary[["bakery_id", "sku_wmape_scaled_pct", "sku_share_distance"]].rename(
        columns={
            "sku_wmape_scaled_pct": "baseline_sku_wmape_scaled_pct",
            "sku_share_distance": "baseline_sku_share_distance",
        }
    )


def _assign_flags(row: pd.Series) -> list[str]:
    flags: list[str] = []
    if row["sku_share_distance"] >= 0.20 or row["max_sku_share_abs_diff"] >= 0.08:
        flags.append("sku_share_distribution")
    if row["runner_wmape_scaled_pct"] >= 35 or row["severe_runner_sku_count"] >= 2:
        flags.append("runner_sku")
    if row["eclair_forecast_share_pct"] >= 2.0 and row["eclair_bias_qty"] >= 150:
        flags.append("eclair_overforecast")
    if row["forecast_only_forecast_share_pct"] >= 3.0:
        flags.append("forecast_only_sku")
    if row["fact_only_fact_share_pct"] >= 2.0:
        flags.append("missed_active_sku")
    if (
        row["category_share_distance"] >= 0.16
        or row["max_category_share_abs_diff"] >= 0.12
    ):
        flags.append("category_mix")
    if abs(row["bias_pct_of_actual_mean"]) >= 12 or row["wmape"] >= 15:
        flags.append("bakery_day_total")
    if row["service_forecast_share_pct"] >= 8 and abs(row["service_bias_qty"]) >= 100:
        flags.append("service_category")
    if row["baseline_improvement_pct_points"] < 8 and row["sku_wmape_scaled_pct"] >= 45:
        flags.append("correction_not_enough")
    return flags


def _risk_level(flags: list[str]) -> str:
    hard_flags = {
        "sku_share_distribution",
        "runner_sku",
        "missed_active_sku",
        "bakery_day_total",
    }
    if hard_flags.intersection(flags) or len(flags) >= 4:
        return "exclude_or_deep_review"
    if flags:
        return "manual_review"
    return "pass"


def _top_problem_skus(df: pd.DataFrame, max_rows: int = 300) -> pd.DataFrame:
    work = df.copy()
    work["is_service_category"] = _contains_any(
        work["category_name"],
        SERVICE_CATEGORY_PATTERNS,
    )
    work["is_eclair"] = (
        work["product_name"]
        .fillna("")
        .astype(str)
        .str.casefold()
        .str.contains(ECLAIR_PATTERN, regex=False)
    )
    work["problem_type"] = np.select(
        [
            work["is_eclair"] & (work["bias_qty"] > 0),
            work["is_service_category"] & (work["bias_qty"].abs() > 0),
            (work["fact_qty"] <= 0) & (work["forecast_variant"] > 0),
            (work["fact_qty"] > 0) & (work["forecast_variant"] <= 0),
            (
                (work["fact_qty"] >= 500)
                & ~work["is_service_category"]
                & (work["bias_qty"] < 0)
            ),
            (
                (work["fact_qty"] >= 500)
                & ~work["is_service_category"]
                & (work["bias_qty"] > 0)
            ),
        ],
        [
            "eclair_overforecast",
            "service_category_bias",
            "forecast_only_sku",
            "missed_active_sku",
            "runner_underforecast",
            "runner_overforecast",
        ],
        default="other_large_error",
    )
    return work.sort_values("abs_err_scaled_fact", ascending=False).head(max_rows)


def _write_markdown(
    summary: pd.DataFrame,
    top_sku: pd.DataFrame,
    top_category: pd.DataFrame,
) -> None:
    risk_counts = (
        summary["risk_level"]
        .value_counts()
        .rename_axis("risk_level")
        .reset_index(name="bakeries")
    )
    flag_counts = (
        summary["risk_flags"]
        .str.get_dummies(sep=";")
        .sum()
        .sort_values(ascending=False)
        .rename_axis("flag")
        .reset_index(name="bakeries")
    )
    high = summary[summary["risk_level"] == "exclude_or_deep_review"].sort_values(
        ["risk_flag_count", "sku_share_distance", "runner_wmape_scaled_pct"],
        ascending=False,
    )

    lines = [
        "# Rollout SKU Risk Audit",
        "",
        "Input files:",
        f"- `{PROD_SKU_PATH.relative_to(REPO_ROOT)}`",
        f"- `{BASELINE_SKU_PATH.relative_to(REPO_ROOT)}`",
        f"- `{BAKERY_DAY_PATH.relative_to(REPO_ROOT)}`",
        "",
        "The audit is bakery-level. It checks SKU allocation quality, "
        "runner SKU, eclairs, forecast-only/missed SKU, category mix, "
        "service-category leakage, and bakery-day total sanity.",
        "",
        "## Risk Counts",
        "",
        risk_counts.to_markdown(index=False),
        "",
        "## Flag Counts",
        "",
        flag_counts.to_markdown(index=False),
        "",
        "## Highest-Risk Bakeries",
        "",
        high[
            [
                "bakery_id",
                "bakery_name",
                "city",
                "risk_flags",
                "sku_wmape_scaled_pct",
                "sku_share_distance",
                "runner_wmape_scaled_pct",
                "eclair_bias_qty",
                "forecast_only_forecast_share_pct",
                "fact_only_fact_share_pct",
                "wmape",
                "bias_pct_of_actual_mean",
            ]
        ]
        .head(30)
        .to_markdown(index=False, floatfmt=".1f"),
        "",
        "## Largest SKU-Level Errors",
        "",
        top_sku[
            [
                "bakery_id",
                "bakery_name",
                "city",
                "product_id",
                "product_name",
                "category_name",
                "problem_type",
                "fact_qty",
                "forecast_variant",
                "bias_qty",
                "abs_err_scaled_fact",
                "recent_days_sold",
            ]
        ]
        .head(40)
        .to_markdown(index=False, floatfmt=".1f"),
        "",
        "## Largest Category Mix Errors",
        "",
        top_category[
            [
                "bakery_id",
                "bakery_name",
                "city",
                "category_name",
                "fact_qty",
                "forecast_qty",
                "bias_qty",
                "share_diff",
                "share_abs_diff",
            ]
        ]
        .head(40)
        .to_markdown(index=False, floatfmt=".3f"),
        "",
    ]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit bakery SKU rollout risks")
    parser.add_argument("--sku-path", default=str(PROD_SKU_PATH))
    parser.add_argument("--baseline-sku-path", default=str(BASELINE_SKU_PATH))
    parser.add_argument("--bakery-day-path", default=str(BAKERY_DAY_PATH))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_summary = output_dir / "bakery_sku_risk_summary.csv"
    out_flags = output_dir / "bakery_sku_risk_flags.csv"
    out_top_sku = output_dir / "top_problem_bakery_sku.csv"
    out_top_category = output_dir / "top_problem_bakery_category.csv"
    out_report = output_dir / "rollout_sku_risk_audit.md"

    sku = _prepare_sku(Path(args.sku_path))
    bakery_day = pd.read_csv(args.bakery_day_path)
    bakery_day = bakery_day[
        [
            "bakery_id",
            "bakery_name",
            "city",
            "n_days",
            "actual_mean",
            "forecast_mean",
            "wmape",
            "bias_pct_of_actual_mean",
        ]
    ]

    summary = bakery_day.merge(
        _bakery_share_metrics(sku),
        on="bakery_id",
        how="left",
        suffixes=("", "_sku"),
    )
    summary = summary.merge(_segment_metrics(sku), on="bakery_id", how="left")
    category_summary, category_detail = _category_metrics(sku)
    summary = summary.merge(category_summary, on="bakery_id", how="left")
    baseline_summary = _bakery_share_metrics(_prepare_sku(Path(args.baseline_sku_path)))
    baseline_summary = baseline_summary[
        ["bakery_id", "sku_wmape_scaled_pct", "sku_share_distance"]
    ].rename(
        columns={
            "sku_wmape_scaled_pct": "baseline_sku_wmape_scaled_pct",
            "sku_share_distance": "baseline_sku_share_distance",
        }
    )
    summary = summary.merge(baseline_summary, on="bakery_id", how="left")

    ratio_pairs = [
        ("runner", "runner_forecast_qty"),
        ("eclair", "eclair_forecast_qty"),
        ("service", "service_forecast_qty"),
        ("forecast_only", "forecast_only_forecast_qty"),
        ("dead_recent_forecast", "dead_recent_forecast_forecast_qty"),
    ]
    for prefix, value_col in ratio_pairs:
        summary[f"{prefix}_forecast_share_pct"] = (
            _safe_div(summary[value_col], summary["sku_forecast_total"]) * 100
        )

    fact_ratio_pairs = [
        ("runner", "runner_fact_qty"),
        ("eclair", "eclair_fact_qty"),
        ("service", "service_fact_qty"),
        ("fact_only", "fact_only_fact_qty"),
    ]
    for prefix, value_col in fact_ratio_pairs:
        summary[f"{prefix}_fact_share_pct"] = (
            _safe_div(summary[value_col], summary["sku_fact_total"]) * 100
        )

    for prefix in ["runner", "eclair", "service"]:
        summary[f"{prefix}_wmape_scaled_pct"] = _safe_div(
            summary[f"{prefix}_abs_err_scaled"],
            summary[f"{prefix}_forecast_qty"],
        ) * 100

    summary["baseline_improvement_pct_points"] = (
        summary["baseline_sku_wmape_scaled_pct"] - summary["sku_wmape_scaled_pct"]
    )
    summary["risk_flag_list"] = summary.apply(_assign_flags, axis=1)
    summary["risk_flags"] = summary["risk_flag_list"].map(lambda flags: ";".join(flags))
    summary["risk_flag_count"] = summary["risk_flag_list"].map(len)
    summary["risk_level"] = summary["risk_flag_list"].map(_risk_level)
    summary = summary.drop(columns=["risk_flag_list"])

    ordered = summary.sort_values(
        [
            "risk_level",
            "risk_flag_count",
            "sku_share_distance",
            "runner_wmape_scaled_pct",
        ],
        ascending=[True, False, False, False],
    )
    ordered.to_csv(out_summary, index=False, encoding="utf-8-sig")

    flags = (
        summary[
            [
                "bakery_id",
                "bakery_name",
                "city",
                "risk_level",
                "risk_flags",
                "risk_flag_count",
            ]
        ]
        .sort_values(["risk_flag_count", "bakery_id"], ascending=[False, True])
    )
    flags.to_csv(out_flags, index=False, encoding="utf-8-sig")

    top_sku = _top_problem_skus(sku)
    top_sku.to_csv(out_top_sku, index=False, encoding="utf-8-sig")
    top_category = category_detail.sort_values("share_abs_diff", ascending=False)
    top_category.to_csv(out_top_category, index=False, encoding="utf-8-sig")
    old_report = OUT_REPORT
    try:
        globals()["OUT_REPORT"] = out_report
        _write_markdown(summary, top_sku, top_category)
    finally:
        globals()["OUT_REPORT"] = old_report

    print(f"Wrote {out_summary.relative_to(REPO_ROOT)}")
    print(f"Wrote {out_flags.relative_to(REPO_ROOT)}")
    print(f"Wrote {out_top_sku.relative_to(REPO_ROOT)}")
    print(f"Wrote {out_top_category.relative_to(REPO_ROOT)}")
    print(f"Wrote {out_report.relative_to(REPO_ROOT)}")
    print(summary["risk_level"].value_counts().to_string())


if __name__ == "__main__":
    main()

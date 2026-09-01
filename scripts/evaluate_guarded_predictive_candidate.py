"""Evaluate forecast shape and canonical metrics for the guarded candidate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports/guarded_predictive_allocation_20260827/rows.parquet"
OUTPUT = ROOT / "reports/guarded_predictive_allocation_20260827"
GROUP = ["date", "bakery_id"]
VARIANTS = {
    "current": "incumbent_sku_forecast",
    "p50_predictive": "p50_predictive",
    "original_floor": "p50_simple_floor",
    "p50_filled": "p50_predictive_filled",
    "filled_unrestricted_floor": "filled_raw_floor",
    "filled_floor_volume_guard": "filled_volume_guard",
    "guarded_candidate": "guarded_predictive_floor",
}


def canonical(actual: pd.Series, forecast: pd.Series) -> dict[str, float]:
    error = forecast - actual
    denominator = float(actual.sum())
    smape_denominator = actual.abs() + forecast.abs()
    ss_res = float(np.square(error).sum())
    ss_tot = float(np.square(actual - actual.mean()).sum())
    return {
        "wape_pct": 100 * float(error.abs().sum()) / denominator,
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "bias_pct": 100 * float(error.sum()) / denominator,
        "smape_pct": 100 * float((2 * error.abs() / smape_denominator.replace(0, np.nan)).mean()),
        "r2": 1 - ss_res / ss_tot,
    }


def main() -> None:
    rows = pd.read_parquet(INPUT)
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    demand_total = rows.groupby(GROUP)["demand"].transform("sum")
    demand_share = rows["demand"] / demand_total.replace(0.0, np.nan)
    summaries = []
    day_parts = []
    for variant, column in VARIANTS.items():
        total = rows.groupby(GROUP)[column].transform("sum")
        share = rows[column] / total.replace(0.0, np.nan)
        day = rows[GROUP].copy()
        day["variant"] = variant
        day["share"] = share.fillna(0.0)
        day["share_error"] = (share - demand_share).abs().fillna(0.0)
        day["product_id"] = rows["product_id"]
        grouped = day.groupby(GROUP, sort=False)
        aggregate = grouped.agg(
            max_share=("share", "max"),
            total_variation=("share_error", lambda values: values.sum() / 2),
        ).reset_index()
        top_idx = grouped["share"].idxmax()
        aggregate["top_product_id"] = rows.loc[top_idx, "product_id"].to_numpy()
        aggregate["variant"] = variant
        day_parts.append(aggregate)

        metrics = canonical(rows["demand"], rows[column])
        summaries.append(
            {
                "variant": variant,
                "volume": float(rows[column].sum()),
                "zero_with_demand": int((rows[column].le(1e-9) & rows["demand"].gt(0)).sum()),
                "demand_on_zero": float(rows.loc[rows[column].le(1e-9) & rows["demand"].gt(0), "demand"].sum()),
                **metrics,
            }
        )
    bakery_day = pd.concat(day_parts, ignore_index=True)
    shape = bakery_day.groupby("variant", as_index=False).agg(
        bakery_days=("bakery_id", "size"),
        median_top_share=("max_share", "median"),
        p90_top_share=("max_share", lambda values: values.quantile(0.90)),
        p99_top_share=("max_share", lambda values: values.quantile(0.99)),
        mean_total_variation=("total_variation", "mean"),
        top_1071_days=("top_product_id", lambda values: values.eq(1071).sum()),
        top_share_ge_20pct=("max_share", lambda values: values.ge(0.20).sum()),
        top_share_ge_30pct=("max_share", lambda values: values.ge(0.30).sum()),
    )
    summary = pd.DataFrame(summaries).merge(shape, on="variant", validate="one_to_one")

    case_rows = rows[(rows["bakery_id"].eq(29)) & (rows["date"].eq(pd.Timestamp("2026-08-23")))].copy()
    case_total = {variant: float(case_rows[column].sum()) for variant, column in VARIANTS.items()}
    case = case_rows[case_rows["product_id"].eq(1071)][["product_id", "demand"]].copy()
    case["demand_share"] = case["demand"] / float(case_rows["demand"].sum())
    for variant, column in VARIANTS.items():
        case[variant] = case_rows.loc[case_rows["product_id"].eq(1071), column].to_numpy()
        case[f"{variant}_share"] = case[variant] / case_total[variant]

    summary.to_csv(OUTPUT / "candidate_shape_and_metrics.csv", index=False, encoding="utf-8-sig")
    bakery_day.to_csv(OUTPUT / "candidate_bakery_day_shape.csv", index=False, encoding="utf-8-sig")
    case.to_csv(OUTPUT / "bakery29_20260823_candidate.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print("\nBakery 29 / SKU 1071")
    print(case.to_string(index=False))


if __name__ == "__main__":
    main()

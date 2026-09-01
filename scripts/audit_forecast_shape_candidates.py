"""Audit forecast shape and recurrence of known SKU-allocation failure modes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "reports/rolling_actual_state_comparison_20260826/rows.parquet"
PRODUCTS = ROOT / "reports/markup_price_mapping_20260826/mapped_products.csv"
OUTPUT = ROOT / "reports/forecast_shape_audit_20260827"

VARIANTS = {
    "reconstructed_demand": "demand",
    "current": "incumbent_sku_forecast",
    "predictive_same_volume": "predictive_forecast",
    "p50_predictive": "p50_predictive",
    "p50_predictive_simple_floor": "p50_simple_floor",
}
KNOWN_DOMINANT_PRODUCT_ID = 1071


def main() -> None:
    rows = pd.read_parquet(ROWS)
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    products = pd.read_csv(PRODUCTS, encoding="utf-8-sig")
    products = products.dropna(subset=["product_id"]).copy()
    products["product_id"] = products["product_id"].astype(int)
    products = products.drop_duplicates("product_id", keep="last")
    name_map = products.set_index("product_id")["product_name"].to_dict()
    rows["product_name"] = rows["product_id"].map(name_map).fillna(rows["product_id"].map(lambda value: f"SKU {value}"))

    group_keys = ["date", "bakery_id"]
    target_total = rows.groupby(group_keys)["demand"].transform("sum")
    rows["demand_share"] = np.where(target_total > 0, rows["demand"] / target_total, 0.0)
    day_parts = []
    sku_parts = []
    for variant, column in VARIANTS.items():
        forecast_total = rows.groupby(group_keys)[column].transform("sum")
        share = np.where(forecast_total > 0, rows[column] / forecast_total, 0.0)
        frame_columns = group_keys + ["product_id", "product_name", "demand", "demand_share"]
        if column != "demand":
            frame_columns.append(column)
        frame = rows[frame_columns].copy()
        frame["variant"] = variant
        frame["forecast"] = frame[column]
        if column != "demand":
            frame = frame.drop(columns=column)
        frame["forecast_total"] = forecast_total
        frame["forecast_share"] = share
        frame["abs_share_error"] = (frame["forecast_share"] - frame["demand_share"]).abs()
        frame["zero_with_demand"] = frame["forecast"].le(1e-9) & frame["demand"].gt(0)
        sku_parts.append(frame)

        grouped = frame.groupby(group_keys, sort=False)
        day = grouped.agg(
            forecast_total=("forecast", "sum"),
            demand_total=("demand", "sum"),
            max_forecast_share=("forecast_share", "max"),
            max_demand_share=("demand_share", "max"),
            share_l1=("abs_share_error", "sum"),
            zero_with_demand=("zero_with_demand", "sum"),
            demand_on_zero_forecast=("demand", lambda values: values[frame.loc[values.index, "zero_with_demand"]].sum()),
            sku_rows=("product_id", "size"),
        ).reset_index()
        top_idx = grouped["forecast_share"].idxmax()
        top = frame.loc[top_idx, group_keys + ["product_id", "product_name", "forecast_share"]].rename(
            columns={
                "product_id": "top_product_id",
                "product_name": "top_product_name",
                "forecast_share": "top_product_share",
            }
        )
        day = day.merge(top, on=group_keys, how="left", validate="one_to_one")
        day["variant"] = variant
        day["total_variation_share"] = day["share_l1"] / 2
        day_parts.append(day)

    sku = pd.concat(sku_parts, ignore_index=True)
    bakery_day = pd.concat(day_parts, ignore_index=True)
    summary = bakery_day.groupby("variant", as_index=False).agg(
        bakery_days=("bakery_id", "size"),
        bakeries=("bakery_id", "nunique"),
        forecast_volume=("forecast_total", "sum"),
        demand_volume=("demand_total", "sum"),
        median_top_share=("max_forecast_share", "median"),
        p90_top_share=("max_forecast_share", lambda values: values.quantile(0.90)),
        p99_top_share=("max_forecast_share", lambda values: values.quantile(0.99)),
        mean_total_variation=("total_variation_share", "mean"),
        median_total_variation=("total_variation_share", "median"),
        zero_with_demand=("zero_with_demand", "sum"),
        demand_on_zero_forecast=("demand_on_zero_forecast", "sum"),
        bakery_days_with_zero_demand_sku=("zero_with_demand", lambda values: values.gt(0).sum()),
        top_1071_days=("top_product_id", lambda values: values.eq(KNOWN_DOMINANT_PRODUCT_ID).sum()),
    )
    for threshold in [0.20, 0.30, 0.40, 0.50]:
        counts = bakery_day.groupby("variant")["max_forecast_share"].apply(lambda values: values.ge(threshold).sum())
        summary[f"top_share_ge_{int(threshold * 100)}pct"] = summary["variant"].map(counts)
    summary["volume_bias_pct"] = 100 * (summary["forecast_volume"] - summary["demand_volume"]) / summary["demand_volume"]

    date = pd.Timestamp("2026-08-23")
    case = sku[(sku["date"].eq(date)) & (sku["bakery_id"].eq(29))].copy()
    case_summary = case.groupby("variant", as_index=False).agg(
        forecast_total=("forecast", "sum"),
        demand_total=("demand", "sum"),
        max_forecast_share=("forecast_share", "max"),
        total_variation_share=("abs_share_error", lambda values: values.sum() / 2),
    )
    product_case = case[case["product_id"].eq(KNOWN_DOMINANT_PRODUCT_ID)][
        ["variant", "product_id", "product_name", "forecast", "forecast_share", "demand", "demand_share"]
    ]
    case_summary = case_summary.merge(product_case, on="variant", how="left", validate="one_to_one")
    case_detail = case.pivot_table(
        index=["product_id", "product_name"],
        columns="variant",
        values=["forecast", "forecast_share"],
        aggfunc="first",
    )
    case_detail.columns = [f"{metric}_{variant}" for metric, variant in case_detail.columns]
    case_detail = case_detail.reset_index().sort_values("forecast_current", ascending=False)

    zero_rows = rows[(rows["predictive_forecast"].le(1e-9)) & (rows["demand"].gt(0))].copy()
    zero_rows["product_name"] = zero_rows["product_id"].map(name_map).fillna(
        zero_rows["product_id"].map(lambda value: f"SKU {value}")
    )
    zero_rows = zero_rows[
        [
            "date", "bakery_id", "product_id", "product_name", "category", "actual_sold", "demand",
            "incumbent_sku_forecast", "predictive_raw", "predictive_forecast", "p50_simple_floor", "history_n",
        ]
    ].sort_values("demand", ascending=False)

    volume = bakery_day.pivot_table(
        index=group_keys, columns="variant", values="forecast_total", aggfunc="first"
    ).reset_index()
    volume["predictive_vs_current_pct"] = 100 * (
        volume["predictive_same_volume"] / volume["current"].replace(0.0, np.nan) - 1
    )
    volume["p50_vs_current_pct"] = 100 * (
        volume["p50_predictive"] / volume["current"].replace(0.0, np.nan) - 1
    )
    volume["floor_vs_p50_pct"] = 100 * (
        volume["p50_predictive_simple_floor"] / volume["p50_predictive"].replace(0.0, np.nan) - 1
    )
    volume_summary = volume[["predictive_vs_current_pct", "p50_vs_current_pct", "floor_vs_p50_pct"]].describe(
        percentiles=[0.1, 0.5, 0.9, 0.99]
    ).T.reset_index().rename(columns={"index": "metric"})

    worst = bakery_day.sort_values(["variant", "max_forecast_share"], ascending=[True, False]).groupby(
        "variant", as_index=False
    ).head(30)
    top_products = bakery_day.groupby(["variant", "top_product_id", "top_product_name"], as_index=False).agg(
        top_days=("bakery_id", "size"),
        mean_top_share=("top_product_share", "mean"),
        max_top_share=("top_product_share", "max"),
    ).sort_values(["variant", "top_days"], ascending=[True, False])

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT / "summary.csv", index=False, encoding="utf-8-sig")
    bakery_day.to_csv(OUTPUT / "bakery_day_diagnostics.csv", index=False, encoding="utf-8-sig")
    case_summary.to_csv(OUTPUT / "bakery29_20260823.csv", index=False, encoding="utf-8-sig")
    case_detail.to_csv(OUTPUT / "bakery29_20260823_sku_detail.csv", index=False, encoding="utf-8-sig")
    zero_rows.to_csv(OUTPUT / "predictive_zero_with_demand.csv", index=False, encoding="utf-8-sig")
    volume_summary.to_csv(OUTPUT / "bakery_volume_change_summary.csv", index=False, encoding="utf-8-sig")
    worst.to_csv(OUTPUT / "worst_concentration_cases.csv", index=False, encoding="utf-8-sig")
    top_products.to_csv(OUTPUT / "top_products.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print("\nBakery 29, 2026-08-23")
    print(case_summary.to_string(index=False))


if __name__ == "__main__":
    main()

"""Daily history audit for bakery 29 and product 1071."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HOURLY = ROOT / ".codex_tmp/rolling_hourly_sales_20260601_20260823.parquet"
LABELS = ROOT / "reports/relaxed_stockout_network_20260826/sku_day_demand.csv"
COMPONENTS = ROOT / ".codex_tmp/rolling_actual_components_20260721_20260823.parquet"
FORECASTS = ROOT / "reports/rolling_actual_state_comparison_20260826/rows.parquet"
OUTPUT = ROOT / "reports/bakery29_sku1071_history_20260827"
BAKERY_ID = 29
PRODUCT_ID = 1071


def main() -> None:
    hourly = pd.read_parquet(HOURLY)
    hourly["date"] = pd.to_datetime(hourly["date"]).dt.normalize()
    forecasts = pd.read_parquet(FORECASTS)
    forecasts["date"] = pd.to_datetime(forecasts["date"]).dt.normalize()
    category_map = forecasts[["product_id", "category"]].drop_duplicates("product_id")
    product_category = forecasts.loc[forecasts["product_id"].eq(PRODUCT_ID), "category"].iloc[0]

    bakery_raw = hourly[hourly["bakery_id"].eq(BAKERY_ID)].copy()
    bakery_total = bakery_raw.groupby("date", as_index=False)["sold"].sum().rename(
        columns={"sold": "bakery_sales"}
    )
    bakery_sales = bakery_raw.merge(
        category_map, on="product_id", how="left"
    )
    daily_products = bakery_sales.groupby(["date", "product_id", "category"], as_index=False)["sold"].sum()
    category_total = daily_products[daily_products["category"].eq(product_category)].groupby(
        "date", as_index=False
    )["sold"].sum().rename(columns={"sold": "category_sales"})
    product = daily_products[daily_products["product_id"].eq(PRODUCT_ID)][["date", "sold"]].rename(
        columns={"sold": "sku_sales"}
    )
    daily = bakery_total.merge(category_total, on="date", how="left").merge(product, on="date", how="left")
    daily[["category_sales", "sku_sales"]] = daily[["category_sales", "sku_sales"]].fillna(0.0)
    daily["sku_bakery_share"] = daily["sku_sales"] / daily["bakery_sales"].replace(0.0, pd.NA)
    daily["sku_category_share"] = daily["sku_sales"] / daily["category_sales"].replace(0.0, pd.NA)
    daily["dow"] = daily["date"].dt.dayofweek

    labels = pd.read_csv(LABELS)
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    labels = labels[(labels["bakery_id"].eq(BAKERY_ID)) & (labels["product_id"].eq(PRODUCT_ID))]
    daily = daily.merge(
        labels[
            [
                "date", "is_clear_stockout", "last_sale_hour", "raw_imputed_demand",
                "imputed_demand", "demand_point_estimate",
            ]
        ],
        on="date",
        how="left",
        validate="one_to_one",
    )

    components = pd.read_parquet(COMPONENTS)
    components["date"] = pd.to_datetime(components["date"]).dt.normalize()
    components = components[
        (components["bakery_id"].eq(BAKERY_ID)) & (components["product_id"].eq(PRODUCT_ID))
    ].copy()
    for column in ["produced", "received", "sent"]:
        components[column] = pd.to_numeric(components[column], errors="coerce").fillna(0.0)
    components["calculated_closing"] = (
        components["produced"] + components["received"] - components["sent"] - components["sold"]
    ).clip(lower=0.0)
    components["calculated_opening"] = components["calculated_closing"].shift(1).fillna(0.0)
    components["available_to_sell"] = (
        components["produced"] + components["received"] - components["sent"] + components["calculated_opening"]
    ).clip(lower=0.0)
    daily = daily.merge(
        components[["date", "produced", "received", "sent", "calculated_opening", "available_to_sell"]],
        on="date",
        how="left",
        validate="one_to_one",
    )

    selected = forecasts[
        (forecasts["bakery_id"].eq(BAKERY_ID)) & (forecasts["product_id"].eq(PRODUCT_ID))
    ][
        [
            "date", "incumbent_sku_forecast", "predictive_forecast", "p50_predictive",
            "p50_simple_floor", "same_weekday_forecast", "causal_trend_forecast", "history_n", "history_p67",
        ]
    ]
    category_forecast = forecasts[
        (forecasts["bakery_id"].eq(BAKERY_ID)) & (forecasts["category"].eq(product_category))
    ].groupby("date", as_index=False).agg(
        current_category_forecast=("incumbent_sku_forecast", "sum"),
        predictive_category_forecast=("predictive_forecast", "sum"),
        p50_category_forecast=("p50_predictive", "sum"),
    )
    daily = daily.merge(selected, on="date", how="left", validate="one_to_one").merge(
        category_forecast, on="date", how="left", validate="one_to_one"
    )

    sunday = daily[daily["dow"].eq(6)].copy()
    weekly = daily.assign(week=daily["date"].dt.to_period("W").astype(str)).groupby("week", as_index=False).agg(
        sku_sales_mean=("sku_sales", "mean"),
        sku_sales_min=("sku_sales", "min"),
        sku_sales_max=("sku_sales", "max"),
        sku_bakery_share_mean=("sku_bakery_share", "mean"),
        stockout_days=("is_clear_stockout", "sum"),
    )
    stockout_summary = daily.groupby("is_clear_stockout", as_index=False).agg(
        days=("date", "size"),
        sales_mean=("sku_sales", "mean"),
        sales_median=("sku_sales", "median"),
        last_sale_hour_mean=("last_sale_hour", "mean"),
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUTPUT / "daily_history.csv", index=False, encoding="utf-8-sig")
    sunday.to_csv(OUTPUT / "sunday_history.csv", index=False, encoding="utf-8-sig")
    weekly.to_csv(OUTPUT / "weekly_summary.csv", index=False, encoding="utf-8-sig")
    stockout_summary.to_csv(OUTPUT / "stockout_summary.csv", index=False, encoding="utf-8-sig")
    print("Sunday history")
    print(
        sunday[
            ["date", "sku_sales", "category_sales", "bakery_sales", "sku_category_share", "sku_bakery_share", "is_clear_stockout"]
        ].to_string(index=False)
    )
    print("\nStockout split")
    print(stockout_summary.to_string(index=False))


if __name__ == "__main__":
    main()

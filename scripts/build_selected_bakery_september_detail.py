"""Export September day/SKU detail for selected pilot bakeries."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.backtest_retrained_p50_comparable import (  # noqa: E402
    add_end_day_lost,
    load_detail,
)

ECONOMICS = (
    ROOT
    / "reports/retrained_p50_all55_sku_safety_20260910"
    / "economics_september_cap_13"
    / "by_bakery_sku_date.csv"
)
FORECAST = (
    ROOT / "reports/retrained_p50_all55_sku_safety_20260910" / "rows_cap_13.parquet"
)
SALE_TIMES = ROOT / ".codex_tmp/sep01_09_sale_times.csv"
OUTPUT = (
    ROOT
    / "reports/retrained_p50_all55_sku_safety_20260910"
    / "selected_bakeries_september_detail"
)
BAKERIES = {99, 246}
KEYS = ["date", "bakery_id", "product_id"]


def main() -> None:
    economics = pd.read_csv(ECONOMICS)
    economics["date"] = pd.to_datetime(economics["date"]).dt.normalize()
    economics = economics[economics["bakery_id"].isin(BAKERIES)].copy()
    forecast = pd.read_parquet(FORECAST)
    forecast["date"] = pd.to_datetime(forecast["date"]).dt.normalize()
    forecast = forecast[forecast["bakery_id"].isin(BAKERIES)]
    actual = add_end_day_lost(load_detail())
    actual = actual[
        actual["bakery_id"].isin(BAKERIES) & actual["date"].dt.month.eq(9)
    ].drop_duplicates(KEYS)
    actual["reconstructed_demand_raw"] = actual["sold_qty"] + actual["lost"]
    sale_times = pd.read_csv(SALE_TIMES)
    sale_times["date"] = pd.to_datetime(sale_times["date"]).dt.normalize()
    sale_times = sale_times[sale_times["bakery_id"].isin(BAKERIES)].drop_duplicates(
        KEYS
    )
    for column in ["first_sale_time", "last_sale_time"]:
        sale_times[column] = pd.to_datetime(
            sale_times[column], errors="coerce"
        ).dt.strftime("%H:%M:%S")
    detail = economics.merge(
        forecast[KEYS + ["direct_forecast", "alpha25_tail_capped"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    detail = detail.merge(
        actual[KEYS + ["sold_qty", "lost", "reconstructed_demand_raw"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    detail = detail.merge(
        sale_times[KEYS + ["first_sale_time", "last_sale_time"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    detail = detail.rename(
        columns={
            "sold_qty": "actual_sales",
            "production_actual_state": "actual_production",
            "alpha25_tail_capped": "model_demand_forecast",
            "production_p50_loss_075": "model_production",
            "lost_actual_state": "actual_unserved_in_simulation",
            "lost": "reconstructed_lost",
            "reconstructed_demand_raw": "reconstructed_demand",
            "gross_profit_p50_loss_075_delta": "gross_profit_delta",
            "expired_strategy_stock_p50_loss_075": "model_expired",
        }
    )
    columns = KEYS + [
        "bakery_name",
        "product_name",
        "first_sale_time",
        "last_sale_time",
        "actual_sales",
        "actual_production",
        "direct_forecast",
        "model_demand_forecast",
        "model_production",
        "reconstructed_demand",
        "reconstructed_lost",
        "actual_unserved_in_simulation",
        "model_expired",
        "gross_profit_delta",
    ]
    detail = detail[columns].sort_values(KEYS)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    detail.to_csv(OUTPUT / "day_sku_detail.csv", index=False, encoding="utf-8-sig")
    for bakery_id, part in detail.groupby("bakery_id"):
        part.to_csv(
            OUTPUT / f"bakery_{bakery_id}_day_sku_detail.csv",
            index=False,
            encoding="utf-8-sig",
        )
    by_day = (
        detail.groupby(["bakery_id", "bakery_name", "date"], as_index=False)
        .agg(
            actual_sales=("actual_sales", "sum"),
            first_sale_time=(
                "first_sale_time",
                lambda values: values.dropna().min() if values.notna().any() else pd.NA,
            ),
            last_sale_time=(
                "last_sale_time",
                lambda values: values.dropna().max() if values.notna().any() else pd.NA,
            ),
            actual_production=("actual_production", "sum"),
            direct_forecast=("direct_forecast", "sum"),
            model_demand_forecast=("model_demand_forecast", "sum"),
            model_production=("model_production", "sum"),
            reconstructed_demand=("reconstructed_demand", "sum"),
            reconstructed_lost=("reconstructed_lost", "sum"),
            model_expired=("model_expired", "sum"),
            gross_profit_delta=("gross_profit_delta", "sum"),
        )
        .sort_values(["bakery_id", "date"])
    )
    by_day.to_csv(OUTPUT / "by_day.csv", index=False, encoding="utf-8-sig")
    by_sku = (
        detail.groupby(
            ["bakery_id", "bakery_name", "product_id", "product_name"],
            as_index=False,
        )
        .agg(
            days=("date", "nunique"),
            first_sale_time=(
                "first_sale_time",
                lambda values: values.dropna().min() if values.notna().any() else pd.NA,
            ),
            last_sale_time=(
                "last_sale_time",
                lambda values: values.dropna().max() if values.notna().any() else pd.NA,
            ),
            actual_sales=("actual_sales", "sum"),
            actual_production=("actual_production", "sum"),
            direct_forecast=("direct_forecast", "sum"),
            model_demand_forecast=("model_demand_forecast", "sum"),
            model_production=("model_production", "sum"),
            reconstructed_demand=("reconstructed_demand", "sum"),
            reconstructed_lost=("reconstructed_lost", "sum"),
            model_expired=("model_expired", "sum"),
            gross_profit_delta=("gross_profit_delta", "sum"),
        )
        .sort_values(["bakery_id", "gross_profit_delta"], ascending=[True, False])
    )
    by_sku.to_csv(OUTPUT / "by_sku.csv", index=False, encoding="utf-8-sig")
    print(f"detail_rows={len(detail)} day_rows={len(by_day)} sku_rows={len(by_sku)}")


if __name__ == "__main__":
    main()

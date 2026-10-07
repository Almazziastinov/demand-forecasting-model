from pathlib import Path

import pandas as pd


REPORT_DIR = Path("reports/direct_reconstructed_economics_sep01_09_20260910")
BAKERY_ID = "22"


def pivot_variants(df: pd.DataFrame, dimensions: list[str]) -> pd.DataFrame:
    metrics = [
        "demand",
        "production",
        "served",
        "lost",
        "expired_strategy_stock",
        "revenue",
        "production_cost",
        "gross_profit",
    ]
    result = (
        df.pivot_table(index=dimensions, columns="variant", values=metrics, aggfunc="sum", fill_value=0)
        .swaplevel(axis=1)
        .sort_index(axis=1)
    )
    result.columns = [f"{metric}_{variant}" for variant, metric in result.columns]
    result = result.reset_index()

    for metric in ["production", "served", "lost", "expired_strategy_stock", "revenue", "production_cost", "gross_profit"]:
        result[f"{metric}_delta"] = (
            result[f"{metric}_direct_alpha_025"] - result[f"{metric}_actual_state"]
        )
    return result


def main() -> None:
    rows = pd.read_parquet(REPORT_DIR / "daily_rows.parquet")
    rows["bakery_id"] = rows["bakery_id"].astype(str)
    rows["product_id"] = rows["product_id"].astype(str)
    target = rows.loc[rows["bakery_id"] == BAKERY_ID].copy()

    sku = pd.read_csv(REPORT_DIR / "by_sku.csv", encoding="utf-8-sig", dtype={"product_id": str})[
        ["product_id", "product_name", "fact_category_name"]
    ]

    by_day = pivot_variants(target, ["date"])
    by_day_sku = pivot_variants(target, ["date", "product_id"]).merge(sku, on="product_id", how="left")
    by_sku = pivot_variants(target, ["product_id"]).merge(sku, on="product_id", how="left")

    ordered = ["date", "product_id", "product_name", "fact_category_name"] + [
        column for column in by_day_sku.columns if column not in {"date", "product_id", "product_name", "fact_category_name"}
    ]
    by_day_sku = by_day_sku[ordered]

    by_day.to_csv(REPORT_DIR / "bakery_22_by_day.csv", index=False, encoding="utf-8-sig")
    by_day_sku.to_csv(REPORT_DIR / "bakery_22_by_day_sku.csv", index=False, encoding="utf-8-sig")
    by_sku.to_csv(REPORT_DIR / "bakery_22_by_sku.csv", index=False, encoding="utf-8-sig")

    print("DAY")
    print(by_day[["date", "demand_actual_state", "production_actual_state", "production_direct_alpha_025", "served_actual_state", "served_direct_alpha_025", "lost_actual_state", "lost_direct_alpha_025", "gross_profit_delta"]].to_csv(index=False))
    print("TOP_WIN_ROWS")
    print(by_day_sku.nlargest(20, "gross_profit_delta")[["date", "product_id", "product_name", "demand_actual_state", "production_actual_state", "production_direct_alpha_025", "served_actual_state", "served_direct_alpha_025", "lost_actual_state", "lost_direct_alpha_025", "gross_profit_delta"]].to_csv(index=False))
    print("TOP_LOSS_ROWS")
    print(by_day_sku.nsmallest(10, "gross_profit_delta")[["date", "product_id", "product_name", "demand_actual_state", "production_actual_state", "production_direct_alpha_025", "served_actual_state", "served_direct_alpha_025", "lost_actual_state", "lost_direct_alpha_025", "gross_profit_delta"]].to_csv(index=False))
    print("SKU_TOTAL")
    print(by_sku.nlargest(15, "gross_profit_delta")[["product_id", "product_name", "demand_actual_state", "production_actual_state", "production_direct_alpha_025", "served_delta", "lost_delta", "gross_profit_delta"]].to_csv(index=False))


if __name__ == "__main__":
    main()

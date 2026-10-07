from pathlib import Path
import os
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_clean_kazan_two_day_economics import (  # noqa: E402
    add_global_segments,
    aggregate,
)
from simulate_two_day_economics import simulate_group  # noqa: E402


KEYS = ["date", "bakery_id", "product_id"]
PERIOD = os.environ.get("ECON_PERIOD", "september")
IS_AUGUST = PERIOD == "august"
DETAIL = ROOT / (
    ".codex_tmp/pilot_recalc_report/detail.csv"
    if IS_AUGUST
    else ".codex_tmp/pilot_management_sep01_09/detail.csv"
)
SALE_TIMES = ROOT / (
    ".codex_tmp/august_sale_times.csv"
    if IS_AUGUST
    else ".codex_tmp/sep01_09_sale_times.csv"
)
MAPPING = ROOT / "reports/markup_price_mapping_20260826/mapped_products.csv"
DEFAULT_P50_ROWS = (
    ROOT
    / "reports"
    / (
        "versioned_direct_scope_p50_loss_0.75_august_20260909"
        if IS_AUGUST
        else "versioned_direct_scope_p50_loss_0.75_september_through_09_20260910"
    )
    / "rows.parquet"
)
P50_ROWS = Path(os.environ.get("P50_ROWS_OVERRIDE", DEFAULT_P50_ROWS))
DEFAULT_OUTPUT = (
    ROOT
    / "reports"
    / (
        "prod_direct_end_of_day_economics_august_folds_20260910"
        if IS_AUGUST
        else "prod_direct_end_of_day_economics_sep01_09_20260910"
    )
)
OUTPUT = Path(os.environ.get("ECON_OUTPUT_OVERRIDE", DEFAULT_OUTPUT))
DISCOUNT = 0.30
ACTUAL_BASELINE = os.environ.get("ECON_ACTUAL_BASELINE", "simulated")


def add_deltas(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
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
    wide = frame.pivot_table(
        index=keys, columns="variant", values=metrics, aggfunc="sum", fill_value=0
    )
    wide.columns = [f"{metric}_{variant}" for metric, variant in wide.columns]
    wide = wide.reset_index()
    for variant in ["prod_direct_forecast", "p50_loss_075"]:
        for metric in metrics[1:]:
            wide[f"{metric}_{variant}_delta"] = (
                wide[f"{metric}_{variant}"] - wide[f"{metric}_actual_state"]
            )
    return wide


def main() -> None:
    detail = pd.read_csv(DETAIL, encoding="utf-8-sig", low_memory=False)
    detail["date"] = pd.to_datetime(detail["business_date"]).dt.normalize()
    if IS_AUGUST:
        detail = detail[detail["date"].dt.month.eq(8)].copy()
    detail = detail.drop_duplicates(KEYS).copy()
    sale_times = pd.read_csv(SALE_TIMES)
    sale_times["date"] = pd.to_datetime(sale_times["date"]).dt.normalize()
    detail = detail.merge(sale_times, on=KEYS, how="left", validate="one_to_one")
    numeric = [
        "forecast_qty",
        "produced_qty",
        "sold_qty",
        "lost_demand_raw_qty",
        "available_to_sell_qty",
        "received_qty",
        "sent_qty",
        "revenue",
        "price",
    ]
    for column in numeric:
        detail[column] = pd.to_numeric(detail[column], errors="coerce")
    detail = detail.dropna(subset=["forecast_qty", "produced_qty", "sold_qty"])
    last_sale = pd.to_datetime(detail["last_sale_time"], errors="coerce", utc=True)
    bakery_end = pd.to_datetime(
        detail["bakery_last_sale_time"], errors="coerce", utc=True
    )
    opening = pd.to_datetime(detail["date"]).dt.tz_localize(
        "Europe/Moscow"
    ).dt.tz_convert("UTC") + pd.Timedelta(hours=7.5)
    elapsed_hours = (last_sale - opening).dt.total_seconds() / 3600
    remaining_hours = (bakery_end - last_sale).dt.total_seconds().clip(lower=0) / 3600
    fully_realized = detail["eligible_lost_demand"].fillna(False).astype(bool)
    eligible_end_day = (
        fully_realized
        & elapsed_hours.ge(2.0)
        & remaining_hours.gt(0.0)
        & detail["sold_qty"].gt(0.0)
    )
    raw_end_day = detail["sold_qty"] / elapsed_hours * remaining_hours
    cap = pd.concat(
        [detail["sold_qty"] * 1.5, pd.Series(15.0, index=detail.index)], axis=1
    ).max(axis=1)
    detail["lost_demand_end_day_qty"] = (
        raw_end_day.where(eligible_end_day, 0.0).clip(lower=0.0).clip(upper=cap)
    )
    detail["received_qty"] = detail["received_qty"].fillna(0.0)
    detail["sent_qty"] = detail["sent_qty"].fillna(0.0)
    detail["opening_stock"] = (
        detail["available_to_sell_qty"].fillna(detail["produced_qty"])
        - detail["produced_qty"]
        - detail["received_qty"]
        + detail["sent_qty"]
    ).clip(lower=0.0)
    detail["produced"] = detail["produced_qty"]
    detail["received"] = detail["received_qty"]
    detail["sent"] = detail["sent_qty"]
    detail["demand"] = detail["sold_qty"] + detail["lost_demand_end_day_qty"]
    detail["prod_forecast"] = detail["forecast_qty"]
    price_history_parts = []
    for path in [
        ROOT / ".codex_tmp/pilot_recalc_report/detail.csv",
        ROOT / ".codex_tmp/pilot_management_sep01_09/detail.csv",
    ]:
        history_part = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        history_part["date"] = pd.to_datetime(
            history_part["business_date"]
        ).dt.normalize()
        history_part["price"] = pd.to_numeric(history_part["price"], errors="coerce")
        price_history_parts.append(history_part[["date", "product_id", "price"]])
    price_history = pd.concat(price_history_parts, ignore_index=True).drop_duplicates()
    product_day_price = (
        price_history.dropna(subset=["price"])
        .groupby(["date", "product_id"], as_index=False)["price"]
        .median()
        .sort_values(["product_id", "date"])
    )
    product_day_price["prior_product_price"] = product_day_price.groupby(
        "product_id", sort=False
    )["price"].transform(
        lambda values: values.shift(1).rolling(28, min_periods=1).median()
    )
    detail = detail.merge(
        product_day_price[["date", "product_id", "prior_product_price"]],
        on=["date", "product_id"],
        how="left",
        validate="many_to_one",
    )

    p50 = pd.read_parquet(P50_ROWS)
    p50["date"] = pd.to_datetime(p50["date"]).dt.normalize()
    p50 = p50[p50["scenario"].eq("calibrated")]
    detail = detail.merge(
        p50[KEYS + ["alpha25_tail_capped"]],
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    detail["p50_loss_075"] = detail["alpha25_tail_capped"]

    source = add_global_segments(detail)
    source = source[source["segment_length"].ge(2)].copy()
    simulations = []
    for variant, plan_column in {
        "actual_state": None,
        "prod_direct_forecast": "prod_forecast",
        "p50_loss_075": "p50_loss_075",
    }.items():
        if variant == "actual_state" and ACTUAL_BASELINE == "checkout":
            frame = source[KEYS + ["demand", "produced", "sold_qty"]].copy()
            frame["production"] = frame["produced"].clip(lower=0.0)
            frame["target_stock"] = frame["production"]
            frame["sold_fresh"] = frame["sold_qty"].clip(lower=0.0)
            frame["sold_yesterday"] = 0.0
            frame["sold_yesterday_initial_stock"] = 0.0
            frame["sold_yesterday_strategy_stock"] = 0.0
            frame["served"] = frame["sold_fresh"]
            frame["lost"] = (frame["demand"] - frame["served"]).clip(lower=0.0)
            frame["expired"] = 0.0
            frame["expired_initial_stock"] = 0.0
            frame["expired_strategy_stock"] = 0.0
            frame["ending_carry"] = 0.0
            frame["segment_id"] = 0
            frame["segment_day"] = 0
            frame = frame.drop(columns=["produced", "sold_qty"])
        else:
            parts = [
                simulate_group(group, plan_column)
                for _, group in source.groupby(["bakery_id", "product_id"], sort=False)
            ]
            frame = pd.concat(parts, ignore_index=True)
        frame["variant"] = variant
        simulations.append(frame)
    rows = pd.concat(simulations, ignore_index=True)
    rows = rows.merge(
        source[
            KEYS
            + [
                "global_segment",
                "segment_end",
                "forecast_run_id",
                "sold_qty",
                "lost_demand_end_day_qty",
                "revenue",
                "price",
                "prior_product_price",
            ]
        ],
        on=KEYS,
        how="left",
        validate="many_to_one",
    )
    rows["terminal_carry"] = rows["ending_carry"].where(
        rows["date"].eq(rows["segment_end"]), 0.0
    )

    mapping = pd.read_csv(MAPPING, encoding="utf-8-sig")
    mapping = mapping[mapping["valid_economics"].fillna(False).astype(bool)].copy()
    mapping["product_id"] = mapping["product_id"].astype(int)
    mapping = mapping.sort_values("unit_price").drop_duplicates(
        "product_id", keep="last"
    )
    rows = rows.merge(
        mapping[
            [
                "product_id",
                "workbook_product_name",
                "workbook_category",
                "unit_price",
                "unit_cost",
            ]
        ],
        on="product_id",
        how="inner",
        validate="many_to_one",
    )
    rows = rows.rename(
        columns={"unit_price": "reference_unit_price", "revenue": "checkout_revenue"}
    )
    rows["unit_price"] = (
        rows["price"]
        .where(rows["price"].gt(0))
        .fillna(rows["prior_product_price"])
        .fillna(rows["reference_unit_price"])
    )
    rows["revenue"] = rows["sold_fresh"] * rows["unit_price"] + rows[
        "sold_yesterday"
    ] * rows["unit_price"] * (1 - DISCOUNT)
    actual_checkout = rows["variant"].eq("actual_state") & (
        ACTUAL_BASELINE == "checkout"
    )
    rows.loc[actual_checkout, "revenue"] = rows.loc[
        actual_checkout, "checkout_revenue"
    ].fillna(
        rows.loc[actual_checkout, "sold_fresh"]
        * rows.loc[actual_checkout, "unit_price"]
    )
    rows["production_cost"] = rows["production"] * rows["unit_cost"]
    rows["gross_profit"] = rows["revenue"] - rows["production_cost"]
    rows["discount_loss"] = rows["sold_yesterday"] * rows["unit_price"] * DISCOUNT

    summary = aggregate(rows, ["variant"])
    actual_gp = float(
        summary.loc[summary["variant"].eq("actual_state"), "gross_profit"].iloc[0]
    )
    summary["gross_profit_delta_vs_actual"] = summary["gross_profit"] - actual_gp
    summary["gross_profit_delta_vs_actual_pct"] = (
        100 * summary["gross_profit_delta_vs_actual"] / actual_gp
    )
    by_bakery = add_deltas(rows, ["bakery_id"])
    by_sku = add_deltas(rows, ["product_id"])
    by_date = add_deltas(rows, ["date"])
    by_bakery_sku_date = add_deltas(rows, ["date", "bakery_id", "product_id"])

    bakery_names = detail[["bakery_id", "bakery_name"]].drop_duplicates("bakery_id")
    product_names = detail[
        ["product_id", "product_name", "fact_category_name"]
    ].drop_duplicates("product_id")
    by_bakery = by_bakery.merge(bakery_names, on="bakery_id", how="left")
    by_sku = by_sku.merge(product_names, on="product_id", how="left")
    by_bakery_sku_date = by_bakery_sku_date.merge(
        bakery_names, on="bakery_id", how="left"
    ).merge(product_names, on="product_id", how="left")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(OUTPUT / "daily_rows.parquet", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False, encoding="utf-8-sig")
    by_bakery.to_csv(OUTPUT / "by_bakery.csv", index=False, encoding="utf-8-sig")
    by_sku.to_csv(OUTPUT / "by_sku.csv", index=False, encoding="utf-8-sig")
    by_date.to_csv(OUTPUT / "by_date.csv", index=False, encoding="utf-8-sig")
    by_bakery_sku_date.to_csv(
        OUTPUT / "by_bakery_sku_date.csv", index=False, encoding="utf-8-sig"
    )
    print(summary.to_string(index=False))
    print("\nTarget row")
    print(
        by_bakery_sku_date[
            (by_bakery_sku_date["date"].eq(pd.Timestamp("2026-09-01")))
            & (by_bakery_sku_date["bakery_id"].eq(22))
            & (by_bakery_sku_date["product_id"].eq(10340))
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

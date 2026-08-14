"""Purchased-bread rows for the pilot Bitrix forecast publisher."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


BAKED_PRODUCT_IDS = {70, 517, 10345, 10524}


def load_mapping(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["product_id"] = pd.to_numeric(frame["product_id"], errors="coerce")
    frame = frame[frame["product_id"].notna()].copy()
    frame["product_id"] = frame["product_id"].astype(int)
    if "exclusion_reason" in frame:
        frame = frame[frame["exclusion_reason"].fillna("").eq("")]
    return frame[~frame["product_id"].isin(BAKED_PRODUCT_IDS)].copy()


def reconstruct_opening_carry(facts: pd.DataFrame, forecast_date: str) -> pd.DataFrame:
    frame = facts.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    target = pd.Timestamp(forecast_date).normalize()
    output = []
    for (bakery_id, product_id), group in frame.groupby(["bakery_id", "product_id"]):
        carry = 0.0
        previous = None
        opening = 0.0
        for row in group.sort_values("date").itertuples(index=False):
            current = pd.Timestamp(row.date)
            if previous is None or current != previous + pd.Timedelta(days=1):
                carry = 0.0
            if current == target:
                opening = carry
                break
            purchase = max(float(row.purchase_qty), 0.0)
            sales = max(float(row.sales_qty), 0.0)
            sales_from_carry = min(sales, carry)
            sales_from_purchase = min(max(sales - sales_from_carry, 0.0), purchase)
            carry = max(purchase - sales_from_purchase, 0.0)
            previous = current
        output.append({"bakery_id": int(bakery_id), "product_id": int(product_id), "opening_carry": opening})
    return pd.DataFrame(output)


def build_purchased_bread_rows(
    client,
    forecast_df: pd.DataFrame,
    bakery_info: dict[int, dict],
    forecast_date: str,
    mapping_path: str | Path,
) -> list[dict]:
    mapping = load_mapping(mapping_path)
    if mapping.empty:
        return []
    bakeries = pd.DataFrame([
        {"bakery_id": bid, "city_scope": info.get("city"), "bakery_name": info.get("name")}
        for bid, info in bakery_info.items()
    ])
    forecast = forecast_df.copy()
    forecast["bakery_id"] = pd.to_numeric(forecast["bakery_id"], errors="coerce")
    forecast["product_id"] = pd.to_numeric(forecast["product_id"], errors="coerce")
    forecast = forecast.dropna(subset=["bakery_id", "product_id"])
    forecast[["bakery_id", "product_id"]] = forecast[["bakery_id", "product_id"]].astype(int)
    forecast = forecast.merge(bakeries, on="bakery_id", how="inner")
    forecast = forecast.merge(mapping, on=["city_scope", "product_id"], how="inner", suffixes=("", "_map"))
    if forecast.empty:
        return []

    bids = sorted(forecast["bakery_id"].unique().tolist())
    pids = sorted(forecast["product_id"].unique().tolist())
    materials = sorted({str(value) for value in forecast["material_id"] if pd.notna(value)})
    history_from = str(pd.Timestamp(forecast_date).date() - pd.Timedelta(days=2))
    sales = client.query_df(
        """select check_date date,toInt64(bakery_id) bakery_id,toInt64(product_id) product_id,
        sum(toFloat64(quantity)) sales_qty from Svezhar.stg_check_lines
        where check_date between toDate(%(history_from)s) and toDate(%(forecast_date)s)
          and toInt64(bakery_id) in %(bids)s and toInt64(product_id) in %(pids)s
          and cash_event_type=%(sale)s and is_deleted=%(no)s and quantity>0
        group by date,bakery_id,product_id""",
        parameters={"history_from": history_from, "forecast_date": forecast_date, "bids": bids,
                    "pids": pids, "sale": "Продажа", "no": "Нет"},
    )
    # Guard: empty result from ClickHouse returns DataFrame without columns
    if sales.empty or "date" not in sales.columns:
        sales = pd.DataFrame(columns=["date", "bakery_id", "product_id", "sales_qty"])

    purchases = client.query_df(
        """with latest as (select order_id,line_id,argMax(delivery_date,_updated_at) date,
        argMax(bakery_id,_updated_at) bakery_id,argMax(status,_updated_at) status,
        argMax(raw_material_id,_updated_at) material_id,argMax(plan_quantity,_updated_at) plan_qty,
        argMax(fact_quantity,_updated_at) fact_qty from Svezhar.fct_orders group by order_id,line_id)
        select date,toInt64OrZero(toString(bakery_id)) bakery_id,material_id,
        sum(multiIf(status=%(received)s and fact_qty>0,toFloat64(fact_qty),
                    status=%(accepted)s and plan_qty>0,toFloat64(plan_qty),0.0)) purchase_qty
        from latest where date between toDate(%(history_from)s) and toDate(%(forecast_date)s)
          and toInt64OrZero(toString(bakery_id)) in %(bids)s and material_id in %(materials)s
          and ((status=%(received)s and fact_qty>0) or (status=%(accepted)s and plan_qty>0))
        group by date,bakery_id,material_id""",
        parameters={"history_from": history_from, "forecast_date": forecast_date, "bids": bids,
                    "materials": materials, "received": "Получена накладная", "accepted": "Принято пекарем"},
    )
    # Guard: empty result from ClickHouse returns DataFrame without columns
    if purchases.empty or "date" not in purchases.columns:
        purchases = pd.DataFrame(columns=["date", "bakery_id", "material_id", "purchase_qty"])

    purchase_map = forecast[["bakery_id", "product_id", "material_id"]].drop_duplicates()
    purchases = purchases.merge(purchase_map, on=["bakery_id", "material_id"], how="inner")
    purchases = purchases.groupby(["date", "bakery_id", "product_id"], as_index=False)["purchase_qty"].sum()
    dates = pd.DataFrame({"date": pd.date_range(history_from, forecast_date)})
    scope = forecast[["bakery_id", "product_id"]].drop_duplicates(); scope["_k"] = 1; dates["_k"] = 1
    facts = scope.merge(dates, on="_k").drop(columns="_k")
    facts = facts.merge(purchases, on=["date", "bakery_id", "product_id"], how="left")
    facts = facts.merge(sales, on=["date", "bakery_id", "product_id"], how="left")
    facts[["purchase_qty", "sales_qty"]] = facts[["purchase_qty", "sales_qty"]].fillna(0.0)
    carry = reconstruct_opening_carry(facts, forecast_date)
    forecast = forecast.merge(carry, on=["bakery_id", "product_id"], how="left")
    forecast["opening_carry"] = forecast["opening_carry"].fillna(0.0).clip(lower=0.0)

    rows = []
    for row in forecast.to_dict("records"):
        qty = max(float(row.get("forecast_qty") or 0.0), 0.0)
        stock = float(row["opening_carry"])
        net = max(qty - stock, 0.0)
        plan = int(math.ceil(net - 1e-9))
        rows.append({"bakery_id": int(row["bakery_id"]), "bakery_name": row["bakery_name"],
                     "category": str(row.get("category_name_map") or row.get("category_name") or "Хлеб"),
                     "product_name": str(row.get("product_name_map") or row.get("product_name") or row["product_id"]),
                     "forecast": round(qty, 1), "yesterday_stock": round(stock, 1),
                     "net_need": round(net, 1), "production_plan": plan,
                     "total_for_sale": round(plan + stock, 1), "kratnost": 1})
    return rows

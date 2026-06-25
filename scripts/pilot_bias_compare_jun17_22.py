"""
Сравнение прогноза с фактом для пилотных пекарен за 17-22 июня.

Три среза:
  pre_cap  - prod_uplifted_bakery_norm_uplift_sku_20260617_h14  (до введения cap-логики)
  cap_bug  - backfill_cap_uplifted_bakery_norm_uplift_sku_20260617_h7 (cap + redistribution bug)
  actual   - mart_sales_60d

Вывод:
  - Суммы по категориям: пироги / остальные / итого
  - Ratio forecast/actual
  - Топ-SKU по отклонению
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import clickhouse_connect
import pandas as pd

HOST = "rc1b-aergg94cc1r6ctr1.mdb.yandexcloud.net"
PORT = 8443
USER = "AlmazZR"
PASSWORD = "Almaz__Ziast0303"
DATABASE = "Svezhar"

PILOT_IDS = [20, 21, 22, 28, 80, 89, 107, 221, 222, 257]
START_DATE = "2026-06-17"
END_DATE = "2026-06-22"

RUN_PRE_CAP = "prod_uplifted_bakery_norm_uplift_sku_20260617_h14"
RUN_CAP_BUG = "backfill_cap_uplifted_bakery_norm_uplift_sku_20260617_h7"

PIE_PATTERN = "пироги сытные|пироги сладкие"


def connect():
    return clickhouse_connect.get_client(
        host=HOST, port=PORT, username=USER, password=PASSWORD,
        database=DATABASE, secure=True, verify=False,
    )


def load_forecast(client, run_id: str) -> pd.DataFrame:
    df = client.query_df(
        """
        select forecast_date, bakery_id, product_id, product_name,
               category_name, forecast_qty
        from sku_forecast_day_embedded
        where run_id = %(run_id)s
          and bakery_id in %(pilot)s
          and forecast_date between %(start)s and %(end)s
        """,
        parameters={
            "run_id": run_id,
            "pilot": PILOT_IDS,
            "start": START_DATE,
            "end": END_DATE,
        },
    )
    return df


def load_actuals(client) -> pd.DataFrame:
    df = client.query_df(
        """
        select
            check_date                             as forecast_date,
            toInt64(bakery_id)                     as bakery_id,
            toInt64(product_id)                    as product_id,
            any(product_name)                      as product_name,
            any(category_name)                     as category_name,
            sum(quantity)                          as actual_qty
        from mart_sales_60d
        where toInt64(bakery_id) in %(pilot)s
          and check_date between %(start)s and %(end)s
        group by forecast_date, bakery_id, product_id
        """,
        parameters={
            "pilot": PILOT_IDS,
            "start": START_DATE,
            "end": END_DATE,
        },
    )
    return df


def category_group(cat: pd.Series) -> pd.Series:
    is_pie = cat.str.lower().str.contains(PIE_PATTERN, na=False)
    return is_pie.map({True: "пироги", False: "остальные"})


def summary_table(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    df = df.copy()
    df["cat_group"] = category_group(df["category_name"])

    rows = []
    for grp, sub in [("все SKU", df), ("пироги", df[df["cat_group"] == "пироги"]), ("остальные", df[df["cat_group"] == "остальные"])]:
        actual  = sub["actual_qty"].sum()
        pre_cap = sub["pre_cap_qty"].sum()
        cap_bug = sub["cap_bug_qty"].sum()
        rows.append({
            "группа": grp,
            "факт": round(actual),
            "до_капа": round(pre_cap),
            "с_капом_баг": round(cap_bug),
            "ratio_до_капа": round(pre_cap / actual, 3) if actual else None,
            "ratio_с_капом": round(cap_bug / actual, 3) if actual else None,
        })
    return pd.DataFrame(rows)


def top_sku_deviation(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    grp = (
        df.groupby(["product_id", "product_name", "category_name"], as_index=False)
        .agg(actual_qty=("actual_qty", "sum"),
             pre_cap_qty=("pre_cap_qty", "sum"),
             cap_bug_qty=("cap_bug_qty", "sum"))
    )
    grp = grp[grp["actual_qty"] >= 10]  # отфильтровать мелочь
    grp["ratio_pre"] = (grp["pre_cap_qty"] / grp["actual_qty"]).round(3)
    grp["ratio_cap"] = (grp["cap_bug_qty"] / grp["actual_qty"]).round(3)
    grp["delta_ratio"] = (grp["ratio_cap"] - grp["ratio_pre"]).round(3)
    return grp.sort_values("delta_ratio", ascending=False).head(n)


def main():
    client = connect()
    print("Загружаем прогнозы из CH...")

    pre_cap_df = load_forecast(client, RUN_PRE_CAP)
    print(f"  pre_cap rows: {len(pre_cap_df)}")

    cap_bug_df = load_forecast(client, RUN_CAP_BUG)
    print(f"  cap_bug rows: {len(cap_bug_df)}")

    actual_df = load_actuals(client)
    print(f"  actual rows:  {len(actual_df)}")

    if pre_cap_df.empty:
        print(f"WARN: нет данных для run {RUN_PRE_CAP}")
    if cap_bug_df.empty:
        print(f"WARN: нет данных для run {RUN_CAP_BUG}")

    key = ["forecast_date", "bakery_id", "product_id"]

    # Мёрджим всё
    merged = (
        actual_df
        .rename(columns={"forecast_date": "forecast_date"})
        .merge(
            pre_cap_df[key + ["category_name", "forecast_qty"]].rename(columns={"forecast_qty": "pre_cap_qty"}),
            on=key, how="outer",
        )
        .merge(
            cap_bug_df[key + ["forecast_qty"]].rename(columns={"forecast_qty": "cap_bug_qty"}),
            on=key, how="outer",
        )
    )
    merged["actual_qty"]  = merged["actual_qty"].fillna(0.0)
    merged["pre_cap_qty"] = merged["pre_cap_qty"].fillna(0.0)
    merged["cap_bug_qty"] = merged["cap_bug_qty"].fillna(0.0)

    # Подтягиваем category из pre_cap там где из actual нет
    if "category_name_x" in merged.columns:
        merged["category_name"] = merged["category_name_x"].combine_first(merged["category_name_y"])
        merged = merged.drop(columns=["category_name_x", "category_name_y"], errors="ignore")

    print("\n" + "=" * 60)
    print(f"ПИЛОТ {PILOT_IDS}")
    print(f"Период {START_DATE} — {END_DATE}")
    print("=" * 60)

    summ = summary_table(merged, "cat_group")
    print("\n--- Суммы по группам ---")
    print(summ.to_string(index=False))

    print("\n--- Топ-20 SKU по росту ratio (cap_bug vs pre_cap) ---")
    top = top_sku_deviation(merged)
    pd.set_option("display.max_colwidth", 40)
    pd.set_option("display.width", 120)
    print(top[["product_id","product_name","category_name","actual_qty","pre_cap_qty","cap_bug_qty","ratio_pre","ratio_cap","delta_ratio"]].to_string(index=False))

    # По пекарням: у кого самый большой delta
    print("\n--- По пекарням: средний ratio_cap по остальным SKU ---")
    per_bak = (
        merged[~merged["category_name"].str.lower().str.contains(PIE_PATTERN, na=False)]
        .groupby("bakery_id", as_index=False)
        .agg(actual=("actual_qty","sum"), pre=("pre_cap_qty","sum"), cap=("cap_bug_qty","sum"))
    )
    per_bak["ratio_pre"] = (per_bak["pre"] / per_bak["actual"]).round(3)
    per_bak["ratio_cap"] = (per_bak["cap"] / per_bak["actual"]).round(3)
    per_bak["delta"]     = (per_bak["ratio_cap"] - per_bak["ratio_pre"]).round(3)
    print(per_bak.sort_values("delta", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()

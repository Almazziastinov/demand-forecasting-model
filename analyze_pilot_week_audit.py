"""
Аудит пилотных пекарен за последнюю неделю (2026-06-23 .. 2026-06-28).

Сравниваем lead-1 прогноз (sku_forecast_day_snapshots, lead_days=1)
с фактом (mart_sales_60d).

Разделы:
  1. Доступность данных — какие даты есть в прогнозе и факте
  2. Общий summary по группам (все / ходовые / пироги сытные)
  3. Топ ходовых SKU — forecast vs fact
  4. Пироги сытные по пекарням — bias, abs error
  5. По пекарням — агрегат ratio
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import clickhouse_connect
import pandas as pd
import numpy as np

HOST = "rc1b-aergg94cc1r6ctr1.mdb.yandexcloud.net"
PORT = 8443
USER = "AlmazZR"
PASSWORD = "Almaz__Ziast0303"
DATABASE = "Svezhar"

PILOT_IDS = [20, 21, 22, 28, 80, 89, 107, 221, 222, 257]
START_DATE = "2026-06-23"
END_DATE = "2026-06-28"

# Ходовые = топ-N по объёму факта за период, порог min_actual
HODOVYE_MIN_ACTUAL = 30   # за весь период по всем пилотным
HODOVYE_TOP_N = 30

PIE_SAVOURY_PATTERN = "пироги сытные"


def connect():
    return clickhouse_connect.get_client(
        host=HOST, port=PORT, username=USER, password=PASSWORD,
        database=DATABASE, secure=True, verify=False,
    )


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------

def load_lead1_forecast(client) -> pd.DataFrame:
    """Lead-1 прогноз из snapshot-таблицы."""
    df = client.query_df(
        """
        select
            forecast_date,
            bakery_id,
            product_id,
            any(product_name)   as product_name,
            any(category_name)  as category_name,
            any(source_run_id)  as source_run_id,
            sum(forecast_qty)   as forecast_qty
        from sku_forecast_day_snapshots
        where lead_days = 1
          and toInt64(bakery_id) in %(pilot)s
          and forecast_date between %(start)s and %(end)s
        group by forecast_date, bakery_id, product_id
        """,
        parameters={"pilot": PILOT_IDS, "start": START_DATE, "end": END_DATE},
    )
    return df


def load_actuals(client) -> pd.DataFrame:
    df = client.query_df(
        """
        select
            check_date                          as forecast_date,
            toInt64(bakery_id)                  as bakery_id,
            toInt64(product_id)                 as product_id,
            any(product_name)                   as product_name,
            any(category_name)                  as category_name,
            sum(quantity)                       as actual_qty
        from mart_sales_60d
        where toInt64(bakery_id) in %(pilot)s
          and check_date between %(start)s and %(end)s
        group by forecast_date, bakery_id, product_id
        """,
        parameters={"pilot": PILOT_IDS, "start": START_DATE, "end": END_DATE},
    )
    return df


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------

def wmape(actual, forecast):
    total_actual = actual.sum()
    if total_actual == 0:
        return np.nan
    return abs(forecast - actual).sum() / total_actual


def bias_pct(actual, forecast):
    total_actual = actual.sum()
    if total_actual == 0:
        return np.nan
    return (forecast.sum() - actual.sum()) / total_actual


# ---------------------------------------------------------------------------
# Разделы аудита
# ---------------------------------------------------------------------------

def section1_availability(forecast_df, actual_df):
    print("\n" + "=" * 65)
    print("1. ДОСТУПНОСТЬ ДАННЫХ")
    print("=" * 65)

    fc_dates = sorted(forecast_df["forecast_date"].astype(str).unique())
    act_dates = sorted(actual_df["forecast_date"].astype(str).unique())
    all_dates = sorted(set(fc_dates) | set(act_dates))

    rows = []
    for d in all_dates:
        fc_bak  = forecast_df[forecast_df["forecast_date"].astype(str) == d]["bakery_id"].nunique()
        act_bak = actual_df[actual_df["forecast_date"].astype(str) == d]["bakery_id"].nunique()

        # run_ids для этой даты
        run_ids = forecast_df[forecast_df["forecast_date"].astype(str) == d]["source_run_id"].unique()
        run_label = ", ".join(sorted(run_ids)[:2]) if len(run_ids) else "—"
        if len(run_ids) > 2:
            run_label += f" (+{len(run_ids)-2})"

        rows.append({
            "дата": d,
            "прогноз_пекарен": fc_bak,
            "факт_пекарен": act_bak,
            "прогноз_sku": len(forecast_df[forecast_df["forecast_date"].astype(str) == d]),
            "факт_sku": len(actual_df[actual_df["forecast_date"].astype(str) == d]),
            "run_id": run_label,
        })
    print(pd.DataFrame(rows).to_string(index=False))


def section2_summary(merged):
    print("\n" + "=" * 65)
    print("2. ОБЩИЙ SUMMARY ПО ГРУППАМ")
    print("=" * 65)

    is_pie = merged["category_name"].str.lower().str.contains(PIE_SAVOURY_PATTERN, na=False)
    total_actual = merged["actual_qty"].sum()

    # Ходовые: по суммарному факту на уровне SKU (product_id)
    sku_actual = merged.groupby("product_id")["actual_qty"].sum()
    hodovye_ids = set(sku_actual[sku_actual >= HODOVYE_MIN_ACTUAL].nlargest(HODOVYE_TOP_N).index)

    is_hodovye = merged["product_id"].isin(hodovye_ids)

    rows = []
    for label, mask in [
        ("Все SKU",           pd.Series([True] * len(merged), index=merged.index)),
        (f"Ходовые (топ-{HODOVYE_TOP_N})", is_hodovye),
        ("Пироги сытные",    is_pie),
        ("Остальные",        ~is_pie & ~is_hodovye),
    ]:
        sub = merged[mask]
        a = sub["actual_qty"].sum()
        f = sub["forecast_qty"].sum()
        rows.append({
            "группа": label,
            "факт": int(round(a)),
            "прогноз": int(round(f)),
            "bias_%": f"{bias_pct(sub['actual_qty'], sub['forecast_qty'])*100:+.1f}%",
            "wMAPE_%": f"{wmape(sub['actual_qty'], sub['forecast_qty'])*100:.1f}%",
            "доля_факта_%": f"{a/total_actual*100:.1f}%" if total_actual else "—",
        })
    print(pd.DataFrame(rows).to_string(index=False))
    return hodovye_ids


def section3_hodovye_sku(merged, hodovye_ids):
    print("\n" + "=" * 65)
    print(f"3. ХОДОВЫЕ SKU (топ-{HODOVYE_TOP_N} по объёму факта)")
    print("=" * 65)

    sub = merged[merged["product_id"].isin(hodovye_ids)]
    grp = (
        sub.groupby(["product_id", "product_name", "category_name"], as_index=False)
        .agg(
            actual_qty=("actual_qty", "sum"),
            forecast_qty=("forecast_qty", "sum"),
        )
        .sort_values("actual_qty", ascending=False)
    )
    grp["bias_%"]   = (grp["forecast_qty"] - grp["actual_qty"]) / grp["actual_qty"].clip(lower=1) * 100
    grp["wMAPE_%"]  = abs(grp["forecast_qty"] - grp["actual_qty"]) / grp["actual_qty"].clip(lower=1) * 100
    grp["ratio"]    = (grp["forecast_qty"] / grp["actual_qty"].clip(lower=1)).round(2)

    grp["bias_%"]  = grp["bias_%"].map(lambda x: f"{x:+.1f}%")
    grp["wMAPE_%"] = grp["wMAPE_%"].map(lambda x: f"{x:.1f}%")

    pd.set_option("display.max_colwidth", 35)
    pd.set_option("display.width", 130)
    print(
        grp[["product_id","product_name","category_name","actual_qty","forecast_qty","ratio","bias_%","wMAPE_%"]]
        .to_string(index=False)
    )


def section4_pies(merged):
    print("\n" + "=" * 65)
    print("4. ПИРОГИ СЫТНЫЕ — по пекарням и датам")
    print("=" * 65)

    pies = merged[
        merged["category_name"].str.lower().str.contains(PIE_SAVOURY_PATTERN, na=False)
    ].copy()

    if pies.empty:
        print("Нет данных по 'пироги сытные' в этом периоде/пилоте")
        return

    print(f"  SKU-пекарня-день записей: {len(pies)}")
    print(f"  Уникальных SKU: {pies['product_id'].nunique()}")
    print(f"  Пекарен: {pies['bakery_id'].nunique()}")

    # По пекарням
    print("\n  По пекарням:")
    per_bak = (
        pies.groupby("bakery_id", as_index=False)
        .agg(actual=("actual_qty","sum"), forecast=("forecast_qty","sum"))
    )
    per_bak["bias_%"] = (per_bak["forecast"] - per_bak["actual"]) / per_bak["actual"].clip(lower=1) * 100
    per_bak["ratio"]  = (per_bak["forecast"] / per_bak["actual"].clip(lower=1)).round(2)
    per_bak = per_bak.sort_values("actual", ascending=False)
    per_bak["bias_%"] = per_bak["bias_%"].map(lambda x: f"{x:+.1f}%")
    print(per_bak.to_string(index=False))

    # По SKU
    print("\n  По SKU (топ по объёму факта):")
    per_sku = (
        pies.groupby(["product_id","product_name"], as_index=False)
        .agg(actual=("actual_qty","sum"), forecast=("forecast_qty","sum"))
        .sort_values("actual", ascending=False)
    )
    per_sku["bias_%"] = (per_sku["forecast"] - per_sku["actual"]) / per_sku["actual"].clip(lower=1) * 100
    per_sku["ratio"]  = (per_sku["forecast"] / per_sku["actual"].clip(lower=1)).round(2)
    per_sku["bias_%"] = per_sku["bias_%"].map(lambda x: f"{x:+.1f}%")
    pd.set_option("display.max_colwidth", 40)
    print(per_sku.to_string(index=False))

    # По дням
    print("\n  По датам:")
    per_day = (
        pies.groupby("forecast_date", as_index=False)
        .agg(actual=("actual_qty","sum"), forecast=("forecast_qty","sum"))
        .sort_values("forecast_date")
    )
    per_day["bias_%"] = (per_day["forecast"] - per_day["actual"]) / per_day["actual"].clip(lower=1) * 100
    per_day["ratio"]  = (per_day["forecast"] / per_day["actual"].clip(lower=1)).round(2)
    per_day["bias_%"] = per_day["bias_%"].map(lambda x: f"{x:+.1f}%")
    print(per_day.to_string(index=False))


def section5_bakeries(merged):
    print("\n" + "=" * 65)
    print("5. ПО ПЕКАРНЯМ — общий ratio и wMAPE")
    print("=" * 65)

    per_bak = (
        merged.groupby("bakery_id", as_index=False)
        .agg(actual=("actual_qty","sum"), forecast=("forecast_qty","sum"))
    )
    per_bak["bias_%"] = (per_bak["forecast"] - per_bak["actual"]) / per_bak["actual"].clip(lower=1) * 100
    per_bak["ratio"]  = (per_bak["forecast"] / per_bak["actual"].clip(lower=1)).round(2)
    per_bak = per_bak.sort_values("actual", ascending=False)
    per_bak["bias_%"] = per_bak["bias_%"].map(lambda x: f"{x:+.1f}%")
    print(per_bak.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Аудит пилота: {PILOT_IDS}")
    print(f"Период: {START_DATE} — {END_DATE}")

    client = connect()
    print("\nЗагружаем данные из ClickHouse...")

    forecast_df = load_lead1_forecast(client)
    print(f"  lead-1 прогноз: {len(forecast_df):,} строк")

    actual_df = load_actuals(client)
    print(f"  факт: {len(actual_df):,} строк")

    # Мёрдж
    key = ["forecast_date", "bakery_id", "product_id"]
    merged = (
        actual_df
        .merge(
            forecast_df[key + ["product_name","category_name","source_run_id","forecast_qty"]],
            on=key, how="outer", suffixes=("_act","_fc"),
        )
    )
    # Сводим product_name / category_name
    for col in ["product_name", "category_name"]:
        col_act, col_fc = f"{col}_act", f"{col}_fc"
        if col_act in merged.columns:
            merged[col] = merged[col_act].combine_first(merged[col_fc])
            merged.drop(columns=[col_act, col_fc], inplace=True, errors="ignore")

    merged["actual_qty"]   = merged["actual_qty"].fillna(0.0)
    merged["forecast_qty"] = merged["forecast_qty"].fillna(0.0)
    merged["source_run_id"] = merged["source_run_id"].fillna("—")

    print(f"  после мёрджа: {len(merged):,} строк\n")

    section1_availability(forecast_df, actual_df)
    hodovye_ids = section2_summary(merged)
    section3_hodovye_sku(merged, hodovye_ids)
    section4_pies(merged)
    section5_bakeries(merged)

    print("\n" + "=" * 65)
    print("Аудит завершён")
    print("=" * 65)


if __name__ == "__main__":
    main()

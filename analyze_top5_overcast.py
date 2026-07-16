"""
Анализ overcast на топ-5 ходовых SKU пилота.

Период: 2026-06-24..2026-06-28 (исключаем 23 июня — задвоенные run_id).

Структура:
  1. Топ-5 SKU: факт vs прогноз, bias по дням
  2. Топ-5 SKU: разбивка по пекарням
  3. Bakery-day уровень: базовый прогноз vs факт (изолируем ошибку модели от SKU-аллокации)
  4. Диагностика: гипотезы о причинах overcast
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
# Исключаем 23 июня — там два run_id одновременно (задвоение)
START_DATE = "2026-06-24"
END_DATE   = "2026-06-28"


def connect():
    return clickhouse_connect.get_client(
        host=HOST, port=PORT, username=USER, password=PASSWORD,
        database=DATABASE, secure=True, verify=False,
    )


def bias_pct(actual, forecast):
    s = actual.sum()
    return (forecast.sum() - s) / s * 100 if s else np.nan


def wmape(actual, forecast):
    s = actual.sum()
    return abs(forecast - actual).sum() / s * 100 if s else np.nan


# ---------------------------------------------------------------------------
# Загрузка SKU-уровня
# ---------------------------------------------------------------------------

def load_sku_forecast(client) -> pd.DataFrame:
    return client.query_df(
        """
        select
            forecast_date,
            toInt64(bakery_id)   as bakery_id,
            toInt64(product_id)  as product_id,
            any(product_name)    as product_name,
            any(category_name)   as category_name,
            any(source_run_id)   as source_run_id,
            sum(forecast_qty)    as forecast_qty
        from sku_forecast_day_snapshots
        where lead_days = 1
          and toInt64(bakery_id) in %(pilot)s
          and forecast_date between %(start)s and %(end)s
        group by forecast_date, bakery_id, product_id
        """,
        parameters={"pilot": PILOT_IDS, "start": START_DATE, "end": END_DATE},
    )


def load_sku_actuals(client) -> pd.DataFrame:
    return client.query_df(
        """
        select
            check_date               as forecast_date,
            toInt64(bakery_id)       as bakery_id,
            toInt64(product_id)      as product_id,
            any(product_name)        as product_name,
            any(category_name)       as category_name,
            sum(quantity)            as actual_qty
        from mart_sales_60d
        where toInt64(bakery_id) in %(pilot)s
          and check_date between %(start)s and %(end)s
        group by forecast_date, bakery_id, product_id
        """,
        parameters={"pilot": PILOT_IDS, "start": START_DATE, "end": END_DATE},
    )


# ---------------------------------------------------------------------------
# Загрузка bakery-day уровня
# ---------------------------------------------------------------------------

def load_bakery_day_forecast(client) -> pd.DataFrame:
    """Прогноз на уровне пекарня-день из snapshot-таблицы."""
    return client.query_df(
        """
        select
            forecast_date,
            toInt64(bakery_id)   as bakery_id,
            any(source_run_id)   as source_run_id,
            sum(forecast_final)  as bakery_forecast_qty
        from bakery_forecast_day_snapshots
        where lead_days = 1
          and toInt64(bakery_id) in %(pilot)s
          and forecast_date between %(start)s and %(end)s
        group by forecast_date, bakery_id
        """,
        parameters={"pilot": PILOT_IDS, "start": START_DATE, "end": END_DATE},
    )


def load_bakery_day_actuals(client) -> pd.DataFrame:
    """Факт на уровне пекарня-день."""
    return client.query_df(
        """
        select
            check_date           as forecast_date,
            toInt64(bakery_id)   as bakery_id,
            sum(quantity)        as bakery_actual_qty
        from mart_sales_60d
        where toInt64(bakery_id) in %(pilot)s
          and check_date between %(start)s and %(end)s
        group by forecast_date, bakery_id
        """,
        parameters={"pilot": PILOT_IDS, "start": START_DATE, "end": END_DATE},
    )


# ---------------------------------------------------------------------------
# Загрузка recent-correction (city-prior) компонента
# ---------------------------------------------------------------------------

def load_recent_correction_context(client) -> pd.DataFrame:
    """
    Смотрим forecast_day_context_embedded — там хранится контекст прогноза:
    температура, поправки и т.д. Если нет поля correction — просто проверим
    что есть.
    """
    try:
        df = client.query_df(
            """
            select *
            from forecast_day_context_embedded
            where forecast_date between %(start)s and %(end)s
            limit 5
            """,
            parameters={"start": START_DATE, "end": END_DATE},
        )
        return df
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})


# ---------------------------------------------------------------------------
# Разделы
# ---------------------------------------------------------------------------

def section1_top5(merged_sku, top5_ids):
    print("\n" + "=" * 65)
    print("1. ТОП-5 ХОДОВЫХ SKU: ОБЩИЙ OVERCAST")
    print("=" * 65)

    sub = merged_sku[merged_sku["product_id"].isin(top5_ids)]
    grp = (
        sub.groupby(["product_id", "product_name", "category_name"], as_index=False)
        .agg(actual=("actual_qty","sum"), forecast=("forecast_qty","sum"))
        .sort_values("actual", ascending=False)
    )
    grp["overcast_qty"] = (grp["forecast"] - grp["actual"]).round(1)
    grp["ratio"]        = (grp["forecast"] / grp["actual"].clip(lower=1)).round(2)
    grp["bias_%"]       = grp.apply(
        lambda r: f"{(r.forecast-r.actual)/r.actual*100:+.1f}%" if r.actual else "—", axis=1
    )
    pd.set_option("display.max_colwidth", 38)
    pd.set_option("display.width", 130)
    print(grp[["product_id","product_name","category_name","actual","forecast","overcast_qty","ratio","bias_%"]].to_string(index=False))

    total_a = sub["actual_qty"].sum()
    total_f = sub["forecast_qty"].sum()
    print(f"\n  ИТОГО топ-5:  факт={total_a:.0f}  прогноз={total_f:.0f}  "
          f"overcast={total_f-total_a:.0f}  bias={bias_pct(sub['actual_qty'],sub['forecast_qty']):+.1f}%  "
          f"wMAPE={wmape(sub['actual_qty'],sub['forecast_qty']):.1f}%")


def section2_top5_by_day(merged_sku, top5_ids, top5_names):
    print("\n" + "=" * 65)
    print("2. ТОП-5 SKU: BIAS ПО ДНЯМ")
    print("=" * 65)

    sub = merged_sku[merged_sku["product_id"].isin(top5_ids)].copy()
    sub["forecast_date"] = sub["forecast_date"].astype(str)

    pivot_act = sub.pivot_table(index="forecast_date", columns="product_id",
                                values="actual_qty", aggfunc="sum")
    pivot_fc  = sub.pivot_table(index="forecast_date", columns="product_id",
                                values="forecast_qty", aggfunc="sum")

    print("\n  Факт по дням:")
    print(pivot_act.rename(columns=top5_names).to_string())

    print("\n  Прогноз по дням:")
    print(pivot_fc.rename(columns=top5_names).to_string())

    ratio_pivot = (pivot_fc / pivot_act.clip(lower=0.1)).round(2)
    print("\n  Ratio (прогноз/факт) по дням:")
    print(ratio_pivot.rename(columns=top5_names).to_string())

    print("\n  Среднедневной ratio по SKU:")
    for pid in top5_ids:
        if pid in ratio_pivot.columns:
            vals = ratio_pivot[pid].dropna()
            print(f"    {top5_names.get(pid, pid)[:35]:35s}  "
                  f"mean={vals.mean():.2f}  min={vals.min():.2f}  max={vals.max():.2f}")


def section3_top5_by_bakery(merged_sku, top5_ids, top5_names):
    print("\n" + "=" * 65)
    print("3. ТОП-5 SKU: РАЗБИВКА ПО ПЕКАРНЯМ")
    print("=" * 65)

    sub = merged_sku[merged_sku["product_id"].isin(top5_ids)]
    grp = (
        sub.groupby(["bakery_id", "product_id", "product_name"], as_index=False)
        .agg(actual=("actual_qty","sum"), forecast=("forecast_qty","sum"))
        .sort_values(["product_id", "actual"], ascending=[True, False])
    )
    grp["ratio"]  = (grp["forecast"] / grp["actual"].clip(lower=1)).round(2)
    grp["bias_%"] = grp.apply(
        lambda r: f"{(r.forecast-r.actual)/r.actual*100:+.1f}%" if r.actual else "нет факта", axis=1
    )
    pd.set_option("display.max_colwidth", 30)
    print(grp[["product_id","product_name","bakery_id","actual","forecast","ratio","bias_%"]].to_string(index=False))


def section4_bakery_day(bak_fc, bak_act):
    print("\n" + "=" * 65)
    print("4. BAKERY-DAY УРОВЕНЬ: ОШИБКА БАЗОВОЙ МОДЕЛИ")
    print("   (до SKU-аллокации)")
    print("=" * 65)

    key = ["forecast_date", "bakery_id"]
    merged = bak_act.merge(bak_fc, on=key, how="outer")
    merged["bakery_actual_qty"]   = merged["bakery_actual_qty"].fillna(0)
    merged["bakery_forecast_qty"] = merged["bakery_forecast_qty"].fillna(0)

    total_a = merged["bakery_actual_qty"].sum()
    total_f = merged["bakery_forecast_qty"].sum()
    print(f"\n  Итого по пилоту (bakery-day):")
    print(f"    факт:     {total_a:,.0f}")
    print(f"    прогноз:  {total_f:,.0f}")
    print(f"    bias:     {(total_f-total_a)/total_a*100:+.1f}%")
    print(f"    wMAPE:    {abs(merged['bakery_forecast_qty']-merged['bakery_actual_qty']).sum()/total_a*100:.1f}%")

    print("\n  По пекарням:")
    per_bak = (
        merged.groupby("bakery_id", as_index=False)
        .agg(actual=("bakery_actual_qty","sum"), forecast=("bakery_forecast_qty","sum"))
        .sort_values("actual", ascending=False)
    )
    per_bak["ratio"]  = (per_bak["forecast"] / per_bak["actual"].clip(lower=1)).round(2)
    per_bak["bias_%"] = ((per_bak["forecast"] - per_bak["actual"]) / per_bak["actual"].clip(lower=1) * 100).map(lambda x: f"{x:+.1f}%")
    print(per_bak.to_string(index=False))

    print("\n  По дням:")
    per_day = (
        merged.groupby("forecast_date", as_index=False)
        .agg(actual=("bakery_actual_qty","sum"), forecast=("bakery_forecast_qty","sum"))
        .sort_values("forecast_date")
    )
    per_day["ratio"]  = (per_day["forecast"] / per_day["actual"].clip(lower=1)).round(2)
    per_day["bias_%"] = ((per_day["forecast"] - per_day["actual"]) / per_day["actual"].clip(lower=1) * 100).map(lambda x: f"{x:+.1f}%")
    print(per_day.to_string(index=False))

    return merged


def section5_diagnosis(merged_sku, bak_merged, top5_ids, top5_names):
    print("\n" + "=" * 65)
    print("5. ДИАГНОСТИКА ПРИЧИН OVERCAST")
    print("=" * 65)

    # 5a: разложение ошибки: базовая модель vs SKU-аллокация
    total_act_sku = merged_sku["actual_qty"].sum()
    total_fc_sku  = merged_sku["forecast_qty"].sum()
    total_act_bak = bak_merged["bakery_actual_qty"].sum()
    total_fc_bak  = bak_merged["bakery_forecast_qty"].sum()

    model_bias_pct  = (total_fc_bak - total_act_bak) / total_act_bak * 100
    sku_bias_pct    = (total_fc_sku - total_act_sku) / total_act_sku * 100

    print(f"\n  Декомпозиция ошибки:")
    print(f"    Bakery-day модель (до аллокации):  bias = {model_bias_pct:+.1f}%")
    print(f"    SKU-аллокация (итого):             bias = {sku_bias_pct:+.1f}%")
    if abs(sku_bias_pct) > abs(model_bias_pct):
        delta = sku_bias_pct - model_bias_pct
        print(f"    => SKU-аллокация добавляет ещё:    {delta:+.1f}%")
    elif abs(model_bias_pct) >= abs(sku_bias_pct) * 0.8:
        print(f"    => Основная ошибка в базовой модели, аллокация не усугубляет")

    # 5b: топ-5 — однородность bias по дням
    sub = merged_sku[merged_sku["product_id"].isin(top5_ids)].copy()
    sub["forecast_date"] = sub["forecast_date"].astype(str)
    print(f"\n  Стабильность overcast по дням (топ-5 SKU):")
    per_day = (
        sub.groupby("forecast_date")
        .apply(lambda g: pd.Series({
            "actual": g["actual_qty"].sum(),
            "forecast": g["forecast_qty"].sum(),
            "bias_%": bias_pct(g["actual_qty"], g["forecast_qty"]),
        }))
        .reset_index()
    )
    per_day["bias_%"] = per_day["bias_%"].map(lambda x: f"{x:+.1f}%")
    print(per_day.to_string(index=False))
    print("  => Если bias стабилен каждый день — это систематика, не случайность")

    # 5c: распределение ratio по SKU-пекарня-день
    sub2 = merged_sku[merged_sku["product_id"].isin(top5_ids)].copy()
    sub2 = sub2[sub2["actual_qty"] > 0]
    sub2["ratio"] = sub2["forecast_qty"] / sub2["actual_qty"]
    print(f"\n  Распределение ratio (SKU x пекарня x день) для топ-5:")
    desc = sub2["ratio"].describe(percentiles=[.1, .25, .5, .75, .9])
    for k, v in desc.items():
        print(f"    {k:6s}: {v:.3f}")
    under_10 = (sub2["ratio"] < 0.1).sum()
    over_2   = (sub2["ratio"] > 2.0).sum()
    between  = len(sub2) - under_10 - over_2
    print(f"    ratio < 0.1 (почти нет прогноза): {under_10}")
    print(f"    ratio 0.1-2.0 (норма):             {between}")
    print(f"    ratio > 2.0  (сильный overcast):   {over_2}")
    print("  => Если много > 2.0 — проблема в outlier-ах, иначе систематический сдвиг")

    # 5d: запросы которые помогут понять дальше
    print("\n" + "-" * 65)
    print("  ГИПОТЕЗЫ:")
    print("-" * 65)

    if model_bias_pct > 30:
        print(f"  [!] H1: БАЗОВАЯ МОДЕЛЬ уже даёт +{model_bias_pct:.0f}%")
        print("      Возможные причины:")
        print("      - city-prior recent correction завысил норму из-за выбросов")
        print("      - обучающий период не отражает текущее поведение (праздники,")
        print("        сезонность, смена ассортимента)")
        print("      - uplift-профиль пекарен 22/222 смещает всё вверх")
    else:
        print(f"  [ok] H1: базовая модель умеренна ({model_bias_pct:+.1f}%)")

    if abs(sku_bias_pct - model_bias_pct) > 15:
        print(f"\n  [!] H2: SKU-АЛЛОКАЦИЯ добавляет ещё {sku_bias_pct-model_bias_pct:+.1f}%")
        print("      Возможные причины:")
        print("      - SKU-профили (доли SKU в пекарне) устарели")
        print("      - новые SKU без истории получают нулевую долю,")
        print("        и их объём перераспределяется на ходовые")
        print("      - sku_hour_share_profile_smoothed завышает доли топ-SKU")
    else:
        print(f"\n  [ok] H2: SKU-аллокация добавляет немного ({sku_bias_pct-model_bias_pct:+.1f}%)")

    print("\n  [?] H3: ДАННЫЕ ФАКТА")
    print("      Проверить: нет ли занижения факта (возвраты, списания)?")
    print("      mart_sales_60d = чистые продажи или включает корректировки?")

    print("\n  [?] H4: НЕДАВНЯЯ КОРРЕКЦИЯ (city-prior, 30 дней)")
    print("      runner_city_prior_soft_weekpart берёт последние 30 дней.")
    print("      Если в мае-июне был аномальный всплеск продаж —")
    print("      prior тянет норму вверх на весь горизонт.")


def section6_recommendations():
    import sys
    out = sys.stdout.buffer
    lines = [
        "",
        "=" * 65,
        "6. ChTO DELATj: VARIANTYi REShENIJ",
        "=" * 65,
        "",
        "  Diaknostika pokazala: oshibka v BAZOVOJ MODELI (+26%),",
        "  SKU-allokaciia dobavliaet 0% — profili neitraljny.",
        "",
        "  A. Ispravitj recent-correction (city-prior):",
        "     A1. Sokratitj okno s 30 do 14 dnej, chtoby ne tianutj anomalii maia.",
        "     A2. Cap na uplift: esli correction > 1.4x ot rolling-medianyi —",
        "         ogranichitj. Proveritj, rabotaet li uzhe sushhestvuiushhaia cap-logika",
        "         v backfill-ranah (backfill_... h1).",
        "     A3. Post-correction downscale: esli 7-dnevnyij rolling-fact",
        "         sistematcheski nizhe modeli — primeniatj scale-factor.",
        "",
        "  B. Byistryij fix bez pereobuchenija (per-SKU bias correction):",
        "     Dlia top-5 SKU poschitat' rolling ratio (fact/forecast)",
        "     za poslednie 14 dnej i primenitj kak multiplikator.",
        "     Prosto, interpretiruemo, ne nuzhen retrain.",
        "     Napimer: ratio = 0.76 => forecast_qty *= 0.76.",
        "",
        "  C. Asymmetric loss (esli budet retrain):",
        "     Perekljuchitj s P50 na P35-P40 dlia hodovyih SKU.",
        "     Sdviget median vniz na ~10-15%.",
        "",
        "  D. Monitoring:",
        "     Alert: esli rolling wMAPE top-10 SKU > 40% za 3 dnia — flag v log.",
        "     Trekate ratio hodovyih otdeljno — oni kritichne dlia biznesa.",
        "",
    ]
    for line in lines:
        out.write((line + "\n").encode("utf-8"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Фокусный аудит: топ-5 ходовых SKU, пилот {PILOT_IDS}")
    print(f"Период: {START_DATE} — {END_DATE} (23 июня исключён — задвоенные run_id)")

    client = connect()
    print("\nЗагружаем данные...")

    sku_fc  = load_sku_forecast(client)
    sku_act = load_sku_actuals(client)
    bak_fc  = load_bakery_day_forecast(client)
    bak_act = load_bakery_day_actuals(client)

    print(f"  SKU прогноз:       {len(sku_fc):,}")
    print(f"  SKU факт:          {len(sku_act):,}")
    print(f"  Bakery-day прогноз:{len(bak_fc):,}")
    print(f"  Bakery-day факт:   {len(bak_act):,}")

    # Мёрдж SKU
    key = ["forecast_date", "bakery_id", "product_id"]
    merged_sku = (
        sku_act
        .merge(
            sku_fc[key + ["product_name","category_name","source_run_id","forecast_qty"]],
            on=key, how="outer", suffixes=("_act","_fc"),
        )
    )
    for col in ["product_name", "category_name"]:
        a, b = f"{col}_act", f"{col}_fc"
        if a in merged_sku.columns:
            merged_sku[col] = merged_sku[a].combine_first(merged_sku[b])
            merged_sku.drop(columns=[a, b], inplace=True, errors="ignore")
    merged_sku["actual_qty"]   = merged_sku["actual_qty"].fillna(0.0)
    merged_sku["forecast_qty"] = merged_sku["forecast_qty"].fillna(0.0)
    merged_sku["source_run_id"] = merged_sku["source_run_id"].fillna("—")

    # Топ-5 по объёму факта
    top5 = (
        merged_sku.groupby(["product_id","product_name"])["actual_qty"]
        .sum()
        .nlargest(5)
        .reset_index()
    )
    top5_ids   = list(top5["product_id"])
    top5_names = dict(zip(top5["product_id"], top5["product_name"]))

    print(f"\nТоп-5 ходовых SKU: {top5_ids}")
    for pid, name in top5_names.items():
        print(f"  {pid}: {name}")

    section1_top5(merged_sku, top5_ids)
    section2_top5_by_day(merged_sku, top5_ids, top5_names)
    section3_top5_by_bakery(merged_sku, top5_ids, top5_names)
    bak_merged = section4_bakery_day(bak_fc, bak_act)
    section5_diagnosis(merged_sku, bak_merged, top5_ids, top5_names)
    section6_recommendations()

    print("=" * 65)
    print("Анализ завершён")
    print("=" * 65)


if __name__ == "__main__":
    main()

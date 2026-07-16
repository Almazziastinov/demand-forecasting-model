"""
Анализ влияния плохой погоды на точность прогноза по пекарням.

Гипотезы:
  H1: В дни плохой погоды (D0) модель систематически завышает прогноз
      (не учла снижение продаж из-за погоды).
  H2: На следующий день (D+1) прогноз занижен, так как модель «увидела»
      низкие продажи D0 через лаговые признаки и экстраполировала падение.

Период: 2026-06-18 .. 2026-07-01 (14 дней).
Погода: из forecast_day_context_embedded (все исторические run_id).
Прогноз: bakery_forecast_day_snapshots, lead_days=1,
         берём max(source_run_id) per date чтобы не дублировать.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from pipelines.forecast_publish.load_forecast_run import get_clickhouse_settings
import clickhouse_connect

ch = get_clickhouse_settings(".env")
c = clickhouse_connect.get_client(
    host=ch["host"], port=int(ch["port"]),
    username=ch["username"], password=ch["password"],
    database=ch["database"], secure=bool(ch["secure"]), verify=False,
)

# ── параметры ───────────────────────────────────────────────────────────
START = "2026-06-18"
END   = "2026-07-01"
MIN_ACTUAL = 10          # минимум продаж для включения пекарни в день
STRONG_DROP_PCT = 0.10   # порог «сильного расхождения» на D0: |error_pct| > 10%

# ── утилиты ─────────────────────────────────────────────────────────────
out = sys.stdout.buffer

def p(s: str) -> None:
    out.write((s + "\n").encode("utf-8"))
    out.flush()

def bias_pct(actual: pd.Series, forecast: pd.Series) -> float:
    d = actual.sum()
    return float((forecast - actual).sum() / d) if d > 0 else float("nan")

def wmape(actual: pd.Series, forecast: pd.Series) -> float:
    d = actual.sum()
    return float(abs(forecast - actual).sum() / d) if d > 0 else float("nan")

def fmt_b(v): return f"{v*100:+.1f}%" if pd.notna(v) and np.isfinite(v) else "—"
def fmt_w(v): return f"{v*100:.1f}%"  if pd.notna(v) and np.isfinite(v) else "—"

p("=" * 72)
p(f"ПОГОДА + ЛАГИ: АНАЛИЗ ТОЧНОСТИ НА УРОВНЕ ПЕКАРНИ  {START}..{END}")
p("=" * 72)

# ── 1. Погода (агрегируем по всем run_id — они возвращают одну и ту же погоду) ──
p("\n[1/4] Погода из forecast_day_context_embedded...")
weather_city = c.query_df("""
    select
        forecast_date,
        city,
        avg(temp_mean)          as temp_mean,
        avg(precipitation)      as precipitation,
        max(is_bad_weather::Int8) as is_bad_weather
    from forecast_day_context_embedded
    where forecast_date between %(s)s and %(e)s
    group by forecast_date, city
    order by forecast_date, city
""", parameters={"s": START, "e": END})
weather_city["forecast_date"] = pd.to_datetime(weather_city["forecast_date"])

weather_day = weather_city.groupby("forecast_date").agg(
    bad_cities=("is_bad_weather", "sum"),
    precip=("precipitation", "mean"),
    temp=("temp_mean", "mean"),
).reset_index()

p("Дата       доб_города  осадки  темп")
for _, r in weather_day.iterrows():
    p(f"  {str(r['forecast_date'].date())}  bad={int(r['bad_cities']):2d}/{weather_city['city'].nunique()}  "
      f"precip={r['precip']:.1f}мм  temp={r['temp']:.1f}°C")

# ── 2. Прогноз (max source_run_id per date чтобы не дублировать) ───────
p("\n[2/4] Прогноз bakery_forecast_day_snapshots lead_days=1...")
forecast_bak = c.query_df("""
    with latest as (
        select forecast_date, max(source_run_id) as run_id
        from bakery_forecast_day_snapshots
        where lead_days = 1
          and forecast_date between %(s)s and %(e)s
          and source_run_id not like 'dev_%%'
        group by forecast_date
    )
    select
        b.forecast_date,
        toInt64(b.bakery_id)   as bakery_id,
        any(b.bakery_name)     as bakery_name,
        any(b.city)            as city,
        sum(b.forecast_final)  as forecast_qty,
        any(l.run_id)          as run_id
    from bakery_forecast_day_snapshots b
    inner join latest l
        on b.forecast_date = l.forecast_date
       and b.source_run_id  = l.run_id
    where b.lead_days = 1
    group by b.forecast_date, bakery_id
""", parameters={"s": START, "e": END})
forecast_bak["forecast_date"] = pd.to_datetime(forecast_bak["forecast_date"])
p(f"  прогноз: {len(forecast_bak)} строк, {forecast_bak['forecast_date'].nunique()} дат, "
  f"{forecast_bak['bakery_id'].nunique()} пекарен")
p(f"  run_id примеры: {forecast_bak['run_id'].unique()[:4].tolist()}")

# ── 3. Факт ─────────────────────────────────────────────────────────────
p("\n[3/4] Факт mart_sales_60d...")
actuals = c.query_df("""
    select
        check_date             as forecast_date,
        toInt64(bakery_id)     as bakery_id,
        any(bakery_name)       as bakery_name,
        any(city)              as city,
        sum(quantity)          as actual_qty
    from mart_sales_60d
    where check_date between %(s)s and %(e)s
    group by forecast_date, bakery_id
""", parameters={"s": START, "e": END})
actuals["forecast_date"] = pd.to_datetime(actuals["forecast_date"])
p(f"  факт: {len(actuals)} строк, {actuals['forecast_date'].nunique()} дат")

# ── 4. Merge ─────────────────────────────────────────────────────────────
key = ["forecast_date", "bakery_id"]
df = actuals.merge(
    forecast_bak[key + ["forecast_qty"]],
    on=key, how="inner",
)
# city из actuals
df = df.rename(columns={"bakery_name_x": "bakery_name", "city_x": "city"})
for col in ["bakery_name", "city"]:
    if col + "_x" in df.columns:
        df[col] = df[col + "_x"]

df = df[df["actual_qty"] >= MIN_ACTUAL].copy()
df["error"]     = df["forecast_qty"] - df["actual_qty"]
df["error_pct"] = df["error"] / df["actual_qty"]

# Присоединяем погоду по (дата, город)
df = df.merge(
    weather_city[["forecast_date", "city", "precipitation", "temp_mean", "is_bad_weather"]],
    on=["forecast_date", "city"], how="left",
)
df["is_bad_weather"] = df["is_bad_weather"].fillna(0).astype(bool)
df["precipitation"]  = df["precipitation"].fillna(0.0)
p(f"\n  итог: {len(df)} строк пекарня×день")

# ═══════════════════════════════════════════════════════════════════════
p("\n" + "=" * 72)
p("РЕЗУЛЬТАТЫ")
p("=" * 72)

# A. Дневная хронология с погодой
p("\n── A. BIAS% ПО ДНЯМ ──────────────────────────────────────────────────")
daily = df.groupby("forecast_date").apply(
    lambda g: pd.Series({
        "actual":     g["actual_qty"].sum(),
        "forecast":   g["forecast_qty"].sum(),
        "bias_pct":   bias_pct(g["actual_qty"], g["forecast_qty"]),
        "wmape":      wmape(g["actual_qty"], g["forecast_qty"]),
        "bad_bak":    int(g["is_bad_weather"].sum()),
        "n_bak":      len(g),
        "precip_avg": g["precipitation"].mean(),
    })
).reset_index()
daily["dow"] = daily["forecast_date"].dt.strftime("%a")
daily["ds"]  = daily["forecast_date"].dt.strftime("%m-%d")

p(f"{'дата':<7} {'дн':3} {'факт':>10} {'прогноз':>10} {'bias':>7} {'wMAPE':>6} "
  f"{'пекарни':>8} {'плохих':>7} {'осадки':>7}")
for _, r in daily.iterrows():
    p(f"{r['ds']:<7} {r['dow']:3} {r['actual']:>10,.0f} {r['forecast']:>10,.0f} "
      f"{fmt_b(r['bias_pct']):>7} {fmt_w(r['wmape']):>6} "
      f"{int(r['n_bak']):>8} {int(r['bad_bak']):>7} {r['precip_avg']:>7.1f}мм")

# B. Плохая vs нормальная погода — агрегат
p("\n── B. ПЛОХАЯ ПОГОДА vs НОРМАЛЬНАЯ ───────────────────────────────────")
bad  = df[df["is_bad_weather"]]
good = df[~df["is_bad_weather"]]
p(f"  Плохая погода ({bad['forecast_date'].nunique()} дат × пекарня = {len(bad)} строк):")
p(f"    bias  = {fmt_b(bias_pct(bad['actual_qty'], bad['forecast_qty']))}")
p(f"    wMAPE = {fmt_w(wmape(bad['actual_qty'], bad['forecast_qty']))}")
p(f"    median error_pct = {fmt_b(bad['error_pct'].median())}")
p(f"  Нормальная погода ({len(good)} строк):")
p(f"    bias  = {fmt_b(bias_pct(good['actual_qty'], good['forecast_qty']))}")
p(f"    wMAPE = {fmt_w(wmape(good['actual_qty'], good['forecast_qty']))}")
p(f"    median error_pct = {fmt_b(good['error_pct'].median())}")

# C. По городам в плохую погоду
p("\n── C. ПЛОХАЯ ПОГОДА ПО ГОРОДАМ ──────────────────────────────────────")
city_bad = bad.groupby("city").apply(
    lambda g: pd.Series({
        "actual":   g["actual_qty"].sum(),
        "forecast": g["forecast_qty"].sum(),
        "bias_pct": bias_pct(g["actual_qty"], g["forecast_qty"]),
        "wmape":    wmape(g["actual_qty"], g["forecast_qty"]),
        "n":        len(g),
    })
).reset_index().sort_values("bias_pct", ascending=False)
for _, r in city_bad.iterrows():
    p(f"  {r['city']:<22} bias={fmt_b(r['bias_pct']):>7}  wMAPE={fmt_w(r['wmape']):>6}  n={int(r['n'])}")

# D. Топ пекарни — наибольший провал в плохую погоду
p("\n── D. ТОП-20 ПЕКАРЕН — ЗАВЫШЕНИЕ В ПЛОХУЮ ПОГОДУ ────────────────────")
bak_bad = bad.groupby(["bakery_id", "bakery_name", "city"]).apply(
    lambda g: pd.Series({
        "actual":   g["actual_qty"].sum(),
        "forecast": g["forecast_qty"].sum(),
        "bias_pct": bias_pct(g["actual_qty"], g["forecast_qty"]),
        "n_bad_days": g["forecast_date"].nunique(),
    })
).reset_index()
bak_bad["abs_bias"] = bak_bad["bias_pct"].abs()
top_bak = bak_bad.sort_values("bias_pct", ascending=False).head(20)
p(f"  {'bak_id':>7} {'city':<22} {'bias':>7} {'факт':>9} {'прогноз':>9} {'дней':>5}")
for _, r in top_bak.iterrows():
    p(f"  {int(r['bakery_id']):>7} {r['city']:<22} {fmt_b(r['bias_pct']):>7} "
      f"{r['actual']:>9,.0f} {r['forecast']:>9,.0f} {int(r['n_bad_days']):>5}")

# E. ГИПОТЕЗА О ЛАГАХ: D0 (плохая) → D+1 (нормальная)
p("\n── E. ГИПОТЕЗА О ЛАГАХ: D0→D+1 ──────────────────────────────────────")

bad_d0 = bad[["forecast_date", "bakery_id", "actual_qty", "forecast_qty", "error_pct",
              "precipitation", "city"]].copy()
bad_d0.columns = ["d0_date", "bakery_id", "d0_actual", "d0_forecast", "d0_error_pct",
                  "d0_precip", "city"]
bad_d0["d1_date"] = bad_d0["d0_date"] + pd.Timedelta(days=1)

# D+1 строки (исключаем строки где D+1 тоже плохая погода)
d1_rows = df[~df["is_bad_weather"]][
    ["forecast_date", "bakery_id", "actual_qty", "forecast_qty", "error_pct"]
].rename(columns={
    "forecast_date": "d1_date",
    "actual_qty": "d1_actual",
    "forecast_qty": "d1_forecast",
    "error_pct": "d1_error_pct",
})

lag = bad_d0.merge(d1_rows, on=["d1_date", "bakery_id"], how="inner")
p(f"  Пар (D0_плохая → D+1_нормальная): {len(lag)}")

if len(lag) > 0:
    # Общая статистика
    d0_bias = lag["d0_error_pct"].mean()
    d1_bias = lag["d1_error_pct"].mean()
    p(f"\n  Средний bias на D0 (плохая):   {fmt_b(d0_bias)}")
    p(f"  Средний bias на D+1 (нормальная): {fmt_b(d1_bias)}")
    p(f"  Медиана bias D0:  {fmt_b(lag['d0_error_pct'].median())}")
    p(f"  Медиана bias D+1: {fmt_b(lag['d1_error_pct'].median())}")

    # Корреляция D0 ↔ D+1
    corr = lag[["d0_error_pct", "d1_error_pct"]].corr().iloc[0, 1]
    p(f"\n  Корреляция bias(D0) ↔ bias(D+1): r={corr:+.3f}")
    p("  (r>0: завысили D0 → завысили D+1 = гипотеза НЕ подтверждается)")
    p("  (r<0: завысили D0 → занизили D+1 = ЛАГИ ТЯНУТ ВНИЗ, гипотеза ✓)")

    # Разбивка: сильный провал D0 vs слабый
    p(f"\n  Разбивка по силе провала D0 (порог bias>+{int(STRONG_DROP_PCT*100)}%):")
    strong = lag[lag["d0_error_pct"] >  STRONG_DROP_PCT]  # модель завысила >10%
    low    = lag[lag["d0_error_pct"] <= STRONG_DROP_PCT]
    p(f"  Сильный провал D0 (n={len(strong)}): D0 bias={fmt_b(strong['d0_error_pct'].mean())}  D+1 bias={fmt_b(strong['d1_error_pct'].mean())}")
    p(f"  Слабый провал D0  (n={len(low)}):   D0 bias={fmt_b(low['d0_error_pct'].mean())}  D+1 bias={fmt_b(low['d1_error_pct'].mean())}")

    # Распределение D+1 bias
    p(f"\n  Распределение bias D+1:")
    pcts = lag["d1_error_pct"]
    p(f"    <-20%: {(pcts < -0.20).sum()} пар  (-10%..-20%: {((pcts>=-0.20)&(pcts<-0.10)).sum()})")
    p(f"    -10%..0%: {((pcts>=-0.10)&(pcts<0)).sum()}  0%..+10%: {((pcts>=0)&(pcts<0.10)).sum()}")
    p(f"    >+10%: {(pcts>=0.10).sum()}  >+20%: {(pcts>=0.20).sum()}")

    # Топ-20 пар: сильный D0-провал, смотрим D+1
    p("\n  Топ-20 пар по D0-провалу:")
    top_lag = lag.sort_values("d0_error_pct", ascending=False).head(20)
    p(f"  {'bak_id':>7} {'d0_дата':>8} {'city':<22} "
      f"{'d0_факт':>9} {'d0_прогноз':>10} {'D0 bias':>8} {'D+1 bias':>9}")
    for _, r in top_lag.iterrows():
        p(f"  {int(r['bakery_id']):>7} {str(r['d0_date'].date()):>8} {r['city']:<22} "
          f"{r['d0_actual']:>9,.0f} {r['d0_forecast']:>10,.0f} "
          f"{fmt_b(r['d0_error_pct']):>8} {fmt_b(r['d1_error_pct']):>9}")

# F. D+2
p("\n── F. D+2 — ВОССТАНОВЛЕНИЕ? ─────────────────────────────────────────")
if len(lag) > 0:
    bad_d0_2 = bad_d0.copy()
    bad_d0_2["d2_date"] = bad_d0_2["d0_date"] + pd.Timedelta(days=2)
    d2_rows = df[~df["is_bad_weather"]][
        ["forecast_date", "bakery_id", "error_pct"]
    ].rename(columns={"forecast_date": "d2_date", "error_pct": "d2_error_pct"})
    lag2 = bad_d0_2.merge(d2_rows, on=["d2_date", "bakery_id"], how="inner")
    if len(lag2) > 0:
        # Три шага для пар где есть D0, D+1, D+2
        triple = lag.merge(
            lag2[["d0_date", "bakery_id", "d2_error_pct"]],
            on=["d0_date", "bakery_id"], how="inner",
        )
        p(f"  Троек D0→D+1→D+2: {len(triple)}")
        p(f"  D0  bias: {fmt_b(triple['d0_error_pct'].mean())}")
        p(f"  D+1 bias: {fmt_b(triple['d1_error_pct'].mean())}")
        p(f"  D+2 bias: {fmt_b(triple['d2_error_pct'].mean())}  ← восстановление если ↓")

# G. Распределение D0 провала по диапазонам осадков
p("\n── G. ЗАВИСИМОСТЬ ПРОВАЛА ОТ КОЛИЧЕСТВА ОСАДКОВ ─────────────────────")
df2 = df.copy()
df2["precip_bucket"] = pd.cut(df2["precipitation"],
    bins=[-0.1, 0, 2, 5, 10, 100],
    labels=["0мм", "0-2мм", "2-5мм", "5-10мм", ">10мм"])
grp = df2.groupby("precip_bucket").apply(
    lambda g: pd.Series({
        "n":        len(g),
        "bias_pct": bias_pct(g["actual_qty"], g["forecast_qty"]),
        "wmape":    wmape(g["actual_qty"], g["forecast_qty"]),
    })
).reset_index()
for _, r in grp.iterrows():
    p(f"  {str(r['precip_bucket']):<8} n={int(r['n']):>5}  bias={fmt_b(r['bias_pct']):>7}  wMAPE={fmt_w(r['wmape']):>6}")

p("\n" + "=" * 72)
p("АНАЛИЗ ЗАВЕРШЁН")
p("=" * 72)

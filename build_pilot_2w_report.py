"""
Двухнедельный отчёт по пилотным пекарням: прогноз vs факт.

Период: 2026-06-23 – 2026-07-06
Вывод:  outputs/pilot_2w_report/report.md
        outputs/pilot_2w_report/charts/*.png
"""
from __future__ import annotations

import os
import sys
import textwrap
from datetime import date, timedelta
from pathlib import Path

import clickhouse_connect
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Конфиг
# ---------------------------------------------------------------------------

HOST     = "rc1b-aergg94cc1r6ctr1.mdb.yandexcloud.net"
PORT     = 8443
USER     = "AlmazZR"
PASSWORD = "Almaz__Ziast0303"
DATABASE = "Svezhar"

PILOT_IDS  = [20, 21, 22, 28, 80, 89, 107, 221, 222, 257]
START_DATE = "2026-06-23"
END_DATE   = "2026-07-06"

BAKERY_NAMES = {
    20:  "Мира 45",
    21:  "Парковая 7",
    22:  "Сибирский тракт 25",
    28:  "Гудованцева 27",
    80:  "Калинина 63",
    89:  "Парина 6",
    107: "Четаева 46А",
    221: "Салиха Батыева 15",
    222: "Габдуллы Тукая 62А",
    257: "Ярмарочная 12 (Чебоксары)",
}

TOP_N_SKU   = 15   # сколько SKU показывать в топ
PIE_PATTERN = "пироги сытные"

OUT_DIR    = Path("outputs/pilot_2w_report")
CHART_DIR  = OUT_DIR / "charts"

DOW_RU = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def connect():
    return clickhouse_connect.get_client(
        host=HOST, port=PORT, username=USER, password=PASSWORD,
        database=DATABASE, secure=True, verify=False,
    )


def wmape(actual: pd.Series, forecast: pd.Series) -> float:
    d = float(actual.sum())
    return float(abs(forecast - actual).sum() / d) if d > 0 else float("nan")


def bias_pct(actual: pd.Series, forecast: pd.Series) -> float:
    d = float(actual.sum())
    return float((forecast.sum() - actual.sum()) / d) if d > 0 else float("nan")


def fmt_bias(v: float) -> str:
    return f"{v * 100:+.1f}%" if not np.isnan(v) else "—"


def fmt_wmape(v: float) -> str:
    return f"{v * 100:.1f}%" if not np.isnan(v) else "—"


def save_fig(fig, name: str) -> str:
    """Сохраняет фигуру и возвращает относительный путь для MD."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    path = CHART_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return f"charts/{name}"


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------

def load_forecast(client) -> pd.DataFrame:
    """Lead-1 прогноз: для каждой forecast_date берём лексически максимальный
    non-dev run, чтобы не суммировать несколько бэкфилл-ранов за одну дату."""
    return client.query_df(
        """
        with latest_run as (
            select forecast_date, max(source_run_id) as run_id
            from sku_forecast_day_snapshots
            where lead_days = 1
              and toInt64(bakery_id) in %(pilot)s
              and forecast_date between %(start)s and %(end)s
              and source_run_id not like 'dev_%%'
            group by forecast_date
        )
        select
            s.forecast_date,
            toInt64(s.bakery_id)  as bakery_id,
            toInt64(s.product_id) as product_id,
            any(s.product_name)   as product_name,
            any(s.category_name)  as category_name,
            any(s.source_run_id)  as source_run_id,
            sum(s.forecast_qty)   as forecast_qty
        from sku_forecast_day_snapshots s
        inner join latest_run l
            on s.forecast_date = l.forecast_date
           and s.source_run_id = l.run_id
        where s.lead_days = 1
          and toInt64(s.bakery_id) in %(pilot)s
        group by s.forecast_date, bakery_id, product_id
        """,
        parameters={"pilot": PILOT_IDS, "start": START_DATE, "end": END_DATE},
    )


def load_actuals(client) -> pd.DataFrame:
    return client.query_df(
        """
        select
            check_date          as forecast_date,
            toInt64(bakery_id)  as bakery_id,
            toInt64(product_id) as product_id,
            any(product_name)   as product_name,
            any(category_name)  as category_name,
            sum(quantity)       as actual_qty
        from mart_sales_60d
        where toInt64(bakery_id) in %(pilot)s
          and check_date between %(start)s and %(end)s
        group by forecast_date, bakery_id, product_id
        """,
        parameters={"pilot": PILOT_IDS, "start": START_DATE, "end": END_DATE},
    )


def load_weather(client) -> pd.DataFrame:
    try:
        return client.query_df(
            """
            select
                forecast_date,
                any(temp_mean)       as temp_mean,
                any(precipitation)   as precipitation,
                any(is_bad_weather)  as is_bad_weather
            from forecast_day_context_embedded
            where forecast_date between %(start)s and %(end)s
            group by forecast_date
            """,
            parameters={"start": START_DATE, "end": END_DATE},
        )
    except Exception:
        return pd.DataFrame()


def load_bakery_day_actuals(client) -> pd.DataFrame:
    """Агрегат по пекарне-дню из факта (для timeline)."""
    return client.query_df(
        """
        select
            check_date         as forecast_date,
            toInt64(bakery_id) as bakery_id,
            sum(quantity)      as actual_qty
        from mart_sales_60d
        where toInt64(bakery_id) in %(pilot)s
          and check_date between %(start)s and %(end)s
        group by forecast_date, bakery_id
        """,
        parameters={"pilot": PILOT_IDS, "start": START_DATE, "end": END_DATE},
    )


def load_bakery_day_forecast(client) -> pd.DataFrame:
    """Агрегат bakery-day из SKU-уровня (избегаем проблем с именами колонок в bakery_forecast_day_snapshots)."""
    df = load_forecast(client)
    if df.empty:
        return pd.DataFrame(columns=["forecast_date", "bakery_id", "forecast_qty"])
    return (
        df.groupby(["forecast_date", "bakery_id"], as_index=False)
        .agg(forecast_qty=("forecast_qty", "sum"))
    )


# ---------------------------------------------------------------------------
# Мёрдж
# ---------------------------------------------------------------------------

def build_merged(forecast_df: pd.DataFrame, actual_df: pd.DataFrame) -> pd.DataFrame:
    key = ["forecast_date", "bakery_id", "product_id"]
    merged = actual_df.merge(
        forecast_df[key + ["product_name", "category_name", "source_run_id", "forecast_qty"]],
        on=key, how="outer", suffixes=("_act", "_fc"),
    )
    for col in ["product_name", "category_name"]:
        a, b = f"{col}_act", f"{col}_fc"
        if a in merged.columns:
            merged[col] = merged[a].combine_first(merged[b])
            merged.drop(columns=[a, b], inplace=True, errors="ignore")
    merged["actual_qty"]   = merged["actual_qty"].fillna(0.0)
    merged["forecast_qty"] = merged["forecast_qty"].fillna(0.0)
    merged["source_run_id"] = merged.get("source_run_id", pd.Series(dtype=str)).fillna("—")
    merged["forecast_date"] = pd.to_datetime(merged["forecast_date"])
    return merged


def build_day_merged(bak_fc: pd.DataFrame, bak_act: pd.DataFrame) -> pd.DataFrame:
    key = ["forecast_date", "bakery_id"]
    m = bak_act.merge(bak_fc, on=key, how="outer")
    m["actual_qty"]   = m["actual_qty"].fillna(0.0)
    m["forecast_qty"] = m["forecast_qty"].fillna(0.0)
    m["forecast_date"] = pd.to_datetime(m["forecast_date"])
    return m


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------

COLORS = {
    "fact":     "#2196F3",
    "forecast": "#FF9800",
    "bias":     "#F44336",
    "neutral":  "#9E9E9E",
    "good":     "#4CAF50",
}


def chart_daily_totals(day_merged: pd.DataFrame, weather: pd.DataFrame) -> str:
    """График суточных итогов по пилоту (факт vs прогноз)."""
    agg = day_merged.groupby("forecast_date").agg(
        actual=("actual_qty", "sum"),
        forecast=("forecast_qty", "sum"),
    ).reset_index()
    agg["bias_pct"] = (agg["forecast"] - agg["actual"]) / agg["actual"].clip(lower=1) * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(agg["forecast_date"], agg["actual"],   "o-", color=COLORS["fact"],     label="Факт",     lw=2)
    ax1.plot(agg["forecast_date"], agg["forecast"], "s--", color=COLORS["forecast"], label="Прогноз", lw=2)
    ax1.fill_between(agg["forecast_date"], agg["actual"], agg["forecast"],
                     alpha=0.12, color=COLORS["bias"])

    # Отметить плохую погоду
    if not weather.empty and "is_bad_weather" in weather.columns:
        w = weather.copy()
        w["forecast_date"] = pd.to_datetime(w["forecast_date"])
        bad = w[w["is_bad_weather"] == 1]["forecast_date"]
        for d in bad:
            ax1.axvline(d, color="#9C27B0", alpha=0.3, lw=6, label=None)
        if len(bad):
            ax1.axvline(bad.iloc[0], color="#9C27B0", alpha=0.3, lw=6, label="Плохая погода")

    ax1.set_ylabel("Продажи (шт)")
    ax1.set_title(f"Пилотные пекарни — суточные итоги ({START_DATE} – {END_DATE})")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Bias-панель
    colors_bar = [COLORS["bias"] if b > 0 else COLORS["good"] for b in agg["bias_pct"]]
    ax2.bar(agg["forecast_date"], agg["bias_pct"], color=colors_bar, alpha=0.8, width=0.7)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("Bias%")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax2.xaxis.set_major_locator(mdates.DayLocator())
    fig.autofmt_xdate(rotation=45)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    return save_fig(fig, "daily_totals.png")


def chart_bakery_summary(day_merged: pd.DataFrame) -> str:
    """Горизонтальный бар: bias% по каждой пекарне за период."""
    agg = day_merged.groupby("bakery_id").agg(
        actual=("actual_qty", "sum"),
        forecast=("forecast_qty", "sum"),
    ).reset_index()
    agg["bias_pct"] = (agg["forecast"] - agg["actual"]) / agg["actual"].clip(lower=1) * 100
    agg["label"] = agg["bakery_id"].map(lambda x: f"{BAKERY_NAMES.get(x, str(x))} ({x})")
    agg = agg.sort_values("bias_pct")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [COLORS["bias"] if v > 0 else COLORS["good"] for v in agg["bias_pct"]]
    bars = ax.barh(agg["label"], agg["bias_pct"], color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Bias% (прогноз − факт) / факт")
    ax.set_title("Bias% по пекарням за весь период")

    for bar, val in zip(bars, agg["bias_pct"]):
        ax.text(val + (0.5 if val >= 0 else -0.5), bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}%", va="center", ha="left" if val >= 0 else "right", fontsize=9)

    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return save_fig(fig, "bakery_bias.png")


def chart_dow_bias(merged: pd.DataFrame) -> str:
    """Bias% по дням недели (агрегат по пилоту)."""
    m = merged.copy()
    m["dow"] = m["forecast_date"].dt.dayofweek
    agg = m.groupby("dow").agg(actual=("actual_qty", "sum"), forecast=("forecast_qty", "sum")).reset_index()
    agg["bias_pct"] = (agg["forecast"] - agg["actual"]) / agg["actual"].clip(lower=1) * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [COLORS["bias"] if v > 0 else COLORS["good"] for v in agg["bias_pct"]]
    ax.bar([DOW_RU[d] for d in agg["dow"]], agg["bias_pct"], color=colors, alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Bias%")
    ax.set_title("Bias% по дням недели")
    ax.grid(alpha=0.3, axis="y")

    for i, (d, v) in enumerate(zip(agg["dow"], agg["bias_pct"])):
        ax.text(i, v + (0.3 if v >= 0 else -0.3), f"{v:+.1f}%", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)

    fig.tight_layout()
    return save_fig(fig, "dow_bias.png")


def chart_top_sku(merged: pd.DataFrame) -> str:
    """Топ-N SKU по abs ошибке: горизонтальный бар с bias%."""
    sku = merged.groupby(["product_id", "product_name", "category_name"]).agg(
        actual=("actual_qty", "sum"),
        forecast=("forecast_qty", "sum"),
    ).reset_index()
    sku["abs_err"] = abs(sku["forecast"] - sku["actual"])
    sku["bias_pct"] = (sku["forecast"] - sku["actual"]) / sku["actual"].clip(lower=1) * 100
    top = sku.nlargest(TOP_N_SKU, "abs_err").sort_values("bias_pct")

    labels = top["product_name"].str[:35].tolist()
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = [COLORS["bias"] if v > 0 else COLORS["good"] for v in top["bias_pct"]]
    ax.barh(labels, top["bias_pct"], color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Bias%")
    ax.set_title(f"Топ-{TOP_N_SKU} SKU по абсолютной ошибке — Bias%")
    ax.grid(alpha=0.3, axis="x")

    for i, (v, a) in enumerate(zip(top["bias_pct"], top["actual"])):
        ax.text(v + (0.5 if v >= 0 else -0.5), i,
                f"{v:+.1f}%  (факт {a:.0f})", va="center",
                ha="left" if v >= 0 else "right", fontsize=8)

    fig.tight_layout()
    return save_fig(fig, "top_sku_bias.png")


def chart_bakery_timelines(day_merged: pd.DataFrame) -> str:
    """Подграфики по каждой пекарне: факт vs прогноз по дням."""
    ids = sorted(day_merged["bakery_id"].unique())
    n = len(ids)
    ncols = 2
    nrows = (n + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3), sharey=False)
    axes = axes.flatten()

    for i, bid in enumerate(ids):
        ax = axes[i]
        sub = day_merged[day_merged["bakery_id"] == bid].sort_values("forecast_date")
        if sub.empty:
            ax.set_visible(False)
            continue
        ax.plot(sub["forecast_date"], sub["actual_qty"],   "o-", color=COLORS["fact"],
                label="Факт", lw=1.5, ms=4)
        ax.plot(sub["forecast_date"], sub["forecast_qty"], "s--", color=COLORS["forecast"],
                label="Прогноз", lw=1.5, ms=4)
        ax.set_title(f"{BAKERY_NAMES.get(bid, bid)} ({bid})", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax.tick_params(axis="x", labelsize=7, rotation=45)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Факт vs Прогноз по пекарням (пекарня-день)", fontsize=12)
    fig.tight_layout()
    return save_fig(fig, "bakery_timelines.png")


# ---------------------------------------------------------------------------
# Генерация MD
# ---------------------------------------------------------------------------

def explain_overall(bp: float, wm: float, weather: pd.DataFrame, merged: pd.DataFrame) -> str:
    """Автоматическое пояснение по общему bias."""
    lines = []
    bp_pct = bp * 100

    if bp_pct > 15:
        lines.append(
            f"**Систематический overforecast ({bp_pct:+.1f}%).**  "
            "Модель предсказывает больше фактических продаж. "
            "Вероятная причина: лаговые признаки (lag30, roll_mean30) отражают "
            "майские значения при прогнозировании июня/июля — "
            "летний сезонный спад не учитывается полностью. "
            "Новый признак `bakery_sales_lag365` добавлен в модель 2026-07-06 "
            "и заработает с нового рана (2026-07-07)."
        )
    elif bp_pct > 5:
        lines.append(
            f"**Умеренный overforecast ({bp_pct:+.1f}%).** "
            "Модель слегка завышает объёмы. Это допустимо (лучше выпечь лишнее, "
            "чем потерять продажу), но стоит следить за динамикой."
        )
    elif bp_pct < -10:
        lines.append(
            f"**Underforecast ({bp_pct:+.1f}%).** "
            "Модель недооценивает спрос — риск упущенных продаж."
        )
    else:
        lines.append(
            f"**Bias в норме ({bp_pct:+.1f}%).** Прогноз близок к факту."
        )

    # Погода
    if not weather.empty and "is_bad_weather" in weather.columns:
        bad_days = int(weather["is_bad_weather"].sum())
        if bad_days > 0:
            lines.append(
                f"За период было **{bad_days} дней с плохой погодой** "
                "(отмечены фиолетовыми полосами на графике). "
                "В эти дни возможен дополнительный spikes или drops продаж."
            )

    return "\n\n".join(lines)


def explain_dow(merged: pd.DataFrame) -> str:
    """Пояснение по DOW-паттерну."""
    m = merged.copy()
    m["dow"] = m["forecast_date"].dt.dayofweek
    agg = m.groupby("dow").agg(actual=("actual_qty", "sum"), forecast=("forecast_qty", "sum"))
    agg["bias"] = (agg["forecast"] - agg["actual"]) / agg["actual"].clip(lower=1) * 100

    worst_over  = agg["bias"].idxmax()
    worst_under = agg["bias"].idxmin()

    lines = []
    if agg.loc[worst_over, "bias"] > 10:
        lines.append(
            f"Наибольший overforecast в **{DOW_RU[worst_over]}** "
            f"({agg.loc[worst_over, 'bias']:+.1f}%). "
            "Вероятно, модель не успевает адаптироваться к спаду спроса в этот день недели."
        )
    if agg.loc[worst_under, "bias"] < -5:
        lines.append(
            f"Underforecast в **{DOW_RU[worst_under]}** "
            f"({agg.loc[worst_under, 'bias']:+.1f}%)."
        )

    # Weekend vs weekday
    weekend_bias = bias_pct(
        m[m["dow"] >= 5]["actual_qty"], m[m["dow"] >= 5]["forecast_qty"]
    ) * 100
    weekday_bias = bias_pct(
        m[m["dow"] < 5]["actual_qty"], m[m["dow"] < 5]["forecast_qty"]
    ) * 100
    lines.append(
        f"Выходные (Сб–Вс): bias **{weekend_bias:+.1f}%** | "
        f"Будни: bias **{weekday_bias:+.1f}%**."
    )

    return "\n\n".join(lines)


def explain_top_sku(merged: pd.DataFrame) -> str:
    """Топ проблемных SKU с пояснением."""
    sku = merged.groupby(["product_id", "product_name", "category_name"]).agg(
        actual=("actual_qty", "sum"),
        forecast=("forecast_qty", "sum"),
    ).reset_index()
    sku["abs_err"] = abs(sku["forecast"] - sku["actual"])
    sku["bias_pct"] = (sku["forecast"] - sku["actual"]) / sku["actual"].clip(lower=1) * 100
    top = sku.nlargest(5, "abs_err")

    lines = ["| SKU | Факт | Прогноз | Bias% | Категория |",
             "|---|---:|---:|---:|---|"]
    for _, r in top.iterrows():
        lines.append(
            f"| {r['product_name'][:40]} | {r['actual']:.0f} | {r['forecast']:.0f} "
            f"| {r['bias_pct']:+.1f}% | {r['category_name']} |"
        )

    return "\n".join(lines)


def explain_bakeries(day_merged: pd.DataFrame) -> str:
    """Пояснение по пекарням с наибольшим расхождением."""
    agg = day_merged.groupby("bakery_id").agg(
        actual=("actual_qty", "sum"),
        forecast=("forecast_qty", "sum"),
    ).reset_index()
    agg["bias_pct"] = (agg["forecast"] - agg["actual"]) / agg["actual"].clip(lower=1) * 100

    lines = ["| Пекарня | Факт | Прогноз | Bias% |",
             "|---|---:|---:|---:|"]
    for _, r in agg.sort_values("bias_pct", ascending=False).iterrows():
        name = BAKERY_NAMES.get(r["bakery_id"], str(int(r["bakery_id"])))
        lines.append(
            f"| {name} ({int(r['bakery_id'])}) | {r['actual']:.0f} | {r['forecast']:.0f} "
            f"| {r['bias_pct']:+.1f}% |"
        )

    worst_over  = agg.loc[agg["bias_pct"].idxmax()]
    worst_under = agg.loc[agg["bias_pct"].idxmin()]

    commentary = []
    if worst_over["bias_pct"] > 20:
        n = BAKERY_NAMES.get(worst_over["bakery_id"], str(int(worst_over["bakery_id"])))
        commentary.append(
            f"**{n}** ({int(worst_over['bakery_id'])}) — наибольший overforecast "
            f"({worst_over['bias_pct']:+.1f}%). Возможные причины: смена графика работы, "
            "временное снижение трафика, перебои с ассортиментом."
        )
    if worst_under["bias_pct"] < -10:
        n = BAKERY_NAMES.get(worst_under["bakery_id"], str(int(worst_under["bakery_id"])))
        commentary.append(
            f"**{n}** ({int(worst_under['bakery_id'])}) — underforecast "
            f"({worst_under['bias_pct']:+.1f}%). Риск упущенных продаж."
        )

    table = "\n".join(lines)
    comment = "\n\n".join(commentary)
    return f"{table}\n\n{comment}" if comment else table


def build_availability_table(forecast_df: pd.DataFrame, actual_df: pd.DataFrame) -> str:
    fc_dates  = set(forecast_df["forecast_date"].astype(str).unique())
    act_dates = set(actual_df["forecast_date"].astype(str).unique())
    all_dates = sorted(fc_dates | act_dates)

    lines = ["| Дата | День | Прог. пекарен | Факт пекарен | Run ID |",
             "|---|---|---:|---:|---|"]
    for d in all_dates:
        dt = pd.to_datetime(d)
        dow = DOW_RU[dt.dayofweek]
        fp = forecast_df[forecast_df["forecast_date"].astype(str) == d]
        ap = actual_df[actual_df["forecast_date"].astype(str) == d]
        fc_bak  = fp["bakery_id"].nunique()
        act_bak = ap["bakery_id"].nunique()
        runs = fp["source_run_id"].unique().tolist()
        run_label = runs[0][:45] if runs else "—"
        if len(runs) > 1:
            run_label += f" (+{len(runs)-1})"
        lines.append(f"| {d} | {dow} | {fc_bak} | {act_bak} | `{run_label}` |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Пилот: {PILOT_IDS}")
    print(f"Период: {START_DATE} – {END_DATE}")
    print("Подключаемся к ClickHouse...")

    client = connect()

    print("Загружаем SKU-уровень прогноза...")
    fc_df  = load_forecast(client)
    print(f"  прогноз: {len(fc_df):,} строк")

    print("Загружаем факт...")
    act_df = load_actuals(client)
    print(f"  факт: {len(act_df):,} строк")

    print("Загружаем bakery-day прогноз...")
    bak_fc  = load_bakery_day_forecast(client)
    print(f"  bakery-day прогноз: {len(bak_fc):,} строк")

    print("Загружаем bakery-day факт...")
    bak_act = load_bakery_day_actuals(client)
    print(f"  bakery-day факт: {len(bak_act):,} строк")

    print("Загружаем погоду...")
    weather = load_weather(client)
    print(f"  погода: {len(weather):,} строк")

    merged    = build_merged(fc_df, act_df)
    day_merge = build_day_merged(bak_fc, bak_act)

    # Общие метрики
    overall_bias  = bias_pct(merged["actual_qty"], merged["forecast_qty"])
    overall_wmape = wmape(merged["actual_qty"], merged["forecast_qty"])
    total_actual  = merged["actual_qty"].sum()
    total_fc      = merged["forecast_qty"].sum()

    is_pie    = merged["category_name"].str.lower().str.contains(PIE_PATTERN, na=False)
    pie_bias  = bias_pct(merged[is_pie]["actual_qty"], merged[is_pie]["forecast_qty"])
    pie_wmape = wmape(merged[is_pie]["actual_qty"],   merged[is_pie]["forecast_qty"])

    # SKU по объёму факта — топ для ходовых
    sku_actual = merged.groupby("product_id")["actual_qty"].sum()
    hodovye_ids = set(sku_actual.nlargest(TOP_N_SKU).index)
    hod_mask  = merged["product_id"].isin(hodovye_ids)
    hod_bias  = bias_pct(merged[hod_mask]["actual_qty"], merged[hod_mask]["forecast_qty"])
    hod_wmape = wmape(merged[hod_mask]["actual_qty"],    merged[hod_mask]["forecast_qty"])

    print("Строим графики...")
    path_daily    = chart_daily_totals(day_merge, weather)
    path_bakeries = chart_bakery_summary(day_merge)
    path_dow      = chart_dow_bias(merged)
    path_sku      = chart_top_sku(merged)
    path_timelines = chart_bakery_timelines(day_merge)
    print(f"  Графики сохранены в {CHART_DIR}")

    # ---------------------------------------------------------------------------
    # MD
    # ---------------------------------------------------------------------------
    avail_table = build_availability_table(fc_df, act_df)
    expl_overall = explain_overall(overall_bias, overall_wmape, weather, merged)
    expl_dow     = explain_dow(merged)
    top_sku_table = explain_top_sku(merged)
    bakery_table  = explain_bakeries(day_merge)

    md = f"""# Отчёт по пилотным пекарням — 2 недели

**Период:** {START_DATE} – {END_DATE}
**Пекарни:** {', '.join(str(x) for x in PILOT_IDS)} ({len(PILOT_IDS)} штук)
**Уровень прогноза:** lead-1 (прогноз сделан за 1 день до факта)
**Дата формирования отчёта:** {date.today()}

---

## Исполнительное резюме

| Метрика | Значение |
|---|---|
| Суммарный факт | **{total_actual:,.0f} шт** |
| Суммарный прогноз | **{total_fc:,.0f} шт** |
| Bias% (все SKU) | **{fmt_bias(overall_bias)}** |
| wMAPE% (все SKU) | **{fmt_wmape(overall_wmape)}** |
| Bias% (ходовые топ-{TOP_N_SKU}) | **{fmt_bias(hod_bias)}** |
| wMAPE% (ходовые топ-{TOP_N_SKU}) | **{fmt_wmape(hod_wmape)}** |
| Bias% (пироги сытные) | **{fmt_bias(pie_bias)}** |
| wMAPE% (пироги сытные) | **{fmt_wmape(pie_wmape)}** |

> **Bias% = (прогноз − факт) / факт.** Положительный = overforecast (прогноз завышен).
> **wMAPE% = |прогноз − факт| / факт** (взвешенная по объёму абсолютная ошибка).

---

## 1. Динамика по дням

![Суточные итоги]({path_daily})

### Пояснение

{expl_overall}

---

## 2. По пекарням

### 2.1 Bias% за весь период

![Bias по пекарням]({path_bakeries})

{bakery_table}

### 2.2 Динамика по каждой пекарне

![Пекарни по дням]({path_timelines})

---

## 3. Анализ по дням недели

![Bias по DOW]({path_dow})

### Пояснение

{expl_dow}

---

## 4. Топ-{TOP_N_SKU} SKU по абсолютной ошибке

![Топ SKU]({path_sku})

### Топ-5 SKU с наибольшим расхождением

{top_sku_table}

---

## 5. Доступность данных по датам

{avail_table}

---

## 6. Контекст и причины расхождений

### 6.1 Сезонный летний спад

Исторически июнь–июль показывают снижение продаж относительно мая:

| Переход | 2025 | 2026 |
|---|---|---|
| Май → Июнь | −4.6% | −2.2% |
| Июнь → Июль | −2.5% | −6.1% |

Модель использует `lag30` и `roll_mean30`, которые отражают предыдущий месяц —
при переходе с мая в июнь они несут более высокие значения и модель завышает прогноз.

**Исправление:** признак `bakery_sales_lag365` добавлен в production-модель 06.07.2026
и даёт модели YoY-якорь на аналогичный период прошлого года.
Первый ран с новым признаком: **07.07.2026** (нейтли-таймер 03:30 UTC).

### 6.2 Коррекция recent (city-prior)

Активный режим recent-коррекции: `runner_city_prior_soft_weekpart`, окно 30 дней.
Коррекция сглаживает систематический сдвиг по пекарне, но не устраняет сезонный тренд.

### 6.3 Ассортиментный эффект

Ассортимент выпекания (`bakeable_products`) пересчитывается еженедельно из `mart_sales_60d`.
Если пекарня временно не продаёт позицию (технические причины, отсутствие ингредиентов),
эта позиция может выпасть из ассортимента — прогноз по ней обнуляется, но факт ≠ 0,
что даёт локальный underforecast.

### 6.4 Сценарий `base_no_sku_uplift`

Активный сценарий с 01.07.2026:
- Базовая bakery-day модель (нет bakery-level uplift)
- Сырые SKU-часовые профили (без floor-uplift множителя)

Floor-uplift был отключён 01.07.2026 т.к. не удалось найти надёжный сигнал
для разделения стокаута и низкого спроса.

---

*Отчёт сформирован автоматически скриптом `build_pilot_2w_report.py`*
"""

    report_path = OUT_DIR / "report.md"
    report_path.write_text(md, encoding="utf-8")
    print(f"\nОтчёт сохранён: {report_path}")
    print(f"Размер: {report_path.stat().st_size:,} байт")


if __name__ == "__main__":
    main()

"""
Анализ упущенных продаж по пилотным пекарням.
Период: 2026-06-23 -- 2026-07-06

Упущенная продажа (missed sale) = все три условия одновременно:
  1. forecast_qty > 0   -- модель ожидала продажи
  2. actual_qty   = 0   -- фактических продаж не было
  3. bakery_active = 1  -- пекарня в этот час торговала (другие SKU продавались)

Признак 3 отличает стокаут от "пекарня закрыта / ещё не открылась".

Выход:
  outputs/pilot_missed_sales/report.md
  outputs/pilot_missed_sales/charts/*.png
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

import clickhouse_connect
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# Минимальный порог прогноза, чтобы считать "ожидали продажу"
MIN_FORECAST_QTY = 0.5
# Топ-N SKU×пекарня для детальных графиков
TOP_N_CHARTS = 12

WORK_HOURS = list(range(6, 22))   # рабочие часы пекарни

OUT_DIR   = Path("outputs/pilot_missed_sales")
CHART_DIR = OUT_DIR / "charts"
DOW_RU    = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]


# ---------------------------------------------------------------------------
# ClickHouse
# ---------------------------------------------------------------------------

def connect():
    return clickhouse_connect.get_client(
        host=HOST, port=PORT, username=USER, password=PASSWORD,
        database=DATABASE, secure=True, verify=False,
    )


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------

def load_hourly_forecast(client) -> pd.DataFrame:
    """
    Lead-1 почасовой прогноз из sku_forecast_hour_snapshots.
    product_name/category_name подтягиваем из sku_forecast_day_snapshots.
    """
    print("  Загружаем почасовой прогноз (sku_forecast_hour_snapshots)...")
    bakeries = ",".join(str(x) for x in PILOT_IDS)
    df = client.query_df(
        f"""
        with latest_run as (
            select forecast_date, max(source_run_id) as run_id
            from sku_forecast_hour_snapshots
            where lead_days = 1
              and toInt64(bakery_id) in ({bakeries})
              and forecast_date between %(start)s and %(end)s
              and source_run_id not like 'dev_%%'
            group by forecast_date
        )
        select
            s.forecast_date,
            toInt64(s.bakery_id)  as bakery_id,
            toInt64(s.product_id) as product_id,
            s.hour                as hour,
            any(s.source_run_id)  as source_run_id,
            sum(s.forecast_qty)   as forecast_qty
        from sku_forecast_hour_snapshots s
        inner join latest_run l
            on s.forecast_date = l.forecast_date
           and s.source_run_id = l.run_id
        where s.lead_days = 1
          and toInt64(s.bakery_id) in ({bakeries})
        group by s.forecast_date, bakery_id, product_id, hour
        """,
        parameters={"start": START_DATE, "end": END_DATE},
    )
    print(f"    {len(df):,} строк")

    # Подтягиваем product_name / category_name из дневных снапшотов
    print("  Загружаем метаданные SKU (sku_forecast_day_snapshots)...")
    meta = client.query_df(
        f"""
        with latest_run as (
            select forecast_date, max(source_run_id) as run_id
            from sku_forecast_day_snapshots
            where lead_days = 1
              and toInt64(bakery_id) in ({bakeries})
              and forecast_date between %(start)s and %(end)s
              and source_run_id not like 'dev_%%'
            group by forecast_date
        )
        select
            toInt64(s.bakery_id)  as bakery_id,
            toInt64(s.product_id) as product_id,
            any(s.product_name)   as product_name,
            any(s.category_name)  as category_name
        from sku_forecast_day_snapshots s
        inner join latest_run l
            on s.forecast_date = l.forecast_date
           and s.source_run_id = l.run_id
        where s.lead_days = 1
          and toInt64(s.bakery_id) in ({bakeries})
        group by bakery_id, product_id
        """,
        parameters={"start": START_DATE, "end": END_DATE},
    )
    print(f"    метаданных: {len(meta):,} строк")

    df = df.merge(meta, on=["bakery_id", "product_id"], how="left")
    df["product_name"]  = df["product_name"].fillna("—")
    df["category_name"] = df["category_name"].fillna("—")
    return df


def load_hourly_actuals(client) -> pd.DataFrame:
    """
    Почасовые фактические продажи из mart_sales_60d.
    check_datetime содержит время транзакции.
    """
    print("  Загружаем почасовой факт (mart_sales_60d)...")
    bakeries = ",".join(str(x) for x in PILOT_IDS)
    df = client.query_df(
        f"""
        select
            check_date              as forecast_date,
            toInt64(bakery_id)      as bakery_id,
            toInt64(product_id)     as product_id,
            any(product_name)       as product_name,
            any(category_name)      as category_name,
            toHour(check_datetime)  as hour,
            sum(quantity)           as actual_qty
        from mart_sales_60d
        where toInt64(bakery_id) in ({bakeries})
          and check_date between %(start)s and %(end)s
        group by forecast_date, bakery_id, product_id, hour
        """,
        parameters={"start": START_DATE, "end": END_DATE},
    )
    print(f"    {len(df):,} строк")
    return df


def load_sku_hour_profile(client) -> pd.DataFrame:
    """
    Продовый профиль из sku_hour_share_profile_smoothed_embedded.
    mean_sku_share_in_hour_norm — нормированная доля часа в суточном объёме SKU.
    Берём только пилотные пекарни, усредняем по DOW (dow=-1 = глобальный).
    """
    print("  Загружаем продовый профиль (sku_hour_share_profile_smoothed_embedded)...")
    bakeries = ",".join(str(x) for x in PILOT_IDS)
    df = client.query_df(
        f"""
        select
            toInt64(bakery_id)           as bakery_id,
            toInt64(product_id)          as product_id,
            hour,
            avg(mean_sku_share_in_hour_norm) as share_norm
        from sku_hour_share_profile_smoothed_embedded
        where toInt64(bakery_id) in ({bakeries})
        group by bakery_id, product_id, hour
        """
    )
    print(f"    {len(df):,} строк")
    return df


def compute_bakery_hourly_activity(actuals: pd.DataFrame) -> pd.DataFrame:
    """
    Для каждой (forecast_date, bakery_id, hour):
    пекарня_активна = продавала хоть что-нибудь (суммарный факт > 0).
    """
    bak_act = (
        actuals.groupby(["forecast_date", "bakery_id", "hour"], as_index=False)
        .agg(bakery_total_qty=("actual_qty", "sum"))
    )
    bak_act["bakery_active"] = (bak_act["bakery_total_qty"] > 0).astype(int)
    return bak_act[["forecast_date", "bakery_id", "hour", "bakery_active"]]


# ---------------------------------------------------------------------------
# Сборка missed-sales датафрейма
# ---------------------------------------------------------------------------

def build_missed_df(
    fc: pd.DataFrame,
    act: pd.DataFrame,
    bak_activity: pd.DataFrame,
) -> pd.DataFrame:
    """
    Полный join прогноза и факта, потом определяем missed.
    """
    fc = fc.copy()
    act = act.copy()

    fc["forecast_date"]  = pd.to_datetime(fc["forecast_date"])
    act["forecast_date"] = pd.to_datetime(act["forecast_date"])
    bak_activity["forecast_date"] = pd.to_datetime(bak_activity["forecast_date"])

    key = ["forecast_date", "bakery_id", "product_id", "hour"]

    merged = fc.merge(
        act[key + ["actual_qty"]],
        on=key,
        how="left",
    )
    merged["actual_qty"]   = merged["actual_qty"].fillna(0.0)
    # product_name / category_name уже в fc после join с метаданными
    if "product_name" not in merged.columns:
        merged["product_name"] = "—"
    if "category_name" not in merged.columns:
        merged["category_name"] = "—"
    merged["product_name"]  = merged["product_name"].fillna("—")
    merged["category_name"] = merged["category_name"].fillna("—")

    # Подтягиваем активность пекарни
    merged = merged.merge(
        bak_activity,
        on=["forecast_date", "bakery_id", "hour"],
        how="left",
    )
    merged["bakery_active"] = merged["bakery_active"].fillna(0).astype(int)

    # Флаг missed sale
    merged["missed"] = (
        (merged["forecast_qty"] >= MIN_FORECAST_QTY) &
        (merged["actual_qty"]   == 0) &
        (merged["bakery_active"] == 1)
    ).astype(int)

    # Добавляем DOW
    merged["dow"] = merged["forecast_date"].dt.dayofweek

    return merged


# ---------------------------------------------------------------------------
# Статистика
# ---------------------------------------------------------------------------

def agg_missed_by_sku(df: pd.DataFrame) -> pd.DataFrame:
    """Агрегат упущенных продаж по (bakery, SKU)."""
    missed = df[df["missed"] == 1]
    grp = (
        missed.groupby(["bakery_id", "product_id", "product_name", "category_name"],
                       as_index=False)
        .agg(
            n_missed_hours=("hour",          "count"),
            n_missed_days =("forecast_date", "nunique"),
            missed_fc_qty =("forecast_qty",  "sum"),
        )
        .sort_values("missed_fc_qty", ascending=False)
    )
    return grp


def agg_missed_by_day(df: pd.DataFrame) -> pd.DataFrame:
    """Упущенные продажи по дням — для calendar view."""
    missed = df[df["missed"] == 1]
    return (
        missed.groupby(["forecast_date", "bakery_id"], as_index=False)
        .agg(n_missed_sku=("product_id", "nunique"),
             missed_fc_qty=("forecast_qty", "sum"))
        .sort_values("forecast_date")
    )


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------

COLORS = {
    "forecast": "#FF9800",
    "actual":   "#2196F3",
    "missed":   "#F44336",
    "active":   "#E8F5E9",
}


def save_fig(fig, name: str) -> str:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    path = CHART_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return f"charts/{name}"


def chart_sku_profile(
    df: pd.DataFrame,
    profile_df: pd.DataFrame,
    bakery_id: int,
    product_id: int,
    product_name: str,
    category_name: str,
) -> str:
    """
    Почасовой профиль одного SKU в одной пекарне.

    Три слоя:
    - 🟠 Продовый профиль (sku_hour_share_profile_smoothed_embedded, 8+ мес)
      — "норма": как SKU ведёт себя в этой пекарне исторически.
      Пересчитывается в абсолютные шт через средний дневной факт.
    - 🔵 Факт 14 дней — реальные продажи за анализируемый период.
    - 🔴 Красная зона — часы с упущенными продажами (≥20% дней).

    Расхождение профиля (🟠) и факта (🔵) в красные часы = аномалия,
    объяснимая стокаутом или ассортиментным пропуском.
    """
    sub = df[(df["bakery_id"] == bakery_id) & (df["product_id"] == product_id)].copy()
    sub = sub[sub["hour"].isin(WORK_HOURS)]

    # --- Факт: среднее по часу за 14 дней ---
    hour_agg = sub.groupby("hour").agg(
        act_mean  =("actual_qty",  "mean"),
        missed_pct=("missed",       "mean"),
        active_pct=("bakery_active","mean"),
    ).reindex(WORK_HOURS, fill_value=0.0)

    # --- Продовый профиль → абсолютные шт ---
    prof = profile_df[
        (profile_df["bakery_id"] == bakery_id) &
        (profile_df["product_id"] == product_id)
    ].set_index("hour")["share_norm"].reindex(WORK_HOURS, fill_value=0.0)

    # Средний дневной факт (только дни, где хоть что-то продалось)
    daily_actual = (
        sub[sub["actual_qty"] > 0]
        .groupby("forecast_date")["actual_qty"].sum()
    )
    mean_day_actual = daily_actual.mean() if len(daily_actual) else 0.0

    # Ожидаемое кол-во в час по профилю
    prof_abs = prof * mean_day_actual

    hours = WORK_HOURS
    x = np.arange(len(hours))

    # --- Матрица missed: дата × час ---
    all_dates = sorted(sub["forecast_date"].unique())
    date_labels = [pd.Timestamp(d).strftime("%d.%m") for d in all_dates]
    missed_matrix = np.zeros((len(all_dates), len(hours)), dtype=float)
    for di, d in enumerate(all_dates):
        day = sub[sub["forecast_date"] == d].set_index("hour")
        for hi, h in enumerate(hours):
            if h in day.index:
                missed_matrix[di, hi] = float(day.loc[h, "missed"])

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(13, 7),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # ── Верхний: профиль ────────────────────────────────────────────────────
    for i, h in enumerate(hours):
        if hour_agg.loc[h, "active_pct"] >= 0.3:
            ax.axvspan(i - 0.5, i + 0.5, color=COLORS["active"], alpha=0.3, zorder=0)
    for i, h in enumerate(hours):
        if hour_agg.loc[h, "missed_pct"] >= 0.2:
            ax.axvspan(i - 0.5, i + 0.5, color=COLORS["missed"], alpha=0.12, zorder=1)

    ax.bar(x, hour_agg["act_mean"], color=COLORS["actual"],
           alpha=0.75, width=0.7, zorder=2, label="Факт (14 дней, среднее)")
    ax.plot(x, [prof_abs[h] for h in hours], "o-", color=COLORS["forecast"],
            lw=2.5, ms=6, zorder=3,
            label="Ожидаемый профиль (8+ мес, sku_hour_share_profile)")

    for i, h in enumerate(hours):
        mp = hour_agg.loc[h, "missed_pct"]
        if mp >= 0.2:
            ax.annotate(
                f"{mp:.0%}\nмissed",
                xy=(i, max(prof_abs[h], hour_agg.loc[h, "act_mean"]) + 0.3),
                ha="center", fontsize=7, color=COLORS["missed"], fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}:00" for h in hours], rotation=45, ha="right")
    ax.set_ylabel("Кол-во шт")
    bname = BAKERY_NAMES.get(bakery_id, str(bakery_id))
    ax.set_title(
        f"{product_name[:60]}\n"
        f"Пекарня: {bname} ({bakery_id})  |  {category_name}",
        fontsize=10,
    )
    patches = [
        mpatches.Patch(color=COLORS["actual"],  alpha=0.75,
                       label="Факт (14 дней, среднее по часу)"),
        plt.Line2D([0], [0], color=COLORS["forecast"], lw=2.5, marker="o", ms=6,
                   label="Продовый профиль: ожидаемый объём в час (8+ мес)"),
        mpatches.Patch(color=COLORS["missed"],  alpha=0.25,
                       label="Часы упущенных продаж (≥20% дней)"),
        mpatches.Patch(color=COLORS["active"],  alpha=0.3,
                       label="Пекарня активна (фон)"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, axis="y")

    # ── Нижний: heatmap дата × час ──────────────────────────────────────────
    # Цвет: красный = missed, светло-серый = нет missed (но пекарня активна),
    #        белый = нет данных
    cmap = matplotlib.colors.ListedColormap(["#F5F5F5", "#EF9A9A"])
    ax2.imshow(missed_matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1,
               interpolation="nearest")

    ax2.set_yticks(range(len(all_dates)))
    ax2.set_yticklabels(date_labels, fontsize=7)
    ax2.set_xticks(range(len(hours)))
    ax2.set_xticklabels([f"{h}:00" for h in hours], rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Дата", fontsize=8)
    ax2.set_title("Даты с упущенными продажами (красный = missed)", fontsize=8)

    # Добавляем DOW-метки рядом с датами
    for di, d in enumerate(all_dates):
        dow = DOW_RU[pd.Timestamp(d).dayofweek]
        ax2.text(-0.7, di, dow, ha="right", va="center", fontsize=6.5,
                 color="#555555", transform=ax2.transData)

    fig.tight_layout()
    return save_fig(fig, f"profile_{bakery_id}_{product_id}.png")


def chart_missed_calendar(missed_day: pd.DataFrame) -> str:
    """
    Heatmap: пекарня × дата, цвет = кол-во missed SKU.
    """
    pivot = missed_day.pivot_table(
        index="bakery_id", columns="forecast_date",
        values="n_missed_sku", aggfunc="sum", fill_value=0,
    )
    pivot.index = [f"{BAKERY_NAMES.get(b, b)} ({b})" for b in pivot.index]
    pivot.columns = [str(d.date()) if hasattr(d, "date") else str(d) for d in pivot.columns]

    fig, ax = plt.subplots(figsize=(max(14, len(pivot.columns) * 0.9), max(4, len(pivot) * 0.55)))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=max(pivot.values.max(), 1))

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Числа в ячейках
    for r in range(len(pivot)):
        for c in range(len(pivot.columns)):
            v = int(pivot.values[r, c])
            if v > 0:
                ax.text(c, r, str(v), ha="center", va="center",
                        fontsize=8, color="white" if v >= 3 else "black", fontweight="bold")

    plt.colorbar(im, ax=ax, label="Кол-во SKU с упущенными продажами")
    ax.set_title(f"Упущенные продажи: пекарня × дата ({START_DATE} – {END_DATE})")
    fig.tight_layout()
    return save_fig(fig, "missed_calendar.png")


def chart_missed_summary_bar(agg: pd.DataFrame) -> str:
    """Топ-15 SKU×пекарня по объёму упущенного прогноза."""
    top = agg.head(15).copy()
    top["label"] = top.apply(
        lambda r: f"{r['product_name'][:30]} / {BAKERY_NAMES.get(r['bakery_id'], r['bakery_id'])}",
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(top["label"], top["missed_fc_qty"],
                   color=COLORS["missed"], alpha=0.8)
    ax.set_xlabel("Суммарный прогноз в часы упущенных продаж (шт)")
    ax.set_title(f"Топ-{len(top)} SKU × пекарня по объёму упущенных продаж")
    ax.grid(alpha=0.3, axis="x")

    for bar, (_, r) in zip(bars, top.iterrows()):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{r['missed_fc_qty']:.0f} шт ({r['n_missed_days']}д, {r['n_missed_hours']}ч)",
                va="center", fontsize=8)

    fig.tight_layout()
    return save_fig(fig, "missed_top_sku.png")


def chart_missed_by_hour(df: pd.DataFrame) -> str:
    """В каких часах чаще всего упущенные продажи (по всем пилотным)."""
    missed = df[df["missed"] == 1]
    hour_counts = missed.groupby("hour").agg(
        n_events=("missed", "count"),
        fc_qty   =("forecast_qty", "sum"),
    ).reindex(WORK_HOURS, fill_value=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    x = np.arange(len(WORK_HOURS))

    ax1.bar(x, hour_counts["n_events"], color=COLORS["missed"], alpha=0.8)
    ax1.set_ylabel("Кол-во missed-событий")
    ax1.set_title("Распределение упущенных продаж по часам дня")
    ax1.grid(alpha=0.3, axis="y")

    ax2.bar(x, hour_counts["fc_qty"], color="#9C27B0", alpha=0.7)
    ax2.set_ylabel("Упущенный прогноз (шт)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{h}:00" for h in WORK_HOURS], rotation=45)
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    return save_fig(fig, "missed_by_hour.png")


def chart_missed_by_dow(df: pd.DataFrame) -> str:
    """В какие дни недели больше упущенных продаж."""
    missed = df[df["missed"] == 1].copy()
    dow_counts = missed.groupby("dow").agg(
        n_events=("missed", "count"),
        fc_qty   =("forecast_qty", "sum"),
    ).reindex(range(7), fill_value=0)

    # Нормируем на кол-во дней каждого DOW в периоде (чтобы не было смещения от разного числа дат)
    all_dates = df["forecast_date"].unique()
    dow_day_counts = pd.Series([
        sum(1 for d in all_dates if pd.Timestamp(d).dayofweek == dow)
        for dow in range(7)
    ], index=range(7)).replace(0, np.nan)
    dow_counts["events_per_day"] = dow_counts["n_events"] / dow_day_counts

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [DOW_RU[i] for i in range(7)]
    ax.bar(labels, dow_counts["events_per_day"].fillna(0),
           color=COLORS["missed"], alpha=0.8)
    ax.set_ylabel("Missed-событий / день (нормировано)")
    ax.set_title("Упущенные продажи по дням недели (на 1 день каждого DOW)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return save_fig(fig, "missed_by_dow.png")


# ---------------------------------------------------------------------------
# Генерация MD
# ---------------------------------------------------------------------------

def generate_md(
    df: pd.DataFrame,
    agg_sku: pd.DataFrame,
    agg_day: pd.DataFrame,
    chart_paths: dict,
    profile_charts: list[dict],
) -> str:

    total_missed_hours = int(df["missed"].sum())
    total_missed_days  = int(df[df["missed"] == 1]["forecast_date"].nunique())
    total_missed_qty   = float(df[df["missed"] == 1]["forecast_qty"].sum())
    n_sku_affected     = int(df[df["missed"] == 1]["product_id"].nunique())
    n_bak_affected     = int(df[df["missed"] == 1]["bakery_id"].nunique())

    # Топ-10 таблица
    top10 = agg_sku.head(10)
    sku_table_rows = "\n".join(
        f"| {BAKERY_NAMES.get(int(r.bakery_id), r.bakery_id)} ({int(r.bakery_id)}) "
        f"| {r.product_name[:40]} | {r.category_name} "
        f"| {int(r.n_missed_days)} | {int(r.n_missed_hours)} | {r.missed_fc_qty:.0f} |"
        for _, r in top10.iterrows()
    )

    # Профильные графики
    profile_md_sections = []
    for p in profile_charts:
        bname = BAKERY_NAMES.get(p["bakery_id"], str(p["bakery_id"]))
        missed_hours_str = ", ".join(f"{h}:00" for h in p["missed_hours"])
        section = f"""### {p["product_name"]} — {bname} ({p["bakery_id"]})

**Категория:** {p["category_name"]}
**Дней с упущенными продажами:** {p["n_missed_days"]}
**Часы упущенных продаж:** {missed_hours_str or "—"}
**Суммарный упущенный прогноз:** {p["missed_fc_qty"]:.0f} шт

![Профиль]({p["chart"]})

**Интерпретация:**
{p["explanation"]}

---"""
        profile_md_sections.append(section)

    profile_block = "\n\n".join(profile_md_sections)

    return f"""# Анализ упущенных продаж — пилотные пекарни

**Период:** {START_DATE} – {END_DATE}
**Пекарни:** {', '.join(str(x) for x in PILOT_IDS)}
**Методология:** Упущенная продажа = прогноз ≥ {MIN_FORECAST_QTY} шт + факт = 0 + пекарня активна в этот час

---

## Итоговые цифры

| Метрика | Значение |
|---|---|
| Всего missed-событий (SKU × пекарня × час × день) | **{total_missed_hours:,}** |
| Дней с упущенными продажами | **{total_missed_days}** из 14 |
| Затронуто уникальных SKU | **{n_sku_affected}** |
| Затронуто пекарен | **{n_bak_affected}** из {len(PILOT_IDS)} |
| Суммарный упущенный прогноз | **{total_missed_qty:,.0f} шт** |

---

## Где происходят упущенные продажи

### По дням периода (heatmap)

Число SKU с упущенными продажами в каждой пекарне за каждый день:

![Календарь упущенных продаж]({chart_paths["calendar"]})

### По часам дня

![Упущенные по часам]({chart_paths["by_hour"]})

Верхняя панель — кол-во missed-событий (SKU × пекарня × день). Нижняя — объём прогноза в эти часы.

**Ключевой паттерн:** Упущенные продажи концентрируются в определённых временных интервалах.
Если пик приходится на послеобеденные часы — скорее всего утренняя партия заканчивалась, а следующая ещё не была готова.
Если пик ранний — возможна неправильная оценка объёма первой партии.

### По дням недели

![Упущенные по DOW]({chart_paths["by_dow"]})

---

## Топ SKU × пекарня по объёму упущенных продаж

![Топ SKU]({chart_paths["top_sku"]})

| Пекарня | SKU | Категория | Дней | Часов | Прогноз (шт) |
|---|---|---|---:|---:|---:|
{sku_table_rows}

---

## Детальные профили топ-{len(profile_charts)} случаев

Для каждого случая показан **средний почасовой профиль** за период:
- 🔵 Синие столбцы = фактические продажи (среднее по дням)
- 🟠 Оранжевая линия = прогноз модели (среднее по дням)
- 🔴 Красная подсветка = часы, где упущенные продажи встречались в ≥20% дней
- 🟢 Зелёный фон = пекарня была активна в этот час
- Аннотация `%` = в скольких % дней был missed в этот час

{profile_block}

---

## Причины упущенных продаж

### 1. Стокаут (основная причина)

Пекарня испекла меньше, чем требовалось по спросу. Товар закончился в середине дня,
и последующие часы, в которые модель ожидала продажи, дали ноль.

**Сигналы стокаута в данных:**
- Продажи идут нормально до часа H, затем обнуляются
- Пекарня продолжает торговать другими SKU в те же часы
- На следующий день продажи возобновляются (это не закрытие точки)

### 2. Ассортиментный пропуск

SKU временно убрали из выпечки (нет ингредиентов, технические причины, решение пекарни).
Модель не знает об этом — прогноз ненулевой, факта нет.

**Сигналы:**
- Нули сразу с начала дня (не с середины)
- Другие SKU той же тесто-группы тоже могут быть занулены

### 3. Запоздалый ран (смещение в часах)

Иногда SKU начинает продаваться позже запланированного часа,
потому что партия вышла из печи позже. Это даёт нули в ранние часы при ненулевом прогнозе.

**Сигналы:**
- Нули в начале дня, потом нормальные продажи
- Общий дневной итог близок к прогнозу, только сдвинут

---

## Что делать с этими данными

1. **Стокаут:** увеличить план выпуска в часы с красной подсветкой — модель даёт правильный сигнал спроса, пекарня просто не выпекает достаточно.
2. **Ассортиментный пропуск:** добавить механизм сигнализации о временном снятии SKU (через шаблон комментариев / план выпекания).
3. **Смещение часа:** скорректировать окно выпечки в плане выпекания — если пик реальных продаж стабильно смещён относительно прогноза.

---

*Сформировано: `build_pilot_missed_sales.py`
Порог missed: forecast ≥ {MIN_FORECAST_QTY} шт, actual = 0, пекарня активна*
"""


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

    fc_hourly   = load_hourly_forecast(client)
    act_hourly  = load_hourly_actuals(client)
    profile_df  = load_sku_hour_profile(client)

    print("  Считаем активность пекарни по часам...")
    bak_activity = compute_bakery_hourly_activity(act_hourly)

    print("  Собираем merged датафрейм...")
    df = build_missed_df(fc_hourly, act_hourly, bak_activity)
    print(f"    итого строк: {len(df):,}, missed: {df['missed'].sum():,}")

    agg_sku = agg_missed_by_sku(df)
    agg_day = agg_missed_by_day(df)

    print("\nСтроим сводные графики...")
    chart_paths = {
        "calendar": chart_missed_calendar(agg_day),
        "top_sku":  chart_missed_summary_bar(agg_sku),
        "by_hour":  chart_missed_by_hour(df),
        "by_dow":   chart_missed_by_dow(df),
    }

    print(f"\nСтроим детальные профили (топ-{TOP_N_CHARTS})...")
    profile_charts = []
    for _, row in agg_sku.head(TOP_N_CHARTS).iterrows():
        bid = int(row["bakery_id"])
        pid = int(row["product_id"])
        pname = row["product_name"]
        cat   = row["category_name"]

        sub = df[(df["bakery_id"] == bid) & (df["product_id"] == pid) & (df["hour"].isin(WORK_HOURS))]

        # Часы с miss ≥ 20% дней
        hour_miss_pct = (
            sub.groupby("hour")["missed"].mean()
            .reindex(WORK_HOURS, fill_value=0.0)
        )
        missed_hours = sorted(hour_miss_pct[hour_miss_pct >= 0.2].index.tolist())

        # Описание: когда именно нули
        if missed_hours:
            min_h, max_h = missed_hours[0], missed_hours[-1]
            if min_h <= 9:
                timing = "Упущенные продажи в начале дня — возможен недостаточный объём первой партии или задержка выпечки."
            elif max_h >= 16:
                timing = "Упущенные продажи в конце дня — вероятно, утренняя партия заканчивалась и повторного выпуска не было."
            else:
                timing = "Упущенные продажи в обеденные часы — возможен стокаут между партиями (пекарня не успела подготовить вторую партию к пиковому спросу)."
        else:
            timing = "Упущенные продажи распределены по дню."

        # Коэффициент покрытия: сколько от прогноза реально продали
        actual_sum   = float(sub["actual_qty"].sum())
        forecast_sum = float(sub["forecast_qty"].sum())
        coverage     = actual_sum / forecast_sum if forecast_sum > 0 else 0.0

        explanation = (
            f"{timing}  \n"
            f"За период пекарня продала **{actual_sum:.0f} шт** против "
            f"прогноза **{forecast_sum:.0f} шт** "
            f"(покрытие прогноза: **{coverage:.0%}**). "
            f"Упущенные продажи зафиксированы в **{int(row['n_missed_days'])} из 14 дней** "
            f"в часы: {', '.join(f'{h}:00' for h in missed_hours) if missed_hours else '—'}."
        )

        chart_path = chart_sku_profile(df, profile_df, bid, pid, pname, cat)
        print(f"    {pname[:40]} / пекарня {bid}")

        profile_charts.append({
            "bakery_id":    bid,
            "product_id":   pid,
            "product_name": pname,
            "category_name": cat,
            "n_missed_days": int(row["n_missed_days"]),
            "n_missed_hours": int(row["n_missed_hours"]),
            "missed_fc_qty":  float(row["missed_fc_qty"]),
            "missed_hours": missed_hours,
            "chart": chart_path,
            "explanation": explanation,
        })

    print("\nГенерируем MD отчёт...")
    md = generate_md(df, agg_sku, agg_day, chart_paths, profile_charts)
    report_path = OUT_DIR / "report.md"
    report_path.write_text(md, encoding="utf-8")
    print(f"Отчёт: {report_path}  ({report_path.stat().st_size:,} байт)")
    print(f"Графиков: {len(list(CHART_DIR.glob('*.png')))}")


if __name__ == "__main__":
    main()

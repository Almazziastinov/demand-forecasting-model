"""
Анализ расхождений прогноза и факта на уровне пекарня-час и вклада SKU.

Период: 2026-06-01 .. 2026-06-29 (lead-1 backfill, base_raw_uplift)
Пекарни: 10 пилотных [20, 21, 22, 28, 80, 89, 107, 221, 222, 257]

Выход:
  outputs/hourly_profile_discrepancy.xlsx  — Excel с листами
  outputs/hourly_profile_discrepancy.md    — MD отчёт с выводами
"""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipelines.forecast_publish.load_forecast_run import create_client  # noqa: E402

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
EXCEL_PATH = OUTPUT_DIR / "hourly_profile_discrepancy.xlsx"
MD_PATH = OUTPUT_DIR / "hourly_profile_discrepancy.md"

PILOT_BAKERIES = [20, 21, 22, 28, 80, 89, 107, 221, 222, 257]
DATE_FROM = "2026-06-01"
DATE_TO = "2026-06-29"
SALES_EVENT_HEX = "D09FD180D0BED0B4D0B0D0B6D0B0"
TOP_SKU_N = 5

DOW_LABELS = {0: "Пн", 1: "Вт", 2: "Ср", 3: "Чт", 4: "Пт", 5: "Сб", 6: "Вс"}


def wmape(forecast: pd.Series, actual: pd.Series) -> float:
    mask = actual > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.abs(forecast[mask] - actual[mask]).sum() / actual[mask].sum())


def bias_pct(forecast: pd.Series, actual: pd.Series) -> float:
    total_actual = actual.sum()
    if total_actual == 0:
        return float("nan")
    return float((forecast.sum() - total_actual) / total_actual)


def load_data(client) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bakeries_in = ",".join(str(b) for b in PILOT_BAKERIES)

    print("Загружаем прогноз (lead-1 SKU-hour)...")
    forecast = client.query_df(f"""
        select
            toDate(h.forecast_date) as date,
            toInt64OrNull(toString(h.bakery_id)) as bakery_id,
            toInt64OrNull(toString(h.product_id)) as product_id,
            any(d.product_name) as product_name,
            any(d.category_name) as category_name,
            toInt16(h.hour) as hour,
            sum(h.forecast_qty) as forecast_qty
        from sku_forecast_hour_snapshots h
        left join sku_forecast_day_snapshots d
          on d.source_run_id = h.source_run_id
         and d.forecast_date = h.forecast_date
         and d.bakery_id = h.bakery_id
         and d.product_id = h.product_id
         and d.lead_days = 1
        where h.lead_days = 1
          and h.source_run_id like %(run_pat)s
          and h.bakery_id in ({bakeries_in})
          and h.forecast_date between %(df)s and %(dt)s
        group by date, bakery_id, product_id, hour
    """, parameters={"df": DATE_FROM, "dt": DATE_TO, "run_pat": "backfill_base_bakery_raw_uplift_sku_%_h1"})
    print(f"  прогноз: {len(forecast):,} строк")

    print("Загружаем факт (mart_sales_60d)...")
    actual = client.query_df(f"""
        select
            check_date as date,
            toInt64OrNull(toString(bakery_id)) as bakery_id,
            toInt64OrNull(toString(product_id)) as product_id,
            toHour(check_datetime) as hour,
            sum(quantity) as actual_qty
        from mart_sales_60d
        where hex(cash_event_type) = %(hex)s
          and check_date between %(df)s and %(dt)s
          and toInt64OrNull(toString(bakery_id)) in ({bakeries_in})
        group by date, bakery_id, product_id, hour
    """, parameters={"hex": SALES_EVENT_HEX, "df": DATE_FROM, "dt": DATE_TO})
    print(f"  факт: {len(actual):,} строк")

    print("Загружаем uplift multiplier...")
    uplift = client.query_df(f"""
        select bakery_id, dow, hour, sku_uplift_multiplier
        from sku_hour_uplift_multiplier_embedded
        where bakery_id in ({bakeries_in})
    """)
    print(f"  uplift: {len(uplift):,} строк")

    print("Загружаем share profile...")
    share = client.query_df(f"""
        select bakery_id, product_id, dow, hour,
               mean_sku_share_in_hour_norm as share_norm,
               mean_sku_share_in_hour as share_raw,
               n_days
        from sku_hour_share_profile_smoothed_embedded
        where bakery_id in ({bakeries_in})
    """)
    print(f"  share: {len(share):,} строк")

    return forecast, actual, uplift, share


def build_sku_hour(forecast: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Джойн прогноза и факта на уровне SKU-час."""
    fc = forecast.copy()
    ac = actual.copy()
    fc["date"] = pd.to_datetime(fc["date"]).dt.date
    ac["date"] = pd.to_datetime(ac["date"]).dt.date

    merged = fc.merge(
        ac[["date", "bakery_id", "product_id", "hour", "actual_qty"]],
        on=["date", "bakery_id", "product_id", "hour"],
        how="outer",
    )
    merged["forecast_qty"] = merged["forecast_qty"].fillna(0.0)
    merged["actual_qty"] = merged["actual_qty"].fillna(0.0)
    merged["delta"] = merged["forecast_qty"] - merged["actual_qty"]
    merged["dow"] = pd.to_datetime(merged["date"]).dt.dayofweek
    return merged


def build_bakery_hour(sku_hour: pd.DataFrame) -> pd.DataFrame:
    """Агрегат на уровне пекарня-дата-час."""
    grp = sku_hour.groupby(["bakery_id", "date", "dow", "hour"]).agg(
        forecast_qty=("forecast_qty", "sum"),
        actual_qty=("actual_qty", "sum"),
    ).reset_index()
    grp["delta"] = grp["forecast_qty"] - grp["actual_qty"]
    return grp


def sheet_dow_hour_heatmap(bh: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """bias% и wMAPE в разрезе DOW x час."""
    rows = []
    for (dow, hour), g in bh.groupby(["dow", "hour"]):
        rows.append({
            "dow": dow,
            "dow_label": DOW_LABELS.get(dow, str(dow)),
            "hour": hour,
            "forecast_total": g["forecast_qty"].sum(),
            "actual_total": g["actual_qty"].sum(),
            "bias_pct": bias_pct(g["forecast_qty"], g["actual_qty"]) * 100,
            "wmape_pct": wmape(g["forecast_qty"], g["actual_qty"]) * 100,
            "n_obs": len(g),
        })
    df = pd.DataFrame(rows)

    bias_pivot = df.pivot_table(index="dow_label", columns="hour", values="bias_pct", aggfunc="mean")
    bias_pivot = bias_pivot.reindex([DOW_LABELS[i] for i in range(7) if DOW_LABELS[i] in bias_pivot.index])
    wmape_pivot = df.pivot_table(index="dow_label", columns="hour", values="wmape_pct", aggfunc="mean")
    wmape_pivot = wmape_pivot.reindex([DOW_LABELS[i] for i in range(7) if DOW_LABELS[i] in wmape_pivot.index])
    return bias_pivot, wmape_pivot, df


def sheet_per_bakery(bh: pd.DataFrame, forecast: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fc_bak = forecast.groupby("bakery_id")[["forecast_qty"]].sum()
    ac_bak = actual.groupby("bakery_id")[["actual_qty"]].sum()
    for bak in PILOT_BAKERIES:
        g = bh[bh["bakery_id"] == bak]
        fc_tot = fc_bak.loc[bak, "forecast_qty"] if bak in fc_bak.index else 0
        ac_tot = ac_bak.loc[bak, "actual_qty"] if bak in ac_bak.index else 0
        rows.append({
            "bakery_id": bak,
            "forecast_total": fc_tot,
            "actual_total": ac_tot,
            "bias_pct": (fc_tot - ac_tot) / ac_tot * 100 if ac_tot > 0 else float("nan"),
            "wmape_pct": wmape(g["forecast_qty"], g["actual_qty"]) * 100,
            "peak_error_hour": g.loc[g["delta"].abs().idxmax(), "hour"] if len(g) > 0 else None,
            "peak_error_dow": DOW_LABELS.get(int(g.loc[g["delta"].abs().idxmax(), "dow"]), "") if len(g) > 0 else None,
        })
    return pd.DataFrame(rows).sort_values("wmape_pct", ascending=False)


def sheet_top_sku_contributors(sku_hour: pd.DataFrame) -> pd.DataFrame:
    """Для каждой пекарни: топ-5 SKU по суммарному |delta| за весь период."""
    grp = sku_hour.groupby(["bakery_id", "product_id", "product_name", "category_name"]).agg(
        forecast_total=("forecast_qty", "sum"),
        actual_total=("actual_qty", "sum"),
        abs_delta_total=("delta", lambda x: x.abs().sum()),
        delta_total=("delta", "sum"),
    ).reset_index()
    grp["bias_pct"] = (grp["delta_total"] / grp["actual_total"].replace(0, np.nan)) * 100

    result = []
    for bak in PILOT_BAKERIES:
        top = grp[grp["bakery_id"] == bak].nlargest(TOP_SKU_N, "abs_delta_total").copy()
        top["rank"] = range(1, len(top) + 1)
        result.append(top)
    return pd.concat(result, ignore_index=True) if result else grp.head(0)


def sheet_profile_shape(sku_hour: pd.DataFrame) -> pd.DataFrame:
    """Форма профиля: нормированный прогноз vs факт по часам."""
    hour_agg = sku_hour.groupby(["bakery_id", "date", "hour"]).agg(
        forecast_qty=("forecast_qty", "sum"),
        actual_qty=("actual_qty", "sum"),
    ).reset_index()

    # нормируем внутри каждого bakery-date до суммы=1
    day_totals = hour_agg.groupby(["bakery_id", "date"]).agg(
        fc_day=("forecast_qty", "sum"),
        ac_day=("actual_qty", "sum"),
    ).reset_index()
    hour_agg = hour_agg.merge(day_totals, on=["bakery_id", "date"])
    hour_agg["fc_share"] = hour_agg["forecast_qty"] / hour_agg["fc_day"].replace(0, np.nan)
    hour_agg["ac_share"] = hour_agg["actual_qty"] / hour_agg["ac_day"].replace(0, np.nan)
    hour_agg["share_delta"] = hour_agg["fc_share"] - hour_agg["ac_share"]

    # среднее по часу
    shape = hour_agg.groupby("hour").agg(
        fc_share_mean=("fc_share", "mean"),
        ac_share_mean=("ac_share", "mean"),
        share_delta_mean=("share_delta", "mean"),
    ).reset_index()
    shape["shape_bias_pct"] = shape["share_delta_mean"] * 100
    return shape


def sheet_uplift_analysis(bh: pd.DataFrame, uplift: pd.DataFrame) -> pd.DataFrame:
    """Сравниваем bias по часам с uplift multiplier."""
    # uplift: bakery_id, dow (-1 = global), hour, sku_uplift_multiplier
    global_uplift = uplift[uplift["dow"] == -1].groupby(["bakery_id", "hour"])["sku_uplift_multiplier"].mean().reset_index()
    global_uplift.rename(columns={"sku_uplift_multiplier": "uplift_mult"}, inplace=True)

    bh_hour = bh.groupby(["bakery_id", "hour"]).agg(
        forecast_total=("forecast_qty", "sum"),
        actual_total=("actual_qty", "sum"),
    ).reset_index()
    bh_hour["bias_pct"] = (bh_hour["forecast_total"] - bh_hour["actual_total"]) / bh_hour["actual_total"].replace(0, np.nan) * 100

    merged = bh_hour.merge(global_uplift, on=["bakery_id", "hour"], how="left")
    # forecast_without_uplift = forecast_total / uplift_mult (примерная оценка)
    merged["forecast_without_uplift"] = merged["forecast_total"] / merged["uplift_mult"].replace(0, np.nan)
    merged["bias_without_uplift_pct"] = (
        (merged["forecast_without_uplift"] - merged["actual_total"]) / merged["actual_total"].replace(0, np.nan) * 100
    )
    return merged.sort_values(["bakery_id", "hour"])


def write_excel(
    bias_pivot, wmape_pivot, dow_hour_df,
    per_bakery, top_sku, profile_shape, uplift_df
):
    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        # 1. Сводка по пекарням
        per_bakery.to_excel(writer, sheet_name="1_По_пекарням", index=False)

        # 2. bias% heatmap DOW x час
        bias_pivot.to_excel(writer, sheet_name="2_Bias_DOW_час")

        # 3. wMAPE heatmap DOW x час
        wmape_pivot.to_excel(writer, sheet_name="3_wMAPE_DOW_час")

        # 4. Детали DOW x час
        dow_hour_df.to_excel(writer, sheet_name="4_DOW_час_детали", index=False)

        # 5. Топ SKU по расхождению
        top_sku.to_excel(writer, sheet_name="5_Топ_SKU_расхождения", index=False)

        # 6. Форма профиля
        profile_shape.to_excel(writer, sheet_name="6_Форма_профиля", index=False)

        # 7. Uplift анализ
        uplift_df.to_excel(writer, sheet_name="7_Uplift_анализ", index=False)

    print(f"Excel: {EXCEL_PATH}")


def write_md(per_bakery, dow_hour_df, top_sku, profile_shape, uplift_df):
    overall_bias = bias_pct(
        pd.Series(per_bakery["forecast_total"].fillna(0)),
        pd.Series(per_bakery["actual_total"].fillna(0)),
    ) * 100
    overall_wmape = wmape(
        pd.Series(per_bakery["forecast_total"].fillna(0)),
        pd.Series(per_bakery["actual_total"].fillna(0)),
    ) * 100

    # worst DOW-hour
    worst = dow_hour_df.reindex(dow_hour_df["wmape_pct"].abs().sort_values(ascending=False).index).head(5)
    worst_rows = "\n".join(
        f"  - {DOW_LABELS.get(int(r.dow), r.dow)} {int(r.hour):02d}:00 — bias {r.bias_pct:+.1f}%, wMAPE {r.wmape_pct:.1f}%"
        for _, r in worst.iterrows()
    )

    # worst bakeries
    worst_bak = per_bakery.head(3)
    worst_bak_rows = "\n".join(
        f"  - Пекарня {int(r.bakery_id)}: bias {r.bias_pct:+.1f}%, wMAPE {r.wmape_pct:.1f}%"
        for _, r in worst_bak.iterrows()
    )

    # profile shape: worst hours
    shape_worst = profile_shape.reindex(profile_shape["share_delta_mean"].abs().sort_values(ascending=False).index).head(5)
    shape_rows = "\n".join(
        f"  - {int(r.hour):02d}:00 — профиль {r.fc_share_mean*100:.1f}% vs факт {r.ac_share_mean*100:.1f}% (delta {r.shape_bias_pct:+.1f}pp)"
        for _, r in shape_worst.iterrows()
    )

    # top sku overall
    top_overall = top_sku.nlargest(10, "abs_delta_total")[["bakery_id","product_name","category_name","delta_total","bias_pct"]]
    top_sku_rows = "\n".join(
        f"  - Пекарня {int(r.bakery_id)} / {r.product_name}: delta {r.delta_total:+.0f} шт, bias {r.bias_pct:+.1f}%"
        for _, r in top_overall.iterrows()
    )

    # uplift effect
    uplift_effect = uplift_df.dropna(subset=["uplift_mult"])
    if len(uplift_effect) > 0:
        avg_uplift = uplift_effect["uplift_mult"].mean()
        corr_bias_uplift = uplift_effect[["uplift_mult", "bias_pct"]].dropna().corr().iloc[0, 1]
        uplift_summary = f"Средний uplift multiplier: {avg_uplift:.3f}. Корреляция uplift↔bias: {corr_bias_uplift:+.2f}."
    else:
        uplift_summary = "Данные uplift недоступны."

    md = dedent(f"""\
    # Анализ расхождений прогноза и факта (часовой уровень)

    **Период:** {DATE_FROM} — {DATE_TO} (lead-1, base_raw_uplift)
    **Пекарни:** {', '.join(str(b) for b in PILOT_BAKERIES)}

    ---

    ## Общие метрики

    | Метрика | Значение |
    |---|---|
    | Bias% (все пекарни) | {overall_bias:+.1f}% |
    | wMAPE% (все пекарни) | {overall_wmape:.1f}% |

    ---

    ## По пекарням (топ-3 по wMAPE)

{worst_bak_rows}

    ---

    ## Расхождения по DOW × час (топ-5 по wMAPE)

{worst_rows}

    ---

    ## Форма профиля: прогноз vs факт (топ-5 расхождений по доле часа)

{shape_rows}

    ---

    ## Топ-10 SKU по вкладу в расхождение

{top_sku_rows}

    ---

    ## Влияние Uplift multiplier

    {uplift_summary}

    Если корреляция uplift↔bias положительная — высокий uplift соответствует высокому перепрогнозу,
    то есть uplift систематически завышает прогноз в часы с уже высоким базовым значением.

    ---

    ## Файлы

    - `outputs/hourly_profile_discrepancy.xlsx` — детальные таблицы и хитмапы
    - Лист 1: сводка по пекарням
    - Лист 2: bias% heatmap DOW × час
    - Лист 3: wMAPE% heatmap DOW × час
    - Лист 5: топ-5 SKU по расхождению для каждой пекарни
    - Лист 6: форма профиля (нормированная доля по часам)
    - Лист 7: uplift анализ по пекарня-час
    """)

    MD_PATH.write_text(md, encoding="utf-8")
    print(f"MD:    {MD_PATH}")


def main():
    client = create_client(str(ROOT / ".env"))
    forecast, actual, uplift, share = load_data(client)

    print("\nСтроим таблицы расхождений...")
    sku_hour = build_sku_hour(forecast, actual)
    bh = build_bakery_hour(sku_hour)

    bias_pivot, wmape_pivot, dow_hour_df = sheet_dow_hour_heatmap(bh)
    per_bakery = sheet_per_bakery(bh, forecast, actual)
    top_sku = sheet_top_sku_contributors(sku_hour)
    profile_shape = sheet_profile_shape(sku_hour)
    uplift_df = sheet_uplift_analysis(bh, uplift)

    print("\nПишем Excel и MD...")
    write_excel(bias_pivot, wmape_pivot, dow_hour_df, per_bakery, top_sku, profile_shape, uplift_df)
    write_md(per_bakery, dow_hour_df, top_sku, profile_shape, uplift_df)
    print("\nГотово.")


if __name__ == "__main__":
    main()

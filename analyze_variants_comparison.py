"""
Сравнение 3 dev-вариантов vs продового прогноза по пилотным пекарням.

Периодика: 2026-06-22 .. 2026-06-28, lead_days=1.

Варианты:
  prod      -- текущий продовый прогноз
  prior14   -- uplifted, prior window 14d
  bias_corr -- uplifted, days=30, + SKU bias correction топ-5
  base_raw  -- base (norm) model + raw uplift на аллокации

Группы анализа:
  - Все SKU
  - Топ-5 ходовых SKU
  - Пироги сытные
  - По пекарням (pivot)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import clickhouse_connect
import pandas as pd
import numpy as np

HOST     = "rc1b-aergg94cc1r6ctr1.mdb.yandexcloud.net"
PORT     = 8443
USER     = "AlmazZR"
PASSWORD = "Almaz__Ziast0303"
DATABASE = "Svezhar"

import argparse as _ap

PILOT_IDS    = [20, 21, 22, 28, 80, 89, 107, 221, 222, 257]
TOP5_SKU_IDS = [1071, 10340, 205, 1076, 57]
PIE_PATTERN  = "пироги сытные"

_parser = _ap.ArgumentParser(add_help=False)
_parser.add_argument("--start",            default="2026-06-01")
_parser.add_argument("--end",              default="2026-06-28")
_parser.add_argument("--variants",         nargs="+", default=["base_raw"])
_parser.add_argument("--prod-run-prefix",  default=None,
                     help="SQL LIKE-паттерн для продового run_id, напр. 'backfill_base_bakery_no_sku_uplift_%%_h1'. "
                          "По умолчанию берётся лексически максимальный non-dev run per date.")
_cli, _ = _parser.parse_known_args()

START_DATE      = _cli.start
END_DATE        = _cli.end
VARIANTS        = _cli.variants
PROD_RUN_PREFIX = _cli.prod_run_prefix


def connect():
    return clickhouse_connect.get_client(
        host=HOST, port=PORT, username=USER, password=PASSWORD,
        database=DATABASE, secure=True, verify=False,
    )


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------

def load_variant_forecast(client, variant: str) -> pd.DataFrame:
    """Загружаем dev-прогноз для одного варианта."""
    run_id_pattern = f"dev_{variant}_%_h1"
    df = client.query_df(
        """
        select
            forecast_date,
            toInt64(bakery_id)  as bakery_id,
            toInt64(product_id) as product_id,
            any(product_name)   as product_name,
            any(category_name)  as category_name,
            sum(forecast_qty)   as forecast_qty
        from sku_forecast_day_snapshots
        where lead_days = 1
          and toInt64(bakery_id) in %(pilot)s
          and forecast_date between %(start)s and %(end)s
          and source_run_id like %(pattern)s
        group by forecast_date, bakery_id, product_id
        """,
        parameters={
            "pilot": PILOT_IDS,
            "start": START_DATE,
            "end": END_DATE,
            "pattern": run_id_pattern,
        },
    )
    return df


def load_prod_forecast(client) -> pd.DataFrame:
    """Берём продовый прогноз.

    Если задан --prod-run-prefix, фильтруем по нему.
    Иначе для каждой forecast_date берём лексически максимальный non-dev run_id
    (исключает задвоение когда за одну дату есть несколько бэкфилл-ранов).
    """
    if PROD_RUN_PREFIX:
        df = client.query_df(
            """
            select
                forecast_date,
                toInt64(bakery_id)  as bakery_id,
                toInt64(product_id) as product_id,
                any(product_name)   as product_name,
                any(category_name)  as category_name,
                sum(forecast_qty)   as forecast_qty
            from sku_forecast_day_snapshots
            where lead_days = 1
              and toInt64(bakery_id) in %(pilot)s
              and forecast_date between %(start)s and %(end)s
              and source_run_id like %(prefix)s
            group by forecast_date, bakery_id, product_id
            """,
            parameters={"pilot": PILOT_IDS, "start": START_DATE, "end": END_DATE,
                        "prefix": PROD_RUN_PREFIX},
        )
    else:
        # Берём только лексически максимальный non-dev run per forecast_date,
        # чтобы не суммировать несколько бэкфилл-ранов за одну дату.
        df = client.query_df(
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
    return df


def load_actuals(client) -> pd.DataFrame:
    df = client.query_df(
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
    return df


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------

def wmape(actual: pd.Series, forecast: pd.Series) -> float:
    denom = actual.sum()
    return float(abs(forecast - actual).sum() / denom) if denom > 0 else float("nan")


def bias_pct(actual: pd.Series, forecast: pd.Series) -> float:
    denom = actual.sum()
    return float((forecast.sum() - actual.sum()) / denom) if denom > 0 else float("nan")


def fmt_bias(v: float) -> str:
    return f"{v*100:+.1f}%" if not np.isnan(v) else "—"


def fmt_wmape(v: float) -> str:
    return f"{v*100:.1f}%" if not np.isnan(v) else "—"


# ---------------------------------------------------------------------------
# Сводные таблицы
# ---------------------------------------------------------------------------

def summary_by_group(merged: pd.DataFrame, all_variants: list[str]) -> pd.DataFrame:
    """bias% и wMAPE по группам для каждого варианта."""
    is_pie  = merged["category_name"].str.lower().str.contains(PIE_PATTERN, na=False)
    is_top5 = merged["product_id"].isin(TOP5_SKU_IDS)

    groups = [
        ("Все SKU",        pd.Series(True, index=merged.index)),
        ("Топ-5 ходовые",  is_top5),
        ("Пироги сытные",  is_pie),
        ("Остальные",      ~is_top5 & ~is_pie),
    ]

    rows = []
    for group_name, mask in groups:
        sub = merged[mask]
        row = {"группа": group_name, "факт": int(sub["actual_qty"].sum())}
        for v in all_variants:
            col = f"forecast_{v}"
            if col not in sub.columns:
                row[f"bias_{v}"] = "—"
                row[f"wmape_{v}"] = "—"
                continue
            row[f"bias_{v}"]  = fmt_bias(bias_pct(sub["actual_qty"], sub[col]))
            row[f"wmape_{v}"] = fmt_wmape(wmape(sub["actual_qty"], sub[col]))
        rows.append(row)
    return pd.DataFrame(rows)


def summary_by_sku(merged: pd.DataFrame, all_variants: list[str]) -> pd.DataFrame:
    """bias% для топ-5 SKU по каждому варианту."""
    sub = merged[merged["product_id"].isin(TOP5_SKU_IDS)].copy()
    grp = sub.groupby(["product_id", "product_name"], as_index=False).agg(
        actual_qty=("actual_qty", "sum"),
        **{f"forecast_{v}": (f"forecast_{v}", "sum") for v in all_variants if f"forecast_{v}" in sub.columns}
    ).sort_values("actual_qty", ascending=False)

    for v in all_variants:
        col = f"forecast_{v}"
        if col in grp.columns:
            grp[f"bias_{v}"] = grp.apply(
                lambda r: fmt_bias(bias_pct(pd.Series([r["actual_qty"]]), pd.Series([r[col]]))), axis=1
            )
    return grp


def summary_by_bakery(merged: pd.DataFrame, all_variants: list[str]) -> pd.DataFrame:
    """bias% по пекарням для каждого варианта."""
    grp = merged.groupby("bakery_id", as_index=False).agg(
        actual_qty=("actual_qty", "sum"),
        **{f"forecast_{v}": (f"forecast_{v}", "sum") for v in all_variants if f"forecast_{v}" in merged.columns}
    ).sort_values("actual_qty", ascending=False)

    for v in all_variants:
        col = f"forecast_{v}"
        if col in grp.columns:
            grp[f"bias_{v}"] = grp.apply(
                lambda r: fmt_bias(bias_pct(pd.Series([r["actual_qty"]]), pd.Series([r[col]]))), axis=1
            )
    return grp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out = sys.stdout.buffer

    def p(s: str):
        out.write((s + "\n").encode("utf-8"))
        out.flush()

    p(f"Сравнение вариантов: пилот {PILOT_IDS}")
    p(f"Период: {START_DATE} — {END_DATE}\n")

    client = connect()

    p("Загружаем данные...")
    actuals = load_actuals(client)
    p(f"  Факт: {len(actuals):,} строк, сумма={actuals['actual_qty'].sum():,.0f}")

    forecasts: dict[str, pd.DataFrame] = {}
    all_variants_found = []

    prod_df = load_prod_forecast(client)
    if not prod_df.empty:
        forecasts["prod"] = prod_df
        all_variants_found.append("prod")
        p(f"  prod: {len(prod_df):,} строк, сумма={prod_df['forecast_qty'].sum():,.0f}")
    else:
        p("  prod: нет данных")

    for v in VARIANTS:
        df = load_variant_forecast(client, v)
        if not df.empty:
            forecasts[v] = df
            all_variants_found.append(v)
            p(f"  {v}: {len(df):,} строк, сумма={df['forecast_qty'].sum():,.0f}")
        else:
            p(f"  {v}: нет данных (бэкфилл ещё не завершён?)")

    if not forecasts:
        p("\nНет данных ни по одному варианту. Запусти после завершения бэкфилла.")
        return

    # Строим общий merged датафрейм: actual + каждый вариант как отдельная колонка
    key = ["forecast_date", "bakery_id", "product_id"]
    merged = actuals.rename(columns={"product_name": "product_name", "category_name": "category_name"})
    for v, df in forecasts.items():
        merged = merged.merge(
            df[key + ["forecast_qty"]].rename(columns={"forecast_qty": f"forecast_{v}"}),
            on=key, how="left",
        )
        # Подтягиваем product_name/category_name если нет
        if f"product_name_{v}" not in merged.columns and "product_name" in df.columns:
            pass  # уже есть из actuals

    for v in all_variants_found:
        col = f"forecast_{v}"
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 35)

    # --- 1. По группам ---
    p("\n" + "=" * 70)
    p("1. СВОДКА ПО ГРУППАМ (bias% / wMAPE%)")
    p("=" * 70)
    grp_df = summary_by_group(merged, all_variants_found)

    # Печатаем в виде блоков: bias и wmape отдельно
    bias_cols  = ["группа", "факт"] + [f"bias_{v}"  for v in all_variants_found if f"bias_{v}"  in grp_df.columns]
    wmape_cols = ["группа", "факт"] + [f"wmape_{v}" for v in all_variants_found if f"wmape_{v}" in grp_df.columns]

    p("\n  BIAS%:")
    p(grp_df[bias_cols].to_string(index=False))
    p("\n  wMAPE%:")
    p(grp_df[wmape_cols].to_string(index=False))

    # --- 2. Топ-5 SKU ---
    p("\n" + "=" * 70)
    p("2. ТОП-5 ХОДОВЫХ SKU — bias% по вариантам")
    p("=" * 70)
    sku_df = summary_by_sku(merged, all_variants_found)
    bias_sku_cols = ["product_id", "product_name", "actual_qty"] + \
                    [f"bias_{v}" for v in all_variants_found if f"bias_{v}" in sku_df.columns]
    p(sku_df[bias_sku_cols].to_string(index=False))

    # --- 3. Пироги сытные отдельно ---
    p("\n" + "=" * 70)
    p("3. ПИРОГИ СЫТНЫЕ — bias% по вариантам")
    p("=" * 70)
    pies = merged[merged["category_name"].str.lower().str.contains(PIE_PATTERN, na=False)]
    if pies.empty:
        p("  Нет данных")
    else:
        pie_grp = pies.groupby(["product_id", "product_name"], as_index=False).agg(
            actual_qty=("actual_qty", "sum"),
            **{f"forecast_{v}": (f"forecast_{v}", "sum")
               for v in all_variants_found if f"forecast_{v}" in pies.columns}
        ).sort_values("actual_qty", ascending=False)
        for v in all_variants_found:
            col = f"forecast_{v}"
            if col in pie_grp.columns:
                pie_grp[f"bias_{v}"] = pie_grp.apply(
                    lambda r: fmt_bias(bias_pct(pd.Series([r["actual_qty"]]), pd.Series([r[col]]))), axis=1
                )
        bias_pie_cols = ["product_id", "product_name", "actual_qty"] + \
                        [f"bias_{v}" for v in all_variants_found if f"bias_{v}" in pie_grp.columns]
        p(pie_grp[bias_pie_cols].to_string(index=False))

    # --- 4. По пекарням ---
    p("\n" + "=" * 70)
    p("4. ПО ПЕКАРНЯМ — bias% по вариантам")
    p("=" * 70)
    bak_df = summary_by_bakery(merged, all_variants_found)
    bias_bak_cols = ["bakery_id", "actual_qty"] + \
                    [f"bias_{v}" for v in all_variants_found if f"bias_{v}" in bak_df.columns]
    p(bak_df[bias_bak_cols].to_string(index=False))

    p("\n" + "=" * 70)
    p("Анализ завершён")
    p("=" * 70)


if __name__ == "__main__":
    main()

"""
Пересборка lead-1 backfill для продовых таблиц за произвольный диапазон дат.

Алгоритм:
  1. Для каждого дня в [date_from, date_to]:
     a. Читает bakery-level lead-1 из prod bakery_forecast_day_snapshots
     b. Пересчитывает SKU аллокацию через allocate_from_clickhouse (фикс cap)
     c. Записывает run в forecast_runs_embedded (prod)
     d. Обновляет sku_forecast_day_snapshots / bakery_forecast_day_snapshots (prod)
        через ReplacingMergeTree — новый generated_at вытеснит старый

Использование:
  python scripts/rebuild_prod_lead1_backfill.py \
      --env-file .env \
      --date-from 2026-06-17 \
      --date-to   2026-06-22 \
      --uplift-profile-version prod_allowlist_22_222_old_else_20260617 \
      --recent-correction-mode runner_city_prior_soft_weekpart
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipelines.forecast_publish.load_forecast_run import (  # noqa: E402
    create_client,
    load_forecast_run,
    load_product_lookup_from_clickhouse,
    prepare_bakery_day,
    prepare_bakery_day_snapshots,
    prepare_sku_day,
    prepare_sku_day_snapshots,
    prepare_sku_hour,
    prepare_sku_hour_snapshots,
)
from pipelines.forecast_publish.table_names import table_name  # noqa: E402
from src.experiments_v2.apply_bakery_profiles_clickhouse import (  # noqa: E402
    allocate_from_clickhouse,
)

SNAPSHOT_TABLES = (
    "bakery_forecast_day_snapshots",
    "sku_forecast_day_snapshots",
    "sku_forecast_hour_snapshots",
)

OUTPUT_DIR = ROOT / "data" / "processed"

DEFAULT_PROFILE_TABLE = "sku_hour_share_profile_smoothed_embedded"
DEFAULT_UPLIFT_TABLE = "sku_hour_uplift_multiplier_embedded"
DEFAULT_ASSORTMENT_TABLE = "assortment_city_products"
DEFAULT_RECENT_SALES_TABLE = "mart_sales_60d"


def load_bakery_lead1_day(client, forecast_date: str) -> pd.DataFrame:
    return client.query_df(
        """
        select
            forecast_date                                  as date,
            bakery_id,
            argMax(bakery_name, generated_at)              as bakery_name,
            argMax(city, generated_at)                     as city,
            argMax(forecast_base, generated_at)            as bakery_day_forecast,
            argMax(forecast_final, generated_at)           as bakery_day_forecast_bias_adj
        from bakery_forecast_day_snapshots
        where lead_days = 1
          and forecast_date = %(date)s
        group by forecast_date, bakery_id
        order by bakery_id
        """,
        parameters={"date": forecast_date},
    )


def _wait_mutations(client, table: str, run_id: str, timeout: int = 120) -> None:
    deadline = time.monotonic() + timeout
    while True:
        count = client.query(
            f"select count() from {table} where source_run_id = %(r)s",
            parameters={"r": run_id},
        ).result_rows[0][0]
        if count == 0:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Cleanup timed out for {run_id} in {table}")
        time.sleep(2)


def rebuild_day(
    client,
    *,
    forecast_date: str,
    env_file: str,
    profile_table: str,
    uplift_table: str,
    uplift_profile_version: str,
    assortment_table: str,
    recent_correction_mode: str,
    recent_correction_days: int,
    recent_sales_table: str,
    table_suffix: str = "",
) -> dict:
    bakery = load_bakery_lead1_day(client, forecast_date)
    if bakery.empty:
        print(f"  SKIP {forecast_date}: no bakery lead-1 snapshots found")
        return {}

    bakery_path = OUTPUT_DIR / f"_backfill_bakery_{forecast_date}.csv"
    bakery.to_csv(bakery_path, index=False, encoding="utf-8-sig")

    allocated = allocate_from_clickhouse(
        bakery_forecast_path=bakery_path,
        bakery_hour_profile_path=OUTPUT_DIR / "bakery_hour_profile.csv",
        output_dir=OUTPUT_DIR,
        env_file=env_file,
        profile_table=table_name(profile_table, table_suffix),
        uplift_table=table_name(uplift_table, table_suffix),
        forecast_col="bakery_day_forecast_bias_adj",
        output_suffix=f"_backfill_{forecast_date}",
        uplift_profile_version=uplift_profile_version,
        recent_correction_mode=recent_correction_mode,
        recent_correction_days=recent_correction_days,
        recent_sales_table=recent_sales_table,
        assortment_table=table_name(assortment_table, table_suffix),
    )

    date_part = forecast_date.replace("-", "")
    run_id = f"backfill_uplifted_bakery_norm_uplift_sku_{date_part}_h1"

    lookup = load_product_lookup_from_clickhouse(
        client, table_name(profile_table, table_suffix)
    )
    bakery_raw = pd.read_csv(bakery_path, encoding="utf-8-sig")
    sku_day_raw = pd.read_csv(allocated["sku_daily"], encoding="utf-8-sig")
    sku_hour_raw = pd.read_csv(allocated["sku_hourly"], encoding="utf-8-sig")

    bakery_day = prepare_bakery_day(bakery_raw, run_id)
    sku_day = prepare_sku_day(sku_day_raw, lookup, run_id)
    sku_hour = prepare_sku_hour(sku_hour_raw, run_id)
    generated_at = pd.Timestamp.utcnow().tz_localize(None)

    # Записываем snapshots (ReplacingMergeTree сам вытеснит старые по generated_at)
    snap_bk = prepare_bakery_day_snapshots(bakery_day, run_id=run_id, generated_at=generated_at)
    snap_sk = prepare_sku_day_snapshots(sku_day, run_id=run_id, generated_at=generated_at)
    snap_hr = prepare_sku_hour_snapshots(sku_hour, run_id=run_id, generated_at=generated_at)

    for frame in (snap_bk, snap_sk, snap_hr):
        dates = pd.to_datetime(frame["forecast_date"], errors="coerce")
        frame["forecast_origin_date"] = (dates - pd.Timedelta(days=1)).dt.date
        frame["lead_days"] = 1

    client.insert_df(table_name("bakery_forecast_day_snapshots", table_suffix), snap_bk)
    client.insert_df(table_name("sku_forecast_day_snapshots", table_suffix), snap_sk)
    client.insert_df(table_name("sku_forecast_hour_snapshots", table_suffix), snap_hr)

    # Создаём/обновляем run-запись
    loaded = load_forecast_run(
        env_file=env_file,
        bakery_path=bakery_path,
        sku_day_path=allocated["sku_daily"],
        sku_hour_path=allocated["sku_hourly"],
        profile_table=table_name(profile_table, table_suffix),
        lookup_source="clickhouse",
        run_id=run_id,
        model_version="bakery_lead1_prod_snapshots_cap_fix",
        profile_version=uplift_profile_version,
        notes=f"Lead-1 backfill {forecast_date} with cap redistribution fix",
        replace_existing=True,
    )

    # Чистим временные файлы
    bakery_path.unlink(missing_ok=True)

    return {
        "run_id": run_id,
        "date": forecast_date,
        "bakery_rows": len(bakery),
        "sku_day_rows": len(sku_day),
        "loaded": loaded,
    }


def daterange(start: str, end: str):
    d = date.fromisoformat(start)
    stop = date.fromisoformat(end)
    while d <= stop:
        yield str(d)
        d += timedelta(days=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--date-from", default="2026-06-17")
    parser.add_argument("--date-to", default="2026-06-22")
    parser.add_argument("--uplift-profile-version", required=True)
    parser.add_argument("--profile-table", default=DEFAULT_PROFILE_TABLE)
    parser.add_argument("--uplift-table", default=DEFAULT_UPLIFT_TABLE)
    parser.add_argument("--assortment-table", default=DEFAULT_ASSORTMENT_TABLE)
    parser.add_argument("--recent-correction-mode", default="runner_city_prior_soft_weekpart")
    parser.add_argument("--recent-correction-days", type=int, default=30)
    parser.add_argument("--recent-sales-table", default=DEFAULT_RECENT_SALES_TABLE)
    parser.add_argument("--table-suffix", default="")
    parser.add_argument("--summary-path", default="reports/prod_lead1_backfill_summary.json")
    args = parser.parse_args()

    client = create_client(args.env_file)
    results = []

    for day in daterange(args.date_from, args.date_to):
        print(f"\n{'='*60}\nProcessing {day}")
        t0 = time.monotonic()
        result = rebuild_day(
            client,
            forecast_date=day,
            env_file=args.env_file,
            profile_table=args.profile_table,
            uplift_table=args.uplift_table,
            uplift_profile_version=args.uplift_profile_version,
            assortment_table=args.assortment_table,
            recent_correction_mode=args.recent_correction_mode,
            recent_correction_days=args.recent_correction_days,
            recent_sales_table=args.recent_sales_table,
            table_suffix=args.table_suffix,
        )
        elapsed = time.monotonic() - t0
        if result:
            print(f"  done: {result['run_id']}  sku_day={result['sku_day_rows']}  t={elapsed:.0f}s")
            results.append(result)

    summary = {"date_from": args.date_from, "date_to": args.date_to, "days": results}
    summary_path = ROOT / args.summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSummary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

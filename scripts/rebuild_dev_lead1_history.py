"""Rebuild dev SKU lead-1 history from production bakery lead-1 snapshots."""

from __future__ import annotations

# ruff: noqa: E501
import argparse
import json
import sys
import time
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
from pipelines.forecast_publish.table_names import (  # noqa: E402
    get_table_suffix_from_env_file,
    table_name,
)
from src.experiments_v2.apply_bakery_profiles_clickhouse import (  # noqa: E402
    allocate_from_clickhouse,
)


SNAPSHOT_TABLES = (
    "bakery_forecast_day_snapshots",
    "sku_forecast_day_snapshots",
    "sku_forecast_hour_snapshots",
)


def load_bakery_lead1(
    client,
    *,
    date_from: str,
    date_to: str,
) -> pd.DataFrame:
    return client.query_df(
        """
        select
            forecast_date as date,
            bakery_id,
            argMax(bakery_name, generated_at) as bakery_name,
            argMax(city, generated_at) as city,
            argMax(forecast_base, generated_at) as bakery_day_forecast,
            argMax(forecast_final, generated_at) as bakery_day_forecast_bias_adj
        from bakery_forecast_day_snapshots
        where lead_days = 1
          and forecast_date between %(date_from)s and %(date_to)s
        group by forecast_date, bakery_id
        order by forecast_date, bakery_id
        """,
        parameters={"date_from": date_from, "date_to": date_to},
    )


def replace_run_snapshots_with_lead1(
    client,
    *,
    env_file: str,
    run_id: str,
    bakery_path: Path,
    sku_day_path: Path,
    sku_hour_path: Path,
    profile_table: str,
) -> dict[str, int]:
    suffix = get_table_suffix_from_env_file(env_file)
    if suffix != "_dev":
        raise ValueError(f"Refusing to rebuild lead-1 outside _dev tables: {suffix}")

    for base_table in SNAPSHOT_TABLES:
        resolved = table_name(base_table, suffix)
        client.command(
            f"alter table {resolved} delete where source_run_id = %(run_id)s",
            parameters={"run_id": run_id},
        )

    deadline = time.monotonic() + 180
    while True:
        remaining = 0
        for base_table in SNAPSHOT_TABLES:
            resolved = table_name(base_table, suffix)
            count = client.query_df(
                f"select count() as rows from {resolved} where source_run_id = %(run_id)s",
                parameters={"run_id": run_id},
            )
            remaining += int(count["rows"].iloc[0])
        if remaining == 0:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Snapshot cleanup timed out for {run_id}: {remaining}")
        time.sleep(2)

    bakery_raw = pd.read_csv(bakery_path, encoding="utf-8-sig")
    sku_day_raw = pd.read_csv(sku_day_path, encoding="utf-8-sig")
    sku_hour_raw = pd.read_csv(sku_hour_path, encoding="utf-8-sig")
    lookup = load_product_lookup_from_clickhouse(client, profile_table)
    bakery_day = prepare_bakery_day(bakery_raw, run_id)
    sku_day = prepare_sku_day(sku_day_raw, lookup, run_id)
    sku_hour = prepare_sku_hour(sku_hour_raw, run_id)
    generated_at = pd.Timestamp.utcnow().tz_localize(None)

    frames = {
        "bakery_forecast_day_snapshots": prepare_bakery_day_snapshots(
            bakery_day, run_id=run_id, generated_at=generated_at
        ),
        "sku_forecast_day_snapshots": prepare_sku_day_snapshots(
            sku_day, run_id=run_id, generated_at=generated_at
        ),
        "sku_forecast_hour_snapshots": prepare_sku_hour_snapshots(
            sku_hour, run_id=run_id, generated_at=generated_at
        ),
    }
    counts = {}
    for base_table, frame in frames.items():
        forecast_dates = pd.to_datetime(frame["forecast_date"], errors="coerce")
        frame["forecast_origin_date"] = (forecast_dates - pd.Timedelta(days=1)).dt.date
        frame["lead_days"] = 1
        client.insert_df(table_name(base_table, suffix), frame)
        counts[base_table] = int(len(frame))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=".env.dev")
    parser.add_argument("--date-from", default="2026-06-01")
    parser.add_argument("--date-to", default="2026-06-14")
    parser.add_argument(
        "--run-id",
        default="dev_lead1_history_20260601_20260614_assortment_v1",
    )
    parser.add_argument(
        "--profile-table", default="sku_hour_share_profile_smoothed_embedded_dev"
    )
    parser.add_argument(
        "--uplift-table", default="sku_hour_uplift_multiplier_embedded_dev"
    )
    parser.add_argument("--uplift-profile-version", required=True)
    parser.add_argument(
        "--assortment-table", default="assortment_city_products_dev"
    )
    parser.add_argument(
        "--recent-correction-mode", default="runner_city_prior_soft_weekpart"
    )
    parser.add_argument("--disable-assortment-renormalization", action="store_true")
    parser.add_argument("--recent-sales-table", default="mart_sales_60d")
    parser.add_argument(
        "--summary-path", default="reports/dev_lead1_assortment_summary.json"
    )
    args = parser.parse_args()

    if get_table_suffix_from_env_file(args.env_file) != "_dev":
        raise ValueError("This script requires FORECAST_TABLE_SUFFIX=_dev")
    client = create_client(args.env_file)
    bakery = load_bakery_lead1(
        client, date_from=args.date_from, date_to=args.date_to
    )
    if bakery.empty:
        raise ValueError("No production bakery lead-1 snapshots found")

    output_dir = ROOT / "data" / "processed"
    bakery_path = output_dir / "bakery_day_forecast_dev_lead1.csv"
    bakery.to_csv(bakery_path, index=False, encoding="utf-8-sig")
    allocated = allocate_from_clickhouse(
        bakery_forecast_path=bakery_path,
        bakery_hour_profile_path=output_dir / "bakery_hour_profile.csv",
        output_dir=output_dir,
        env_file=args.env_file,
        profile_table=args.profile_table,
        uplift_table=args.uplift_table,
        forecast_col="bakery_day_forecast_bias_adj",
        output_suffix="dev_lead1_assortment",
        uplift_profile_version=args.uplift_profile_version,
        recent_correction_mode=args.recent_correction_mode,
        recent_sales_table=args.recent_sales_table,
        assortment_table=args.assortment_table,
        disable_assortment_renormalization=args.disable_assortment_renormalization,
    )
    loaded = load_forecast_run(
        env_file=args.env_file,
        bakery_path=bakery_path,
        sku_day_path=allocated["sku_daily"],
        sku_hour_path=allocated["sku_hourly"],
        profile_table=args.profile_table,
        lookup_source="clickhouse",
        run_id=args.run_id,
        model_version="bakery_lead1_prod_snapshots",
        profile_version=args.uplift_profile_version,
        notes="Lead-1 history rebuilt with assortment-aware dev allocation",
        replace_existing=True,
    )
    snapshot_counts = replace_run_snapshots_with_lead1(
        client,
        env_file=args.env_file,
        run_id=args.run_id,
        bakery_path=bakery_path,
        sku_day_path=allocated["sku_daily"],
        sku_hour_path=allocated["sku_hourly"],
        profile_table=args.profile_table,
    )
    summary = {
        "run_id": args.run_id,
        "date_from": args.date_from,
        "date_to": args.date_to,
        "bakery_source_rows": int(len(bakery)),
        "loaded": loaded,
        "lead1_snapshot_rows": snapshot_counts,
        "allocation_summary_path": str(allocated["summary"]),
    }
    summary_path = ROOT / args.summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

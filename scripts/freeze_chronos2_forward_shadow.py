"""Freeze a read-only, local Chronos-2 shadow input from current facts.

This does not publish forecasts or alter ClickHouse. It covers only bakeries
present in the original 55-bakery research panel and records extraction time.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.export_clickhouse_checks import create_client  # noqa: E402
from scripts.run_chronos2_fixed_origin import _sha256  # noqa: E402
from src.model_tournament.fixed_origin import validate_flows  # noqa: E402


def freeze_frames(
    historical: pd.DataFrame,
    sales: pd.DataFrame,
    release: pd.DataFrame,
    *,
    origin: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Append only facts after the historical panel's last date."""
    historical = validate_flows(historical)
    historical_end = historical["date"].max()
    if historical_end >= origin:
        raise ValueError("Historical panel already reaches the requested origin")
    start = historical_end + pd.Timedelta(days=1)
    keys = ["date", "bakery_id", "product_id"]
    for frame, value in ((sales, "observed_sales_qty"), (release, "release_qty")):
        if not set(keys + [value]).issubset(frame.columns):
            raise ValueError(f"Current facts lack {value} or SKU-day keys")
        if frame.duplicated(keys).any():
            raise ValueError(f"Current {value} facts have duplicate SKU-day keys")
    current = sales.merge(release, on=keys, how="outer", validate="one_to_one")
    current["date"] = pd.to_datetime(current["date"]).dt.normalize()
    if not current["date"].between(start, origin).all():
        raise ValueError("Current facts extend outside the frozen gap")
    current["observed_sales_qty"] = current["observed_sales_qty"].fillna(0.0)
    current["release_qty"] = current["release_qty"].fillna(0.0)
    if current.empty or current["date"].max() != origin:
        raise ValueError("No current facts for the forecast origin")
    needed = keys + ["observed_sales_qty", "release_qty"]
    combined = validate_flows(pd.concat([historical[needed], current[needed]]))
    coverage = (
        current.groupby("date")
        .agg(rows=("product_id", "size"), bakeries=("bakery_id", "nunique"))
        .reset_index()
    )
    if len(coverage) != (origin - start).days + 1:
        raise ValueError("Current fact interval has a missing calendar day")
    return combined, {
        "historical_end": str(historical_end.date()),
        "current_start": str(start.date()),
        "current_end": str(origin.date()),
        "current_rows": len(current),
        "current_daily_min_bakeries": int(coverage["bakeries"].min()),
        "current_daily_max_bakeries": int(coverage["bakeries"].max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-panel", type=Path, required=True)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    origin = pd.Timestamp(args.origin).normalize()
    now_msk = datetime.now(ZoneInfo("Europe/Moscow"))
    if origin.date() >= now_msk.date():
        raise ValueError("Origin must be a completed Moscow business date")

    historical = pd.read_parquet(
        args.historical_panel,
        columns=[
            "date", "bakery_id", "product_id", "observed_sales_qty", "release_qty"
        ],
    )
    bakery_ids = tuple(sorted(historical["bakery_id"].astype(int).unique()))
    if not bakery_ids:
        raise ValueError("Historical panel has no bakeries")
    start = pd.Timestamp(historical["date"].max()).normalize() + pd.Timedelta(days=1)
    if start > origin:
        raise ValueError("Historical panel has no new period to freeze")
    params = {
        "start": str(start.date()),
        "end": str(origin.date()),
        "bakery_ids": bakery_ids,
    }
    extraction_started = datetime.now(ZoneInfo("UTC"))
    client = create_client(args.env_file)
    sales = client.query_df(
        """
        SELECT check_date AS date,
               toInt64OrZero(toString(bakery_id)) AS bakery_id,
               toInt64OrZero(toString(product_id)) AS product_id,
               sum(toFloat64(quantity)) AS observed_sales_qty
        FROM (
            SELECT DISTINCT check_datetime, check_date, bakery_id, product_id,
                            quantity, line_amount
            FROM Svezhar.fct_check_lines
            WHERE hex(cash_event_type) = 'D09FD180D0BED0B4D0B0D0B6D0B0'
              AND check_date BETWEEN toDate(%(start)s) AND toDate(%(end)s)
              AND toInt64OrZero(toString(bakery_id)) IN %(bakery_ids)s
        )
        GROUP BY date, bakery_id, product_id
        """,
        parameters=params,
    )
    release = client.query_df(
        """
        SELECT rd AS date, toInt64OrZero(toString(bid)) AS bakery_id,
               toInt64OrZero(toString(pid)) AS product_id,
               sum(qty) AS release_qty
        FROM (
            SELECT argMax(release_date, _updated_at) AS rd,
                   argMax(bakery_id, _updated_at) AS bid,
                   argMax(product_id, _updated_at) AS pid,
                   toFloat64(argMax(quantity, _updated_at)) AS qty,
                   argMax(is_deleted, _updated_at) AS deleted
            FROM Svezhar.fct_production_release
            WHERE release_date BETWEEN toDate(%(start)s) AND toDate(%(end)s)
            GROUP BY release_id, line_id
            HAVING deleted NOT IN ('1', 'true', 'Да')
        )
        WHERE toInt64OrZero(toString(bid)) IN %(bakery_ids)s
        GROUP BY date, bakery_id, product_id
        """,
        parameters=params,
    )
    active = client.query_df(
        """
        SELECT run_id, generated_at, model_version
        FROM forecast_runs_embedded WHERE status = 'active'
        ORDER BY generated_at DESC LIMIT 1
        """
    )
    if len(active) != 1:
        raise ValueError("Could not identify one active forecast run")
    run_id = str(active.iloc[0]["run_id"])
    incumbent = client.query_df(
        """
        SELECT forecast_date AS date,
               toInt64OrZero(toString(bakery_id)) AS bakery_id,
               toInt64OrZero(toString(product_id)) AS product_id,
               sum(toFloat64(forecast_qty)) AS prediction
        FROM sku_forecast_day_embedded
        WHERE run_id = %(run_id)s
          AND forecast_date BETWEEN toDate(%(from_date)s) AND toDate(%(to_date)s)
          AND toInt64OrZero(toString(bakery_id)) IN %(bakery_ids)s
        GROUP BY date, bakery_id, product_id
        """,
        parameters={
            "run_id": run_id,
            "from_date": str((origin + pd.Timedelta(days=1)).date()),
            "to_date": str((origin + pd.Timedelta(days=14)).date()),
            "bakery_ids": bakery_ids,
        },
    )
    extraction_finished = datetime.now(ZoneInfo("UTC"))
    combined, coverage = freeze_frames(historical, sales, release, origin=origin)
    if incumbent.empty or incumbent.duplicated(
        ["date", "bakery_id", "product_id"]
    ).any():
        raise ValueError("Incumbent forecast is absent or has duplicate keys")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    input_file = args.output_dir / "frozen_flows.parquet"
    combined.to_parquet(input_file, index=False)
    incumbent.to_parquet(args.output_dir / "incumbent.parquet", index=False)
    metadata = {
        "production_write": False,
        "origin": str(origin.date()),
        "extraction_started_utc": extraction_started.isoformat(),
        "extraction_finished_utc": extraction_finished.isoformat(),
        "first_prospective_date": str(now_msk.date() + pd.Timedelta(days=1)),
        "historical_panel": str(args.historical_panel.resolve()),
        "historical_panel_sha256": _sha256(args.historical_panel),
        "frozen_flows_sha256": _sha256(input_file),
        "bakery_count": len(bakery_ids),
        "active_run_id_at_extraction": run_id,
        "active_run_model_version": str(active.iloc[0]["model_version"]),
        "active_run_generated_at": str(active.iloc[0]["generated_at"]),
        "incumbent_rows": len(incumbent),
        "as_of_origin_fact_arrival_verified": False,
        "source_queries_are_not_atomic_snapshot": True,
        **coverage,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

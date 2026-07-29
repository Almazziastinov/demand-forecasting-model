"""Rolling backtest for conservative systematic SKU corrections."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import clickhouse_connect
import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments_v2.sku_systematic_correction import (  # noqa: E402
    CorrectionConfig,
    apply_category_neutral_corrections,
    build_correction_registry,
    forecast_metrics,
)


PILOT_BAKERY_IDS = [20, 21, 22, 28, 80, 89, 107, 221, 222, 257]
PILOT_CATEGORIES = ["Выпечка сладкая", "Выпечка сытная"]
# These recently introduced products are handled by the separate new-SKU
# workflow and must not enter the mature-SKU correction registry.
EXCLUDED_NEW_PRODUCT_IDS = {11573, 11574}
DEFAULT_OUTPUT = ROOT / "reports" / "sku_systematic_correction_backtest"


def create_client(env_file: str | Path):
    load_dotenv(env_file)
    return clickhouse_connect.get_client(
        host=os.environ["HOST"],
        port=int(os.environ["PORT"]),
        user=os.environ["USER"],
        password=os.environ["PASSWORD"],
        database=os.environ["DATABASE"],
        secure=True,
    )


def load_frame(
    client,
    *,
    date_from: pd.Timestamp,
    date_to: pd.Timestamp,
    excluded_product_ids: set[int] | None = EXCLUDED_NEW_PRODUCT_IDS,
) -> pd.DataFrame:
    fact = client.query_df(
        """
        select
            dt as date,
            toInt64(bakery_id) as bakery_id,
            any(bakery_name) as bakery_name,
            toInt64(product_id) as product_id,
            any(product_name) as product_name,
            any(category_name) as category_name,
            sum(qty_sold) as sold_qty,
            sum(qty_produced) as produced_qty,
            max(last_sale_time) as last_sale_time
        from Svezhar.mart_zero_sales_60d
        where dt between toDate(%(date_from)s) and toDate(%(date_to)s)
          and toInt64(bakery_id) in %(bakery_ids)s
        group by date, bakery_id, product_id
        """,
        parameters={
            "date_from": str(date_from.date()),
            "date_to": str(date_to.date()),
            "bakery_ids": PILOT_BAKERY_IDS,
        },
    )
    fact = fact[fact["category_name"].isin(PILOT_CATEGORIES)].copy()
    forecast = client.query_df(
        """
        select
            forecast_date as date,
            bakery_id,
            product_id,
            argMax(forecast_qty, generated_at) as forecast_qty
        from Svezhar.sku_forecast_day_snapshots
        where forecast_date between toDate(%(date_from)s) and toDate(%(date_to)s)
          and lead_days = 1
          and bakery_id in %(bakery_ids)s
        group by date, bakery_id, product_id
        """,
        parameters={
            "date_from": str(date_from.date()),
            "date_to": str(date_to.date()),
            "bakery_ids": PILOT_BAKERY_IDS,
        },
    )
    frame = fact.merge(
        forecast,
        on=["date", "bakery_id", "product_id"],
        how="left",
        validate="one_to_one",
    )
    if excluded_product_ids:
        frame = frame[
            ~frame["product_id"].isin(excluded_product_ids)
        ].copy()
    frame["forecast_qty"] = frame["forecast_qty"].fillna(0.0)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    last_sale = pd.to_datetime(frame["last_sale_time"], errors="coerce")
    day_start = last_sale.dt.normalize() + pd.Timedelta(hours=7)
    day_end = last_sale.dt.normalize() + pd.Timedelta(hours=19)
    elapsed_hours = (last_sale - day_start).dt.total_seconds() / 3600.0
    remaining_hours = (
        (day_end - last_sale).dt.total_seconds() / 3600.0
    ).clip(lower=0.0)
    full_realization = (
        (frame["produced_qty"] > 0)
        & (frame["sold_qty"] >= frame["produced_qty"] - 0.01)
        & (elapsed_hours >= 2.0)
        & (last_sale < day_end)
    )
    raw_lost = np.where(
        full_realization,
        frame["sold_qty"] / elapsed_hours * remaining_hours,
        0.0,
    )
    conservative_cap = np.maximum(frame["sold_qty"] * 1.5, 15.0)
    frame["lost_demand_qty"] = np.minimum(raw_lost, conservative_cap)
    frame["demand_qty"] = frame["sold_qty"] + frame["lost_demand_qty"]
    return frame


def rolling_backtest(
    frame: pd.DataFrame,
    *,
    date_to: pd.Timestamp,
    test_days: int,
    config: CorrectionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_start = date_to - pd.Timedelta(days=test_days - 1)
    outputs: list[pd.DataFrame] = []
    registries: list[pd.DataFrame] = []
    for forecast_date in pd.date_range(test_start, date_to, freq="D"):
        registry = build_correction_registry(
            frame,
            as_of_date=forecast_date,
            config=config,
        )
        if not registry.empty:
            registry = registry.copy()
            registry["registry_as_of"] = forecast_date
            registries.append(registry)
        day = frame[frame["date"].eq(forecast_date)].copy()
        if day.empty:
            continue
        outputs.append(apply_category_neutral_corrections(day, registry))
    result = pd.concat(outputs, ignore_index=True)
    registry_history = (
        pd.concat(registries, ignore_index=True)
        if registries
        else pd.DataFrame()
    )
    return result, registry_history


def build_summary(result: pd.DataFrame, registry: pd.DataFrame) -> dict:
    baseline = forecast_metrics(result, forecast_col="forecast_qty")
    corrected = forecast_metrics(result, forecast_col="corrected_forecast_qty")
    return {
        "definition": {
            "old_sku_guard": "minimum observed days and age",
            "demand": "sales plus conservative lost-demand estimate",
            "information_boundary": "strictly before each forecast date",
            "category_total_preserved": True,
        },
        "baseline": baseline,
        "corrected": corrected,
        "delta": {
            "wape_pct": corrected["wape_pct"] - baseline["wape_pct"],
            "abs_bias_pct": abs(corrected["bias_pct"]) - abs(baseline["bias_pct"]),
            "underforecast_qty": (
                corrected["underforecast_qty"] - baseline["underforecast_qty"]
            ),
            "overforecast_qty": (
                corrected["overforecast_qty"] - baseline["overforecast_qty"]
            ),
        },
        "registry_dates": (
            int(registry["registry_as_of"].nunique()) if not registry.empty else 0
        ),
        "registry_pairs": (
            int(
                registry[["bakery_id", "product_id"]]
                .drop_duplicates()
                .shape[0]
            )
            if not registry.empty
            else 0
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--date-to", default="2026-07-28")
    parser.add_argument("--test-days", type=int, default=28)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    date_to = pd.Timestamp(args.date_to).normalize()
    config = CorrectionConfig()
    date_from = date_to - pd.Timedelta(days=config.history_days + args.test_days)
    client = create_client(args.env_file)
    frame = load_frame(client, date_from=date_from, date_to=date_to)
    result, registry = rolling_backtest(
        frame,
        date_to=date_to,
        test_days=args.test_days,
        config=config,
    )
    summary = build_summary(result, registry)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        output / "backtest_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )
    registry.to_csv(
        output / "registry_history.csv",
        index=False,
        encoding="utf-8-sig",
    )
    final_registry = build_correction_registry(
        frame,
        as_of_date=date_to + pd.Timedelta(days=1),
        config=config,
    )
    final_registry.to_csv(
        output / "current_registry.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

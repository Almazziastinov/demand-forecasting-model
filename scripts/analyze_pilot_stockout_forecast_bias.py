"""Compare lead-1 forecast bias on clear stockout and confirmed non-stockout SKU-days.

Observed sales are censored on stockout days, so positive bias in that group is
reported as forecast headroom over realized sales, not as forecast error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.export_clickhouse_checks import create_client  # noqa: E402


DEFAULT_SIGNALS_PATH = (
    ROOT / "reports" / "pilot_mart_zero_demand_reconstruction" / "daily_stockout_signals.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "pilot_stockout_forecast_bias"
PILOT_BAKERY_IDS = [16, 20, 21, 22, 28, 80, 89, 107, 221, 222, 257]


def load_signals(path: Path) -> pd.DataFrame:
    signals = pd.read_csv(path, encoding="utf-8-sig")
    signals["date"] = pd.to_datetime(signals["date"], errors="coerce").dt.normalize()
    for column in [
        "balance_consistent",
        "hourly_daily_sales_agree",
        "is_inventory_stockout",
        "is_strong_temporal_stockout",
    ]:
        signals[column] = signals[column].fillna(False).astype(bool)
    signals["stockout_group"] = np.select(
        [
            signals["is_strong_temporal_stockout"],
            signals["balance_consistent"]
            & signals["hourly_daily_sales_agree"]
            & ~signals["is_inventory_stockout"],
        ],
        ["clear_stockout", "confirmed_non_stockout"],
        default="ambiguous",
    )
    return signals


def load_latest_lead1_forecasts(
    client,
    *,
    date_from: pd.Timestamp,
    date_to: pd.Timestamp,
) -> pd.DataFrame:
    params = {
        "date_from": str(date_from.date()),
        "date_to": str(date_to.date()),
        "bakery_ids": PILOT_BAKERY_IDS,
    }
    forecast = client.query_df(
        """
        select
            forecast_date as date,
            bakery_id,
            product_id,
            argMax(forecast_qty, generated_at) as forecast_qty,
            argMax(source_run_id, generated_at) as source_run_id,
            max(generated_at) as latest_generated_at
        from sku_forecast_day_snapshots
        where lead_days = 1
          and forecast_date between %(date_from)s and %(date_to)s
          and bakery_id in %(bakery_ids)s
        group by date, bakery_id, product_id
        """,
        parameters=params,
    )
    forecast["date"] = pd.to_datetime(forecast["date"], errors="coerce").dt.normalize()
    return forecast


def build_comparison(signals: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    eligible = signals[signals["stockout_group"] != "ambiguous"].copy()
    eligible = eligible[eligible["date"].isin(forecast["date"].dropna().unique())].copy()
    keys = ["date", "bakery_id", "product_id"]
    comparison = eligible.merge(forecast, on=keys, how="left", validate="one_to_one")
    comparison["has_forecast"] = comparison["forecast_qty"].notna()
    comparison["forecast_qty"] = comparison["forecast_qty"].fillna(0.0)
    comparison["bias_qty"] = comparison["forecast_qty"] - comparison["daily_sold"]
    comparison["forecast_to_sales_ratio"] = comparison["forecast_qty"] / comparison[
        "daily_sold"
    ].replace(0.0, np.nan)
    comparison["bias_pct_of_sales"] = 100.0 * comparison["bias_qty"] / comparison[
        "daily_sold"
    ].replace(0.0, np.nan)
    return comparison


def summarize_group(frame: pd.DataFrame) -> dict[str, Any]:
    actual = float(frame["daily_sold"].sum())
    forecast = float(frame["forecast_qty"].sum())
    ratios = frame["forecast_to_sales_ratio"].dropna()
    covered = frame[frame["has_forecast"]]
    covered_actual = float(covered["daily_sold"].sum())
    covered_forecast = float(covered["forecast_qty"].sum())
    return {
        "sku_days": int(len(frame)),
        "dates": int(frame["date"].nunique()),
        "bakeries": int(frame["bakery_id"].nunique()),
        "forecast_coverage": float(frame["has_forecast"].mean()) if len(frame) else 0.0,
        "observed_sales": actual,
        "forecast_qty": forecast,
        "total_bias_qty": forecast - actual,
        "aggregate_bias_pct_of_sales": 100.0 * (forecast - actual) / actual if actual else None,
        "covered_aggregate_bias_pct_of_sales": (
            100.0 * (covered_forecast - covered_actual) / covered_actual
            if covered_actual
            else None
        ),
        "mean_bias_qty_per_sku_day": float(frame["bias_qty"].mean()) if len(frame) else None,
        "median_bias_qty_per_sku_day": float(frame["bias_qty"].median()) if len(frame) else None,
        "median_forecast_to_sales_ratio": float(ratios.median()) if len(ratios) else None,
        "positive_bias_share": float((frame["bias_qty"] > 0).mean()) if len(frame) else None,
        "below_observed_share": float((frame["bias_qty"] < 0).mean()) if len(frame) else None,
    }


def summarize_by(comparison: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in comparison.groupby(columns, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(columns, key_values, strict=True))
        row.update(summarize_group(group))
        rows.append(row)
    return pd.DataFrame(rows)


def select_manual_cases(comparison: pd.DataFrame, *, per_group: int = 50) -> pd.DataFrame:
    selected = []
    for group_name, group in comparison.groupby("stockout_group"):
        positive = group.sort_values("bias_qty", ascending=False).head(per_group).copy()
        positive["case_type"] = "largest_positive_bias"
        negative = group.sort_values("bias_qty").head(per_group).copy()
        negative["case_type"] = "largest_negative_bias"
        selected.extend([positive, negative])
    return pd.concat(selected, ignore_index=True).sort_values(
        ["stockout_group", "case_type", "bias_qty"],
        ascending=[True, True, False],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze forecast bias by clear stockout status")
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--signals-path", default=str(DEFAULT_SIGNALS_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signals = load_signals(Path(args.signals_path))
    client = create_client(args.env_file)
    forecast = load_latest_lead1_forecasts(
        client,
        date_from=signals["date"].min(),
        date_to=signals["date"].max(),
    )
    comparison = build_comparison(signals, forecast)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_group = summarize_by(comparison, ["stockout_group"])
    by_bakery = summarize_by(comparison, ["stockout_group", "bakery_id", "bakery_name"])
    by_category = summarize_by(comparison, ["stockout_group", "category_name"])
    by_date = summarize_by(comparison, ["stockout_group", "date"])
    manual = select_manual_cases(comparison)

    comparison.to_csv(output_dir / "sku_day_comparison.csv", index=False, encoding="utf-8-sig")
    by_group.to_csv(output_dir / "by_group.csv", index=False, encoding="utf-8-sig")
    by_bakery.to_csv(output_dir / "by_bakery.csv", index=False, encoding="utf-8-sig")
    by_category.to_csv(output_dir / "by_category.csv", index=False, encoding="utf-8-sig")
    by_date.to_csv(output_dir / "by_date.csv", index=False, encoding="utf-8-sig")
    manual.to_csv(output_dir / "manual_cases.csv", index=False, encoding="utf-8-sig")

    summary = {
        "forecast_date_min": str(forecast["date"].min().date()) if len(forecast) else None,
        "forecast_date_max": str(forecast["date"].max().date()) if len(forecast) else None,
        "forecast_rows": int(len(forecast)),
        "eligible_rows": int(len(comparison)),
        "ambiguous_signal_rows_excluded": int((signals["stockout_group"] == "ambiguous").sum()),
        "groups": by_group.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

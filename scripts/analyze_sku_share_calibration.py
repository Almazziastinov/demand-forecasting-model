"""Measure SKU allocation-share calibration by stockout status."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_stockout_allocation_failures import (  # noqa: E402
    load_bakery_context,
)
from scripts.export_clickhouse_checks import create_client  # noqa: E402

DEFAULT_INPUT = (
    ROOT / "reports" / "pilot_stockout_forecast_bias" / "sku_day_comparison.csv"
)
DEFAULT_OUTPUT = ROOT / "reports" / "sku_share_calibration"


def build_share_comparison(
    frame: pd.DataFrame, bakery: pd.DataFrame, sku_totals: pd.DataFrame
) -> pd.DataFrame:
    work = frame.merge(
        bakery, on=["date", "bakery_id"], how="left", validate="many_to_one"
    )
    work = work.merge(
        sku_totals, on=["date", "bakery_id"], how="left", validate="many_to_one"
    )
    work["forecast_share"] = work["forecast_qty"] / work[
        "sku_forecast_total_qty"
    ].replace(0.0, pd.NA)
    work["observed_share"] = work["daily_sold"] / work["bakery_actual_qty"].replace(
        0.0, pd.NA
    )
    work["allocated_qty_at_actual_bakery_total"] = (
        work["forecast_share"] * work["bakery_actual_qty"]
    )
    work["allocation_bias_qty"] = (
        work["allocated_qty_at_actual_bakery_total"] - work["daily_sold"]
    )
    work["allocation_bias_pct"] = (
        100.0 * work["allocation_bias_qty"] / work["daily_sold"].replace(0.0, pd.NA)
    )
    return work


def summarize_group(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(columns + ["stockout_group"], dropna=False):
        values = keys if isinstance(keys, tuple) else (keys,)
        sold = float(group["daily_sold"].sum())
        allocated = float(group["allocated_qty_at_actual_bakery_total"].sum())
        row = dict(zip(columns + ["stockout_group"], values, strict=True))
        row.update(
            {
                "sku_days": int(len(group)),
                "bakeries": int(group["bakery_id"].nunique()),
                "observed_sales": sold,
                "allocated_qty_at_actual_bakery_total": allocated,
                "allocation_bias_qty": allocated - sold,
                "allocation_bias_pct": 100.0 * (allocated - sold) / sold
                if sold
                else None,
                "median_daily_bias_pct": float(group["allocation_bias_pct"].median()),
                "below_observed_share": float(
                    (group["allocation_bias_qty"] < -0.5).mean()
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def classify_skus(summary: pd.DataFrame) -> pd.DataFrame:
    index = ["product_id", "product_name", "category_name"]
    values = [
        "sku_days",
        "observed_sales",
        "allocation_bias_qty",
        "allocation_bias_pct",
        "median_daily_bias_pct",
        "below_observed_share",
    ]
    wide = summary.pivot(index=index, columns="stockout_group", values=values)
    wide.columns = [f"{metric}__{group}" for metric, group in wide.columns]
    wide = wide.reset_index()
    normal_bias = wide.get("allocation_bias_pct__confirmed_non_stockout")
    stockout_bias = wide.get("allocation_bias_pct__clear_stockout")
    normal_days = wide.get("sku_days__confirmed_non_stockout", 0)
    wide["calibration_diagnosis"] = "insufficient_or_mixed"
    enough = normal_days.fillna(0).ge(5)
    wide.loc[enough & normal_bias.lt(-5.0), "calibration_diagnosis"] = (
        "mean_share_underallocated"
    )
    wide.loc[
        enough & normal_bias.ge(-5.0) & stockout_bias.lt(-5.0),
        "calibration_diagnosis",
    ] = "regime_shift_not_captured"
    wide.loc[
        enough & normal_bias.ge(-5.0) & stockout_bias.ge(-5.0),
        "calibration_diagnosis",
    ] = "not_systematically_underallocated"
    return wide


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SKU share calibration")
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input, encoding="utf-8-sig")
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame[
        frame["stockout_group"].isin(["clear_stockout", "confirmed_non_stockout"])
    ]
    bakery, totals = load_bakery_context(
        create_client(args.env_file),
        str(frame["date"].min().date()),
        str(frame["date"].max().date()),
        sorted(frame["bakery_id"].unique().tolist()),
    )
    comparison = build_share_comparison(frame, bakery, totals)
    by_status = summarize_group(comparison, [])
    by_sku_status = summarize_group(
        comparison, ["product_id", "product_name", "category_name"]
    )
    by_pair_status = summarize_group(
        comparison, ["bakery_id", "bakery_name", "product_id", "product_name"]
    )
    sku = classify_skus(by_sku_status)
    counts = sku["calibration_diagnosis"].value_counts().to_dict()
    payload = {
        "rows": int(len(comparison)),
        "sku_count": int(len(sku)),
        "diagnosis_counts": {str(k): int(v) for k, v in counts.items()},
        "status_summary": by_status.to_dict(orient="records"),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(
        output / "sku_day_share_comparison.csv", index=False, encoding="utf-8-sig"
    )
    by_sku_status.to_csv(
        output / "sku_by_status.csv", index=False, encoding="utf-8-sig"
    )
    by_pair_status.to_csv(
        output / "bakery_sku_by_status.csv", index=False, encoding="utf-8-sig"
    )
    sku.to_csv(
        output / "sku_calibration_diagnosis.csv", index=False, encoding="utf-8-sig"
    )
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

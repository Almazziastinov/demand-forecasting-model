"""Separate accepted stockouts into model and bakery execution causes.

The accepted stockout population is the existing ``clear_stockout`` group:
the SKU stopped selling at least two hours earlier than usual while the bakery
continued trading.  Since post-stockout demand is censored, this analysis only
uses conclusions that are identifiable from observed sales, forecast and
production quantities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "reports" / "pilot_stockout_forecast_bias" / "sku_day_comparison.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "pilot_stockout_responsibility"

MODEL_UNDERFORECAST = "confirmed_model_underforecast"
BAKERY_UNDERPRODUCTION = "bakery_produced_below_forecast"
FORECAST_MET_STILL_STOCKOUT = "forecast_met_but_stockout"


def classify_stockouts(
    comparison: pd.DataFrame, *, tolerance: float = 0.5
) -> pd.DataFrame:
    """Classify accepted stockouts using only observable lower bounds."""
    stockouts = comparison[comparison["stockout_group"] == "clear_stockout"].copy()
    stockouts["forecast_qty"] = pd.to_numeric(
        stockouts["forecast_qty"], errors="coerce"
    ).fillna(0.0)
    stockouts["daily_sold"] = pd.to_numeric(
        stockouts["daily_sold"], errors="coerce"
    ).fillna(0.0)
    stockouts["qty_produced"] = pd.to_numeric(
        stockouts["qty_produced"], errors="coerce"
    ).fillna(0.0)

    proven_model_miss = stockouts["forecast_qty"] < stockouts["daily_sold"] - tolerance
    execution_gap = (~proven_model_miss) & (
        stockouts["qty_produced"] < stockouts["forecast_qty"] - tolerance
    )
    stockouts["responsibility_group"] = FORECAST_MET_STILL_STOCKOUT
    stockouts.loc[execution_gap, "responsibility_group"] = BAKERY_UNDERPRODUCTION
    stockouts.loc[proven_model_miss, "responsibility_group"] = MODEL_UNDERFORECAST

    stockouts["confirmed_model_shortfall_qty"] = 0.0
    stockouts.loc[proven_model_miss, "confirmed_model_shortfall_qty"] = (
        stockouts.loc[proven_model_miss, "daily_sold"]
        - stockouts.loc[proven_model_miss, "forecast_qty"]
    )
    stockouts["bakery_execution_gap_qty"] = 0.0
    stockouts.loc[execution_gap, "bakery_execution_gap_qty"] = (
        stockouts.loc[execution_gap, "forecast_qty"]
        - stockouts.loc[execution_gap, "qty_produced"]
    )
    stockouts["forecast_headroom_over_observed_qty"] = (
        stockouts["forecast_qty"] - stockouts["daily_sold"]
    )
    stockouts["production_to_forecast_ratio"] = stockouts["qty_produced"] / stockouts[
        "forecast_qty"
    ].replace(0.0, pd.NA)
    return stockouts


def summarize_group(frame: pd.DataFrame, *, total: int) -> dict[str, Any]:
    production_ratio = frame["production_to_forecast_ratio"].dropna()
    return {
        "stockout_cases": int(len(frame)),
        "share_of_stockouts": float(len(frame) / total) if total else 0.0,
        "bakeries": int(frame["bakery_id"].nunique()),
        "products": int(frame["product_id"].nunique()),
        "observed_sales": float(frame["daily_sold"].sum()),
        "forecast_qty": float(frame["forecast_qty"].sum()),
        "produced_qty": float(frame["qty_produced"].sum()),
        "confirmed_model_shortfall_qty": float(
            frame["confirmed_model_shortfall_qty"].sum()
        ),
        "bakery_execution_gap_qty": float(frame["bakery_execution_gap_qty"].sum()),
        "median_hours_early": float(frame["last_hour_gap"].median()),
        "median_bakery_sales_after_last": float(
            frame["bakery_sales_after_last"].median()
        ),
        "median_production_to_forecast_ratio": (
            float(production_ratio.median()) if len(production_ratio) else None
        ),
    }


def summarize(stockouts: pd.DataFrame) -> pd.DataFrame:
    total = len(stockouts)
    rows = []
    for name, group in stockouts.groupby("responsibility_group", sort=False):
        row = {"responsibility_group": name}
        row.update(summarize_group(group, total=total))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("stockout_cases", ascending=False)


def summarize_dimension(stockouts: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    total = len(stockouts)
    rows = []
    for keys, group in stockouts.groupby(
        columns + ["responsibility_group"], dropna=False
    ):
        values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(columns + ["responsibility_group"], values, strict=True))
        row.update(summarize_group(group, total=total))
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attribute accepted stockouts to model or execution"
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--tolerance", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = pd.read_csv(args.input, encoding="utf-8-sig")
    stockouts = classify_stockouts(comparison, tolerance=args.tolerance)
    by_group = summarize(stockouts)
    by_bakery = summarize_dimension(stockouts, ["bakery_id", "bakery_name"])
    by_product = summarize_dimension(
        stockouts, ["product_id", "product_name", "category_name"]
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stockouts.to_csv(
        output_dir / "stockout_cases_classified.csv", index=False, encoding="utf-8-sig"
    )
    by_group.to_csv(output_dir / "by_group.csv", index=False, encoding="utf-8-sig")
    by_bakery.to_csv(output_dir / "by_bakery.csv", index=False, encoding="utf-8-sig")
    by_product.to_csv(output_dir / "by_product.csv", index=False, encoding="utf-8-sig")

    payload = {
        "definition": {
            "accepted_group": "clear_stockout",
            "manual_confirmation_required": False,
            "quantity_tolerance": args.tolerance,
            "interpretation": (
                "Observed sales are a lower bound on latent demand after stockout."
            ),
        },
        "stockout_cases": int(len(stockouts)),
        "groups": by_group.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

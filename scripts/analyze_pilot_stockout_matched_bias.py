"""Match clear stockout SKU-days to comparable confirmed non-stockout days."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = (
    ROOT / "reports" / "pilot_stockout_forecast_bias" / "sku_day_comparison.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "pilot_stockout_matched_bias"
MATCH_KEYS = ["bakery_id", "product_id", "dow"]


def load_comparison(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    for column in [
        "daily_sold",
        "forecast_qty",
        "bias_qty",
        "qty_produced",
        "last_sale_hour",
        "normal_last_hour",
        "normal_days",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def build_matches(
    comparison: pd.DataFrame,
    *,
    max_date_gap_days: int = 28,
    production_tolerance: float = 0.25,
    matches_per_case: int = 3,
) -> pd.DataFrame:
    stockouts = comparison[comparison["stockout_group"] == "clear_stockout"].copy()
    controls = comparison[
        comparison["stockout_group"] == "confirmed_non_stockout"
    ].copy()
    control_groups = {
        keys: group.copy()
        for keys, group in controls.groupby(MATCH_KEYS, sort=False)
    }
    matches: list[pd.DataFrame] = []

    for stockout_index, stockout in stockouts.iterrows():
        keys = tuple(stockout[column] for column in MATCH_KEYS)
        candidates = control_groups.get(keys)
        if candidates is None or candidates.empty or stockout["qty_produced"] <= 0:
            continue
        work = candidates.copy()
        work["date_gap_days"] = (work["date"] - stockout["date"]).abs().dt.days
        work["production_gap_ratio"] = (
            (work["qty_produced"] - stockout["qty_produced"]).abs()
            / max(float(stockout["qty_produced"]), 1.0)
        )
        work = work[
            work["date_gap_days"].le(max_date_gap_days)
            & work["production_gap_ratio"].le(production_tolerance)
        ].sort_values(["date_gap_days", "production_gap_ratio", "date"])
        if work.empty:
            continue
        selected = work.head(matches_per_case).copy()
        selected["stockout_index"] = stockout_index
        selected["stockout_date"] = stockout["date"]
        selected["stockout_sold"] = stockout["daily_sold"]
        selected["stockout_forecast"] = stockout["forecast_qty"]
        selected["stockout_produced"] = stockout["qty_produced"]
        selected["stockout_stock_balance"] = stockout["stock_balance"]
        selected["stockout_bias_qty"] = stockout["bias_qty"]
        selected["stockout_last_sale_hour"] = stockout["last_sale_hour"]
        selected["stockout_normal_last_hour"] = stockout["normal_last_hour"]
        selected["stockout_bakery_sales_after_last"] = stockout[
            "bakery_sales_after_last"
        ]
        selected["stockout_normal_days"] = stockout["normal_days"]
        selected["stockout_source_run_id"] = stockout["source_run_id"]
        selected["stockout_forecast_to_sales"] = (
            stockout["forecast_qty"] / stockout["daily_sold"]
            if stockout["daily_sold"] > 0
            else np.nan
        )
        selected["control_forecast_to_sales"] = selected["forecast_qty"] / selected[
            "daily_sold"
        ].replace(0.0, np.nan)
        matches.append(selected)

    if not matches:
        return pd.DataFrame()
    return pd.concat(matches, ignore_index=True)


def aggregate_matched_cases(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame()
    group_columns = [
        "stockout_index",
        "stockout_date",
        "bakery_id",
        "bakery_name",
        "product_id",
        "product_name",
        "category_name",
        "dow",
        "stockout_sold",
        "stockout_forecast",
        "stockout_produced",
        "stockout_stock_balance",
        "stockout_bias_qty",
        "stockout_last_sale_hour",
        "stockout_normal_last_hour",
        "stockout_bakery_sales_after_last",
        "stockout_normal_days",
        "stockout_source_run_id",
        "stockout_forecast_to_sales",
    ]
    cases = matches.groupby(group_columns, as_index=False, dropna=False).agg(
        matched_days=("date", "nunique"),
        control_sold_median=("daily_sold", "median"),
        control_forecast_median=("forecast_qty", "median"),
        control_produced_median=("qty_produced", "median"),
        control_bias_qty_median=("bias_qty", "median"),
        control_forecast_to_sales_median=("control_forecast_to_sales", "median"),
        max_match_date_gap=("date_gap_days", "max"),
        max_production_gap_ratio=("production_gap_ratio", "max"),
    )
    cases["forecast_to_sales_ratio_delta"] = (
        cases["stockout_forecast_to_sales"]
        - cases["control_forecast_to_sales_median"]
    )
    cases["forecast_minus_produced"] = (
        cases["stockout_forecast"] - cases["stockout_produced"]
    )
    cases["forecast_below_observed"] = (
        cases["stockout_forecast"] < cases["stockout_sold"]
    )
    cases["forecast_not_above_produced"] = (
        cases["stockout_forecast"] <= cases["stockout_produced"]
    )
    cases["stockout_ratio_below_control"] = (
        cases["forecast_to_sales_ratio_delta"] < 0
    )
    return cases


def summarize(cases: pd.DataFrame, *, total_stockout_cases: int) -> dict[str, Any]:
    if cases.empty:
        return {
            "total_clear_stockout_cases": total_stockout_cases,
            "matched_cases": 0,
        }
    return {
        "total_clear_stockout_cases": total_stockout_cases,
        "matched_cases": int(len(cases)),
        "matched_case_coverage": float(len(cases) / total_stockout_cases)
        if total_stockout_cases
        else 0.0,
        "median_stockout_forecast_to_sales": float(
            cases["stockout_forecast_to_sales"].median()
        ),
        "median_matched_non_stockout_forecast_to_sales": float(
            cases["control_forecast_to_sales_median"].median()
        ),
        "median_ratio_delta_stockout_minus_control": float(
            cases["forecast_to_sales_ratio_delta"].median()
        ),
        "stockout_ratio_below_matched_control_share": float(
            cases["stockout_ratio_below_control"].mean()
        ),
        "forecast_below_observed_cases": int(cases["forecast_below_observed"].sum()),
        "forecast_below_observed_share": float(cases["forecast_below_observed"].mean()),
        "forecast_not_above_produced_cases": int(
            cases["forecast_not_above_produced"].sum()
        ),
        "forecast_not_above_produced_share": float(
            cases["forecast_not_above_produced"].mean()
        ),
        "median_forecast_minus_produced": float(cases["forecast_minus_produced"].median()),
    }


def select_manual_cases(cases: pd.DataFrame, *, count: int = 20) -> pd.DataFrame:
    if cases.empty:
        return cases
    guaranteed_under = cases.sort_values("stockout_bias_qty").head(count).copy()
    guaranteed_under["review_bucket"] = "forecast_below_observed"
    weakest_relative = cases.sort_values("forecast_to_sales_ratio_delta").head(count).copy()
    weakest_relative["review_bucket"] = "largest_gap_vs_matched_non_stockout"
    strongest_headroom = cases.sort_values("stockout_bias_qty", ascending=False).head(count).copy()
    strongest_headroom["review_bucket"] = "largest_stockout_headroom"
    return (
        pd.concat([guaranteed_under, weakest_relative, strongest_headroom], ignore_index=True)
        .drop_duplicates(["stockout_index", "review_bucket"])
        .sort_values(["review_bucket", "stockout_bias_qty"])
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matched bias analysis for clear stockout SKU-days")
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-date-gap-days", type=int, default=28)
    parser.add_argument("--production-tolerance", type=float, default=0.25)
    parser.add_argument("--matches-per-case", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = load_comparison(Path(args.input_path))
    matches = build_matches(
        comparison,
        max_date_gap_days=args.max_date_gap_days,
        production_tolerance=args.production_tolerance,
        matches_per_case=args.matches_per_case,
    )
    cases = aggregate_matched_cases(matches)
    stockout_count = int((comparison["stockout_group"] == "clear_stockout").sum())
    summary = summarize(cases, total_stockout_cases=stockout_count)
    summary.update(
        {
            "max_date_gap_days": args.max_date_gap_days,
            "production_tolerance": args.production_tolerance,
            "matches_per_case": args.matches_per_case,
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matches.to_csv(output_dir / "matched_control_rows.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(output_dir / "matched_cases.csv", index=False, encoding="utf-8-sig")
    select_manual_cases(cases).to_csv(
        output_dir / "manual_review_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

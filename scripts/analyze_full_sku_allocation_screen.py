"""Screen all researched pilot SKU/bakery pairs for allocation problems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "reports" / "sku_share_calibration" / "sku_day_share_comparison.csv"
)
DEFAULT_OUTPUT = ROOT / "reports" / "full_sku_allocation_screen"
FIX_DATE = pd.Timestamp("2026-07-15")


def aggregate_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate clean non-stockout evidence at bakery/SKU level."""
    normal = frame[frame["stockout_group"].eq("confirmed_non_stockout")].copy()
    pairs = (
        normal.groupby(
            ["bakery_id", "bakery_name", "product_id", "product_name"],
            as_index=False,
        )
        .agg(
            normal_days=("date", "size"),
            observed_sales=("daily_sold", "sum"),
            allocated_qty=("allocated_qty_at_actual_bakery_total", "sum"),
            forecast_qty=("forecast_qty", "sum"),
        )
    )
    pairs["allocation_bias_qty"] = pairs["allocated_qty"] - pairs["observed_sales"]
    pairs["allocation_bias_pct"] = np.where(
        pairs["observed_sales"].gt(0),
        100.0 * pairs["allocation_bias_qty"] / pairs["observed_sales"],
        np.nan,
    )
    pairs["confirmed_deficit_qty"] = (-pairs["allocation_bias_qty"]).clip(lower=0)
    pairs["has_enough_evidence"] = pairs["normal_days"].ge(5)
    pairs["issue_type"] = "no_material_issue"
    problem = pairs["has_enough_evidence"] & pairs["allocation_bias_pct"].lt(-10.0)
    missing = problem & pairs["allocated_qty"].abs().lt(0.01)
    pairs.loc[missing, "issue_type"] = "missing_allocation"
    pairs.loc[problem & ~missing, "issue_type"] = "persistent_local_underallocation"
    pairs.loc[~pairs["has_enough_evidence"], "issue_type"] = "insufficient_evidence"
    return pairs


def aggregate_skus(frame: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Combine pair problems with stockout-vs-normal SKU regime evidence."""
    status = (
        frame.groupby(["product_id", "product_name", "stockout_group"], as_index=False)
        .agg(
            sku_days=("date", "size"),
            observed_sales=("daily_sold", "sum"),
            allocated_qty=("allocated_qty_at_actual_bakery_total", "sum"),
        )
    )
    status["allocation_bias_pct"] = np.where(
        status["observed_sales"].gt(0),
        100.0
        * (status["allocated_qty"] - status["observed_sales"])
        / status["observed_sales"],
        np.nan,
    )
    wide = status.pivot(
        index=["product_id", "product_name"],
        columns="stockout_group",
        values=["sku_days", "observed_sales", "allocation_bias_pct"],
    )
    wide.columns = [f"{metric}__{group}" for metric, group in wide.columns]
    wide = wide.reset_index()

    problems = pairs[pairs["issue_type"].isin(
        ["missing_allocation", "persistent_local_underallocation"]
    )]
    pair_summary = (
        problems.groupby(["product_id", "product_name"], as_index=False)
        .agg(
            problem_bakeries=("bakery_id", "nunique"),
            confirmed_pair_deficit_qty=("confirmed_deficit_qty", "sum"),
            missing_allocation_pairs=(
                "issue_type",
                lambda s: int(s.eq("missing_allocation").sum()),
            ),
            underallocated_pairs=(
                "issue_type",
                lambda s: int(s.eq("persistent_local_underallocation").sum()),
            ),
        )
    )
    result = wide.merge(pair_summary, on=["product_id", "product_name"], how="left")
    for column in [
        "problem_bakeries",
        "confirmed_pair_deficit_qty",
        "missing_allocation_pairs",
        "underallocated_pairs",
    ]:
        result[column] = result[column].fillna(0)
    normal_bias = result["allocation_bias_pct__confirmed_non_stockout"]
    stockout_bias = result["allocation_bias_pct__clear_stockout"]
    result["sku_issue_type"] = "no_material_issue"
    result.loc[result["missing_allocation_pairs"].gt(0), "sku_issue_type"] = (
        "missing_allocation"
    )
    result.loc[
        result["missing_allocation_pairs"].eq(0)
        & result["underallocated_pairs"].gt(0),
        "sku_issue_type",
    ] = "persistent_local_underallocation"
    result.loc[
        result["missing_allocation_pairs"].eq(0)
        & result["underallocated_pairs"].eq(0)
        & normal_bias.ge(-5.0)
        & stockout_bias.lt(-5.0),
        "sku_issue_type",
    ] = "stockout_regime_shift"
    return result.sort_values(
        ["confirmed_pair_deficit_qty", "problem_bakeries"], ascending=False
    )


def summarize_fix_window(frame: pd.DataFrame) -> pd.DataFrame:
    """Show whether zero allocation remains after the 2026-07-15 ordering fix."""
    work = frame[frame["stockout_group"].eq("confirmed_non_stockout")].copy()
    work["period"] = np.where(work["date"].lt(FIX_DATE), "before_fix", "after_fix")
    work["missing_allocation"] = work["forecast_share"].fillna(0).abs().lt(1e-12)
    return (
        work.groupby(["period", "bakery_id"], as_index=False)
        .agg(
            sku_days=("date", "size"),
            missing_allocation_days=("missing_allocation", "sum"),
            sales_on_missing_days=(
                "daily_sold",
                lambda s: float(s[work.loc[s.index, "missing_allocation"]].sum()),
            ),
        )
        .sort_values(["period", "sales_on_missing_days"], ascending=[True, False])
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen all pilot SKU allocation")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input, encoding="utf-8-sig")
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    pairs = aggregate_pairs(frame)
    skus = aggregate_skus(frame, pairs)
    fix_window = summarize_fix_window(frame)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output / "bakery_sku_screen.csv", index=False, encoding="utf-8-sig")
    skus.to_csv(output / "sku_screen.csv", index=False, encoding="utf-8-sig")
    fix_window.to_csv(
        output / "missing_allocation_fix_window.csv",
        index=False,
        encoding="utf-8-sig",
    )
    problem_pairs = pairs[pairs["issue_type"].isin(
        ["missing_allocation", "persistent_local_underallocation"]
    )]
    payload = {
        "scope": {
            "rows": int(len(frame)),
            "skus": int(frame["product_id"].nunique()),
            "bakeries": int(frame["bakery_id"].nunique()),
            "date_min": str(frame["date"].min().date()),
            "date_max": str(frame["date"].max().date()),
        },
        "eligible_pairs": int(pairs["has_enough_evidence"].sum()),
        "problem_pairs": int(len(problem_pairs)),
        "problem_skus": int(problem_pairs["product_id"].nunique()),
        "pair_issue_counts": {
            str(key): int(value)
            for key, value in pairs["issue_type"].value_counts().items()
        },
        "pair_issue_deficit_qty": {
            str(key): float(value)
            for key, value in problem_pairs.groupby("issue_type")[
                "confirmed_deficit_qty"
            ].sum().items()
        },
        "sku_issue_counts": {
            str(key): int(value)
            for key, value in skus["sku_issue_type"].value_counts().items()
        },
    }
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

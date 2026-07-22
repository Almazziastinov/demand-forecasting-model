"""Decompose reconstructed stockout demand into surplus reallocation and volume gaps."""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.export_clickhouse_checks import create_client  # noqa: E402

DEFAULT_BALANCE = ROOT / "reports/pilot_mart_zero_stockout_balance/inventory_balance.csv"
DEFAULT_ADJUSTMENTS = ROOT / "reports/demand_adjusted_stockout_history_all_cases/case_adjustments.csv"
DEFAULT_STOCKOUTS = ROOT / "reports/pilot_stockout_responsibility/stockout_cases_classified.csv"
DEFAULT_OUTPUT = ROOT / "reports/stockout_surplus_coverage"
KEYS = ["date", "bakery_id", "product_id"]
DAY_KEYS = ["date", "bakery_id"]


def load_two_day_products(env_file: str | Path) -> pd.DataFrame:
    client = create_client(env_file)
    frame = client.query_df(
        """
        select
            toInt64(product_id) as product_id_int,
            max(toUInt8(is_two_day)) as is_two_day,
            any(product_name) as meta_product_name
        from baking_sku_meta
        where is_active = 1
          and product_id != ''
        group by product_id_int
        """
    )
    return frame.rename(columns={"product_id_int": "product_id"})


def prepare_surplus_rows(
    balance: pd.DataFrame,
    stockouts: pd.DataFrame,
    two_day_products: pd.DataFrame,
    *,
    reserve_units: float = 1.0,
) -> pd.DataFrame:
    work = balance.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    stockout_keys = set(
        map(
            tuple,
            stockouts.assign(date=pd.to_datetime(stockouts["date"]).dt.normalize())[
                KEYS
            ].astype({"bakery_id": int, "product_id": int}).to_numpy(),
        )
    )
    work["is_clear_stockout_sku"] = [
        (row.date, int(row.bakery_id), int(row.product_id)) in stockout_keys
        for row in work.itertuples()
    ]
    meta = two_day_products[["product_id", "is_two_day"]].copy()
    meta["product_id"] = pd.to_numeric(meta["product_id"], errors="coerce").astype("Int64")
    work = work.merge(meta, on="product_id", how="left", validate="many_to_one")
    work["is_two_day"] = work["is_two_day"].fillna(0).astype(bool)
    closing = pd.to_numeric(work["stock_balance"], errors="coerce").fillna(0.0)
    work["surplus_after_reserve"] = (closing - reserve_units).clip(lower=0.0)
    non_recipient = ~work["is_clear_stockout_sku"]
    balanced = work["balance_is_consistent"].astype(bool)
    hourly_agree = work["hourly_daily_sales_agree"].astype(bool)
    one_day = ~work["is_two_day"]
    work["strict_usable_surplus"] = np.where(
        non_recipient & balanced & hourly_agree & one_day,
        work["surplus_after_reserve"],
        0.0,
    )
    work["balance_only_surplus"] = np.where(
        non_recipient & balanced & one_day,
        work["surplus_after_reserve"],
        0.0,
    )
    work["all_product_surplus"] = np.where(
        non_recipient & balanced & hourly_agree,
        work["surplus_after_reserve"],
        0.0,
    )
    last_sale = pd.to_datetime(work.get("last_sale_time"), errors="coerce")
    work["donor_last_sale_hour"] = last_sale.dt.hour
    return work


def classify_coverage(deficit: float, surplus: float) -> str:
    if deficit <= 0:
        return "no_reconstructed_deficit"
    ratio = surplus / deficit
    if ratio <= 0.10:
        return "volume_shortage_supported"
    if ratio < 0.90:
        return "mixed_supported"
    if ratio <= 1.10:
        return "allocation_balanced_supported"
    return "allocation_plus_excess_supported"


def build_day_coverage(
    adjustments: pd.DataFrame,
    surplus_rows: pd.DataFrame,
) -> pd.DataFrame:
    cases = adjustments.copy()
    cases["date"] = pd.to_datetime(cases["date"]).dt.normalize()
    deficit = cases.groupby(DAY_KEYS, as_index=False).agg(
        reconstructed_deficit=("imputed_demand", "sum"),
        recipient_cases=("product_id", "size"),
        recipient_products=("product_id", "nunique"),
        earliest_stockout_hour=("last_sale_hour", "min"),
        latest_stockout_hour=("last_sale_hour", "max"),
    )
    donors = surplus_rows.groupby(DAY_KEYS, as_index=False).agg(
        strict_usable_surplus=("strict_usable_surplus", "sum"),
        balance_only_surplus=("balance_only_surplus", "sum"),
        all_product_surplus=("all_product_surplus", "sum"),
        donor_products=("strict_usable_surplus", lambda value: int(value.gt(0).sum())),
    )
    temporal = surplus_rows.merge(
        deficit[DAY_KEYS + ["latest_stockout_hour"]],
        on=DAY_KEYS,
        how="inner",
        validate="many_to_one",
    )
    temporal["late_confirmed_surplus"] = np.where(
        temporal["donor_last_sale_hour"].ge(temporal["latest_stockout_hour"]),
        temporal["strict_usable_surplus"],
        0.0,
    )
    temporal = temporal.groupby(DAY_KEYS, as_index=False)[
        "late_confirmed_surplus"
    ].sum()
    donors = donors.merge(temporal, on=DAY_KEYS, how="left")
    result = deficit.merge(donors, on=DAY_KEYS, how="left")
    surplus_columns = [
        "strict_usable_surplus",
        "balance_only_surplus",
        "all_product_surplus",
        "late_confirmed_surplus",
    ]
    result[surplus_columns + ["donor_products"]] = result[
        surplus_columns + ["donor_products"]
    ].fillna(0.0)
    for column in surplus_columns:
        suffix = column.removesuffix("_surplus")
        result[f"{suffix}_coverage"] = result[column] / result[
            "reconstructed_deficit"
        ].replace(0.0, np.nan)
        result[f"{suffix}_allocation_component"] = np.minimum(
            result[column], result["reconstructed_deficit"]
        )
        result[f"{suffix}_volume_gap"] = (
            result["reconstructed_deficit"] - result[column]
        ).clip(lower=0.0)
        result[f"{suffix}_excess"] = (
            result[column] - result["reconstructed_deficit"]
        ).clip(lower=0.0)
        result[f"{suffix}_mechanism"] = [
            classify_coverage(deficit_value, surplus_value)
            for deficit_value, surplus_value in zip(
                result["reconstructed_deficit"], result[column], strict=True
            )
        ]
    return result.sort_values(DAY_KEYS).reset_index(drop=True)


def build_mechanism_comparison(
    adjustments: pd.DataFrame,
    day_coverage: pd.DataFrame,
) -> pd.DataFrame:
    cases = adjustments.copy()
    cases["date"] = pd.to_datetime(cases["date"]).dt.normalize()
    merged = cases.merge(
        day_coverage[DAY_KEYS + ["strict_usable_mechanism", "strict_usable_coverage"]],
        on=DAY_KEYS,
        how="left",
        validate="many_to_one",
    )
    merged["bakery_day_key"] = (
        merged["date"].astype(str) + "|" + merged["bakery_id"].astype(str)
    )
    return (
        merged.groupby(
            ["robust_case_type", "strict_usable_mechanism"],
            dropna=False,
            as_index=False,
        )
        .agg(
            cases=("product_id", "size"),
            bakery_days=("bakery_day_key", "nunique"),
            reconstructed_deficit=("imputed_demand", "sum"),
            median_coverage=("strict_usable_coverage", "median"),
        )
        .sort_values(["robust_case_type", "cases"], ascending=[True, False])
    )


def build_surplus_context_comparison(
    surplus_rows: pd.DataFrame,
    adjustments: pd.DataFrame,
) -> pd.DataFrame:
    """Compare strict closing surplus on stockout and non-stockout bakery-days."""
    day_surplus = surplus_rows.groupby(DAY_KEYS, as_index=False).agg(
        strict_usable_surplus=("strict_usable_surplus", "sum"),
        donor_products=("strict_usable_surplus", lambda value: int(value.gt(0).sum())),
    )
    adjusted_days = adjustments.copy()
    adjusted_days["date"] = pd.to_datetime(adjusted_days["date"]).dt.normalize()
    adjusted_days = adjusted_days[DAY_KEYS].drop_duplicates()
    adjusted_days["has_reconstructed_stockout"] = True
    day_surplus = day_surplus.merge(adjusted_days, on=DAY_KEYS, how="left")
    day_surplus["has_reconstructed_stockout"] = day_surplus[
        "has_reconstructed_stockout"
    ].eq(True)

    date_min = adjusted_days["date"].min()
    date_max = adjusted_days["date"].max()
    day_surplus = day_surplus[
        day_surplus["date"].between(date_min, date_max, inclusive="both")
    ].copy()
    return (
        day_surplus.groupby("has_reconstructed_stockout", as_index=False)
        .agg(
            bakery_days=("date", "size"),
            mean_surplus=("strict_usable_surplus", "mean"),
            median_surplus=("strict_usable_surplus", "median"),
            p75_surplus=("strict_usable_surplus", lambda value: value.quantile(0.75)),
            mean_donor_products=("donor_products", "mean"),
        )
        .sort_values("has_reconstructed_stockout", ascending=False)
    )


def build_summary(day_coverage: pd.DataFrame, donor_rows: pd.DataFrame) -> dict:
    valid = day_coverage[day_coverage["reconstructed_deficit"].gt(0)].copy()
    mechanism_counts = valid["strict_usable_mechanism"].value_counts().to_dict()
    late_mechanism_counts = valid["late_confirmed_mechanism"].value_counts().to_dict()
    total_deficit = float(valid["reconstructed_deficit"].sum())
    allocation = float(valid["strict_usable_allocation_component"].sum())
    volume_gap = float(valid["strict_usable_volume_gap"].sum())
    return {
        "bakery_days": int(len(day_coverage)),
        "bakery_days_with_positive_deficit": int(len(valid)),
        "date_min": str(day_coverage["date"].min().date()),
        "date_max": str(day_coverage["date"].max().date()),
        "reconstructed_deficit_units": total_deficit,
        "strict_usable_surplus_units": float(valid["strict_usable_surplus"].sum()),
        "allocation_component_units": allocation,
        "volume_gap_units": volume_gap,
        "allocation_component_share": allocation / total_deficit if total_deficit else None,
        "volume_gap_share": volume_gap / total_deficit if total_deficit else None,
        "median_strict_coverage": float(valid["strict_usable_coverage"].median()),
        "late_confirmed_surplus_units": float(
            valid["late_confirmed_surplus"].sum()
        ),
        "late_confirmed_allocation_component_units": float(
            valid["late_confirmed_allocation_component"].sum()
        ),
        "late_confirmed_allocation_component_share": float(
            valid["late_confirmed_allocation_component"].sum() / total_deficit
        )
        if total_deficit
        else None,
        "median_late_confirmed_coverage": float(
            valid["late_confirmed_coverage"].median()
        ),
        "mechanism_counts": mechanism_counts,
        "late_confirmed_mechanism_counts": late_mechanism_counts,
        "strict_donor_rows": int(donor_rows["strict_usable_surplus"].gt(0).sum()),
        "strict_donor_products": int(
            donor_rows.loc[donor_rows["strict_usable_surplus"].gt(0), "product_id"].nunique()
        ),
        "production_write": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--balance", type=Path, default=DEFAULT_BALANCE)
    parser.add_argument("--adjustments", type=Path, default=DEFAULT_ADJUSTMENTS)
    parser.add_argument("--stockouts", type=Path, default=DEFAULT_STOCKOUTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reserve-units", type=float, default=1.0)
    args = parser.parse_args()

    balance = pd.read_csv(args.balance, encoding="utf-8-sig")
    adjustments = pd.read_csv(args.adjustments, encoding="utf-8-sig")
    stockouts = pd.read_csv(args.stockouts, encoding="utf-8-sig")
    two_day = load_two_day_products(args.env_file)
    surplus_rows = prepare_surplus_rows(
        balance,
        stockouts,
        two_day,
        reserve_units=args.reserve_units,
    )
    relevant_days = adjustments.assign(
        date=pd.to_datetime(adjustments["date"]).dt.normalize()
    )[DAY_KEYS].drop_duplicates()
    donors = surplus_rows.merge(relevant_days, on=DAY_KEYS, how="inner")
    coverage = build_day_coverage(adjustments, donors)
    comparison = build_mechanism_comparison(adjustments, coverage)
    summary = build_summary(coverage, donors)
    context_comparison = build_surplus_context_comparison(surplus_rows, adjustments)
    bakery_summary = coverage.groupby("bakery_id", as_index=False).agg(
        bakery_days=("date", "size"),
        reconstructed_deficit=("reconstructed_deficit", "sum"),
        allocation_component=("strict_usable_allocation_component", "sum"),
        volume_gap=("strict_usable_volume_gap", "sum"),
        late_confirmed_allocation_component=(
            "late_confirmed_allocation_component",
            "sum",
        ),
    )
    bakery_summary["allocation_share"] = (
        bakery_summary["allocation_component"]
        / bakery_summary["reconstructed_deficit"]
    )
    bakery_summary["late_confirmed_allocation_share"] = (
        bakery_summary["late_confirmed_allocation_component"]
        / bakery_summary["reconstructed_deficit"]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(args.output_dir / "bakery_day_coverage.csv", index=False, encoding="utf-8-sig")
    donors[donors["strict_usable_surplus"].gt(0)].sort_values(
        "strict_usable_surplus", ascending=False
    ).to_csv(args.output_dir / "donor_rows.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(args.output_dir / "mechanism_comparison.csv", index=False, encoding="utf-8-sig")
    context_comparison.to_csv(
        args.output_dir / "surplus_context_comparison.csv", index=False, encoding="utf-8-sig"
    )
    bakery_summary.to_csv(
        args.output_dir / "bakery_summary.csv", index=False, encoding="utf-8-sig"
    )
    coverage.sort_values("reconstructed_deficit", ascending=False).head(100).to_csv(
        args.output_dir / "manual_review_top_deficit.csv", index=False, encoding="utf-8-sig"
    )
    donor_top = (
        donors.groupby(["product_id", "product_name"], as_index=False)["strict_usable_surplus"]
        .sum()
        .sort_values("strict_usable_surplus", ascending=False)
    )
    donor_top.to_csv(args.output_dir / "top_donor_products.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

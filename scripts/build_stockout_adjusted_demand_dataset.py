"""Build an offline demand dataset with stockout-censored observations restored."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_demand_adjusted_stockout_history import (  # noqa: E402
    build_adjusted_history,
    load_hourly_sales,
    reconstruct_cases,
)
from scripts.export_clickhouse_checks import create_client  # noqa: E402

DEFAULT_STOCKOUTS = (
    ROOT / "reports/pilot_stockout_responsibility/stockout_cases_classified.csv"
)
DEFAULT_OUTPUT = ROOT / "reports/stockout_adjusted_demand_dataset"
KEYS = ["date", "bakery_id", "product_id"]
SOURCE_COLUMNS = [
    "qty_produced",
    "stock_balance",
    "balance_consistent",
    "hourly_daily_sales_agree",
    "is_reliable_inventory_stockout",
    "is_strong_temporal_stockout",
    "normal_days",
    "normal_daily_sold",
    "normal_last_hour",
    "last_hour_gap",
    "bakery_sales_after_last",
]
DATASET_SCHEMA = {
    "contract": "stockout_censored_demand_v1",
    "grain": "date x bakery_id x product_id with positive observed sales",
    "primary_key": KEYS,
    "fields": {
        "demand_lower_bound": (
            "Observed sales; exact on clean rows and a lower bound on stockout rows."
        ),
        "imputed_demand": (
            "Estimated post-stockout demand, never taken from another SKU."
        ),
        "demand_point_estimate": "demand_lower_bound + imputed_demand.",
        "demand_upper_guardrail": (
            "Observed sales plus the configured per-case reconstruction cap."
        ),
        "is_clear_stockout": "Accepted observable stockout signal for this SKU-day.",
        "reconstruction_confidence": (
            "Evidence for the imputation: observed, high, medium, or insufficient."
        ),
        "suggested_training_weight": (
            "Offline starting weight; not a production policy."
        ),
        "target_source": (
            "Observed, reconstructed, or censored-without-estimate provenance."
        ),
    },
    "allocation_assumption": False,
}


def classify_reconstruction_confidence(reference_days: pd.Series) -> pd.Series:
    """Rate reconstruction evidence, separately from stockout confidence."""
    return pd.Series(
        np.select(
            [reference_days.ge(5), reference_days.ge(3)],
            ["high", "medium"],
            default="insufficient",
        ),
        index=reference_days.index,
        dtype="string",
    )


def build_demand_dataset(
    hourly: pd.DataFrame,
    audit: pd.DataFrame,
    stockouts: pd.DataFrame,
) -> pd.DataFrame:
    """Create one explicit observed/censored demand contract per SKU-day."""
    daily_sku, _, _ = build_adjusted_history(hourly, audit)
    result = daily_sku.rename(
        columns={
            "observed_sales": "demand_lower_bound",
            "demand_adjusted_sales": "demand_point_estimate",
        }
    ).drop(columns=["dow"])

    cases = audit[
        KEYS
        + [
            "last_sale_hour",
            "reference_days",
            "raw_imputed_demand",
            "case_cap",
            "imputed_demand",
        ]
    ].copy()
    source = stockouts.copy()
    source["date"] = pd.to_datetime(source["date"]).dt.normalize()
    available_source = [column for column in SOURCE_COLUMNS if column in source]
    cases = cases.merge(
        source[KEYS + available_source].drop_duplicates(KEYS),
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    cases["is_clear_stockout"] = True
    cases["reconstruction_confidence"] = classify_reconstruction_confidence(
        cases["reference_days"]
    )

    result = result.drop(columns=["imputed_demand"]).merge(
        cases,
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    result["is_clear_stockout"] = result["is_clear_stockout"].eq(True)
    numeric_fill = [
        "reference_days",
        "raw_imputed_demand",
        "case_cap",
        "imputed_demand",
    ]
    result[numeric_fill] = result[numeric_fill].fillna(0.0)
    result["reconstruction_confidence"] = result[
        "reconstruction_confidence"
    ].fillna("observed")
    reconstructed = result["is_clear_stockout"] & result["imputed_demand"].gt(0)
    insufficient = result["is_clear_stockout"] & result[
        "reconstruction_confidence"
    ].eq("insufficient")
    result["target_source"] = np.select(
        [reconstructed, insufficient, result["is_clear_stockout"]],
        [
            "reconstructed_hourly_profile",
            "censored_insufficient_history",
            "censored_no_post_cutoff_estimate",
        ],
        default="observed_sales",
    )
    result["suggested_training_weight"] = result[
        "reconstruction_confidence"
    ].map({"observed": 1.0, "high": 0.8, "medium": 0.5, "insufficient": 0.0})
    no_estimate = result["is_clear_stockout"] & ~reconstructed
    result.loc[no_estimate, "suggested_training_weight"] = 0.0
    result["training_eligible"] = result["suggested_training_weight"].gt(0)
    result["demand_upper_guardrail"] = np.where(
        result["is_clear_stockout"],
        result["demand_lower_bound"] + result["case_cap"],
        result["demand_lower_bound"],
    )
    result["imputation_ratio"] = result["imputed_demand"] / result[
        "demand_lower_bound"
    ].clip(lower=1.0)
    result["is_case_cap_binding"] = (
        result["is_clear_stockout"]
        & result["reference_days"].ge(3)
        & result["raw_imputed_demand"].gt(result["case_cap"] + 1e-9)
    )
    ordered = [
        "date",
        "bakery_id",
        "product_id",
        "product_name",
        "demand_lower_bound",
        "imputed_demand",
        "demand_point_estimate",
        "demand_upper_guardrail",
        "is_clear_stockout",
        "target_source",
        "reconstruction_confidence",
        "suggested_training_weight",
        "training_eligible",
        "last_sale_hour",
        "reference_days",
        "raw_imputed_demand",
        "case_cap",
        "imputation_ratio",
        "is_case_cap_binding",
    ]
    ordered += [column for column in available_source if column not in ordered]
    return result[ordered].sort_values(KEYS).reset_index(drop=True)


def summarize_dataset(dataset: pd.DataFrame) -> dict[str, object]:
    stockouts = dataset[dataset["is_clear_stockout"]]
    adjusted = stockouts[stockouts["imputed_demand"].gt(0)]
    duplicate_rows = int(dataset.duplicated(KEYS).sum())
    point_below_observed = int(
        dataset["demand_point_estimate"].lt(dataset["demand_lower_bound"]).sum()
    )
    return {
        "dataset_contract": "stockout_censored_demand_v1",
        "rows": int(len(dataset)),
        "date_min": str(pd.to_datetime(dataset["date"]).min().date()),
        "date_max": str(pd.to_datetime(dataset["date"]).max().date()),
        "bakeries": int(dataset["bakery_id"].nunique()),
        "products": int(dataset["product_id"].nunique()),
        "clear_stockout_rows": int(len(stockouts)),
        "adjusted_stockout_rows": int(len(adjusted)),
        "unadjusted_censored_rows": int(len(stockouts) - len(adjusted)),
        "imputed_demand_units": float(dataset["imputed_demand"].sum()),
        "observed_demand_units": float(dataset["demand_lower_bound"].sum()),
        "point_demand_units": float(dataset["demand_point_estimate"].sum()),
        "point_uplift_pct": float(
            dataset["imputed_demand"].sum()
            / dataset["demand_lower_bound"].sum()
            * 100.0
        ),
        "cap_binding_stockout_rows": int(stockouts["is_case_cap_binding"].sum()),
        "cap_binding_stockout_share": float(stockouts["is_case_cap_binding"].mean()),
        "confidence_counts": stockouts["reconstruction_confidence"]
        .value_counts()
        .to_dict(),
        "target_source_counts": dataset["target_source"].value_counts().to_dict(),
        "quality": {
            "duplicate_key_rows": duplicate_rows,
            "negative_observed_rows": int(dataset["demand_lower_bound"].lt(0).sum()),
            "negative_imputation_rows": int(dataset["imputed_demand"].lt(0).sum()),
            "point_below_observed_rows": point_below_observed,
        },
        "contains_allocation_assumption": False,
        "production_write": False,
    }


def build_stockout_diagnostics(
    dataset: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    stockouts = dataset[dataset["is_clear_stockout"]].copy()
    return (
        stockouts.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            stockout_rows=("product_id", "size"),
            adjusted_rows=("imputed_demand", lambda value: int(value.gt(0).sum())),
            observed_demand=("demand_lower_bound", "sum"),
            imputed_demand=("imputed_demand", "sum"),
            median_imputation=("imputed_demand", "median"),
            median_imputation_ratio=("imputation_ratio", "median"),
            cap_binding_rows=("is_case_cap_binding", "sum"),
        )
        .sort_values("imputed_demand", ascending=False)
    )


def build_cap_sensitivity(audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    observed = pd.to_numeric(audit["daily_sold_observed"], errors="coerce").fillna(0.0)
    raw = pd.to_numeric(audit["raw_imputed_demand"], errors="coerce").fillna(0.0)
    eligible = pd.to_numeric(audit["reference_days"], errors="coerce").ge(3)
    for ratio in (0.50, 0.75, 1.00):
        for unit_cap in (10.0, 20.0):
            cap = np.minimum(unit_cap, np.maximum(observed, 4.0) * ratio)
            estimate = np.where(eligible, np.minimum(raw, cap), 0.0)
            rows.append(
                {
                    "max_case_uplift_ratio": ratio,
                    "max_case_uplift_units": unit_cap,
                    "imputed_demand_units": float(estimate.sum()),
                    "cap_binding_rows": int((eligible & raw.gt(cap + 1e-9)).sum()),
                    "adjusted_rows": int(np.count_nonzero(estimate > 0)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--stockouts", type=Path, default=DEFAULT_STOCKOUTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lookback-days", type=int, default=42)
    parser.add_argument("--min-reference-days", type=int, default=3)
    parser.add_argument("--max-case-uplift-ratio", type=float, default=0.75)
    parser.add_argument("--max-case-uplift-units", type=float, default=20.0)
    args = parser.parse_args()

    stockouts = pd.read_csv(args.stockouts, encoding="utf-8-sig")
    stockouts["date"] = pd.to_datetime(stockouts["date"]).dt.normalize()
    if "stockout_group" in stockouts:
        stockouts = stockouts[stockouts["stockout_group"].eq("clear_stockout")].copy()
    if stockouts.empty:
        raise ValueError("No clear stockout rows were found")

    client = create_client(args.env_file)
    history_start = stockouts["date"].min() - pd.Timedelta(
        days=args.lookback_days
    )
    hourly = load_hourly_sales(
        client,
        bakery_ids=sorted(stockouts["bakery_id"].astype(int).unique().tolist()),
        date_from=str(history_start.date()),
        date_to=str(stockouts["date"].max().date()),
    )
    audit, hourly_audit = reconstruct_cases(
        hourly,
        stockouts,
        stockouts,
        lookback_days=args.lookback_days,
        min_reference_days=args.min_reference_days,
        max_case_uplift_ratio=args.max_case_uplift_ratio,
        max_case_uplift_units=args.max_case_uplift_units,
    )
    dataset = build_demand_dataset(hourly, audit, stockouts)
    summary = summarize_dataset(dataset)
    summary["reconstruction_parameters"] = {
        "lookback_days": args.lookback_days,
        "min_reference_days": args.min_reference_days,
        "max_case_uplift_ratio": args.max_case_uplift_ratio,
        "max_case_uplift_units": args.max_case_uplift_units,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(
        args.output_dir / "sku_day_demand.csv", index=False, encoding="utf-8-sig"
    )
    audit.to_csv(
        args.output_dir / "stockout_reconstruction_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    hourly_audit.to_csv(
        args.output_dir / "stockout_hourly_imputation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    build_stockout_diagnostics(dataset, ["reconstruction_confidence"]).to_csv(
        args.output_dir / "diagnostics_by_confidence.csv",
        index=False,
        encoding="utf-8-sig",
    )
    build_stockout_diagnostics(dataset, ["bakery_id"]).to_csv(
        args.output_dir / "diagnostics_by_bakery.csv",
        index=False,
        encoding="utf-8-sig",
    )
    build_stockout_diagnostics(dataset, ["product_id", "product_name"]).to_csv(
        args.output_dir / "diagnostics_by_product.csv",
        index=False,
        encoding="utf-8-sig",
    )
    build_cap_sensitivity(audit).to_csv(
        args.output_dir / "cap_sensitivity.csv", index=False, encoding="utf-8-sig"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "schema.json").write_text(
        json.dumps(DATASET_SCHEMA, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

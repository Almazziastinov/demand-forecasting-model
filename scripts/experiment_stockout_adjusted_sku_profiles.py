"""End-to-end SKU backtest for conservative stockout-adjusted demand."""

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

from scripts.experiment_demand_adjusted_profiles import (  # noqa: E402
    HOUR_KEYS,
    apply_hourly_adjustments,
    build_scored_rows,
    build_serving_profiles,
    load_hourly_raw,
)
from src.experiments_v2.build_sku_hour_share_profile import (  # noqa: E402
    BAKERY_ID_COL,
    DATE_COL,
    DOW_COL,
    HOUR_COL,
    PRODUCT_ID_COL,
    build_sku_hour_share_profile,
)

DEFAULT_RAW = ROOT / "data/raw/pilot_stg_check_lines_2026-04-30_2026-07-19.csv"
DEFAULT_DEMAND = ROOT / "reports/stockout_adjusted_demand_dataset/sku_day_demand.csv"
DEFAULT_HOURLY_IMPUTATION = (
    ROOT
    / "reports/stockout_adjusted_demand_dataset/stockout_hourly_imputation.csv"
)
DEFAULT_BAKERY_PREDICTIONS = (
    ROOT / "reports/stockout_adjusted_bakery_target_experiment/predictions.csv"
)
DEFAULT_OUTPUT = ROOT / "reports/stockout_adjusted_sku_profile_experiment"
BASELINE = "observed_total_observed_profile"
BAKERY_ONLY = "conservative_total_observed_profile"
PROFILE_ONLY = "observed_total_conservative_profile"
END_TO_END = "conservative_total_conservative_profile"
GUARDED_END_TO_END = "conservative_total_guarded_profile"
BAKERY_BASELINE_VARIANT = "observed_sales_target"
BAKERY_CONSERVATIVE_VARIANT = "conservative_reconstructed_target"
DAY_KEYS = [DATE_COL, BAKERY_ID_COL]
SKU_DAY_KEYS = [DATE_COL, BAKERY_ID_COL, PRODUCT_ID_COL]


def build_conservative_hourly_adjustments(
    demand: pd.DataFrame,
    hourly_imputation: pd.DataFrame,
) -> pd.DataFrame:
    stockouts = demand[demand["is_clear_stockout"].astype(bool)].copy()
    observed = pd.to_numeric(
        stockouts["demand_lower_bound"], errors="coerce"
    ).fillna(0.0)
    raw = pd.to_numeric(stockouts["raw_imputed_demand"], errors="coerce").fillna(
        0.0
    )
    references = pd.to_numeric(stockouts["reference_days"], errors="coerce").fillna(
        0.0
    )
    cap = np.minimum(10.0, np.maximum(observed, 4.0) * 0.50)
    stockouts["conservative_imputed"] = np.where(
        references.ge(3), np.minimum(raw, cap), 0.0
    )
    case_totals = hourly_imputation.groupby(SKU_DAY_KEYS, as_index=False).agg(
        original_hourly_imputed=("imputed_demand", "sum")
    )
    scale = stockouts[
        SKU_DAY_KEYS + ["conservative_imputed"]
    ].merge(case_totals, on=SKU_DAY_KEYS, how="left", validate="one_to_one")
    scale["original_hourly_imputed"] = scale["original_hourly_imputed"].fillna(0.0)
    scale["conservative_scale"] = np.where(
        scale["original_hourly_imputed"].gt(0),
        scale["conservative_imputed"] / scale["original_hourly_imputed"],
        0.0,
    )
    result = hourly_imputation.merge(
        scale[SKU_DAY_KEYS + ["conservative_scale"]],
        on=SKU_DAY_KEYS,
        how="inner",
        validate="many_to_one",
    )
    result["imputed_demand"] = (
        result["imputed_demand"] * result["conservative_scale"]
    )
    return result[HOUR_KEYS + ["imputed_demand"]]


def build_evaluation_targets(demand: pd.DataFrame) -> pd.DataFrame:
    stockouts = demand[demand["is_clear_stockout"].astype(bool)].copy()
    observed = pd.to_numeric(
        stockouts["demand_lower_bound"], errors="coerce"
    ).fillna(0.0)
    raw = pd.to_numeric(stockouts["raw_imputed_demand"], errors="coerce").fillna(
        0.0
    )
    references = pd.to_numeric(stockouts["reference_days"], errors="coerce").fillna(
        0.0
    )
    cap = np.minimum(10.0, np.maximum(observed, 4.0) * 0.50)
    stockouts["conservative_imputed"] = np.where(
        references.ge(3), np.minimum(raw, cap), 0.0
    )
    stockouts["is_clear_stockout"] = True
    return stockouts[
        SKU_DAY_KEYS
        + ["is_clear_stockout", "imputed_demand", "conservative_imputed"]
    ].drop_duplicates(SKU_DAY_KEYS)


def normalize_to_bakery_prediction(
    scored: pd.DataFrame,
    bakery_predictions: pd.DataFrame,
) -> pd.DataFrame:
    sku_day = scored.groupby(SKU_DAY_KEYS, as_index=False).agg(
        observed_sales=("actual_qty", "sum"),
        profile_quantity=("predicted_qty", "sum"),
    )
    profile_total = sku_day.groupby(DAY_KEYS)["profile_quantity"].transform("sum")
    sku_day["profile_share"] = np.where(
        profile_total.gt(0), sku_day["profile_quantity"] / profile_total, 0.0
    )
    prediction = bakery_predictions[DAY_KEYS + ["prediction"]].copy()
    result = sku_day.merge(
        prediction,
        on=DAY_KEYS,
        how="inner",
        validate="many_to_one",
    )
    result["predicted_demand"] = result["profile_share"] * result["prediction"]
    return result


def attach_targets_and_scopes(
    scored: pd.DataFrame,
    evaluation_targets: pd.DataFrame,
    adjusted_pairs: set[tuple[int, int]],
) -> pd.DataFrame:
    result = scored.merge(
        evaluation_targets,
        on=SKU_DAY_KEYS,
        how="left",
        validate="one_to_one",
    )
    result[["imputed_demand", "conservative_imputed"]] = result[
        ["imputed_demand", "conservative_imputed"]
    ].fillna(0.0)
    result["is_stockout_sku_day"] = result["is_clear_stockout"].eq(True)
    result["is_adjusted_pair"] = [
        (int(row.bakery_id), int(row.product_id)) in adjusted_pairs
        for row in result.itertuples()
    ]
    result["conservative_target"] = (
        result["observed_sales"] + result["conservative_imputed"]
    )
    result["full_point_target"] = (
        result["observed_sales"] + result["imputed_demand"]
    )
    return result


def summarize_variant(scored: pd.DataFrame, *, variant: str) -> pd.DataFrame:
    scopes = {
        "all_sku_days_observed_sales": (
            pd.Series(True, index=scored.index),
            "observed_sales",
        ),
        "clean_sku_days_observed_sales": (
            ~scored["is_stockout_sku_day"],
            "observed_sales",
        ),
        "adjusted_pairs_clean_sku_days": (
            scored["is_adjusted_pair"] & ~scored["is_stockout_sku_day"],
            "observed_sales",
        ),
        "stockout_sku_days_observed_lower_bound": (
            scored["is_stockout_sku_day"],
            "observed_sales",
        ),
        "stockout_sku_days_conservative_target": (
            scored["is_stockout_sku_day"],
            "conservative_target",
        ),
        "stockout_sku_days_full_point": (
            scored["is_stockout_sku_day"],
            "full_point_target",
        ),
    }
    rows: list[dict[str, object]] = []
    for scope, (mask, target_column) in scopes.items():
        part = scored[mask]
        error = part["predicted_demand"] - part[target_column]
        target = float(part[target_column].sum())
        rows.append(
            {
                "variant": variant,
                "scope": scope,
                "sku_days": int(len(part)),
                "target_qty": target,
                "predicted_qty": float(part["predicted_demand"].sum()),
                "bias_qty": float(error.sum()),
                "mean_bias": float(error.mean()) if len(part) else None,
                "mae": float(error.abs().mean()) if len(part) else None,
                "underforecast_qty": float((-error).clip(lower=0).sum()),
                "overforecast_qty": float(error.clip(lower=0).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = metrics[metrics["variant"].eq(BASELINE)].set_index(
        ["cutoff", "scope"]
    )
    rows = []
    for row in metrics[~metrics["variant"].eq(BASELINE)].itertuples(index=False):
        base = baseline.loc[(row.cutoff, row.scope)]
        rows.append(
            {
                "cutoff": row.cutoff,
                "variant": row.variant,
                "scope": row.scope,
                "bias_qty_delta": float(row.bias_qty - base["bias_qty"]),
                "abs_bias_delta": float(abs(row.bias_qty) - abs(base["bias_qty"])),
                "mae_delta": float(row.mae - base["mae"]),
                "underforecast_qty_delta": float(
                    row.underforecast_qty - base["underforecast_qty"]
                ),
                "overforecast_qty_delta": float(
                    row.overforecast_qty - base["overforecast_qty"]
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_deltas(deltas: pd.DataFrame) -> pd.DataFrame:
    return deltas.groupby(["variant", "scope"], as_index=False).agg(
        cutoffs=("cutoff", "nunique"),
        abs_bias_wins=("abs_bias_delta", lambda value: int(value.lt(0).sum())),
        mae_wins=("mae_delta", lambda value: int(value.lt(0).sum())),
        mean_bias_qty_delta=("bias_qty_delta", "mean"),
        mean_abs_bias_delta=("abs_bias_delta", "mean"),
        mean_mae_delta=("mae_delta", "mean"),
        mean_underforecast_qty_delta=("underforecast_qty_delta", "mean"),
        mean_overforecast_qty_delta=("overforecast_qty_delta", "mean"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--demand", type=Path, default=DEFAULT_DEMAND)
    parser.add_argument(
        "--hourly-imputation", type=Path, default=DEFAULT_HOURLY_IMPUTATION
    )
    parser.add_argument(
        "--bakery-predictions", type=Path, default=DEFAULT_BAKERY_PREDICTIONS
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cutoffs",
        nargs="+",
        default=["2026-06-21", "2026-07-05"],
    )
    parser.add_argument("--holdout-days", type=int, default=14)
    args = parser.parse_args()

    demand = pd.read_csv(args.demand, encoding="utf-8-sig", low_memory=False)
    demand[DATE_COL] = pd.to_datetime(demand[DATE_COL]).dt.normalize()
    demand[[BAKERY_ID_COL, PRODUCT_ID_COL]] = demand[
        [BAKERY_ID_COL, PRODUCT_ID_COL]
    ].astype(int)
    hourly_imputation = pd.read_csv(
        args.hourly_imputation, encoding="utf-8-sig"
    )
    hourly_imputation[DATE_COL] = pd.to_datetime(
        hourly_imputation[DATE_COL]
    ).dt.normalize()
    conservative_adjustments = build_conservative_hourly_adjustments(
        demand, hourly_imputation
    )
    evaluation_targets = build_evaluation_targets(demand)
    bakery_predictions = pd.read_csv(
        args.bakery_predictions, encoding="utf-8-sig"
    )
    bakery_predictions[DATE_COL] = pd.to_datetime(
        bakery_predictions[DATE_COL]
    ).dt.normalize()
    bakery_ids = set(demand[BAKERY_ID_COL].unique())
    max_holdout = max(pd.Timestamp(value) for value in args.cutoffs) + pd.Timedelta(
        days=args.holdout_days
    )
    hourly = load_hourly_raw(
        args.raw,
        bakery_ids=bakery_ids,
        date_from=pd.Timestamp("2026-05-01"),
        date_to=max_holdout,
    )

    metric_parts = []
    scored_parts = []
    profile_diagnostics = []
    for cutoff_value in args.cutoffs:
        cutoff = pd.Timestamp(cutoff_value)
        holdout_start = cutoff + pd.Timedelta(days=1)
        holdout_end = cutoff + pd.Timedelta(days=args.holdout_days)
        train = hourly[hourly[DATE_COL].le(cutoff)].copy()
        holdout = hourly[hourly[DATE_COL].between(holdout_start, holdout_end)].copy()
        train_adjustments = conservative_adjustments[
            conservative_adjustments[DATE_COL].le(cutoff)
        ].copy()
        adjusted_train = apply_hourly_adjustments(train, train_adjustments)
        baseline_profile, _ = build_sku_hour_share_profile(train)
        adjusted_profile, _ = build_sku_hour_share_profile(adjusted_train)
        baseline_exact, _ = build_serving_profiles(baseline_profile)
        triple_columns = [BAKERY_ID_COL, DOW_COL, HOUR_COL]
        baseline_exact_triples = set(
            map(
                tuple,
                baseline_exact[triple_columns].drop_duplicates().to_numpy(),
            )
        )
        adjusted_pairs = set(
            map(
                tuple,
                train_adjustments.loc[
                    train_adjustments["imputed_demand"].gt(0),
                    [BAKERY_ID_COL, PRODUCT_ID_COL],
                ]
                .drop_duplicates()
                .astype(int)
                .to_numpy(),
            )
        )
        base_bakery = bakery_predictions[
            bakery_predictions["cutoff"].eq(cutoff.date().isoformat())
            & bakery_predictions["variant"].eq(BAKERY_BASELINE_VARIANT)
        ]
        conservative_bakery = bakery_predictions[
            bakery_predictions["cutoff"].eq(cutoff.date().isoformat())
            & bakery_predictions["variant"].eq(BAKERY_CONSERVATIVE_VARIANT)
        ]
        variants = [
            (BASELINE, baseline_profile, base_bakery, None, None),
            (BAKERY_ONLY, baseline_profile, conservative_bakery, None, None),
            (PROFILE_ONLY, adjusted_profile, base_bakery, None, None),
            (END_TO_END, adjusted_profile, conservative_bakery, None, None),
            (
                GUARDED_END_TO_END,
                adjusted_profile,
                conservative_bakery,
                baseline_exact_triples,
                baseline_profile,
            ),
        ]
        for (
            variant,
            profile,
            day_prediction,
            allowed_exact,
            fallback_source,
        ) in variants:
            profile_scored = build_scored_rows(
                profile,
                holdout,
                allowed_exact_triples=allowed_exact,
                fallback_source_profile=fallback_source,
            )
            scored = attach_targets_and_scopes(
                normalize_to_bakery_prediction(profile_scored, day_prediction),
                evaluation_targets[
                    evaluation_targets[DATE_COL].between(
                        holdout_start, holdout_end
                    )
                ],
                adjusted_pairs,
            )
            scored["variant"] = variant
            scored["cutoff"] = cutoff.date().isoformat()
            scored_parts.append(scored)
            metrics = summarize_variant(scored, variant=variant)
            metrics["cutoff"] = cutoff.date().isoformat()
            metric_parts.append(metrics)
        profile_diagnostics.append(
            {
                "cutoff": cutoff.date().isoformat(),
                "train_imputed_units": float(
                    train_adjustments["imputed_demand"].sum()
                ),
                "adjusted_pairs": int(len(adjusted_pairs)),
                "baseline_profile_rows": int(len(baseline_profile)),
                "adjusted_profile_rows": int(len(adjusted_profile)),
            }
        )

    metrics = pd.concat(metric_parts, ignore_index=True)
    scored = pd.concat(scored_parts, ignore_index=True)
    deltas = build_deltas(metrics)
    summary = summarize_deltas(deltas)
    diagnostics = pd.DataFrame(profile_diagnostics)
    stockout_delivery = (
        scored[scored["is_stockout_sku_day"]]
        .groupby(["cutoff", "variant"], as_index=False)
        .agg(
            stockout_sku_days=(PRODUCT_ID_COL, "size"),
            observed_sales=("observed_sales", "sum"),
            conservative_target=("conservative_target", "sum"),
            predicted_demand=("predicted_demand", "sum"),
        )
    )
    baseline_delivery = stockout_delivery[
        stockout_delivery["variant"].eq(BASELINE)
    ][["cutoff", "predicted_demand"]].rename(
        columns={"predicted_demand": "baseline_predicted_demand"}
    )
    stockout_delivery = stockout_delivery.merge(
        baseline_delivery, on="cutoff", how="left", validate="many_to_one"
    )
    stockout_delivery["delivered_uplift_vs_baseline"] = (
        stockout_delivery["predicted_demand"]
        - stockout_delivery["baseline_predicted_demand"]
    )
    baseline_sku = scored[scored["variant"].eq(BASELINE)][
        ["cutoff", *SKU_DAY_KEYS, "predicted_demand"]
    ].rename(columns={"predicted_demand": "baseline_sku_prediction"})
    uplift_destination = scored.merge(
        baseline_sku,
        on=["cutoff", *SKU_DAY_KEYS],
        how="left",
        validate="many_to_one",
    )
    uplift_destination["prediction_delta"] = (
        uplift_destination["predicted_demand"]
        - uplift_destination["baseline_sku_prediction"]
    )
    uplift_destination_summary = (
        uplift_destination.groupby(
            ["cutoff", "variant", "is_stockout_sku_day"],
            as_index=False,
        )
        .agg(
            sku_days=(PRODUCT_ID_COL, "size"),
            net_prediction_delta=("prediction_delta", "sum"),
            positive_prediction_delta=(
                "prediction_delta",
                lambda value: float(value.clip(lower=0).sum()),
            ),
            negative_prediction_delta=(
                "prediction_delta",
                lambda value: float((-value).clip(lower=0).sum()),
            ),
        )
    )
    total_destination = uplift_destination_summary.groupby(
        ["cutoff", "variant"], as_index=False
    )["net_prediction_delta"].sum().rename(
        columns={"net_prediction_delta": "total_prediction_delta"}
    )
    stockout_delivery = stockout_delivery.merge(
        total_destination,
        on=["cutoff", "variant"],
        how="left",
        validate="one_to_one",
    )
    stockout_delivery["baseline_gap_to_conservative_target"] = (
        stockout_delivery["conservative_target"]
        - stockout_delivery["baseline_predicted_demand"]
    )
    stockout_delivery["delivered_share_of_reconstructed_gap"] = (
        stockout_delivery["delivered_uplift_vs_baseline"]
        / stockout_delivery["baseline_gap_to_conservative_target"].replace(0.0, np.nan)
    )
    stockout_delivery["stockout_share_of_total_prediction_delta"] = (
        stockout_delivery["delivered_uplift_vs_baseline"]
        / stockout_delivery["total_prediction_delta"].replace(0.0, np.nan)
    )
    uplift_by_product = (
        uplift_destination[
            uplift_destination["variant"].isin([END_TO_END, GUARDED_END_TO_END])
        ]
        .groupby(
            ["variant", PRODUCT_ID_COL, "is_stockout_sku_day"],
            as_index=False,
        )
        .agg(
            sku_days=(DATE_COL, "size"),
            net_prediction_delta=("prediction_delta", "sum"),
            positive_prediction_delta=(
                "prediction_delta",
                lambda value: float(value.clip(lower=0).sum()),
            ),
        )
        .sort_values(["variant", "positive_prediction_delta"], ascending=[True, False])
    )
    if "product_name" in demand:
        product_names = demand.groupby(PRODUCT_ID_COL, as_index=False).agg(
            product_name=("product_name", "last")
        )
        product_names["product_name"] = product_names["product_name"].str.strip()
        uplift_by_product = uplift_by_product.merge(
            product_names,
            on=PRODUCT_ID_COL,
            how="left",
            validate="many_to_one",
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    deltas.to_csv(args.output_dir / "deltas.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(
        args.output_dir / "profile_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stockout_delivery.to_csv(
        args.output_dir / "stockout_delivery.csv",
        index=False,
        encoding="utf-8-sig",
    )
    uplift_destination_summary.to_csv(
        args.output_dir / "uplift_destination_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    uplift_by_product.to_csv(
        args.output_dir / "uplift_by_product.csv",
        index=False,
        encoding="utf-8-sig",
    )
    scored.to_csv(
        args.output_dir / "scored_sku_days.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report = {
        "cutoffs": args.cutoffs,
        "holdout_days": args.holdout_days,
        "variants": [
            BASELINE,
            BAKERY_ONLY,
            PROFILE_ONLY,
            END_TO_END,
            GUARDED_END_TO_END,
        ],
        "summary": summary.to_dict("records"),
        "production_write": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

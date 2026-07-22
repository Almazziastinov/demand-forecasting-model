"""Rolling bakery-day target experiment for stockout-adjusted demand variants."""

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

from scripts.experiment_demand_adjusted_bakery_target import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_RAW,
    apply_training_adjustments,
    extend_with_pilot_actuals,
    score_variant,
)
from scripts.experiment_demand_adjusted_profiles import load_hourly_raw  # noqa: E402
from src.experiments_v2.bakery_day_forecast import (  # noqa: E402
    BAKERY_ID_COL,
    DATE_COL,
    TARGET_COL,
)

DEFAULT_DEMAND = ROOT / "reports/stockout_adjusted_demand_dataset/sku_day_demand.csv"
DEFAULT_OUTPUT = ROOT / "reports/stockout_adjusted_bakery_target_experiment"
BASELINE = "observed_sales_target"
WEIGHTED = "weighted_reconstructed_target"
CONSERVATIVE = "conservative_reconstructed_target"
VARIANTS = [BASELINE, WEIGHTED, CONSERVATIVE]


def build_adjustment_variants(demand: pd.DataFrame) -> pd.DataFrame:
    stockouts = demand[demand["is_clear_stockout"].astype(bool)].copy()
    observed = pd.to_numeric(stockouts["demand_lower_bound"], errors="coerce").fillna(0.0)
    raw = pd.to_numeric(stockouts["raw_imputed_demand"], errors="coerce").fillna(0.0)
    reference_days = pd.to_numeric(stockouts["reference_days"], errors="coerce").fillna(0.0)
    weighted = (
        pd.to_numeric(stockouts["imputed_demand"], errors="coerce").fillna(0.0)
        * pd.to_numeric(stockouts["suggested_training_weight"], errors="coerce").fillna(0.0)
    )
    conservative_cap = np.minimum(10.0, np.maximum(observed, 4.0) * 0.50)
    conservative = np.where(
        reference_days.ge(3), np.minimum(raw, conservative_cap), 0.0
    )
    parts = []
    for variant, addition in [
        (WEIGHTED, weighted),
        (CONSERVATIVE, conservative),
    ]:
        part = stockouts[[DATE_COL, BAKERY_ID_COL]].copy()
        part["variant"] = variant
        part["imputed_demand"] = addition
        parts.append(part)
    return (
        pd.concat(parts, ignore_index=True)
        .groupby(["variant", DATE_COL, BAKERY_ID_COL], as_index=False)[
            "imputed_demand"
        ]
        .sum()
    )


def build_evaluation_adjustments(demand: pd.DataFrame) -> pd.DataFrame:
    stockouts = demand[demand["is_clear_stockout"].astype(bool)].copy()
    observed = pd.to_numeric(stockouts["demand_lower_bound"], errors="coerce").fillna(0.0)
    raw = pd.to_numeric(stockouts["raw_imputed_demand"], errors="coerce").fillna(0.0)
    reference_days = pd.to_numeric(stockouts["reference_days"], errors="coerce").fillna(0.0)
    conservative_cap = np.minimum(10.0, np.maximum(observed, 4.0) * 0.50)
    stockouts["conservative_imputed"] = np.where(
        reference_days.ge(3), np.minimum(raw, conservative_cap), 0.0
    )
    return stockouts.groupby([DATE_COL, BAKERY_ID_COL], as_index=False).agg(
        full_imputed=("imputed_demand", "sum"),
        conservative_imputed=("conservative_imputed", "sum"),
        stockout_skus=("product_id", "size"),
    )


def summarize_predictions(
    predictions: pd.DataFrame,
    evaluation_adjustments: pd.DataFrame,
    *,
    variant: str,
) -> pd.DataFrame:
    work = predictions.merge(
        evaluation_adjustments,
        on=[DATE_COL, BAKERY_ID_COL],
        how="left",
        validate="one_to_one",
    )
    work[["full_imputed", "conservative_imputed", "stockout_skus"]] = work[
        ["full_imputed", "conservative_imputed", "stockout_skus"]
    ].fillna(0.0)
    work["is_stockout_day"] = work["stockout_skus"].gt(0)
    work["full_point_demand"] = work[TARGET_COL] + work["full_imputed"]
    work["conservative_point_demand"] = (
        work[TARGET_COL] + work["conservative_imputed"]
    )
    scopes = {
        "all_days_observed_sales": (pd.Series(True, index=work.index), TARGET_COL),
        "clean_days_observed_sales": (~work["is_stockout_day"], TARGET_COL),
        "stockout_days_observed_lower_bound": (work["is_stockout_day"], TARGET_COL),
        "stockout_days_conservative_point": (
            work["is_stockout_day"],
            "conservative_point_demand",
        ),
        "stockout_days_full_point": (work["is_stockout_day"], "full_point_demand"),
    }
    rows: list[dict[str, object]] = []
    for scope, (mask, target_column) in scopes.items():
        part = work[mask]
        actual = float(part[target_column].sum())
        error = part["prediction"] - part[target_column]
        rows.append(
            {
                "variant": variant,
                "scope": scope,
                "rows": int(len(part)),
                "target_qty": actual,
                "predicted_qty": float(part["prediction"].sum()),
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
        mean_bias_qty_delta=("bias_qty_delta", "mean"),
        mean_abs_bias_delta=("abs_bias_delta", "mean"),
        mean_underforecast_qty_delta=("underforecast_qty_delta", "mean"),
        mean_overforecast_qty_delta=("overforecast_qty_delta", "mean"),
    )


def select_non_overlapping_cutoffs(
    cutoffs: pd.Series,
    *,
    holdout_days: int,
) -> list[str]:
    selected: list[pd.Timestamp] = []
    for cutoff in sorted(pd.to_datetime(cutoffs.unique())):
        if not selected or cutoff >= selected[-1] + pd.Timedelta(days=holdout_days):
            selected.append(cutoff)
    return [cutoff.date().isoformat() for cutoff in selected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--demand", type=Path, default=DEFAULT_DEMAND)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cutoffs",
        nargs="+",
        default=["2026-06-21", "2026-06-28", "2026-07-05"],
    )
    parser.add_argument("--holdout-days", type=int, default=14)
    args = parser.parse_args()

    demand = pd.read_csv(args.demand, encoding="utf-8-sig", low_memory=False)
    demand[DATE_COL] = pd.to_datetime(demand[DATE_COL]).dt.normalize()
    demand[BAKERY_ID_COL] = demand[BAKERY_ID_COL].astype(int)
    pilot_ids = set(demand[BAKERY_ID_COL].unique())
    adjustment_variants = build_adjustment_variants(demand)
    evaluation_adjustments = build_evaluation_adjustments(demand)

    base = pd.read_csv(args.dataset, encoding="utf-8-sig")
    base[DATE_COL] = pd.to_datetime(base[DATE_COL]).dt.normalize()
    max_holdout = max(pd.Timestamp(value) for value in args.cutoffs) + pd.Timedelta(
        days=args.holdout_days
    )
    hourly = load_hourly_raw(
        args.raw,
        bakery_ids=pilot_ids,
        date_from=base[DATE_COL].max() + pd.Timedelta(days=1),
        date_to=max_holdout,
    )
    base = extend_with_pilot_actuals(base, hourly)
    observed_target = base[[DATE_COL, BAKERY_ID_COL, TARGET_COL]].rename(
        columns={TARGET_COL: "observed_target"}
    )

    prediction_parts = []
    for cutoff_value in args.cutoffs:
        cutoff = pd.Timestamp(cutoff_value)
        holdout_end = cutoff + pd.Timedelta(days=args.holdout_days)
        frames = {BASELINE: base}
        for variant in [WEIGHTED, CONSERVATIVE]:
            additions = adjustment_variants[
                adjustment_variants["variant"].eq(variant)
            ]
            frames[variant] = apply_training_adjustments(
                base, additions, train_end=cutoff
            )
        for variant, frame in frames.items():
            part = score_variant(
                frame,
                train_end=cutoff,
                holdout_end=holdout_end,
                pilot_ids=pilot_ids,
            )
            part = (
                part.drop(columns=TARGET_COL)
                .merge(
                    observed_target,
                    on=[DATE_COL, BAKERY_ID_COL],
                    how="left",
                    validate="one_to_one",
                )
                .rename(columns={"observed_target": TARGET_COL})
            )
            part["variant"] = variant
            part["cutoff"] = cutoff.date().isoformat()
            prediction_parts.append(part)
    predictions = pd.concat(prediction_parts, ignore_index=True)

    metric_parts = []
    for (cutoff, variant), part in predictions.groupby(["cutoff", "variant"]):
        metrics = summarize_predictions(
            part,
            evaluation_adjustments,
            variant=variant,
        )
        metrics["cutoff"] = cutoff
        metric_parts.append(metrics)
    metrics = pd.concat(metric_parts, ignore_index=True)
    deltas = build_deltas(metrics)
    summary = summarize_deltas(deltas)
    non_overlapping_cutoffs = select_non_overlapping_cutoffs(
        deltas["cutoff"], holdout_days=args.holdout_days
    )
    non_overlapping_summary = summarize_deltas(
        deltas[deltas["cutoff"].isin(non_overlapping_cutoffs)]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        args.output_dir / "predictions.csv", index=False, encoding="utf-8-sig"
    )
    metrics.to_csv(args.output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    deltas.to_csv(args.output_dir / "deltas.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    non_overlapping_summary.to_csv(
        args.output_dir / "non_overlapping_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "rolling_windows": summary.to_dict("records"),
                "non_overlapping_cutoffs": non_overlapping_cutoffs,
                "non_overlapping_windows": non_overlapping_summary.to_dict("records"),
                "production_write": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

"""Evaluate a non-overlapping pair-level gate for adjusted SKU profiles."""

from __future__ import annotations

# ruff: noqa: E501

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASELINE = "observed_sales_profile"
ADJUSTED = "demand_adjusted_guarded_routing"
KEYS = ["date", "bakery_id", "dow", "hour", "product_id"]
PAIR_KEYS = ["bakery_id", "product_id"]
CONTEXT_KEYS = ["date", "bakery_id", "dow", "hour"]


def build_pair_evidence(scored: pd.DataFrame) -> pd.DataFrame:
    """Measure adjusted-minus-baseline absolute error on clean SKU-days."""
    work = scored[
        scored["variant"].isin([BASELINE, ADJUSTED])
        & ~scored["is_stockout_sku_day"].astype(bool)
        & scored["is_adjusted_pair"].astype(bool)
    ].copy()
    daily = (
        work.groupby(["variant", "date", *PAIR_KEYS], as_index=False)
        .agg(actual_qty=("actual_qty", "sum"), predicted_qty=("predicted_qty", "sum"))
    )
    daily["abs_error"] = (daily["predicted_qty"] - daily["actual_qty"]).abs()
    pair = (
        daily.groupby(["variant", *PAIR_KEYS], as_index=False)
        .agg(
            clean_days=("date", "nunique"),
            actual_qty=("actual_qty", "sum"),
            abs_error=("abs_error", "sum"),
        )
        .pivot(index=PAIR_KEYS, columns="variant")
    )
    pair.columns = [f"{metric}_{variant}" for metric, variant in pair.columns]
    pair = pair.reset_index()
    pair["clean_days"] = pair[
        [f"clean_days_{BASELINE}", f"clean_days_{ADJUSTED}"]
    ].min(axis=1)
    pair["abs_error_delta"] = pair[f"abs_error_{ADJUSTED}"] - pair[f"abs_error_{BASELINE}"]
    pair["relative_error_delta"] = pair["abs_error_delta"] / pair[
        f"actual_qty_{BASELINE}"
    ].replace(0, np.nan)
    return pair.sort_values("abs_error_delta").reset_index(drop=True)


def eligible_pairs(
    evidence: pd.DataFrame,
    *,
    min_clean_days: int,
    min_gain_qty: float,
) -> set[tuple[int, int]]:
    selected = evidence[
        evidence["clean_days"].ge(min_clean_days)
        & evidence["abs_error_delta"].le(-min_gain_qty)
    ]
    return set(map(tuple, selected[PAIR_KEYS].astype(int).to_numpy()))


def apply_pair_gate(
    scored: pd.DataFrame,
    allowed_pairs: set[tuple[int, int]],
) -> pd.DataFrame:
    """Choose adjusted predictions by pair, then restore bakery-hour totals."""
    base = scored[scored["variant"].eq(BASELINE)].copy()
    adjusted = scored[scored["variant"].eq(ADJUSTED)][
        [*KEYS, "predicted_qty"]
    ].rename(columns={"predicted_qty": "adjusted_prediction"})
    work = base.merge(adjusted, on=KEYS, how="left", validate="one_to_one")
    work["is_pair_eligible"] = [
        (int(row.bakery_id), int(row.product_id)) in allowed_pairs
        for row in work.itertuples()
    ]
    work["selected_prediction"] = np.where(
        work["is_pair_eligible"],
        work["adjusted_prediction"].fillna(work["predicted_qty"]),
        work["predicted_qty"],
    )
    selected_total = work.groupby(CONTEXT_KEYS)["selected_prediction"].transform("sum")
    scale = work["bakery_hour_sales"] / selected_total.replace(0, np.nan)
    work["predicted_qty"] = (work["selected_prediction"] * scale).fillna(
        work["predicted_qty"]
    )
    work["error"] = work["predicted_qty"] - work["actual_qty"]
    work["abs_error"] = work["error"].abs()
    work["variant"] = "pair_gate"
    return work.drop(columns=["adjusted_prediction", "selected_prediction"])


def summarize(scored: pd.DataFrame, *, variant: str) -> pd.DataFrame:
    scopes = {
        "all_holdout": pd.Series(True, index=scored.index),
        "clean_sku_days": ~scored["is_stockout_sku_day"].astype(bool),
        "adjusted_pairs_clean_sku_days": (
            scored["is_adjusted_pair"].astype(bool)
            & ~scored["is_stockout_sku_day"].astype(bool)
        ),
        "eligible_pairs_clean_sku_days": (
            scored.get("is_pair_eligible", False)
            & ~scored["is_stockout_sku_day"].astype(bool)
        ),
    }
    rows = []
    for scope, mask in scopes.items():
        part = scored[mask]
        daily = part.groupby(["date", *PAIR_KEYS], as_index=False).agg(
            actual_qty=("actual_qty", "sum"),
            predicted_qty=("predicted_qty", "sum"),
        )
        error = daily["predicted_qty"] - daily["actual_qty"]
        actual = float(daily["actual_qty"].sum())
        rows.append(
            {
                "variant": variant,
                "scope": scope,
                "rows": int(len(part)),
                "sku_days": int(len(daily)),
                "actual_qty": actual,
                "predicted_qty": float(daily["predicted_qty"].sum()),
                "bias_qty": float(error.sum()),
                "sku_day_wape": float(error.abs().sum() / actual) if actual > 0 else None,
                "underforecast_qty": float((-error).clip(lower=0).sum()),
                "overforecast_qty": float(error.clip(lower=0).sum()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_fold(
    evidence_scored: pd.DataFrame,
    target_scored: pd.DataFrame,
    *,
    min_clean_days: int,
    min_gain_qty: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    evidence = build_pair_evidence(evidence_scored)
    allowed = eligible_pairs(
        evidence,
        min_clean_days=min_clean_days,
        min_gain_qty=min_gain_qty,
    )
    gated = apply_pair_gate(target_scored, allowed)
    baseline = target_scored[target_scored["variant"].eq(BASELINE)].copy()
    baseline["is_pair_eligible"] = [
        (int(row.bakery_id), int(row.product_id)) in allowed
        for row in baseline.itertuples()
    ]
    adjusted = target_scored[target_scored["variant"].eq(ADJUSTED)].copy()
    adjusted["is_pair_eligible"] = [
        (int(row.bakery_id), int(row.product_id)) in allowed
        for row in adjusted.itertuples()
    ]
    metrics = pd.concat(
        [
            summarize(baseline, variant=BASELINE),
            summarize(adjusted, variant=ADJUSTED),
            summarize(gated, variant="pair_gate"),
        ],
        ignore_index=True,
    )
    evidence["is_eligible"] = [
        (int(row.bakery_id), int(row.product_id)) in allowed
        for row in evidence.itertuples()
    ]
    return metrics, evidence, gated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", action="append", nargs=3, metavar=("NAME", "EVIDENCE_DIR", "TARGET_DIR"), required=True)
    parser.add_argument("--min-clean-days", nargs="+", type=int, default=[2, 3, 5, 7])
    parser.add_argument("--min-gain-qty", nargs="+", type=float, default=[0.0, 2.0, 5.0, 10.0])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    metric_parts = []
    evidence_parts = []
    for fold_name, evidence_dir, target_dir in args.fold:
        evidence_scored = pd.read_csv(Path(evidence_dir) / "scored_rows.csv")
        target_scored = pd.read_csv(Path(target_dir) / "scored_rows.csv")
        evidence_scored["date"] = pd.to_datetime(evidence_scored["date"])
        target_scored["date"] = pd.to_datetime(target_scored["date"])
        for min_days in args.min_clean_days:
            for min_gain in args.min_gain_qty:
                metrics, evidence, _ = evaluate_fold(
                    evidence_scored,
                    target_scored,
                    min_clean_days=min_days,
                    min_gain_qty=min_gain,
                )
                metrics["fold"] = fold_name
                metrics["min_clean_days"] = min_days
                metrics["min_gain_qty"] = min_gain
                metric_parts.append(metrics)
                evidence["fold"] = fold_name
                evidence["min_clean_days"] = min_days
                evidence["min_gain_qty"] = min_gain
                evidence_parts.append(evidence)

    metrics = pd.concat(metric_parts, ignore_index=True)
    evidence = pd.concat(evidence_parts, ignore_index=True)
    lookup = metrics.set_index(["fold", "min_clean_days", "min_gain_qty", "variant", "scope"])
    deltas = []
    parameter_rows = metrics[["fold", "min_clean_days", "min_gain_qty"]].drop_duplicates()
    for row in parameter_rows.itertuples(index=False):
        for scope in metrics["scope"].unique():
            base = lookup.loc[(row.fold, row.min_clean_days, row.min_gain_qty, BASELINE, scope)]
            gate = lookup.loc[(row.fold, row.min_clean_days, row.min_gain_qty, "pair_gate", scope)]
            adjusted = lookup.loc[(row.fold, row.min_clean_days, row.min_gain_qty, ADJUSTED, scope)]
            deltas.append(
                {
                    "fold": row.fold,
                    "min_clean_days": row.min_clean_days,
                    "min_gain_qty": row.min_gain_qty,
                    "scope": scope,
                    "eligible_pairs": int(
                        evidence[
                            evidence["fold"].eq(row.fold)
                            & evidence["min_clean_days"].eq(row.min_clean_days)
                            & evidence["min_gain_qty"].eq(row.min_gain_qty)
                            & evidence["is_eligible"]
                        ].shape[0]
                    ),
                    "gate_sku_day_wape_delta": float(gate["sku_day_wape"] - base["sku_day_wape"]),
                    "full_adjusted_sku_day_wape_delta": float(adjusted["sku_day_wape"] - base["sku_day_wape"]),
                    "gate_underforecast_qty_delta": float(gate["underforecast_qty"] - base["underforecast_qty"]),
                    "gate_overforecast_qty_delta": float(gate["overforecast_qty"] - base["overforecast_qty"]),
                }
            )
    deltas = pd.DataFrame(deltas)
    aggregate = deltas.groupby(["min_clean_days", "min_gain_qty", "scope"], as_index=False).agg(
        folds=("fold", "nunique"),
        wins=("gate_sku_day_wape_delta", lambda value: int(value.lt(0).sum())),
        mean_gate_sku_day_wape_delta=("gate_sku_day_wape_delta", "mean"),
        mean_full_adjusted_sku_day_wape_delta=("full_adjusted_sku_day_wape_delta", "mean"),
        mean_eligible_pairs=("eligible_pairs", "mean"),
        mean_underforecast_qty_delta=("gate_underforecast_qty_delta", "mean"),
        mean_overforecast_qty_delta=("gate_overforecast_qty_delta", "mean"),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    evidence.to_csv(args.output_dir / "pair_evidence.csv", index=False, encoding="utf-8-sig")
    deltas.to_csv(args.output_dir / "fold_deltas.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(args.output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    best = aggregate[aggregate["scope"].eq("clean_sku_days")].sort_values(
        ["wins", "mean_gate_sku_day_wape_delta"], ascending=[False, True]
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(best.head(10).to_dict("records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(best.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

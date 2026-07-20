"""Offline walk-forward model for constrained bakery/SKU allocation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_non_stockout_share_allocation import add_bakery_actual  # noqa: E402
from scripts.export_clickhouse_checks import create_client  # noqa: E402

DEFAULT_INPUT = ROOT / "reports/pilot_stockout_forecast_bias/sku_day_comparison.csv"
DEFAULT_OUTPUT = ROOT / "reports/dynamic_sku_allocation_experiment"
FEATURES = [
    "bakery_id",
    "product_id",
    "category_code",
    "dow",
    "baseline_share",
    "pair_log_ratio",
    "pair_dow_log_ratio",
    "product_log_ratio",
    "category_log_ratio",
    "pair_ratio_std",
    "pair_history_days",
]


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    work = work.sort_values(["date", "bakery_id", "product_id"]).reset_index(drop=True)
    total = work.groupby(["date", "bakery_id"])["forecast_qty"].transform("sum")
    work["baseline_total"] = total
    work["baseline_share"] = work["forecast_qty"] / total.replace(0.0, np.nan)
    work["observed_share"] = work["daily_sold"] / work["bakery_actual_qty"].replace(
        0.0, np.nan
    )
    ratio = work["observed_share"] / work["baseline_share"].replace(0.0, np.nan)
    work["target_log_ratio"] = np.log(ratio.clip(0.1, 10.0))
    categories = sorted(work["category_name"].fillna("unknown").unique())
    mapping = {name: index for index, name in enumerate(categories)}
    work["category_code"] = work["category_name"].fillna("unknown").map(mapping)
    return work


def _history_aggregate(
    history: pd.DataFrame, keys: list[str], prefix: str
) -> pd.DataFrame:
    return history.groupby(keys, as_index=False).agg(
        **{
            f"{prefix}_log_ratio": ("target_log_ratio", "median"),
            f"{prefix}_history_days": ("date", "nunique"),
            f"{prefix}_ratio_std": ("target_log_ratio", "std"),
        }
    )


def build_lagged_features(
    frame: pd.DataFrame, *, lookback_days: int = 42
) -> pd.DataFrame:
    work = frame.copy()
    outputs = []
    for current_date in sorted(work["date"].unique()):
        current_date = pd.Timestamp(current_date)
        current = work[work["date"].eq(current_date)].copy()
        history = work[
            work["date"].lt(current_date)
            & work["date"].ge(current_date - pd.Timedelta(days=lookback_days))
            & work["stockout_group"].eq("confirmed_non_stockout")
            & work["target_log_ratio"].notna()
        ]
        specs = [
            (["bakery_id", "product_id"], "pair"),
            (["bakery_id", "product_id", "dow"], "pair_dow"),
            (["product_id"], "product"),
            (["bakery_id", "category_code"], "category"),
        ]
        for keys, prefix in specs:
            aggregate = _history_aggregate(history, keys, prefix)
            current = current.merge(aggregate, on=keys, how="left")
        outputs.append(current)
    result = pd.concat(outputs, ignore_index=True)
    result["pair_ratio_std"] = result["pair_ratio_std"].fillna(0.0)
    result["pair_history_days"] = result["pair_history_days"].fillna(0.0)
    for column in [
        "pair_log_ratio",
        "pair_dow_log_ratio",
        "product_log_ratio",
        "category_log_ratio",
    ]:
        result[column] = result[column].fillna(0.0)
    return result


def predict_walk_forward(
    frame: pd.DataFrame,
    *,
    min_train_rows: int = 500,
    retrain_days: int = 7,
) -> pd.DataFrame:
    work = frame.copy()
    work["model_log_ratio"] = np.nan
    model = None
    last_train_date: pd.Timestamp | None = None
    for current_date in sorted(work["date"].unique()):
        current_date = pd.Timestamp(current_date)
        train = work[
            work["date"].lt(current_date)
            & work["stockout_group"].eq("confirmed_non_stockout")
            & work["target_log_ratio"].notna()
        ]
        should_train = len(train) >= min_train_rows and (
            model is None
            or last_train_date is None
            or (current_date - last_train_date).days >= retrain_days
        )
        if should_train:
            model = lgb.LGBMRegressor(
                objective="huber",
                n_estimators=180,
                learning_rate=0.04,
                num_leaves=24,
                min_child_samples=40,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                verbosity=-1,
                random_state=42,
            )
            model.fit(train[FEATURES], train["target_log_ratio"])
            last_train_date = current_date
        mask = work["date"].eq(current_date)
        if model is None:
            work.loc[mask, "model_log_ratio"] = work.loc[mask, "pair_log_ratio"]
        else:
            work.loc[mask, "model_log_ratio"] = model.predict(work.loc[mask, FEATURES])
    work["model_log_ratio"] = work["model_log_ratio"].clip(np.log(0.25), np.log(4.0))
    return work


def apply_constrained_allocation(
    frame: pd.DataFrame, *, correction_column: str, strength: float
) -> pd.DataFrame:
    work = frame.copy()
    correction = np.exp(work[correction_column] * strength).clip(0.25, 4.0)
    work["allocation_weight"] = work["baseline_share"] * correction
    weight_total = work.groupby(["date", "bakery_id"])["allocation_weight"].transform(
        "sum"
    )
    work["adjusted_share"] = work["allocation_weight"] / weight_total.replace(
        0.0, np.nan
    )
    work["adjusted_forecast_qty"] = work["baseline_total"] * work["adjusted_share"]
    return work


def evaluate(frame: pd.DataFrame) -> dict[str, float | int]:
    stockout = frame[frame["stockout_group"].eq("clear_stockout")]
    normal = frame[frame["stockout_group"].eq("confirmed_non_stockout")]
    before = (stockout["daily_sold"] - stockout["forecast_qty"]).clip(lower=0.0)
    after = (stockout["daily_sold"] - stockout["adjusted_forecast_qty"]).clip(
        lower=0.0
    )
    normal_before_error = normal["forecast_qty"] - normal["daily_sold"]
    normal_after_error = normal["adjusted_forecast_qty"] - normal["daily_sold"]
    original_totals = frame.groupby(["date", "bakery_id"])["forecast_qty"].sum()
    adjusted_totals = frame.groupby(["date", "bakery_id"])[
        "adjusted_forecast_qty"
    ].sum()
    return {
        "baseline_shortfall_qty": float(before.sum()),
        "adjusted_shortfall_qty": float(after.sum()),
        "underforecast_cases_removed": int(((before > 0.5) & (after <= 0.5)).sum()),
        "new_underforecast_cases": int(((before <= 0.5) & (after > 0.5)).sum()),
        "normal_bias_before": float(normal_before_error.mean()),
        "normal_bias_after": float(normal_after_error.mean()),
        "normal_mae_before": float(normal_before_error.abs().mean()),
        "normal_mae_after": float(normal_after_error.abs().mean()),
        "max_bakery_total_delta": float((original_totals - adjusted_totals).abs().max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    frame = pd.read_csv(args.input, encoding="utf-8-sig")
    frame = frame[
        frame["stockout_group"].isin(["clear_stockout", "confirmed_non_stockout"])
    ]
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = add_bakery_actual(frame, create_client(args.env_file))
    featured = build_lagged_features(prepare_frame(frame))
    predicted = predict_walk_forward(featured)
    scenarios: list[dict[str, object]] = []
    details: dict[str, pd.DataFrame] = {}
    for correction in ["pair_log_ratio", "model_log_ratio"]:
        for strength in [0.25, 0.50, 0.75, 1.00]:
            name = f"{correction}_strength_{strength:.2f}"
            adjusted = apply_constrained_allocation(
                predicted, correction_column=correction, strength=strength
            )
            row: dict[str, object] = {
                "scenario": name,
                "correction": correction,
                "strength": strength,
            }
            row.update(evaluate(adjusted))
            row["net_cases_improved"] = int(row["underforecast_cases_removed"]) - int(
                row["new_underforecast_cases"]
            )
            scenarios.append(row)
            details[name] = adjusted
    comparison = pd.DataFrame(scenarios).sort_values(
        ["net_cases_improved", "adjusted_shortfall_qty", "normal_mae_after"],
        ascending=[False, True, True],
    )
    best = str(comparison.iloc[0]["scenario"])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output / "scenario_comparison.csv", index=False)
    details[best].to_csv(output / "best_scenario_rows.csv", index=False, encoding="utf-8-sig")
    summary = {
        "best_scenario": best,
        "best_metrics": comparison.iloc[0].to_dict(),
        "training": "walk_forward_non_stockout_only",
        "bakery_total_constrained": True,
        "production_write": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

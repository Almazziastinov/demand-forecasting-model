"""Expanding walk-forward validation for direct allocation, uplift and floor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_direct_bakery_sku_allocation import (
    DAY,
    HISTORY,
    INPUT,
    KEYS,
    build_features,
    fit_predict,
)
from build_direct_uplift_floor_candidates import (
    LABELS,
    add_floor_reference,
    fit_uplift,
    normalize_to_volume,
)
from select_direct_adaptive_floor import balance, make_plan


ROOT = Path(__file__).resolve().parents[1]
OPERATIONAL = ROOT / "reports/rolling_actual_state_comparison_20260826/rows.parquet"
OUTPUT = ROOT / "reports/rolling_direct_uplift_floor_20260827"
FEATURE_CACHE = ROOT / ".codex_tmp/direct_bakery_sku_features_20260827.parquet"
SCENARIOS = {"conservative": 0.5, "calibrated": 1.0, "upper": 1.5}
FOLDS = [
    {
        "name": "2026-07-20",
        "train_end": "2026-07-21",
        "dates": pd.date_range("2026-07-22", "2026-07-26"),
    },
    {
        "name": "2026-07-27",
        "train_end": "2026-07-26",
        "dates": pd.date_range("2026-07-27", "2026-08-02"),
    },
    {
        "name": "2026-08-10",
        "train_end": "2026-08-10",
        "dates": pd.to_datetime(["2026-08-11", "2026-08-12", "2026-08-13"]),
    },
    {
        "name": "2026-08-17",
        "train_end": "2026-08-16",
        "dates": pd.to_datetime(
            ["2026-08-17", "2026-08-18", "2026-08-21", "2026-08-22", "2026-08-23"]
        ),
    },
]


def load_features() -> pd.DataFrame:
    if FEATURE_CACHE.exists():
        return pd.read_parquet(FEATURE_CACHE)
    snapshot = pd.read_parquet(INPUT).rename(
        columns={"forecast_qty": "incumbent_sku_forecast"}
    )
    history = pd.read_parquet(HISTORY)
    for frame in (snapshot, history):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["product_id"] = pd.to_numeric(frame["product_id"]).astype("int64")
    features = build_features(snapshot, history)
    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURE_CACHE, index=False)
    return features


def build_predictions(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    enriched = features.merge(
        labels[KEYS + ["is_clear_stockout", "imputed_demand"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    enriched["is_clear_stockout"] = enriched["is_clear_stockout"].fillna(False)
    enriched["imputed_demand"] = enriched["imputed_demand"].fillna(0.0)
    parts = []
    for fold in FOLDS:
        dates = [str(date.date()) for date in fold["dates"]]
        direct = fit_predict(features, fold["train_end"], dates)
        train = enriched[enriched["date"].le(pd.Timestamp(fold["train_end"]))]
        probability, conditional, expected = fit_uplift(train, direct)
        direct["predicted_stockout_probability"] = probability
        direct["predicted_lost_if_stockout"] = conditional
        direct["predictive_uplift"] = expected
        direct["rolling_fold"] = fold["name"]
        parts.append(direct)
    return pd.concat(parts, ignore_index=True)


def candidate_grid(history: pd.DataFrame, scenario_rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    actual_under = balance(
        history["available_to_sell"].to_numpy(), history["scenario_demand"].to_numpy()
    )["underbake"]
    for min_n in [4, 6, 8]:
        for rate in [0.25, 0.50, 0.75]:
            for lost in [1.0, 2.0, 4.0]:
                for scale in [0.80, 0.90, 1.00]:
                    for unit_cap in [5.0, 10.0, 15.0]:
                        for relative_cap in [0.10, 0.25]:
                            params = {
                                "min_n": min_n,
                                "min_stockout_rate": rate,
                                "min_lost_mean": lost,
                                "scale": scale,
                                "unit_cap": unit_cap,
                                "relative_cap": relative_cap,
                            }
                            result = balance(
                                make_plan(history, params),
                                history["scenario_demand"].to_numpy(),
                            )
                            records.append(
                                {
                                    **params,
                                    **result,
                                    "passes_actual_under": result["underbake"]
                                    <= actual_under,
                                }
                            )
    grid = pd.DataFrame(records)
    feasible = grid[grid["passes_actual_under"]].sort_values(
        ["surplus", "underbake", "imbalance"]
    )
    return (
        feasible
        if not feasible.empty
        else grid.sort_values(["underbake", "surplus", "imbalance"])
    )


def add_scenario(
    predictions: pd.DataFrame,
    operational: pd.DataFrame,
    labels: pd.DataFrame,
    name: str,
    loss_scale: float,
) -> tuple[pd.DataFrame, list[dict]]:
    rows = predictions.merge(operational, on=KEYS, how="inner", validate="one_to_one")
    rows = rows.merge(
        labels[KEYS + ["demand_lower_bound", "imputed_demand"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    rows["imputed_demand"] = rows["imputed_demand"].fillna(0.0)
    rows["scenario_demand"] = (
        rows["demand_lower_bound"].fillna(0.0) + loss_scale * rows["imputed_demand"]
    )
    rows["direct_p50"] = rows["direct_forecast"] * rows["p50_factor"]
    raw = rows["direct_raw_demand"] + loss_scale * rows["predictive_uplift"]
    volume = rows.groupby(DAY)["direct_p50"].transform("sum")
    rows["direct_uplift_p50"] = normalize_to_volume(raw, rows, volume)

    floor_labels = labels.copy()
    floor_labels["demand_point_estimate"] = (
        floor_labels["demand_lower_bound"] + loss_scale * floor_labels["imputed_demand"]
    )
    floor_labels["imputed_demand"] *= loss_scale
    rows = add_floor_reference(rows, floor_labels)
    rows["direct_uplift_adaptive_floor"] = rows["direct_uplift_p50"]
    selections = []
    fold_names = [fold["name"] for fold in FOLDS]
    for index, fold_name in enumerate(fold_names):
        if index == 0:
            selections.append(
                {
                    "scenario": name,
                    "fold": fold_name,
                    "status": "calibration_start_no_floor",
                }
            )
            continue
        prior = rows[rows["rolling_fold"].isin(fold_names[:index])]
        selected = candidate_grid(prior, rows).iloc[0]
        params = {
            key: float(selected[key])
            for key in [
                "min_n",
                "min_stockout_rate",
                "min_lost_mean",
                "scale",
                "unit_cap",
                "relative_cap",
            ]
        }
        mask = rows["rolling_fold"].eq(fold_name)
        rows.loc[mask, "direct_uplift_adaptive_floor"] = make_plan(rows[mask], params)
        selections.append(
            {
                "scenario": name,
                "fold": fold_name,
                "status": "selected_on_prior_folds",
                **params,
            }
        )
    rows["scenario"] = name
    return rows, selections


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (scenario, fold), part in rows.groupby(["scenario", "rolling_fold"]):
        for variant in [
            "direct_p50",
            "direct_uplift_p50",
            "direct_uplift_adaptive_floor",
        ]:
            result = balance(
                part[variant].to_numpy(), part["scenario_demand"].to_numpy()
            )
            recognized = np.minimum(
                part["imputed_demand"] * SCENARIOS[scenario],
                (part[variant] - part["actual_sold"]).clip(lower=0),
            ).sum()
            total = part.groupby(DAY)[variant].transform("sum")
            top = (
                (part[variant] / total.replace(0, np.nan))
                .groupby([part["date"], part["bakery_id"]])
                .max()
            )
            records.append(
                {
                    "scenario": scenario,
                    "fold": fold,
                    "variant": variant,
                    **result,
                    "wape_pct": 100
                    * result["imbalance"]
                    / part["scenario_demand"].sum(),
                    "recognized_lost": float(recognized),
                    "top_share_max": float(top.max()),
                    "top_share_ge20": int(top.ge(0.20).sum()),
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    features = load_features()
    labels = pd.read_csv(LABELS, encoding="utf-8-sig")
    operational = pd.read_parquet(OPERATIONAL)[
        KEYS + ["p50_factor", "available_to_sell"]
    ]
    for frame in (features, labels, operational):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["product_id"] = pd.to_numeric(frame["product_id"]).astype("int64")
    predictions = build_predictions(features, labels)
    scenario_parts = []
    selections = []
    for name, scale in SCENARIOS.items():
        scenario, selected = add_scenario(predictions, operational, labels, name, scale)
        scenario_parts.append(scenario)
        selections.extend(selected)
    rows = pd.concat(scenario_parts, ignore_index=True)
    metrics = summarize(rows)
    evaluation = metrics[~metrics["fold"].eq(FOLDS[0]["name"])]
    aggregate = evaluation.groupby(["scenario", "variant"], as_index=False).agg(
        volume=("volume", "sum"),
        surplus=("surplus", "sum"),
        underbake=("underbake", "sum"),
        imbalance=("imbalance", "sum"),
        recognized_lost=("recognized_lost", "sum"),
        folds=("fold", "nunique"),
        max_top_share=("top_share_max", "max"),
        bakery_days_ge20=("top_share_ge20", "sum"),
    )
    demand_totals = (
        rows[~rows["rolling_fold"].eq(FOLDS[0]["name"])]
        .groupby("scenario")["scenario_demand"]
        .sum()
    )
    aggregate["wape_pct"] = (
        aggregate["imbalance"] / aggregate["scenario"].map(demand_totals) * 100
    )
    summary = {
        "folds": [
            {**fold, "dates": [str(date.date()) for date in fold["dates"]]}
            for fold in FOLDS
        ],
        "evaluation_folds": [fold["name"] for fold in FOLDS[1:]],
        "scenarios": SCENARIOS,
        "production_write": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(OUTPUT / "rows.parquet", index=False)
    metrics.to_csv(OUTPUT / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(
        OUTPUT / "aggregate_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(selections).to_csv(
        OUTPUT / "floor_selections.csv", index=False, encoding="utf-8-sig"
    )
    (OUTPUT / "design.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()

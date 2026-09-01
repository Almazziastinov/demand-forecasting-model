"""Build causal direct allocation variants with expected-loss uplift and adaptive floor."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from backtest_direct_bakery_sku_allocation import (
    CATEGORICAL,
    DAY,
    FEATURES,
    HISTORY,
    INPUT,
    KEYS,
    build_features,
)


ROOT = Path(__file__).resolve().parents[1]
DIRECT = ROOT / "reports/direct_bakery_sku_allocation_20260827/predictions.parquet"
OPERATIONAL = ROOT / "reports/rolling_actual_state_comparison_20260826/rows.parquet"
LABELS = ROOT / "reports/calibrated_stockout_network_20260826/sku_day_demand.csv"
OUTPUT = ROOT / "reports/direct_uplift_floor_20260827"
UPLIFT_FEATURES = FEATURES


def fit_uplift(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.Series, pd.Series, pd.Series]:
    classifier = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=180,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=150,
        reg_lambda=4.0,
        random_state=43,
        verbosity=-1,
    )
    classifier.fit(
        train[UPLIFT_FEATURES],
        train["is_clear_stockout"].astype(int),
        categorical_feature=CATEGORICAL,
    )
    positive = train[train["imputed_demand"].gt(0)].copy()
    severity = lgb.LGBMRegressor(
        objective="huber",
        n_estimators=180,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=100,
        reg_lambda=4.0,
        random_state=44,
        verbosity=-1,
    )
    severity.fit(
        positive[UPLIFT_FEATURES],
        np.log1p(positive["imputed_demand"]),
        categorical_feature=CATEGORICAL,
    )
    probability = pd.Series(
        classifier.predict_proba(test[UPLIFT_FEATURES])[:, 1], index=test.index
    )
    conditional = pd.Series(
        np.expm1(severity.predict(test[UPLIFT_FEATURES])).clip(min=0),
        index=test.index,
    )
    expected = probability * conditional
    return probability, conditional, expected


def add_floor_reference(rows: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    history = labels.copy()
    history["dow"] = history["date"].dt.dayofweek
    for date, day in rows.groupby("date", sort=True):
        date = pd.Timestamp(date)
        sample = history[
            history["date"].between(
                date - pd.Timedelta(days=56), date - pd.Timedelta(days=1)
            )
            & history["dow"].eq(date.dayofweek)
            & history["demand_point_estimate"].gt(0)
        ]
        reference = sample.groupby(["bakery_id", "product_id"], as_index=False).agg(
            floor_history_n=("demand_point_estimate", "size"),
            floor_demand_p67=(
                "demand_point_estimate",
                lambda values: values.quantile(0.67),
            ),
            historical_stockout_rate=("is_clear_stockout", "mean"),
            historical_lost_mean=("imputed_demand", "mean"),
        )
        outputs.append(
            day.merge(
                reference,
                on=["bakery_id", "product_id"],
                how="left",
                validate="many_to_one",
            )
        )
    result = pd.concat(outputs, ignore_index=True)
    result["floor_history_n"] = result["floor_history_n"].fillna(0).astype(int)
    for column in [
        "floor_demand_p67",
        "historical_stockout_rate",
        "historical_lost_mean",
    ]:
        result[column] = result[column].fillna(0.0)
    return result


def normalize_to_volume(
    raw: pd.Series, rows: pd.DataFrame, volume: pd.Series
) -> pd.Series:
    denominator = raw.groupby([rows["date"], rows["bakery_id"]]).transform("sum")
    share = raw / denominator.replace(0, np.nan)
    return share.fillna(0.0) * volume


def build_fold(
    features: pd.DataFrame,
    direct: pd.DataFrame,
    labels: pd.DataFrame,
    train_end: str,
    fold: str,
) -> pd.DataFrame:
    enriched = features.merge(
        labels[KEYS + ["is_clear_stockout", "imputed_demand"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    enriched["is_clear_stockout"] = enriched["is_clear_stockout"].fillna(False)
    enriched["imputed_demand"] = enriched["imputed_demand"].fillna(0.0)
    test = direct[direct["fold"].eq(fold)].copy()
    train = enriched[enriched["date"].le(pd.Timestamp(train_end))].copy()

    probability, conditional, expected = fit_uplift(train, test)
    test["predicted_stockout_probability"] = probability
    test["predicted_lost_if_stockout"] = conditional
    test["predictive_uplift"] = expected
    test["uplift_raw_demand"] = test["direct_raw_demand"] + expected
    return test


def main() -> None:
    snapshot = pd.read_parquet(INPUT).rename(
        columns={"forecast_qty": "incumbent_sku_forecast"}
    )
    history = pd.read_parquet(HISTORY)
    direct = pd.read_parquet(DIRECT)
    operational = pd.read_parquet(OPERATIONAL)[KEYS + ["p50_factor", "demand"]]
    labels = pd.read_csv(LABELS, encoding="utf-8-sig")
    for frame in (snapshot, history, direct, operational, labels):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["product_id"] = pd.to_numeric(frame["product_id"]).astype("int64")

    features = build_features(snapshot, history)
    blocked = build_fold(features, direct, labels, "2026-07-21", "blocked")
    current = build_fold(features, direct, labels, "2026-08-10", "current")
    rows = pd.concat([blocked, current], ignore_index=True)
    rows = rows.merge(operational, on=KEYS, how="inner", validate="one_to_one")
    rows["direct_p50"] = rows["direct_forecast"] * rows["p50_factor"]
    p50_volume = rows.groupby(DAY)["direct_p50"].transform("sum")
    rows["direct_uplift_p50"] = normalize_to_volume(
        rows["uplift_raw_demand"], rows, p50_volume
    )
    rows = add_floor_reference(rows, labels)
    eligible = (
        rows["floor_history_n"].ge(6)
        & rows["historical_stockout_rate"].ge(0.10)
        & rows["floor_demand_p67"].gt(rows["direct_uplift_p50"])
    )
    cap = np.minimum(
        rows["direct_uplift_p50"] + 15.0,
        rows["direct_uplift_p50"] * 1.25,
    )
    floor_target = np.minimum(rows["floor_demand_p67"], cap)
    rows["direct_uplift_adaptive_floor"] = np.where(
        eligible,
        np.maximum(rows["direct_uplift_p50"], floor_target),
        rows["direct_uplift_p50"],
    )
    rows["adaptive_floor_eligible"] = eligible
    rows["adaptive_floor_increment"] = (
        rows["direct_uplift_adaptive_floor"] - rows["direct_uplift_p50"]
    )

    variants = [
        "direct_p50",
        "direct_uplift_p50",
        "direct_uplift_adaptive_floor",
    ]
    actual = rows["demand"].sum()
    metrics = {}
    for variant in variants:
        error = rows[variant] - rows["demand"]
        metrics[variant] = {
            "volume": float(rows[variant].sum()),
            "wape_pct": float(100 * error.abs().sum() / actual),
            "bias_pct": float(100 * error.sum() / actual),
            "surplus": float(error.clip(lower=0).sum()),
            "underbake": float((-error).clip(lower=0).sum()),
        }
    summary = {
        "scope": {
            "dates": int(rows["date"].nunique()),
            "bakery_days": int(rows.groupby(DAY).ngroups),
            "sku_rows": int(len(rows)),
        },
        "metrics": metrics,
        "uplift": {
            "expected_units_before_normalization": float(
                rows["predictive_uplift"].sum()
            ),
            "mean_stockout_probability": float(
                rows["predicted_stockout_probability"].mean()
            ),
        },
        "adaptive_floor": {
            "eligible_rows": int(eligible.sum()),
            "increased_rows": int(rows["adaptive_floor_increment"].gt(0).sum()),
            "increment": float(rows["adaptive_floor_increment"].sum()),
        },
        "production_write": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(OUTPUT / "rows.parquet", index=False)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Train a research-only Direct model on economically gated corrected demand."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.backtest_direct_bakery_sku_allocation import (  # noqa: E402
    CATEGORICAL,
    FEATURES,
)


FEATURE_CACHE = ROOT / ".codex_tmp/direct_bakery_sku_features_20260827.parquet"
LABELS = ROOT / "reports/weighted_lost_demand_labels_20260914/floor_history.parquet"
BASE_ARTIFACTS = ROOT / "models/direct_alpha_025_v1"
OUTPUT = ROOT / "models/direct_corrected_demand_gate30_v1"
CONTINUATION_GATE = 0.30


def main() -> None:
    features = pd.read_parquet(FEATURE_CACHE)
    labels = pd.read_parquet(LABELS)
    for frame in (features, labels):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["product_id"] = frame["product_id"].astype("int64")

    train_end = min(features["date"].max(), labels["date"].max())
    train = features[features["date"].le(train_end)].copy()
    label_columns = [
        "date",
        "bakery_id",
        "product_id",
        "is_clear_stockout",
        "continuation_probability",
        "hierarchical_imputed_demand",
    ]
    train = train.merge(
        labels[label_columns],
        on=["date", "bakery_id", "product_id"],
        how="left",
        validate="one_to_one",
    )
    train["is_clear_stockout"] = train["is_clear_stockout"].fillna(False)
    train["continuation_probability"] = train["continuation_probability"].fillna(0.0)
    train["hierarchical_imputed_demand"] = (
        train["hierarchical_imputed_demand"].fillna(0.0).clip(lower=0.0)
    )
    train["is_economic_lost"] = (
        train["is_clear_stockout"]
        & train["continuation_probability"].ge(CONTINUATION_GATE)
        & train["hierarchical_imputed_demand"].gt(0.0)
    )
    train["economic_lost_demand"] = train["hierarchical_imputed_demand"].where(
        train["is_economic_lost"], 0.0
    )
    train["corrected_demand"] = train["actual_sold"] + train["economic_lost_demand"]

    direct = lgb.LGBMRegressor(
        objective="poisson",
        n_estimators=240,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=120,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        random_state=42,
        verbosity=-1,
    )
    direct.fit(train[FEATURES], train["corrected_demand"], categorical_feature=CATEGORICAL)

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
        train[FEATURES],
        train["is_economic_lost"].astype(int),
        categorical_feature=CATEGORICAL,
    )

    positive = train[train["economic_lost_demand"].gt(0.0)].copy()
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
        positive[FEATURES],
        np.log1p(positive["economic_lost_demand"]),
        categorical_feature=CATEGORICAL,
    )

    floor_history = labels.copy()
    floor_history["is_clear_stockout"] = (
        floor_history["is_clear_stockout"].fillna(False)
        & floor_history["continuation_probability"].ge(CONTINUATION_GATE)
        & floor_history["hierarchical_imputed_demand"].gt(0.0)
    )
    floor_history["imputed_demand"] = floor_history["hierarchical_imputed_demand"].where(
        floor_history["is_clear_stockout"], 0.0
    )
    floor_history["demand_point_estimate"] = (
        floor_history["demand_lower_bound"] + floor_history["imputed_demand"]
    )

    base_metadata = json.loads((BASE_ARTIFACTS / "metadata.json").read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(direct, OUTPUT / "direct_model.joblib")
    joblib.dump(classifier, OUTPUT / "stockout_classifier.joblib")
    joblib.dump(severity, OUTPUT / "lost_severity_model.joblib")
    floor_history.to_parquet(OUTPUT / "floor_history.parquet", index=False)
    metadata = {
        **base_metadata,
        "version": "direct_corrected_demand_gate30_v1",
        "train_end": str(train_end.date()),
        "train_rows": int(len(train)),
        "positive_loss_rows": int(len(positive)),
        "continuation_gate": CONTINUATION_GATE,
        "observed_sales_sum": float(train["actual_sold"].sum()),
        "economic_lost_sum": float(train["economic_lost_demand"].sum()),
        "corrected_demand_sum": float(train["corrected_demand"].sum()),
        "direct_target": "actual_sold + economically gated hierarchical lost demand",
        "base_direct_model_fixed": False,
        "source_artifact": str(BASE_ARTIFACTS),
        "production_write": False,
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    public = {key: value for key, value in metadata.items() if key not in {"mappings", "p50_factors"}}
    print(json.dumps(public, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Train research Direct loss artifacts on probability-weighted labels."""

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
OUTPUT = ROOT / "models/direct_alpha_025_weighted_loss_v1"


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
        "imputed_demand",
    ]
    train = train.merge(labels[label_columns], on=["date", "bakery_id", "product_id"], how="left")
    train["is_clear_stockout"] = train["is_clear_stockout"].fillna(False)
    for column in ["continuation_probability", "hierarchical_imputed_demand", "imputed_demand"]:
        train[column] = train[column].fillna(0.0).clip(lower=0.0)

    # Observed-sales Direct is intentionally fixed to isolate the new lost-demand layer.
    direct = joblib.load(BASE_ARTIFACTS / "direct_model.joblib")
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
    target = train["is_clear_stockout"].astype(int)
    classifier_weight = np.where(
        target.eq(1), train["continuation_probability"].clip(lower=0.02), 1.0
    )
    classifier.fit(
        train[FEATURES],
        target,
        sample_weight=classifier_weight,
        categorical_feature=CATEGORICAL,
    )

    positive = train[train["hierarchical_imputed_demand"].gt(0.0)].copy()
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
        np.log1p(positive["hierarchical_imputed_demand"]),
        sample_weight=positive["continuation_probability"].clip(lower=0.02),
        categorical_feature=CATEGORICAL,
    )

    base_metadata = json.loads((BASE_ARTIFACTS / "metadata.json").read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(direct, OUTPUT / "direct_model.joblib")
    joblib.dump(classifier, OUTPUT / "stockout_classifier.joblib")
    joblib.dump(severity, OUTPUT / "lost_severity_model.joblib")
    labels.to_parquet(OUTPUT / "floor_history.parquet", index=False)
    metadata = {
        **base_metadata,
        "version": "direct_alpha_025_weighted_loss_v1",
        "train_end": str(train_end.date()),
        "train_rows": int(len(train)),
        "positive_loss_rows": int(len(positive)),
        "weighted_loss_sum": float(train["imputed_demand"].sum()),
        "mean_positive_probability": float(positive["continuation_probability"].mean()),
        "base_direct_model_fixed": True,
        "source_artifact": str(BASE_ARTIFACTS),
        "production_write": False,
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in metadata.items() if key not in {"mappings", "p50_factors"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

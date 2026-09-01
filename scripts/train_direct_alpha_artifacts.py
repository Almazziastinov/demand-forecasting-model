"""Train and save the frozen Direct alpha=.25 integration artifacts."""

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
LABELS = ROOT / "reports/calibrated_stockout_network_20260826/sku_day_demand.csv"
P50_ROWS = ROOT / "reports/rolling_actual_state_comparison_20260826/rows.parquet"
OUTPUT = ROOT / "models/direct_alpha_025_v1"


def main() -> None:
    features = pd.read_parquet(FEATURE_CACHE)
    labels = pd.read_csv(LABELS, encoding="utf-8-sig")
    for frame in (features, labels):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["product_id"] = frame["product_id"].astype("int64")
    train_end = min(features["date"].max(), labels["date"].max())
    train = features[features["date"].le(train_end)].copy()
    train = train.merge(
        labels[
            [
                "date",
                "bakery_id",
                "product_id",
                "is_clear_stockout",
                "imputed_demand",
            ]
        ],
        on=["date", "bakery_id", "product_id"],
        how="left",
        validate="one_to_one",
    )
    train["is_clear_stockout"] = train["is_clear_stockout"].fillna(False)
    train["imputed_demand"] = train["imputed_demand"].fillna(0.0)

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
    direct.fit(train[FEATURES], train["actual_sold"], categorical_feature=CATEGORICAL)

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
        train["is_clear_stockout"].astype(int),
        categorical_feature=CATEGORICAL,
    )
    positive = train[train["imputed_demand"].gt(0.0)].copy()
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
        np.log1p(positive["imputed_demand"]),
        categorical_feature=CATEGORICAL,
    )

    mappings = {}
    for source, target in [
        ("bakery_id", "bakery_code"),
        ("product_id", "product_code"),
        ("category", "category_code"),
    ]:
        pairs = train[[source, target]].drop_duplicates(source)
        mappings[source] = dict(
            zip(pairs[source].astype(str), pairs[target].astype(int))
        )
    p50 = pd.read_parquet(P50_ROWS)[["bakery_id", "p50_factor"]].dropna()
    p50_factors = p50.groupby("bakery_id")["p50_factor"].median().to_dict()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(direct, OUTPUT / "direct_model.joblib")
    joblib.dump(classifier, OUTPUT / "stockout_classifier.joblib")
    joblib.dump(severity, OUTPUT / "lost_severity_model.joblib")
    labels.to_csv(
        OUTPUT / "floor_history.csv.gz",
        index=False,
        encoding="utf-8",
        compression="gzip",
    )
    metadata = {
        "version": "direct_alpha_025_v1",
        "train_end": str(train_end.date()),
        "train_rows": int(len(train)),
        "positive_loss_rows": int(len(positive)),
        "features": FEATURES,
        "categorical": CATEGORICAL,
        "mappings": mappings,
        "p50_factors": {str(key): float(value) for key, value in p50_factors.items()},
        "p50_fallback": float(p50["p50_factor"].median()),
        "alpha": 0.25,
        "production_write": False,
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    public_metadata = {
        key: value
        for key, value in metadata.items()
        if key not in {"mappings", "p50_factors"}
    }
    print(json.dumps(public_metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

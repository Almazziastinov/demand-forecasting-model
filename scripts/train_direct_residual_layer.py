"""Train and evaluate a causal residual-risk layer on top of Direct alpha=.25."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_direct_risk_reallocation import (  # noqa: E402
    Candidate,
    build_candidate,
    evaluate,
)
from src.experiments_v2.direct_alpha_allocation import (  # noqa: E402
    DirectAlphaAllocationConfig,
    build_selected_direct_plan,
)


ROLLING = ROOT / "reports/rolling_direct_uplift_floor_20260827/rows.parquet"
TEST_INPUT = (
    ROOT / "reports/three_prod_models_august_holdout_20260911/evaluation_input.parquet"
)
DIRECT_DAYS = ROOT / "reports/three_prod_models_august_holdout_20260911/direct_days"
ECONOMICS_DIR = ROOT / "reports/three_prod_models_august_holdout_20260911"
OUTPUT = ROOT / "reports/direct_residual_layer_20260911"
KEYS = ["date", "bakery_id", "product_id"]
TRAIN_END = pd.Timestamp("2026-08-13")
VALIDATION_START = pd.Timestamp("2026-08-17")
VALIDATION_END = pd.Timestamp("2026-08-23")

FEATURES = [
    "bakery_code",
    "product_code",
    "category_code",
    "dow",
    "log_bakery_total",
    "recent_7_mean",
    "prior_7_mean",
    "broad_56_mean",
    "same_weekday_4_mean",
    "presence_28",
    "recent_7_share",
    "broad_56_share",
    "same_weekday_4_share",
    "historical_category_share",
    "recent_trend",
    "selected_sku_forecast",
    "direct_p50",
    "direct_alpha_floor",
    "floor_demand_p67",
    "floor_history_n",
    "historical_stockout_rate",
    "historical_lost_mean",
    "predicted_stockout_probability",
    "predicted_lost_if_stockout",
    "predictive_uplift",
    "plan_share",
    "plan_to_recent_ratio",
    "plan_to_broad_ratio",
    "plan_to_weekday_ratio",
    "plan_minus_floor",
]
CATEGORICAL = ["bakery_code", "product_code", "category_code", "dow"]


def add_derived_features(rows: pd.DataFrame) -> pd.DataFrame:
    result = rows.copy()
    day_total = result.groupby(["date", "bakery_id"])[
        "selected_sku_forecast"
    ].transform("sum")
    result["plan_share"] = result["selected_sku_forecast"] / day_total.replace(
        0.0, np.nan
    )
    for source, target in [
        ("recent_7_mean", "plan_to_recent_ratio"),
        ("broad_56_mean", "plan_to_broad_ratio"),
        ("same_weekday_4_mean", "plan_to_weekday_ratio"),
    ]:
        result[target] = result["selected_sku_forecast"] / result[source].replace(
            0.0, np.nan
        )
        result[target] = result[target].clip(0.0, 20.0).fillna(20.0)
    result["plan_minus_floor"] = (
        result["selected_sku_forecast"] - result["floor_demand_p67"]
    )
    result["plan_share"] = result["plan_share"].fillna(0.0)
    return result


def build_training_rows() -> pd.DataFrame:
    rows = pd.read_parquet(ROLLING)
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    rows = rows[rows["scenario"].eq("calibrated")].copy()
    rows["loss_scale"] = 1.0
    rows = build_selected_direct_plan(rows, DirectAlphaAllocationConfig())
    rows = add_derived_features(rows)
    rows["residual_qty"] = (
        rows["scenario_demand"] - rows["selected_sku_forecast"]
    ).clip(lower=0.0)
    rows["residual_case"] = rows["residual_qty"].ge(1.0).astype("int8")
    rows["certain_residual"] = (
        rows["actual_sold"] - rows["selected_sku_forecast"]
    ).ge(1.0)
    rows["label_weight"] = np.where(
        rows["residual_case"].eq(1),
        np.where(rows["certain_residual"], 1.0, 0.35),
        0.70,
    )
    return rows


def build_test_rows(universe: pd.DataFrame) -> pd.DataFrame:
    columns = list(
        dict.fromkeys(
            KEYS
            + FEATURES
            + [
                "selected_sku_forecast",
                "direct_p50",
                "floor_demand_p67",
                "predicted_stockout_probability",
                "predicted_lost_if_stockout",
                "predictive_uplift",
            ]
        )
    )
    base_columns = [column for column in columns if column not in {
        "plan_share",
        "plan_to_recent_ratio",
        "plan_to_broad_ratio",
        "plan_to_weekday_ratio",
        "plan_minus_floor",
    }]
    frames = [
        pd.read_parquet(path, columns=base_columns)
        for path in sorted(DIRECT_DAYS.glob("*/shadow_rows.parquet"))
    ]
    features = pd.concat(frames, ignore_index=True)
    features["date"] = pd.to_datetime(features["date"]).dt.normalize()
    test = universe.merge(features, on=KEYS, how="left", validate="one_to_one")
    return add_derived_features(test)


def fit_models(
    train: pd.DataFrame, validation: pd.DataFrame
) -> tuple[LGBMClassifier, LGBMRegressor]:
    classifier = LGBMClassifier(
        objective="binary",
        n_estimators=350,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=200,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=42,
        verbosity=-1,
    )
    classifier.fit(
        train[FEATURES],
        train["residual_case"],
        sample_weight=train["label_weight"],
        categorical_feature=CATEGORICAL,
        eval_set=[(validation[FEATURES], validation["residual_case"])],
        callbacks=[],
    )
    positive = train[train["residual_case"].eq(1)]
    severity = LGBMRegressor(
        objective="regression_l1",
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=150,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=42,
        verbosity=-1,
    )
    severity.fit(
        positive[FEATURES],
        np.log1p(positive["residual_qty"]),
        sample_weight=positive["label_weight"],
        categorical_feature=CATEGORICAL,
    )
    return classifier, severity

def score_model(
    rows: pd.DataFrame, classifier: LGBMClassifier, severity: LGBMRegressor
) -> pd.DataFrame:
    result = rows.copy()
    result["residual_probability"] = classifier.predict_proba(result[FEATURES])[:, 1]
    result["residual_severity"] = np.expm1(
        severity.predict(result[FEATURES])
    ).clip(min=0.0)
    result["residual_expected"] = (
        result["residual_probability"] * result["residual_severity"]
    )
    return result


def classification_metrics(rows: pd.DataFrame, label: str) -> dict[str, float | str]:
    return {
        "split": label,
        "rows": int(len(rows)),
        "positive_rate_pct": 100 * float(rows["residual_case"].mean()),
        "roc_auc": float(
            roc_auc_score(rows["residual_case"], rows["residual_probability"])
        ),
        "average_precision": float(
            average_precision_score(
                rows["residual_case"], rows["residual_probability"]
            )
        ),
    }


def quantile_candidates(validation: pd.DataFrame) -> list[Candidate]:
    probability = validation["residual_probability"]
    return [
        Candidate(
            "residual_q90_q20_cap3",
            float(probability.quantile(0.90)),
            float(probability.quantile(0.20)),
            3.0,
            0.10,
        ),
        Candidate(
            "residual_q85_q25_cap3",
            float(probability.quantile(0.85)),
            float(probability.quantile(0.25)),
            3.0,
            0.15,
        ),
        Candidate(
            "residual_q80_q30_cap5",
            float(probability.quantile(0.80)),
            float(probability.quantile(0.30)),
            5.0,
            0.20,
        ),
        Candidate(
            "residual_q75_q40_cap5",
            float(probability.quantile(0.75)),
            float(probability.quantile(0.40)),
            5.0,
            0.20,
        ),
    ]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    history = build_training_rows()
    train = history[history["date"].le(TRAIN_END)].copy()
    validation = history[
        history["date"].between(VALIDATION_START, VALIDATION_END)
    ].copy()
    classifier, severity = fit_models(train, validation)
    validation = score_model(validation, classifier, severity)

    economics_input = pd.read_parquet(TEST_INPUT)
    economics_input["date"] = pd.to_datetime(economics_input["date"]).dt.normalize()
    test = build_test_rows(economics_input[KEYS])
    test = score_model(test, classifier, severity)
    test_labels = economics_input[KEYS + ["demand", "observed_sales_qty"]]
    test_scored = test.merge(test_labels, on=KEYS, validate="one_to_one")
    test_scored["residual_qty"] = (
        test_scored["demand"] - test_scored["selected_sku_forecast"]
    ).clip(lower=0.0)
    test_scored["residual_case"] = test_scored["residual_qty"].ge(1.0).astype("int8")

    metric_rows = [
        classification_metrics(validation, "validation_2026-08-17_23"),
        classification_metrics(test_scored, "test_2026-08-24_31"),
    ]
    pd.DataFrame(metric_rows).to_csv(
        OUTPUT / "classifier_metrics.csv", index=False, encoding="utf-8-sig"
    )

    actual = pd.read_parquet(ECONOMICS_DIR / "economics_actual_state.parquet")
    actual_cases = actual["lost"].ge(1.0)
    actual_gp = float(actual["gross_profit"].sum())
    direct = pd.read_parquet(
        ECONOMICS_DIR / "economics_direct_alpha_025_v1.parquet"
    )
    records = [
        {
            "variant": "direct_alpha_025_v1",
            "transferred_qty": 0.0,
            "lost_cases": int(direct["lost"].ge(1.0).sum()),
            "case_delta_vs_actual": int(
                direct["lost"].ge(1.0).sum() - actual_cases.sum()
            ),
            "lost_qty": float(direct["lost"].sum()),
            "writeoff_qty": float(direct["writeoff"].sum()),
            "service_pct": 100 * direct["served"].sum() / direct["demand"].sum(),
            "gross_profit": float(direct["gross_profit"].sum()),
            "gp_delta_vs_actual": float(direct["gross_profit"].sum() - actual_gp),
        }
    ]

    allocator_features = test.copy()
    allocator_features["predicted_stockout_probability"] = allocator_features[
        "residual_probability"
    ]
    allocator_features["predicted_lost_if_stockout"] = allocator_features[
        "residual_severity"
    ]
    allocator_features["predictive_uplift"] = allocator_features[
        "residual_expected"
    ]
    candidates = quantile_candidates(validation)
    for index, candidate in enumerate(candidates):
        plan_column = f"residual_plan_{index}"
        plan = build_candidate(allocator_features, candidate)
        economics_input[plan_column] = plan
        transferred = 0.5 * float(
            (plan - allocator_features["selected_sku_forecast"]).abs().sum()
        )
        record, simulation = evaluate(
            economics_input,
            plan_column,
            candidate.name,
            actual_cases,
            actual_gp,
            transferred,
        )
        records.append(record)
        simulation.to_parquet(OUTPUT / f"economics_{candidate.name}.parquet")

    summary = pd.DataFrame(records).sort_values(
        ["lost_cases", "gross_profit"], ascending=[True, False]
    )
    summary.to_csv(OUTPUT / "economics_summary.csv", index=False, encoding="utf-8-sig")
    test_scored.to_parquet(OUTPUT / "test_scores.parquet", index=False)
    joblib.dump(classifier, OUTPUT / "residual_classifier.joblib")
    joblib.dump(severity, OUTPUT / "residual_severity.joblib")
    metadata = {
        "train_end": str(TRAIN_END.date()),
        "validation": [str(VALIDATION_START.date()), str(VALIDATION_END.date())],
        "test": ["2026-08-24", "2026-08-31"],
        "features": FEATURES,
        "label": "scenario_demand - selected Direct plan >= 1",
        "label_weights": {
            "certain_positive": 1.0,
            "reconstructed_only_positive": 0.35,
            "negative": 0.70,
        },
        "candidates": [asdict(candidate) for candidate in candidates],
        "production_write": False,
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

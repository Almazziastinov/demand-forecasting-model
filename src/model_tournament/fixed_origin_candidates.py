"""Audited legacy-objective adapters and new fixed-origin candidates."""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


BASELINE_ID = "weighted_weekday_sales_formula"
P60_ID = "lgbm_quantile_p60_fixed_origin"
RESIDUAL_ID = "lgbm_weighted_residual_fixed_origin"
LEGACY_L2_ID = "legacy_exp01_lgbm_l2_adapted"
LEGACY_TWEEDIE_ID = "legacy_exp40_tweedie_adapted"
LEGACY_LOG_ID = "legacy_exp41_log_target_adapted"
LEGACY_P50_ID = "legacy_exp43_quantile_p50_adapted"
CATBOOST_ID = "new_catboost_mae"
HURDLE_ID = "new_lgbm_hurdle_mean"
HURDLE_MEDIAN_ID = "new_lgbm_hurdle_median"

DEFAULT_CANDIDATES = (
    LEGACY_L2_ID,
    LEGACY_TWEEDIE_ID,
    LEGACY_LOG_ID,
    LEGACY_P50_ID,
    P60_ID,
    RESIDUAL_ID,
    CATBOOST_ID,
    HURDLE_ID,
    HURDLE_MEDIAN_ID,
)

PROVENANCE = {
    BASELINE_ID: {
        "class": "formula_baseline",
        "source": "pipelines/forecast_publish/weighted_weekday_production.py",
        "parity": "sales formula only; not production demand or scope",
    },
    LEGACY_L2_ID: {
        "class": "adapted_legacy_objective",
        "source": "src/experiments_v2/01_baseline_8m/run.py",
        "parity": (
            "L2 objective on common fixed-origin features; "
            "not original feature set or parameters"
        ),
    },
    LEGACY_TWEEDIE_ID: {
        "class": "adapted_legacy_objective",
        "source": "src/experiments_v2/40_tweedie/run.py",
        "parity": (
            "native Tweedie objective, variance power 1.5; common fixed-origin features"
        ),
    },
    LEGACY_LOG_ID: {
        "class": "adapted_legacy_objective",
        "source": "src/experiments_v2/41_log_target/run.py",
        "parity": "log1p/expm1 target path; common fixed-origin features",
    },
    LEGACY_P50_ID: {
        "class": "adapted_legacy_objective",
        "source": "src/experiments_v2/43_quantile/run.py",
        "parity": "native P50 objective; not original features or test split",
    },
    P60_ID: {
        "class": "corrected_legacy_idea",
        "source": "src/experiments_v2/42_asymmetric_loss/run.py",
        "parity": "native P60 quantile replaces broken custom asymmetric objective",
    },
    RESIDUAL_ID: {
        "class": "new_candidate",
        "source": "src/model_tournament/fixed_origin_candidates.py",
        "parity": "L1 residual over sales weighted-weekday formula",
    },
    CATBOOST_ID: {
        "class": "new_candidate",
        "source": "src/model_tournament/fixed_origin_candidates.py",
        "parity": "CatBoost MAE with categorical bakery/product IDs",
    },
    HURDLE_ID: {
        "class": "new_candidate",
        "source": "src/model_tournament/fixed_origin_candidates.py",
        "parity": "Pr(sales>0) times conditional positive-sales mean; uncalibrated",
    },
    HURDLE_MEDIAN_ID: {
        "class": "new_candidate",
        "source": "src/model_tournament/fixed_origin_candidates.py",
        "parity": (
            "Zero-inflated median from Pr(sales>0) and conditional positive-sales "
            "quantiles; probability and quantiles are uncalibrated"
        ),
    },
}


def _lightgbm_params(n_estimators: int) -> dict[str, object]:
    return {
        "n_estimators": n_estimators,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "random_state": 42,
        "n_jobs": 4,
        "verbosity": -1,
    }


def _zero_inflated_median(
    probability: np.ndarray, positive_quantiles: np.ndarray
) -> np.ndarray:
    """Approximate the mixture median from positive-demand quantiles.

    For a point mass of ``1-p`` at zero, the unconditional median is zero when
    ``p <= 0.5``. Otherwise its conditional-positive quantile level is
    ``1 - 0.5/p``. A zero anchor avoids a jump at ``p == 0.5``.
    """
    probability = np.asarray(probability, dtype=float)
    quantiles = np.maximum(np.asarray(positive_quantiles, dtype=float), 0.0)
    if quantiles.shape != (len(probability), 3):
        raise ValueError("Expected positive quantiles with shape (rows, 3)")
    quantiles = np.maximum.accumulate(quantiles, axis=1)
    level = np.where(probability > 0.5, 1.0 - 0.5 / np.maximum(probability, 1e-12), 0.0)
    q10, q30, q50 = quantiles.T
    median = np.where(
        level <= 0.1,
        q10 * level / 0.1,
        np.where(
            level <= 0.3,
            q10 + (q30 - q10) * (level - 0.1) / 0.2,
            q30 + (q50 - q30) * (level - 0.3) / 0.2,
        ),
    )
    return np.where(probability > 0.5, median, 0.0)


def fit_predict_candidates(
    X_train: pd.DataFrame,
    X_eval: pd.DataFrame,
    y_train: np.ndarray,
    baseline_train: np.ndarray,
    baseline_eval: np.ndarray,
    *,
    model_ids: tuple[str, ...] = DEFAULT_CANDIDATES,
    n_estimators: int = 160,
) -> dict[str, np.ndarray]:
    """Fit every selected model on identical rows without touching eval labels."""
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("Candidate model IDs must be unique")
    unknown = sorted(set(model_ids) - set(DEFAULT_CANDIDATES))
    if unknown:
        raise ValueError(f"Unknown candidate models: {unknown}")
    if n_estimators < 1:
        raise ValueError("n_estimators must be positive")
    params = _lightgbm_params(n_estimators)
    predictions: dict[str, np.ndarray] = {}

    for model_id in model_ids:
        if model_id == CATBOOST_ID:
            cat_train = X_train.copy()
            cat_eval = X_eval.copy()
            for column in ("bakery_id", "product_id"):
                cat_train[column] = cat_train[column].astype(str)
                cat_eval[column] = cat_eval[column].astype(str)
            model = CatBoostRegressor(
                iterations=n_estimators,
                depth=6,
                learning_rate=0.05,
                loss_function="MAE",
                random_seed=42,
                thread_count=4,
                verbose=False,
                allow_writing_files=False,
            )
            model.fit(cat_train, y_train, cat_features=["bakery_id", "product_id"])
            values = model.predict(cat_eval)
        elif model_id in (HURDLE_ID, HURDLE_MEDIAN_ID):
            positive = y_train > 0
            if not positive.any() or positive.all():
                probability = np.full(len(X_eval), float(positive.mean()))
            else:
                classifier = LGBMClassifier(**params, objective="binary")
                classifier.fit(X_train, positive.astype(int))
                probability = classifier.predict_proba(X_eval)[:, 1]
            if positive.any() and model_id == HURDLE_MEDIAN_ID:
                positive_quantiles = []
                for alpha in (0.1, 0.3, 0.5):
                    quantile_model = LGBMRegressor(
                        **params, objective="quantile", alpha=alpha
                    )
                    quantile_model.fit(X_train.loc[positive], y_train[positive])
                    positive_quantiles.append(quantile_model.predict(X_eval))
                values = _zero_inflated_median(
                    probability, np.column_stack(positive_quantiles)
                )
            elif positive.any():
                conditional = LGBMRegressor(**params, objective="regression")
                conditional.fit(X_train.loc[positive], y_train[positive])
                positive_mean = np.maximum(conditional.predict(X_eval), 0.0)
                values = probability * positive_mean
            else:
                values = np.zeros(len(X_eval), dtype=float)
        elif model_id == RESIDUAL_ID:
            model = LGBMRegressor(**params, objective="regression_l1")
            model.fit(X_train, y_train - baseline_train)
            values = baseline_eval + model.predict(X_eval)
        elif model_id == LEGACY_LOG_ID:
            model = LGBMRegressor(**params, objective="regression")
            model.fit(X_train, np.log1p(y_train))
            values = np.expm1(model.predict(X_eval))
        else:
            objective = "regression"
            extra: dict[str, float] = {}
            if model_id == LEGACY_TWEEDIE_ID:
                objective = "tweedie"
                extra["tweedie_variance_power"] = 1.5
            elif model_id in (LEGACY_P50_ID, P60_ID):
                objective = "quantile"
                extra["alpha"] = 0.5 if model_id == LEGACY_P50_ID else 0.6
            model = LGBMRegressor(**params, objective=objective, **extra)
            model.fit(X_train, y_train)
            values = model.predict(X_eval)
        predicted = np.maximum(np.asarray(values, dtype=float), 0.0)
        if len(predicted) != len(X_eval) or not np.isfinite(predicted).all():
            raise ValueError(f"Candidate {model_id!r} produced invalid predictions")
        predictions[model_id] = predicted
    return predictions

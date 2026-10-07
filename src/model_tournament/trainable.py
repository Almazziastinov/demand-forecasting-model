"""Freshly trained candidates using only tournament-built causal features."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.model_tournament.baselines import predict_baseline


QUANTILE_PATTERN = re.compile(r"lgbm_quantile_p(\d{2})$")
TRAINABLE_MODELS = ("lgbm_quantile_p60", "lgbm_weighted_residual_v1")


def build_causal_features(
    panel: pd.DataFrame,
    *,
    target_col: str,
    key_cols: list[str],
) -> pd.DataFrame:
    """Build features for a daily lead-one forecast.

    Every target-derived input is computed through the same dated history
    lookup used by the deterministic baselines.
    """
    features = pd.DataFrame(index=panel.index)
    for key in key_cols:
        features[key] = panel[key].astype("category")
    features["day_of_week"] = panel["date"].dt.dayofweek
    features["month"] = panel["date"].dt.month
    features["day_of_month"] = panel["date"].dt.day
    for name in ("lag1", "lag7", "mean7", "mean14", "weighted_weekday_formula_v1"):
        features[name] = predict_baseline(
            name,
            panel,
            target_col=target_col,
            key_cols=key_cols,
            lead_days=1,
        )
    return features


def train_and_predict(
    model_id: str,
    panel: pd.DataFrame,
    *,
    target_col: str,
    key_cols: list[str],
    evaluation_mask: pd.Series,
    train_end: pd.Timestamp,
    n_estimators: int = 160,
) -> tuple[pd.Series, dict[str, object]]:
    """Fit one model before the holdout and score its complete evaluation set."""
    if model_id not in TRAINABLE_MODELS:
        raise ValueError(f"Unknown trainable model: {model_id}")
    if train_end >= panel.loc[evaluation_mask, "date"].min():
        raise ValueError("train_end must precede every evaluation date")
    train_mask = panel["date"].le(train_end)
    if int(train_mask.sum()) < 30:
        raise ValueError("At least 30 historical rows are required for training")

    features = build_causal_features(
        panel, target_col=target_col, key_cols=key_cols
    )
    X_train = features.loc[train_mask]
    X_eval = features.loc[evaluation_mask]
    y_train = panel.loc[train_mask, target_col].to_numpy(dtype=float)
    base_params = {
        "n_estimators": n_estimators,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "random_state": 42,
        "n_jobs": 4,
        "verbosity": -1,
    }
    if model_id == "lgbm_weighted_residual_v1":
        baseline_train = X_train["weighted_weekday_formula_v1"].to_numpy(dtype=float)
        y_fit = y_train - baseline_train
        objective = "regression_l1"
        alpha = None
    else:
        match = QUANTILE_PATTERN.fullmatch(model_id)
        if match is None:
            raise ValueError(f"Invalid quantile model ID: {model_id}")
        alpha = int(match.group(1)) / 100.0
        if not 0.0 < alpha < 1.0:
            raise ValueError("Quantile alpha must be in (0, 1)")
        objective = "quantile"
        y_fit = y_train

    params = {**base_params, "objective": objective}
    if alpha is not None:
        params["alpha"] = alpha
    model = LGBMRegressor(**params)
    model.fit(X_train, y_fit)
    prediction = model.predict(X_eval)
    if model_id == "lgbm_weighted_residual_v1":
        prediction += X_eval["weighted_weekday_formula_v1"].to_numpy(dtype=float)
    prediction = np.maximum(prediction, 0.0)

    result = pd.Series(np.nan, index=panel.index, dtype=float)
    result.loc[evaluation_mask] = prediction
    metadata: dict[str, object] = {
        "model_id": model_id,
        "model_class": "freshly_trained_on_panel",
        "objective": objective,
        "alpha": alpha,
        "train_end": str(train_end.date()),
        "train_rows": int(train_mask.sum()),
        "evaluation_rows": int(evaluation_mask.sum()),
        "feature_names": list(features.columns),
        "n_estimators": n_estimators,
        "history_rule": "target date minus at least one day",
    }
    return result, metadata

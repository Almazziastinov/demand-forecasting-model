"""
Experiment D2: Two-Stage with Soft Routing
Instead of hard argmax routing, use classifier probabilities
to weighted-average predictions from all 3 per-bin regressors.

pred = p_low * regressor_low(x) + p_med * regressor_med(x) + p_high * regressor_high(x)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import mean_absolute_error

from src.experiments.common import (
    load_data, MODEL_PARAMS, BASELINE_MAE,
    wmape, print_metrics, print_category_metrics,
    train_lgbm, predict_clipped, save_predictions,
)


BINS = {"low": (0, 2), "medium": (3, 10), "high": (11, None)}
BIN_LABELS = list(BINS.keys())


def assign_bin(y):
    y = np.asarray(y)
    bins = np.full(len(y), "medium", dtype=object)
    bins[y <= 2] = "low"
    bins[(y >= 3) & (y <= 10)] = "medium"
    bins[y >= 11] = "high"
    return bins


def main():
    print("=" * 65)
    print("  EXPERIMENT D2: SOFT ROUTING (weighted by classifier proba)")
    print("=" * 65)

    df, X_train, y_train, X_test, y_test, features = load_data()

    # --- Baseline ---
    print("\n--- Baseline (global model, v6-best) ---")
    bl_model = train_lgbm(X_train, y_train)
    bl_pred = predict_clipped(bl_model, X_test)
    bl_mae, _, _ = print_metrics("Baseline", y_test, bl_pred)

    # --- Stage 1: Classifier ---
    print("\n--- Stage 1: Classifier ---")
    y_train_bins = assign_bin(y_train)

    clf_params = {
        "n_estimators": 1000,
        "learning_rate": 0.02,
        "num_leaves": 63,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    clf = LGBMClassifier(**clf_params)
    clf.fit(X_train, y_train_bins)

    # Probabilities on test
    proba = clf.predict_proba(X_test)  # shape (n, 3)
    classes = clf.classes_  # e.g. ['high', 'low', 'medium']
    print(f"  Classifier classes: {classes.tolist()}")

    # Map class names to column indices
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"  Mean probabilities on test:")
    for b in BIN_LABELS:
        idx = class_to_idx[b]
        print(f"    p_{b}: mean={proba[:, idx].mean():.3f}  "
              f"min={proba[:, idx].min():.3f}  max={proba[:, idx].max():.3f}")

    # --- Stage 2: Per-bin regressors ---
    print("\n--- Stage 2: Per-bin regressors ---")
    regressors = {}
    test_preds = {}

    for b in BIN_LABELS:
        train_mask = y_train_bins == b
        X_tr_bin = X_train[train_mask]
        y_tr_bin = y_train[train_mask]

        model = train_lgbm(X_tr_bin, y_tr_bin)
        pred = predict_clipped(model, X_test)

        regressors[b] = model
        test_preds[b] = pred

        mae_all = mean_absolute_error(y_test, pred)
        print(f"  regressor_{b}: trained on {len(X_tr_bin):,} samples, "
              f"MAE on full test = {mae_all:.4f}")

    # --- Soft routing ---
    print("\n--- Soft routing ---")
    y_pred_soft = np.zeros(len(y_test))
    for b in BIN_LABELS:
        idx = class_to_idx[b]
        y_pred_soft += proba[:, idx] * test_preds[b]
    y_pred_soft = np.maximum(y_pred_soft, 0)

    # --- Hard routing (for comparison) ---
    pred_bins = clf.predict(X_test)
    y_pred_hard = np.zeros(len(y_test))
    for b in BIN_LABELS:
        mask = pred_bins == b
        y_pred_hard[mask] = test_preds[b][mask]

    # --- Oracle ---
    y_test_bins = assign_bin(y_test)
    y_pred_oracle = np.zeros(len(y_test))
    for b in BIN_LABELS:
        mask = y_test_bins == b
        y_pred_oracle[mask] = test_preds[b][mask]

    # --- Metrics ---
    print("\n--- Itogo ---")
    soft_mae, soft_wmape, soft_bias = print_metrics(
        "Soft Routing", y_test, y_pred_soft
    )
    hard_mae, _, _ = print_metrics("Hard Routing", y_test, y_pred_hard)
    oracle_mae, _, _ = print_metrics("Oracle", y_test, y_pred_oracle)

    print("\n--- Po kategoriyam (soft routing) ---")
    print_category_metrics(y_test, y_pred_soft, X_test["Категория"])

    print(f"\n  Sravnenie:")
    print(f"    Baseline:     {bl_mae:.4f}")
    print(f"    Hard routing: {hard_mae:.4f} ({hard_mae - bl_mae:+.4f})")
    print(f"    Soft routing: {soft_mae:.4f} ({soft_mae - bl_mae:+.4f})")
    print(f"    Oracle:       {oracle_mae:.4f} ({oracle_mae - bl_mae:+.4f})")

    # Save
    save_predictions(
        X_test, y_test, y_pred_soft,
        "reports/exp_d2_predictions.csv",
        extra_cols={
            "pred_baseline": bl_pred,
            "pred_hard": y_pred_hard,
            "pred_oracle": y_pred_oracle,
            "p_low": proba[:, class_to_idx["low"]],
            "p_medium": proba[:, class_to_idx["medium"]],
            "p_high": proba[:, class_to_idx["high"]],
        },
    )

    summary = pd.DataFrame([{
        "experiment": "D2_soft_routing",
        "mae": soft_mae,
        "wmape": soft_wmape,
        "bias": soft_bias,
        "baseline_mae": BASELINE_MAE,
        "delta": soft_mae - BASELINE_MAE,
    }])
    summary.to_csv("reports/exp_d2_summary.csv", index=False)

    print("\nGotovo!")


if __name__ == "__main__":
    main()

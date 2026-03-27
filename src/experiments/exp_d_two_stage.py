"""
Experiment D: Two-Stage Model
Stage 1: Classify demand into bins (low / medium / high)
Stage 2: Separate LightGBM regressors per demand bin
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import mean_absolute_error, classification_report

from src.experiments.common import (
    load_data, MODEL_PARAMS, BASELINE_MAE, FEATURES, TARGET,
    wmape, print_metrics, print_category_metrics,
    train_lgbm, predict_clipped, save_predictions,
)


# Demand bins
BINS = {
    "low": (0, 2),      # 0-2 inclusive
    "medium": (3, 10),   # 3-10 inclusive
    "high": (11, None),  # 11+
}
BIN_LABELS = list(BINS.keys())


def assign_bin(y):
    """Assign demand level bin to each sample."""
    y = np.asarray(y)
    bins = np.full(len(y), "medium", dtype=object)
    bins[y <= 2] = "low"
    bins[(y >= 3) & (y <= 10)] = "medium"
    bins[y >= 11] = "high"
    return bins


def main():
    print("=" * 65)
    print("  EXPERIMENT D: TWO-STAGE (classify + per-bin regress)")
    print("=" * 65)

    df, X_train, y_train, X_test, y_test, features = load_data()

    # --- Baseline ---
    print("\n--- Baseline (global model, v6-best) ---")
    bl_model = train_lgbm(X_train, y_train)
    bl_pred = predict_clipped(bl_model, X_test)
    bl_mae, _, _ = print_metrics("Baseline", y_test, bl_pred)

    # --- Stage 1: Classifier ---
    print("\n--- Stage 1: Demand level classifier ---")
    y_train_bins = assign_bin(y_train)
    y_test_bins = assign_bin(y_test)

    # Distribution
    for b in BIN_LABELS:
        n_tr = (y_train_bins == b).sum()
        n_te = (y_test_bins == b).sum()
        print(f"  {b:>7}: train={n_tr:,} ({n_tr/len(y_train)*100:.1f}%)  "
              f"test={n_te:,} ({n_te/len(y_test)*100:.1f}%)")

    # LightGBM classifier
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

    # Classify test
    pred_bins = clf.predict(X_test)
    print(f"\n  Classifier accuracy:")
    for b in BIN_LABELS:
        true_mask = y_test_bins == b
        pred_mask = pred_bins == b
        if true_mask.sum() > 0:
            acc = (pred_bins[true_mask] == b).sum() / true_mask.sum()
            print(f"    {b:>7}: accuracy={acc:.3f} "
                  f"(true={true_mask.sum()}, pred={pred_mask.sum()})")

    # --- Stage 2: Per-bin regressors ---
    print("\n--- Stage 2: Per-bin regressors ---")
    y_pred_combined = np.zeros(len(y_test))

    for b in BIN_LABELS:
        train_mask = y_train_bins == b
        test_mask = pred_bins == b  # route by classifier prediction

        X_tr_bin = X_train[train_mask]
        y_tr_bin = y_train[train_mask]
        X_te_bin = X_test[test_mask]

        if test_mask.sum() == 0:
            print(f"  [{b}] Net dannykh v teste (po klassifikatoru)")
            continue

        model = train_lgbm(X_tr_bin, y_tr_bin)
        pred = predict_clipped(model, X_te_bin)

        y_te_bin = y_test[test_mask]
        mae_bin = mean_absolute_error(y_te_bin, pred)
        bias_bin = np.mean(y_te_bin.values - pred)

        print(f"  [{b:>7}] train={len(X_tr_bin):,} test={test_mask.sum():,}  "
              f"MAE={mae_bin:.4f} bias={bias_bin:+.3f}")

        y_pred_combined[test_mask] = pred

    # --- Also try: use TRUE bins for routing (oracle) ---
    print("\n--- Oracle routing (true bins) ---")
    y_pred_oracle = np.zeros(len(y_test))

    for b in BIN_LABELS:
        train_mask = y_train_bins == b
        test_mask = y_test_bins == b

        X_tr_bin = X_train[train_mask]
        y_tr_bin = y_train[train_mask]
        X_te_bin = X_test[test_mask]

        if test_mask.sum() == 0:
            continue

        model = train_lgbm(X_tr_bin, y_tr_bin)
        pred = predict_clipped(model, X_te_bin)

        mae_bin = mean_absolute_error(y_test[test_mask], pred)
        print(f"  [{b:>7}] test={test_mask.sum():,}  MAE={mae_bin:.4f}")

        y_pred_oracle[test_mask] = pred

    # --- Metrics ---
    print("\n--- Itogo ---")
    overall_mae, overall_wmape, overall_bias = print_metrics(
        "Two-Stage (classifier routing)", y_test, y_pred_combined
    )
    oracle_mae, oracle_wmape, oracle_bias = print_metrics(
        "Two-Stage (oracle routing)", y_test, y_pred_oracle
    )

    print("\n--- Po kategoriyam (classifier routing) ---")
    print_category_metrics(y_test, y_pred_combined, X_test["Категория"])

    print(f"\n  Baseline MAE:   {bl_mae:.4f}")
    print(f"  Two-Stage MAE:  {overall_mae:.4f} (classifier)")
    print(f"  Two-Stage MAE:  {oracle_mae:.4f} (oracle)")
    print(f"  Delta (clf):    {overall_mae - bl_mae:+.4f}")
    print(f"  Delta (oracle): {oracle_mae - bl_mae:+.4f}")

    # Save
    save_predictions(
        X_test, y_test, y_pred_combined,
        "reports/exp_d_predictions.csv",
        extra_cols={
            "pred_baseline": bl_pred,
            "pred_oracle": y_pred_oracle,
            "pred_bin": pred_bins,
            "true_bin": y_test_bins,
        },
    )

    summary = pd.DataFrame([{
        "experiment": "D_two_stage",
        "mae": overall_mae,
        "wmape": overall_wmape,
        "bias": overall_bias,
        "baseline_mae": BASELINE_MAE,
        "delta": overall_mae - BASELINE_MAE,
    }])
    summary.to_csv("reports/exp_d_summary.csv", index=False)

    print("\nGotovo!")


if __name__ == "__main__":
    main()

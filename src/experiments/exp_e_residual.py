"""
Experiment E: Residual Correction
1. Train global model (v6-best) on full train
2. Compute train residuals (actual - predicted)
3. Train per-category correction models on residuals
4. Final = global_pred + category_correction
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

from src.experiments.common import (
    load_data, MODEL_PARAMS, BASELINE_MAE,
    wmape, print_metrics, print_category_metrics,
    train_lgbm, predict_clipped, save_predictions,
)


def main():
    print("=" * 65)
    print("  EXPERIMENT E: RESIDUAL CORRECTION (global + per-category)")
    print("=" * 65)

    df, X_train, y_train, X_test, y_test, features = load_data()

    # --- Baseline / Global model ---
    print("\n--- Global model (v6-best) ---")
    global_model = train_lgbm(X_train, y_train)
    global_pred_train = predict_clipped(global_model, X_train)
    global_pred_test = predict_clipped(global_model, X_test)

    bl_mae, _, _ = print_metrics("Global (baseline)", y_test, global_pred_test)

    # --- Train residuals ---
    print("\n--- Residualy na train ---")
    residuals_train = y_train.values - global_pred_train

    print(f"  Mean residual:  {residuals_train.mean():+.4f}")
    print(f"  Std residual:   {residuals_train.std():.4f}")
    print(f"  Min residual:   {residuals_train.min():.4f}")
    print(f"  Max residual:   {residuals_train.max():.4f}")

    # Per-category residual stats
    categories_train = X_train["Категория"]
    print(f"\n  Residualy po kategoriyam:")
    for cat in sorted(categories_train.unique()):
        mask = categories_train == cat
        r = residuals_train[mask.values]
        print(f"    {str(cat):<25} mean={r.mean():+.4f} std={r.std():.4f} n={len(r):,}")

    # --- Per-category residual models ---
    print("\n--- Per-category residual models ---")
    # Use lighter params for residual models (residuals are smaller, less complex)
    residual_params = MODEL_PARAMS.copy()
    residual_params["n_estimators"] = 1000
    residual_params["num_leaves"] = 63
    residual_params["max_depth"] = 5
    residual_params["learning_rate"] = 0.02

    categories = X_train["Категория"].cat.categories.tolist()
    y_pred_corrected = global_pred_test.copy()

    for cat in categories:
        train_mask = X_train["Категория"] == cat
        test_mask = X_test["Категория"] == cat

        X_tr_cat = X_train[train_mask]
        r_tr_cat = residuals_train[train_mask.values]  # residuals as target
        X_te_cat = X_test[test_mask]

        if test_mask.sum() == 0:
            print(f"  [{cat}] Net dannykh v teste")
            continue

        # Train residual correction model
        res_model = LGBMRegressor(**residual_params)
        res_model.fit(X_tr_cat, r_tr_cat)

        # Predict correction
        correction = res_model.predict(X_te_cat)

        # Apply: global + correction, clip >= 0
        corrected = np.maximum(global_pred_test[test_mask.values] + correction, 0)

        y_te_cat = y_test[test_mask]
        mae_before = mean_absolute_error(y_te_cat, global_pred_test[test_mask.values])
        mae_after = mean_absolute_error(y_te_cat, corrected)
        delta = mae_after - mae_before

        print(f"  [{cat}] n_test={test_mask.sum():,}  "
              f"MAE: {mae_before:.4f} -> {mae_after:.4f} (delta={delta:+.4f})  "
              f"mean_correction={correction.mean():+.3f}")

        y_pred_corrected[test_mask.values] = corrected

    # --- Also try: simple bias correction per category (no model) ---
    print("\n--- Simple bias correction (mean residual per category) ---")
    y_pred_simple_bias = global_pred_test.copy()

    for cat in categories:
        train_mask = X_train["Категория"] == cat
        test_mask = X_test["Категория"] == cat

        if test_mask.sum() == 0:
            continue

        mean_residual = residuals_train[train_mask.values].mean()
        y_pred_simple_bias[test_mask.values] = np.maximum(
            global_pred_test[test_mask.values] + mean_residual, 0
        )

        y_te_cat = y_test[test_mask]
        mae = mean_absolute_error(y_te_cat, y_pred_simple_bias[test_mask.values])
        print(f"  [{cat}] bias={mean_residual:+.4f}  MAE={mae:.4f}")

    # --- Metrics ---
    print("\n--- Itogo ---")
    overall_mae, overall_wmape, overall_bias = print_metrics(
        "Residual Model Correction", y_test, y_pred_corrected
    )
    simple_mae, simple_wmape, simple_bias = print_metrics(
        "Simple Bias Correction", y_test, y_pred_simple_bias
    )

    print("\n--- Po kategoriyam (residual model) ---")
    print_category_metrics(y_test, y_pred_corrected, X_test["Категория"])

    print(f"\n  Baseline MAE:        {bl_mae:.4f}")
    print(f"  Residual Model MAE:  {overall_mae:.4f} (delta={overall_mae - bl_mae:+.4f})")
    print(f"  Simple Bias MAE:     {simple_mae:.4f} (delta={simple_mae - bl_mae:+.4f})")

    # Save
    save_predictions(
        X_test, y_test, y_pred_corrected,
        "reports/exp_e_predictions.csv",
        extra_cols={
            "pred_baseline": global_pred_test,
            "pred_simple_bias": y_pred_simple_bias,
        },
    )

    summary = pd.DataFrame([{
        "experiment": "E_residual",
        "mae": overall_mae,
        "wmape": overall_wmape,
        "bias": overall_bias,
        "baseline_mae": BASELINE_MAE,
        "delta": overall_mae - BASELINE_MAE,
    }])
    summary.to_csv("reports/exp_e_summary.csv", index=False)

    print("\nGotovo!")


if __name__ == "__main__":
    main()

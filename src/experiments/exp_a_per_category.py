"""
Experiment A: Per-Category Models
Train separate LightGBM per product category (5 models).
Hypothesis: specialized models improve worst categories (e.g. Vypechka sytnaya MAE 5.2).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.experiments.common import (
    load_data, MODEL_PARAMS, BASELINE_MAE,
    wmape, print_metrics, print_category_metrics,
    train_lgbm, predict_clipped, save_predictions,
)


def main():
    print("=" * 65)
    print("  EXPERIMENT A: PER-CATEGORY MODELS (5 otdel'nykh LightGBM)")
    print("=" * 65)

    df, X_train, y_train, X_test, y_test, features = load_data()

    # --- Baseline: single global model ---
    print("\n--- Baseline (global model, v6-best) ---")
    from src.experiments.common import train_lgbm
    bl_model = train_lgbm(X_train, y_train)
    bl_pred = predict_clipped(bl_model, X_test)
    bl_mae, _, _ = print_metrics("Baseline", y_test, bl_pred)

    # --- Per-category models ---
    print("\n--- Per-category models ---")
    categories = X_train["Категория"].cat.categories.tolist()
    print(f"  Kategorij: {len(categories)}")

    y_pred_combined = np.zeros(len(y_test))
    cat_results = []

    for cat in categories:
        # Split by category
        train_mask = X_train["Категория"] == cat
        test_mask = X_test["Категория"] == cat

        X_tr_cat = X_train[train_mask]
        y_tr_cat = y_train[train_mask]
        X_te_cat = X_test[test_mask]
        y_te_cat = y_test[test_mask]

        if len(X_te_cat) == 0:
            print(f"  [{cat}] Net dannykh v teste, propuskayem")
            continue

        # Train per-category model
        model = train_lgbm(X_tr_cat, y_tr_cat)
        pred = predict_clipped(model, X_te_cat)

        mae_cat = mean_absolute_error(y_te_cat, pred)
        bias_cat = np.mean(y_te_cat.values - pred)
        bl_pred_cat = bl_pred[test_mask.values]
        bl_mae_cat = mean_absolute_error(y_te_cat, bl_pred_cat)
        delta = mae_cat - bl_mae_cat

        print(f"  [{cat}] train={len(X_tr_cat):,} test={len(X_te_cat):,}  "
              f"MAE={mae_cat:.4f} (baseline={bl_mae_cat:.4f}, delta={delta:+.4f})  "
              f"bias={bias_cat:+.3f}")

        y_pred_combined[test_mask.values] = pred
        cat_results.append({
            "Категория": cat,
            "n_train": len(X_tr_cat),
            "n_test": len(X_te_cat),
            "mae_per_cat": mae_cat,
            "mae_baseline": bl_mae_cat,
            "delta": delta,
        })

    # --- Overall metrics ---
    print("\n--- Itogo (combined) ---")
    overall_mae, overall_wmape, overall_bias = print_metrics(
        "Per-Category Combined", y_test, y_pred_combined
    )

    print("\n--- Po kategoriyam ---")
    print_category_metrics(y_test, y_pred_combined, X_test["Категория"])

    print(f"\n  Baseline MAE:     {bl_mae:.4f}")
    print(f"  Per-Category MAE: {overall_mae:.4f}")
    print(f"  Delta:            {overall_mae - bl_mae:+.4f}")
    if overall_mae < bl_mae:
        print(f"  -> Per-category LUCHSHE na {bl_mae - overall_mae:.4f}")
    else:
        print(f"  -> Baseline luchshe na {overall_mae - bl_mae:.4f}")

    # Save
    save_predictions(
        X_test, y_test, y_pred_combined,
        "reports/exp_a_predictions.csv",
        extra_cols={"pred_baseline": bl_pred},
    )

    # Save summary
    summary = pd.DataFrame([{
        "experiment": "A_per_category",
        "mae": overall_mae,
        "wmape": overall_wmape,
        "bias": overall_bias,
        "baseline_mae": BASELINE_MAE,
        "delta": overall_mae - BASELINE_MAE,
    }])
    summary.to_csv("reports/exp_a_summary.csv", index=False)

    print("\nGotovo!")


if __name__ == "__main__":
    main()

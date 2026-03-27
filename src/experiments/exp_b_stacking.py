"""
Experiment B: Stacking (LightGBM + Ridge)
L1 base models (3-fold temporal split for meta-features):
  - LightGBM with v6-best params
  - LightGBM with simpler params (num_leaves=31)
  - Ridge regression (numeric features only)
L2 meta-model: Ridge on L1 out-of-fold predictions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from src.experiments.common import (
    load_data, MODEL_PARAMS, BASELINE_MAE, CATEGORICAL_COLS,
    wmape, print_metrics, print_category_metrics,
    train_lgbm, predict_clipped, save_predictions,
)


def get_numeric_features(X, features):
    """Return only numeric (non-categorical) feature names."""
    cat_cols = set(CATEGORICAL_COLS)
    return [f for f in features if f not in cat_cols and f in X.columns]


def temporal_3fold(df_full, target_col="Продано"):
    """Create 3 temporal folds from training data.
    Returns list of (train_idx, val_idx) tuples.
    """
    dates = df_full["Дата"].sort_values().unique()
    n = len(dates)
    fold_size = n // 4  # ~25% for each validation fold

    folds = []
    for i in range(3):
        val_start = dates[fold_size * (i + 1)]
        val_end = dates[min(fold_size * (i + 2), n) - 1]
        train_mask = df_full["Дата"] < val_start
        val_mask = (df_full["Дата"] >= val_start) & (df_full["Дата"] <= val_end)
        folds.append((
            df_full.index[train_mask].tolist(),
            df_full.index[val_mask].tolist(),
        ))
        print(f"    Fold {i+1}: train={train_mask.sum():,} val={val_mask.sum():,} "
              f"(val: {val_start.date()}..{val_end.date()})")

    return folds


def main():
    print("=" * 65)
    print("  EXPERIMENT B: STACKING (LightGBM + Ridge)")
    print("=" * 65)

    df, X_train, y_train, X_test, y_test, features = load_data()

    # --- Baseline ---
    print("\n--- Baseline (global model, v6-best) ---")
    bl_model = train_lgbm(X_train, y_train)
    bl_pred = predict_clipped(bl_model, X_test)
    bl_mae, _, _ = print_metrics("Baseline", y_test, bl_pred)

    # --- Prepare folds ---
    print("\n--- Temporal 3-fold split ---")
    train_df = df[df["Дата"] < (df["Дата"].max() - pd.Timedelta(days=2))].copy()
    train_df = train_df.reset_index(drop=True)
    folds = temporal_3fold(train_df)

    numeric_features = get_numeric_features(X_train, features)
    print(f"  Numeric features dlia Ridge: {len(numeric_features)}")

    # --- L1: Generate out-of-fold predictions ---
    print("\n--- L1: Out-of-fold predictions ---")
    oof_lgbm_full = np.zeros(len(train_df))
    oof_lgbm_simple = np.zeros(len(train_df))
    oof_ridge = np.zeros(len(train_df))

    test_preds_lgbm_full = np.zeros(len(X_test))
    test_preds_lgbm_simple = np.zeros(len(X_test))
    test_preds_ridge = np.zeros(len(X_test))

    simple_params = MODEL_PARAMS.copy()
    simple_params["num_leaves"] = 31
    simple_params["max_depth"] = 5
    simple_params["n_estimators"] = 1500

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        print(f"\n  Fold {fold_i + 1}:")
        X_tr_fold = train_df.loc[train_idx, features]
        y_tr_fold = train_df.loc[train_idx, "Продано"]
        X_val_fold = train_df.loc[val_idx, features]
        y_val_fold = train_df.loc[val_idx, "Продано"]

        # L1-1: LightGBM full
        m1 = LGBMRegressor(**MODEL_PARAMS)
        m1.fit(X_tr_fold, y_tr_fold)
        oof_lgbm_full[val_idx] = np.maximum(m1.predict(X_val_fold), 0)
        test_preds_lgbm_full += np.maximum(m1.predict(X_test), 0) / 3
        mae1 = mean_absolute_error(y_val_fold, oof_lgbm_full[val_idx])
        print(f"    LightGBM full:   val MAE = {mae1:.4f}")

        # L1-2: LightGBM simple
        m2 = LGBMRegressor(**simple_params)
        m2.fit(X_tr_fold, y_tr_fold)
        oof_lgbm_simple[val_idx] = np.maximum(m2.predict(X_val_fold), 0)
        test_preds_lgbm_simple += np.maximum(m2.predict(X_test), 0) / 3
        mae2 = mean_absolute_error(y_val_fold, oof_lgbm_simple[val_idx])
        print(f"    LightGBM simple: val MAE = {mae2:.4f}")

        # L1-3: Ridge (numeric only)
        scaler = StandardScaler()
        X_tr_num = scaler.fit_transform(X_tr_fold[numeric_features].fillna(0))
        X_val_num = scaler.transform(X_val_fold[numeric_features].fillna(0))
        X_te_num = scaler.transform(X_test[numeric_features].fillna(0))

        m3 = Ridge(alpha=1.0)
        m3.fit(X_tr_num, y_tr_fold)
        oof_ridge[val_idx] = np.maximum(m3.predict(X_val_num), 0)
        test_preds_ridge += np.maximum(m3.predict(X_te_num), 0) / 3
        mae3 = mean_absolute_error(y_val_fold, oof_ridge[val_idx])
        print(f"    Ridge:           val MAE = {mae3:.4f}")

    # --- L2: Meta-model ---
    print("\n--- L2: Meta-model (Ridge) ---")
    # Only use rows that were in validation folds
    val_mask = (oof_lgbm_full != 0) | (oof_lgbm_simple != 0) | (oof_ridge != 0)
    meta_train = np.column_stack([
        oof_lgbm_full[val_mask],
        oof_lgbm_simple[val_mask],
        oof_ridge[val_mask],
    ])
    meta_y = train_df.loc[val_mask, "Продано"].values

    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_train, meta_y)
    print(f"  L2 koefficienty: {meta_model.coef_.round(4)}")
    print(f"  L2 intercept:    {meta_model.intercept_:.4f}")

    # --- Final test predictions ---
    meta_test = np.column_stack([
        test_preds_lgbm_full,
        test_preds_lgbm_simple,
        test_preds_ridge,
    ])
    y_pred_stacking = np.maximum(meta_model.predict(meta_test), 0)

    # --- Metrics ---
    print("\n--- Itogo ---")
    overall_mae, overall_wmape, overall_bias = print_metrics(
        "Stacking (LGB+LGB_simple+Ridge)", y_test, y_pred_stacking
    )

    # Individual L1 model metrics on test
    print("\n  Otdel'nye L1 modeli na teste:")
    for name, pred in [
        ("LightGBM full", test_preds_lgbm_full),
        ("LightGBM simple", test_preds_lgbm_simple),
        ("Ridge", test_preds_ridge),
    ]:
        mae = mean_absolute_error(y_test, pred)
        print(f"    {name:<20} MAE = {mae:.4f}")

    print("\n--- Po kategoriyam ---")
    print_category_metrics(y_test, y_pred_stacking, X_test["Категория"])

    print(f"\n  Baseline MAE: {bl_mae:.4f}")
    print(f"  Stacking MAE: {overall_mae:.4f}")
    print(f"  Delta:        {overall_mae - bl_mae:+.4f}")

    # Save
    save_predictions(
        X_test, y_test, y_pred_stacking,
        "reports/exp_b_predictions.csv",
        extra_cols={
            "pred_baseline": bl_pred,
            "pred_lgbm_full": test_preds_lgbm_full,
            "pred_lgbm_simple": test_preds_lgbm_simple,
            "pred_ridge": test_preds_ridge,
        },
    )

    summary = pd.DataFrame([{
        "experiment": "B_stacking",
        "mae": overall_mae,
        "wmape": overall_wmape,
        "bias": overall_bias,
        "baseline_mae": BASELINE_MAE,
        "delta": overall_mae - BASELINE_MAE,
    }])
    summary.to_csv("reports/exp_b_summary.csv", index=False)

    print("\nGotovo!")


if __name__ == "__main__":
    main()

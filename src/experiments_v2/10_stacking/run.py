"""
Experiment 10: Stacking Ensemble (demand target).

Level 1 base models (3-fold OOF):
  - LightGBM Quantile P50 (current best)
  - CatBoost Quantile P50 (native categorical handling)
  - Ridge (linear baseline)

Level 2 meta-learner:
  - Ridge on 3 OOF predictions -> final prediction

Compared vs Baseline V3 (exp 60: MAE 2.8816, demand+price+P50).

Usage:
  .venv/Scripts/python.exe src/experiments_v2/10_stacking/run.py
"""

import sys
import os
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from src.config import MODEL_PARAMS, TEST_DAYS
from src.experiments_v2.common import (
    DEMAND_8M_PATH, FEATURES_V3, CATEGORICAL_COLS_V2, DEMAND_TARGET,
    wmape, print_metrics, print_category_metrics,
    predict_clipped, save_results,
)

EXP_DIR = Path(__file__).resolve().parent
BASELINE_V3_MAE = 2.8816

N_FOLDS = 3


def get_lgbm_params():
    p = MODEL_PARAMS.copy()
    p["objective"] = "quantile"
    p["alpha"] = 0.5
    p.pop("metric", None)
    return p


def main():
    print("=" * 60)
    print("  EXPERIMENT 10: Stacking Ensemble (demand target)")
    print("  Level 1: LightGBM P50 + CatBoost P50 + Ridge")
    print("  Level 2: Ridge meta-learner")
    print("  Baseline V3: MAE 2.8816")
    print("=" * 60)
    t_start = time.time()

    # --- Load ---
    print(f"\n[1/5] Loading data...")
    if not DEMAND_8M_PATH.exists():
        print(f"  ERROR: {DEMAND_8M_PATH} not found!")
        return
    df = pd.read_csv(str(DEMAND_8M_PATH), encoding="utf-8-sig")
    df["Дата"] = pd.to_datetime(df["Дата"])
    print(f"  Shape: {df.shape}, Days: {df['Дата'].nunique()}, "
          f"Bakeries: {df['Пекарня'].nunique()}")

    # --- Features ---
    print(f"\n[2/5] Preparing features...")
    available = [f for f in FEATURES_V3 if f in df.columns]
    print(f"  Using {len(available)} features")

    # LightGBM uses category dtype
    for col in CATEGORICAL_COLS_V2:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Categorical column names for CatBoost
    cat_col_names = [c for c in CATEGORICAL_COLS_V2 if c in available]

    # --- Split ---
    print(f"\n[3/5] Train/test split (last {TEST_DAYS} days)...")
    test_start = df["Дата"].max() - pd.Timedelta(days=TEST_DAYS - 1)
    train = df[df["Дата"] < test_start].copy()
    test = df[df["Дата"] >= test_start].copy()
    print(f"  Train: {len(train):,} rows, Test: {len(test):,} rows")

    X_train = train[available].copy()
    y_train = train[DEMAND_TARGET]
    X_test = test[available].copy()
    y_test = test[DEMAND_TARGET].values

    # Prepare CatBoost-compatible copy (string categories, no NaN in cats)
    X_train_cb = X_train.copy()
    X_test_cb = X_test.copy()
    for col in cat_col_names:
        X_train_cb[col] = X_train_cb[col].astype(str)
        X_test_cb[col] = X_test_cb[col].astype(str)

    # Prepare Ridge-compatible copy (numeric only)
    X_train_ridge = X_train.select_dtypes(include=[np.number]).fillna(0)
    X_test_ridge = X_test.select_dtypes(include=[np.number]).fillna(0)

    # --- OOF Training ---
    print(f"\n[4/5] Training {N_FOLDS}-fold OOF...")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    lgbm_params = get_lgbm_params()

    oof_lgbm = np.zeros(len(X_train))
    oof_cb = np.zeros(len(X_train))
    oof_ridge = np.zeros(len(X_train))
    test_lgbm = np.zeros(len(X_test))
    test_cb = np.zeros(len(X_test))
    test_ridge = np.zeros(len(X_test))

    t_oof = time.time()

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\n    Fold {fold + 1}/{N_FOLDS} ({len(tr_idx):,} train, {len(val_idx):,} val)...",
              flush=True)

        y_tr = y_train.iloc[tr_idx]
        y_val = y_train.iloc[val_idx]

        # --- LightGBM ---
        print(f"      LightGBM...", end=" ", flush=True)
        t = time.time()
        lgbm = LGBMRegressor(**lgbm_params)
        lgbm.fit(X_train.iloc[tr_idx], y_tr)
        oof_lgbm[val_idx] = np.maximum(lgbm.predict(X_train.iloc[val_idx]), 0)
        test_lgbm += np.maximum(lgbm.predict(X_test), 0) / N_FOLDS
        mae_l = mean_absolute_error(y_val, oof_lgbm[val_idx])
        print(f"MAE={mae_l:.4f} ({time.time()-t:.0f}s)", flush=True)

        # --- CatBoost ---
        print(f"      CatBoost...", end=" ", flush=True)
        t = time.time()
        cb = CatBoostRegressor(
            iterations=500,
            learning_rate=0.08,
            depth=7,
            loss_function="Quantile:alpha=0.5",
            verbose=0,
            random_seed=42,
            thread_count=-1,
        )
        cb.fit(X_train_cb.iloc[tr_idx], y_tr, cat_features=cat_col_names)
        oof_cb[val_idx] = np.maximum(cb.predict(X_train_cb.iloc[val_idx]), 0)
        test_cb += np.maximum(cb.predict(X_test_cb), 0) / N_FOLDS
        mae_c = mean_absolute_error(y_val, oof_cb[val_idx])
        print(f"MAE={mae_c:.4f} ({time.time()-t:.0f}s)", flush=True)

        # --- Ridge ---
        print(f"      Ridge...", end=" ", flush=True)
        t = time.time()
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_ridge.iloc[tr_idx], y_tr)
        oof_ridge[val_idx] = np.maximum(ridge.predict(X_train_ridge.iloc[val_idx]), 0)
        test_ridge += np.maximum(ridge.predict(X_test_ridge), 0) / N_FOLDS
        mae_r = mean_absolute_error(y_val, oof_ridge[val_idx])
        print(f"MAE={mae_r:.4f} ({time.time()-t:.0f}s)", flush=True)

    time_oof = time.time() - t_oof
    print(f"\n  Total OOF time: {time_oof:.0f}s")

    # OOF metrics
    print(f"\n  OOF metrics (full train):")
    for name, oof in [("LightGBM", oof_lgbm), ("CatBoost", oof_cb), ("Ridge", oof_ridge)]:
        mae = mean_absolute_error(y_train, oof)
        print(f"    {name:<15} MAE={mae:.4f}")

    # --- Level 2: Meta-learner ---
    print(f"\n[5/5] Training meta-learner (Ridge)...")
    meta_X_train = np.column_stack([oof_lgbm, oof_cb, oof_ridge])
    meta_X_test = np.column_stack([test_lgbm, test_cb, test_ridge])

    meta = Ridge(alpha=1.0)
    meta.fit(meta_X_train, y_train)
    pred_stack = np.maximum(meta.predict(meta_X_test), 0)

    print(f"  Meta weights: LightGBM={meta.coef_[0]:.3f}, "
          f"CatBoost={meta.coef_[1]:.3f}, Ridge={meta.coef_[2]:.3f}, "
          f"intercept={meta.intercept_:.3f}")

    # --- Evaluation ---
    print(f"\n  === RESULTS (vs {DEMAND_TARGET}) ===")

    results = {}
    for name, pred in [
        ("LightGBM P50", test_lgbm),
        ("CatBoost P50", test_cb),
        ("Ridge", test_ridge),
        ("Stacking", pred_stack),
    ]:
        mae = mean_absolute_error(y_test, pred)
        wm = wmape(y_test, pred)
        bias = np.mean(y_test - pred)
        delta = mae - BASELINE_V3_MAE
        print(f"\n  {name}:")
        print(f"    MAE={mae:.4f} (delta vs V3: {delta:+.4f}), "
              f"WMAPE={wm:.2f}%, Bias={bias:+.4f}")
        results[name] = {"mae": mae, "wmape": wm, "bias": bias}

    best_name = min(results, key=lambda k: results[k]["mae"])
    print(f"\n  Winner: {best_name} (MAE {results[best_name]['mae']:.4f})")

    # High demand
    print(f"\n  === High demand analysis ===")
    for threshold in [15, 50, 100]:
        mask = y_test >= threshold
        if mask.sum() > 0:
            mae_s = mean_absolute_error(y_test[mask], pred_stack[mask])
            mae_l = mean_absolute_error(y_test[mask], test_lgbm[mask])
            print(f"    >= {threshold}: N={mask.sum()}, "
                  f"Stack MAE={mae_s:.2f}, LightGBM MAE={mae_l:.2f} "
                  f"(delta {mae_s - mae_l:+.2f})")

    # Per-category
    print(f"\n  === Per-category metrics (Stacking) ===")
    print_category_metrics(y_test, pred_stack, test["Категория"].values)

    # --- Save ---
    print(f"\n  Saving results...")
    mae_stack = results["Stacking"]["mae"]
    wm_stack = results["Stacking"]["wmape"]
    rmse_stack = np.sqrt(mean_squared_error(y_test, pred_stack))
    bias_stack = results["Stacking"]["bias"]
    r2_stack = r2_score(y_test, pred_stack)

    metrics = {
        "experiment": "10_stacking",
        "target": DEMAND_TARGET,
        "method": "Stacking (LightGBM P50 + CatBoost P50 + Ridge -> Ridge meta)",
        "n_folds": N_FOLDS,
        "mae": round(mae_stack, 4),
        "wmape": round(wm_stack, 2),
        "rmse": round(rmse_stack, 4),
        "bias": round(bias_stack, 4),
        "r2": round(r2_stack, 4),
        "delta_vs_v3": round(mae_stack - BASELINE_V3_MAE, 4),
        "base_model_results": {
            name: {k: round(v, 4) for k, v in r.items()}
            for name, r in results.items()
        },
        "meta_weights": {
            "lgbm": round(meta.coef_[0], 4),
            "catboost": round(meta.coef_[1], 4),
            "ridge": round(meta.coef_[2], 4),
            "intercept": round(meta.intercept_, 4),
        },
        "train_rows": len(train),
        "test_rows": len(test),
        "n_features": len(available),
        "baseline_v3_mae": BASELINE_V3_MAE,
        "train_time_oof_s": round(time_oof, 1),
    }

    predictions = pd.DataFrame({
        "Дата": test["Дата"].values,
        "Пекарня": test["Пекарня"].values,
        "Номенклатура": test["Номенклатура"].values,
        "Категория": test["Категория"].values,
        "fact_demand": y_test,
        "pred_lgbm": np.round(test_lgbm, 2),
        "pred_catboost": np.round(test_cb, 2),
        "pred_ridge": np.round(test_ridge, 2),
        "pred_stacking": np.round(pred_stack, 2),
        "abs_err_stacking": np.round(np.abs(y_test - pred_stack), 2),
    })

    save_results(EXP_DIR, metrics, predictions)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s")
    print("  Done!")


if __name__ == "__main__":
    main()

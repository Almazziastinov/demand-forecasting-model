"""
Experiment 46: Variance-weighted training.

sample_weight = f(product_std). Two variants:
  a) weight proportional to std  -- focus on high demand (high variance)
  b) weight proportional to 1/std -- focus on predictable products (low variance)

We run both and compare.

Usage:
  .venv/Scripts/python.exe src/experiments_v2/46_variance_weighted/run.py
"""

import sys, os, time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import TARGET, MODEL_PARAMS, TEST_DAYS
from src.experiments_v2.common import (
    DAILY_8M_PATH, FEATURES_V2, CATEGORICAL_COLS_V2,
    wmape, print_metrics, print_category_metrics,
    predict_clipped, save_results,
)

EXP_DIR = Path(__file__).resolve().parent
BASELINE_MAE = 2.2904
BASELINE_WMAPE = 25.91


def main():
    print("=" * 60)
    print("  EXPERIMENT 46: Variance-weighted training")
    print("=" * 60)
    t_start = time.time()

    # Load
    df = pd.read_csv(str(DAILY_8M_PATH), encoding="utf-8-sig")
    df["Дата"] = pd.to_datetime(df["Дата"])
    print(f"  Shape: {df.shape}, Days: {df['Дата'].nunique()}")

    available = [f for f in FEATURES_V2 if f in df.columns]
    for col in CATEGORICAL_COLS_V2:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Split
    test_start = df["Дата"].max() - pd.Timedelta(days=TEST_DAYS - 1)
    train = df[df["Дата"] < test_start].copy()
    test = df[df["Дата"] >= test_start].copy()
    print(f"  Train: {len(train):,}, Test: {len(test):,}")

    # Compute product std from train
    product_std = train.groupby("Номенклатура")[TARGET].std().fillna(1.0)
    train["product_std"] = train["Номенклатура"].map(product_std).fillna(1.0)
    test["product_std"] = test["Номенклатура"].map(product_std).fillna(1.0)

    X_train, y_train = train[available], train[TARGET]
    X_test, y_test = test[available], test[TARGET]

    results = {}
    total_train_time = 0

    # --- Variant A: weight proportional to std (focus on high demand) ---
    print(f"\n  --- Variant A: weight ~ std (focus on hard cases) ---")
    w_a = train["product_std"].values
    w_a = w_a / w_a.mean()  # normalize to mean=1
    print(f"  Weight stats: mean={w_a.mean():.2f}, min={w_a.min():.4f}, max={w_a.max():.2f}")

    t_train = time.time()
    model_a = LGBMRegressor(**MODEL_PARAMS)
    model_a.fit(X_train, y_train, sample_weight=w_a)
    tt = time.time() - t_train
    total_train_time += tt
    print(f"  Training time: {tt:.0f}s")

    y_pred_a = predict_clipped(model_a, X_test)
    mae_a = mean_absolute_error(y_test, y_pred_a)
    wm_a = wmape(y_test, y_pred_a)
    bias_a = np.mean(np.asarray(y_test) - y_pred_a)
    print(f"  MAE={mae_a:.4f}, WMAPE={wm_a:.2f}%, Bias={bias_a:+.4f}")
    results["A_std"] = {"mae": mae_a, "wmape": wm_a, "bias": bias_a}

    # --- Variant B: weight proportional to 1/std (focus on predictable) ---
    print(f"\n  --- Variant B: weight ~ 1/std (focus on easy wins) ---")
    w_b = 1.0 / (train["product_std"].values + 0.1)
    w_b = w_b / w_b.mean()  # normalize
    print(f"  Weight stats: mean={w_b.mean():.2f}, min={w_b.min():.4f}, max={w_b.max():.2f}")

    t_train = time.time()
    model_b = LGBMRegressor(**MODEL_PARAMS)
    model_b.fit(X_train, y_train, sample_weight=w_b)
    tt = time.time() - t_train
    total_train_time += tt
    print(f"  Training time: {tt:.0f}s")

    y_pred_b = predict_clipped(model_b, X_test)
    mae_b = mean_absolute_error(y_test, y_pred_b)
    wm_b = wmape(y_test, y_pred_b)
    bias_b = np.mean(np.asarray(y_test) - y_pred_b)
    print(f"  MAE={mae_b:.4f}, WMAPE={wm_b:.2f}%, Bias={bias_b:+.4f}")
    results["B_inv_std"] = {"mae": mae_b, "wmape": wm_b, "bias": bias_b}

    # --- Best variant ---
    best_name = "A_std" if mae_a <= mae_b else "B_inv_std"
    best_pred = y_pred_a if mae_a <= mae_b else y_pred_b
    best = results[best_name]
    print(f"\n  === BEST VARIANT: {best_name} ===")
    print(f"    MAE={best['mae']:.4f}, WMAPE={best['wmape']:.2f}%, Bias={best['bias']:+.4f}")

    mae = best["mae"]
    wm = best["wmape"]
    rmse = np.sqrt(mean_squared_error(y_test, best_pred))
    bias = np.mean(np.asarray(y_test) - best_pred)
    r2 = r2_score(y_test, best_pred)

    print(f"    RMSE = {rmse:.4f}, R2 = {r2:.4f}")
    print(f"    Delta vs baseline: MAE {mae - BASELINE_MAE:+.4f}, WMAPE {wm - BASELINE_WMAPE:+.2f}%")

    # High demand
    y_t = np.asarray(y_test, dtype=float)
    y_p = np.asarray(best_pred, dtype=float)
    for threshold in [15, 50, 100]:
        mask = y_t >= threshold
        if mask.sum() > 0:
            mae_h = mean_absolute_error(y_t[mask], y_p[mask])
            bias_h = np.mean(y_t[mask] - y_p[mask])
            print(f"    Demand >= {threshold}: N={mask.sum()}, MAE={mae_h:.2f}, Bias={bias_h:+.2f}")

    # Per-category
    print(f"\n  === Per-category metrics (best: {best_name}) ===")
    print_category_metrics(y_test, best_pred, test["Категория"].values)

    # Save
    metrics = {
        "experiment": "46_variance_weighted",
        "best_variant": best_name,
        "mae_A_std": round(mae_a, 4), "wmape_A_std": round(wm_a, 2),
        "mae_B_inv_std": round(mae_b, 4), "wmape_B_inv_std": round(wm_b, 2),
        "mae": round(mae, 4), "wmape": round(wm, 2),
        "rmse": round(rmse, 4), "bias": round(bias, 4), "r2": round(r2, 4),
        "train_rows": len(train), "test_rows": len(test),
        "train_time_s": round(total_train_time, 1),
        "baseline_01_mae": BASELINE_MAE, "baseline_01_wmape": BASELINE_WMAPE,
    }
    predictions = pd.DataFrame({
        "Дата": test["Дата"].values, "Пекарня": test["Пекарня"].values,
        "Номенклатура": test["Номенклатура"].values, "Категория": test["Категория"].values,
        "fact": y_test.values, "pred": np.round(best_pred, 2),
        "pred_A_std": np.round(y_pred_a, 2), "pred_B_inv_std": np.round(y_pred_b, 2),
        "abs_error": np.round(np.abs(y_test.values - best_pred), 2),
    })
    save_results(EXP_DIR, metrics, predictions)

    print(f"\n  Total time: {time.time() - t_start:.0f}s")
    print("  Done!")


if __name__ == "__main__":
    main()

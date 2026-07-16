"""
Experiment 68: Two approaches to fix lag contamination on post-bad-weather days.

Problem confirmed in exp 67:
  - Days after bad weather: MAE=3.0009, Bias=-1.22  (N=15850, 12.7% of test rows)
  - Normal days:             MAE=2.8640, Bias=-0.68
  - Global MAE improvement was ~0 because affected days are rare in test period (April)
  - But in June/July with real storms, effect is material (Sibirsky Trakt 25: -15.8% bias on D+1)

Two approaches tested here:

  VARIANT A — Post-processing correction:
    Train base model (V3 features).
    On training data, compute per-bakery correction factor for post-bad-weather days:
      correction[bakery] = mean(actual / pred) where is_bad_weather_lag1=1
    At prediction time, multiply forecast by correction factor when is_bad_weather_lag1=1.

  VARIANT B — Data augmentation (lag cleaning):
    For training rows where is_bad_weather_lag1=1, create augmented copies where
    lag features are replaced by "clean" estimates (bakery DOW-mean):
      lag1_aug = bakery_sales_dow_mean  (the DOW-expected value, not weather-depressed actual)
      demand_lag1_aug = same
    This teaches the model: "when yesterday was bad weather, ignore the depressed lag1"
    Model is trained on original + augmented rows combined.

Usage:
  .venv/Scripts/python.exe src/experiments_v2/68_weather_correction/run.py
"""

import sys
import os
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.experiments_v2.common import (
    DEMAND_8M_PATH, FEATURES_V3, CATEGORICAL_COLS_V2, DEMAND_TARGET,
    wmape, print_category_metrics,
    train_quantile, train_lgbm, predict_clipped, save_results,
)
from src.config import TEST_DAYS

EXP_DIR = Path(__file__).resolve().parent

BASELINE_V3_MAE = 2.8816  # exp 60
EXP67_MAE       = 2.8814  # exp 67 (weather-lag, minimal gain)

# Lag-related features that get contaminated by bad weather
LAG_FEATURES = [
    "sales_lag1", "sales_lag2",
    "demand_lag1", "demand_lag2",
    "sales_roll_mean3",
    "demand_roll_mean3",
]


def build_weather_lag_col(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_bad_weather_lag1 column (needed for both variants)."""
    df = df.copy()
    df["Дата"] = pd.to_datetime(df["Дата"])
    df = df.sort_values(["Пекарня", "Дата"])
    df["is_bad_weather_lag1"] = (
        df.groupby("Пекарня")["is_bad_weather"].shift(1).fillna(0).astype(int)
    )
    return df


def evaluate(y_true, y_pred, label, is_bad_lag1=None):
    mae  = mean_absolute_error(y_true, y_pred)
    wm   = wmape(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    print(f"  {label:<40} MAE={mae:.4f}  WMAPE={wm:.2f}%  Bias={bias:+.5f}")
    if is_bad_lag1 is not None:
        for val, lbl in [(0, "  normal yesterday   "), (1, "  bad weather yesterday")]:
            mask = is_bad_lag1 == val
            if mask.sum() > 0:
                mae_s  = mean_absolute_error(y_true[mask], y_pred[mask])
                bias_s = np.mean(y_pred[mask] - y_true[mask])
                print(f"    {lbl}  N={mask.sum():6d}  MAE={mae_s:.4f}  Bias={bias_s:+.5f}")
    return mae


# ─────────────────────────────────────────────────────────────────────────────
# VARIANT A: Post-processing correction
# ─────────────────────────────────────────────────────────────────────────────

def run_variant_a(train, test, available, y_train, y_test):
    print("\n" + "=" * 60)
    print("  VARIANT A: Post-processing correction")
    print("=" * 60)

    # Train base P50 model on V3 features (no weather-lag)
    t0 = time.time()
    model = train_quantile(train[available], y_train, alpha=0.5)
    print(f"  Train time: {time.time()-t0:.0f}s")

    pred_train = predict_clipped(model, train[available])
    pred_test  = predict_clipped(model, test[available])

    # Compute correction factor per-bakery on TRAIN data
    train_df = train.copy()
    train_df["pred"]   = pred_train
    train_df["actual"] = y_train.values
    train_df["ratio"]  = train_df["actual"] / np.maximum(train_df["pred"], 1.0)

    bad_mask_train = train_df["is_bad_weather_lag1"] == 1

    # Global correction factor
    global_cf = train_df.loc[bad_mask_train, "ratio"].mean()
    print(f"\n  Global correction factor (bad_lag1=1): {global_cf:.4f}")

    # Per-bakery correction factor (with fallback to global)
    bakery_cf = (
        train_df[bad_mask_train]
        .groupby("Пекарня")["ratio"]
        .mean()
        .rename("cf")
    )
    print(f"  Per-bakery CF: mean={bakery_cf.mean():.3f}  "
          f"min={bakery_cf.min():.3f}  max={bakery_cf.max():.3f}  N={len(bakery_cf)}")

    # Apply correction to test
    test_df = test.copy()
    test_df["pred_base"] = pred_test
    test_df = test_df.merge(bakery_cf, on="Пекарня", how="left")
    test_df["cf"] = test_df["cf"].fillna(global_cf)

    # Apply only where is_bad_weather_lag1 = 1
    test_df["pred_corrected"] = test_df["pred_base"].copy()
    mask = test_df["is_bad_weather_lag1"] == 1
    test_df.loc[mask, "pred_corrected"] = (
        test_df.loc[mask, "pred_base"] * test_df.loc[mask, "cf"]
    ).clip(lower=0)

    y_test_arr = y_test.values
    bad_lag1   = test_df["is_bad_weather_lag1"].values

    print("\n  --- Results ---")
    evaluate(y_test_arr, pred_test,                          "Base P50 (no correction)  ", bad_lag1)
    mae_a = evaluate(y_test_arr, test_df["pred_corrected"].values, "A: Per-bakery corrected   ", bad_lag1)

    return mae_a, test_df["pred_corrected"].values, global_cf, bakery_cf


# ─────────────────────────────────────────────────────────────────────────────
# VARIANT B: Data augmentation (lag cleaning)
# ─────────────────────────────────────────────────────────────────────────────

def run_variant_b(train, test, available, y_train, y_test):
    print("\n" + "=" * 60)
    print("  VARIANT B: Data augmentation (lag cleaning)")
    print("=" * 60)

    train_aug = train.copy()

    # For rows where is_bad_weather_lag1=1, create augmented copies with cleaned lags
    bad_rows = train_aug[train_aug["is_bad_weather_lag1"] == 1].copy()
    print(f"  Post-bad-weather rows in train: {len(bad_rows):,} ({len(bad_rows)/len(train)*100:.1f}%)")

    # Replace lag features with 7-day rolling mean as proxy for "expected" lag
    # (bakery_sales_dow_mean not present in this dataset; roll_mean7 is a stable estimate)
    clean_proxy = bad_rows["sales_roll_mean7"]
    demand_proxy = bad_rows["demand_roll_mean7"]
    for feat in LAG_FEATURES:
        if feat not in bad_rows.columns:
            continue
        if feat.startswith("demand_"):
            bad_rows[feat] = demand_proxy
        else:
            bad_rows[feat] = clean_proxy

    # Combine: original + augmented bad-weather rows
    train_combined = pd.concat([train_aug, bad_rows], ignore_index=True)
    y_combined     = train_combined[DEMAND_TARGET]
    print(f"  Combined train size: {len(train_combined):,} (+{len(bad_rows):,} augmented)")

    for col in CATEGORICAL_COLS_V2:
        if col in train_combined.columns:
            train_combined[col] = train_combined[col].astype("category")

    t0 = time.time()
    model_b = train_quantile(train_combined[available], y_combined, alpha=0.5)
    print(f"  Train time: {time.time()-t0:.0f}s")

    pred_test_b = predict_clipped(model_b, test[available])
    y_test_arr  = y_test.values
    bad_lag1    = test["is_bad_weather_lag1"].values

    print("\n  --- Results ---")
    mae_b = evaluate(y_test_arr, pred_test_b, "B: Augmented model (P50)  ", bad_lag1)

    return mae_b, pred_test_b


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  EXPERIMENT 68: Post-processing vs Augmentation")
    print("  Fixing lag contamination on post-bad-weather days")
    print("=" * 60)
    t_start = time.time()

    # --- Load & prepare ---
    print(f"\n[1/4] Loading data...")
    df = pd.read_csv(str(DEMAND_8M_PATH), encoding="utf-8-sig")
    df["Дата"] = pd.to_datetime(df["Дата"])
    df = df.sort_values(["Пекарня", "Дата"])
    df = build_weather_lag_col(df)
    print(f"  Shape: {df.shape}")
    print(f"  is_bad_weather_lag1=1: {df['is_bad_weather_lag1'].mean()*100:.1f}% of rows")

    # Features: V3 only (no extra weather-lag — they didn't help in exp67)
    available = [f for f in FEATURES_V3 if f in df.columns]
    for col in CATEGORICAL_COLS_V2:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # --- Split ---
    print(f"\n[2/4] Train/test split...")
    test_start = df["Дата"].max() - pd.Timedelta(days=TEST_DAYS - 1)
    train = df[df["Дата"] < test_start].copy()
    test  = df[df["Дата"] >= test_start].copy()
    print(f"  Train: {len(train):,} rows  |  Test: {len(test):,} rows")

    y_train = train[DEMAND_TARGET]
    y_test  = test[DEMAND_TARGET]

    # --- Baseline: plain V3 P50 (for reference within this run) ---
    print(f"\n[3/4] Baseline (V3 P50, no correction)...")
    t0 = time.time()
    model_base  = train_quantile(train[available], y_train, alpha=0.5)
    pred_base   = predict_clipped(model_base, test[available])
    mae_base    = mean_absolute_error(y_test.values, pred_base)
    bad_lag1    = test["is_bad_weather_lag1"].values
    print(f"  MAE={mae_base:.4f}  Time={time.time()-t0:.0f}s")
    evaluate(y_test.values, pred_base, "Baseline V3 P50           ", bad_lag1)

    # --- Variant A ---
    print(f"\n[4a/4] Variant A...")
    mae_a, pred_a, global_cf, bakery_cf = run_variant_a(train, test, available, y_train, y_test)

    # --- Variant B ---
    print(f"\n[4b/4] Variant B...")
    mae_b, pred_b = run_variant_b(train, test, available, y_train, y_test)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Approach':<42} {'MAE':>8}  {'Delta vs V3':>12}")
    print(f"  {'-'*64}")
    print(f"  {'exp 60 (V3, P50)':<42} {BASELINE_V3_MAE:>8.4f}  {'ref':>12}")
    print(f"  {'exp 67 (weather-lag feats)':<42} {EXP67_MAE:>8.4f}  {EXP67_MAE-BASELINE_V3_MAE:>+12.4f}")
    print(f"  {'exp 68 baseline (this run)':<42} {mae_base:>8.4f}  {mae_base-BASELINE_V3_MAE:>+12.4f}")
    print(f"  {'exp 68A (post-process correction)':<42} {mae_a:>8.4f}  {mae_a-BASELINE_V3_MAE:>+12.4f}")
    print(f"  {'exp 68B (lag augmentation)':<42} {mae_b:>8.4f}  {mae_b-BASELINE_V3_MAE:>+12.4f}")

    # Effect specifically on bad-weather-lag=1 rows
    print(f"\n  Effect on post-bad-weather rows (N={bad_lag1.sum()}):")
    for preds, label in [
        (pred_base, "baseline "),
        (pred_a,    "68A corrected"),
        (pred_b,    "68B augmented"),
    ]:
        mask = bad_lag1 == 1
        mae_bw  = mean_absolute_error(y_test.values[mask], preds[mask])
        bias_bw = np.mean(preds[mask] - y_test.values[mask])
        print(f"    {label:<16} MAE={mae_bw:.4f}  Bias={bias_bw:+.4f}")

    # --- Save ---
    metrics = {
        "experiment":    "68_weather_correction",
        "mae_baseline":  round(mae_base, 4),
        "mae_variant_a": round(mae_a, 4),
        "mae_variant_b": round(mae_b, 4),
        "delta_a_vs_v3": round(mae_a - BASELINE_V3_MAE, 4),
        "delta_b_vs_v3": round(mae_b - BASELINE_V3_MAE, 4),
        "global_correction_factor": round(float(global_cf), 4),
        "n_bakeries_with_cf": int(len(bakery_cf)),
        "bad_lag1_pct_test": round(float(bad_lag1.mean() * 100), 1),
        "train_rows":    len(train),
        "test_rows":     len(test),
        "n_features":    len(available),
        "total_time_s":  round(time.time() - t_start, 1),
    }

    predictions = pd.DataFrame({
        "Дата":              test["Дата"].values,
        "Пекарня":           test["Пекарня"].values,
        "Номенклатура":      test["Номенклатура"].values,
        "fact_demand":       y_test.values,
        "pred_base":         np.round(pred_base, 2),
        "pred_A":            np.round(pred_a, 2),
        "pred_B":            np.round(pred_b, 2),
        "is_bad_weather_lag1": bad_lag1,
    })

    save_results(EXP_DIR, metrics, predictions)
    print(f"\n  Total time: {time.time()-t_start:.0f}s")
    print("  Done!")


if __name__ == "__main__":
    main()

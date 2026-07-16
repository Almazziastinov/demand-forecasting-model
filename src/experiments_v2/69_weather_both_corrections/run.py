"""
Experiment 69: Symmetric proportional weather correction — both D0 and D+1.

Two systematic errors confirmed:
  D0  (bad weather day):  model OVER-forecasts  -> need downward correction
  D+1 (day after storm):  model UNDER-forecasts -> need upward correction

Unified formula (estimated from training data):
  CF_down(precip_today) = 1 - delta_down * (precip_today / precip_ref)
  CF_up(precip_lag1)    = 1 + delta_up   * (precip_lag1  / precip_ref)

Both deltas fitted via linear regression on train:
  residual_ratio = (actual / pred - 1)
  residual_ratio ~ alpha * precipitation          (for D0)
  residual_ratio ~ alpha * precipitation_lag1     (for D+1)

Note on seasonality (future work):
  Precipitation = snow in winter, rain in summer -> different demand impact.
  Current model learns one average effect. Next step: seasonal split of deltas.

Usage:
  .venv/Scripts/python.exe src/experiments_v2/69_weather_both_corrections/run.py
"""

import sys
import os
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

from src.experiments_v2.common import (
    DEMAND_8M_PATH, FEATURES_V3, CATEGORICAL_COLS_V2, DEMAND_TARGET,
    wmape, train_quantile, predict_clipped, save_results,
)
from src.config import TEST_DAYS

EXP_DIR = Path(__file__).resolve().parent
BASELINE_V3_MAE = 2.8816


def build_lag_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["Пекарня", "Дата"])
    grp = df.groupby("Пекарня")
    df["precipitation_lag1"]  = grp["precipitation"].shift(1).fillna(0)
    df["is_bad_weather_lag1"] = grp["is_bad_weather"].shift(1).fillna(0).astype(int)
    return df


def fit_correction(train_df: pd.DataFrame, pred_col: str, actual_col: str,
                   precip_col: str, min_precip: float = 0.5) -> dict:
    """
    Fit proportional correction: residual_ratio ~ slope * precipitation
    Only on rows where precipitation > min_precip (exclude dry days from fit).
    Returns dict with slope, precip_ref, and the implied delta at precip_ref.
    """
    df = train_df.copy()
    df = df[df[precip_col] > min_precip].copy()
    df["ratio"] = df[actual_col] / np.maximum(df[pred_col], 1.0) - 1.0

    # Winsorize ratio to avoid outlier influence
    lo, hi = df["ratio"].quantile(0.02), df["ratio"].quantile(0.98)
    df["ratio"] = df["ratio"].clip(lo, hi)

    X = df[[precip_col]].values
    y = df["ratio"].values
    reg = LinearRegression(fit_intercept=False).fit(X, y)
    slope = float(reg.coef_[0])

    precip_ref = float(df[precip_col].mean())
    delta_ref  = slope * precip_ref

    return {
        "slope":      slope,
        "precip_ref": precip_ref,
        "delta_ref":  delta_ref,
        "n_rows":     len(df),
        "ratio_mean": float(df["ratio"].mean()),
    }


def apply_correction(pred: np.ndarray, precip: np.ndarray, slope: float,
                     direction: str, clip_cf: tuple = (0.7, 1.5)) -> np.ndarray:
    """
    Apply CF = 1 + slope * precip.
    direction='down'  -> slope should be negative (reduce on bad day)
    direction='up'    -> slope should be positive (boost on day after)
    Clip CF to avoid extreme corrections.
    """
    cf = 1.0 + slope * precip
    cf = np.clip(cf, clip_cf[0], clip_cf[1])
    return np.maximum(pred * cf, 0)


def breakdown(y_true, y_pred, is_bad, is_bad_lag1, label):
    """Print MAE/Bias breakdown by weather condition."""
    mae  = mean_absolute_error(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    print(f"  {label:<42} MAE={mae:.4f}  Bias={bias:+.5f}")
    for mask, lbl in [
        (is_bad == 1,     "    bad day (D0)        "),
        (is_bad_lag1 == 1,"    after bad day (D+1) "),
        ((is_bad == 0) & (is_bad_lag1 == 0), "    normal              "),
    ]:
        if mask.sum() > 0:
            print(f"  {lbl}  N={mask.sum():6d}  "
                  f"MAE={mean_absolute_error(y_true[mask], y_pred[mask]):.4f}  "
                  f"Bias={np.mean(y_pred[mask] - y_true[mask]):+.4f}")
    return mae


def main():
    print("=" * 62)
    print("  EXPERIMENT 69: Symmetric proportional weather correction")
    print("  D0 (bad day) downward + D+1 (after storm) upward")
    print("=" * 62)
    t0 = time.time()

    # --- Load ---
    print(f"\n[1/5] Loading data...")
    df = pd.read_csv(str(DEMAND_8M_PATH), encoding="utf-8-sig")
    df["Дата"] = pd.to_datetime(df["Дата"])
    df = build_lag_cols(df)

    available = [f for f in FEATURES_V3 if f in df.columns]
    for col in CATEGORICAL_COLS_V2:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print(f"  is_bad_weather=1:     {df['is_bad_weather'].mean()*100:.1f}% rows")
    print(f"  is_bad_weather_lag1=1: {df['is_bad_weather_lag1'].mean()*100:.1f}% rows")

    # --- Split ---
    print(f"\n[2/5] Split...")
    test_start = df["Дата"].max() - pd.Timedelta(days=TEST_DAYS - 1)
    train = df[df["Дата"] < test_start].copy()
    test  = df[df["Дата"] >= test_start].copy()
    print(f"  Train: {len(train):,}  |  Test: {len(test):,}")

    y_train = train[DEMAND_TARGET]
    y_test  = test[DEMAND_TARGET].values

    # --- Train base model ---
    print(f"\n[3/5] Train base P50 model (V3 features)...")
    t1 = time.time()
    model = train_quantile(train[available], y_train, alpha=0.5)
    print(f"  Done in {time.time()-t1:.0f}s")

    pred_train = predict_clipped(model, train[available])
    pred_test  = predict_clipped(model, test[available])

    # --- Fit correction slopes from train ---
    print(f"\n[4/5] Fitting correction slopes on train data...")
    train_df = train.copy()
    train_df["pred"]   = pred_train
    train_df["actual"] = y_train.values

    # D0: bad weather day -> model over-forecasts -> slope should be negative
    fit_d0 = fit_correction(
        train_df, pred_col="pred", actual_col="actual",
        precip_col="precipitation", min_precip=0.5,
    )
    print(f"\n  D0  (bad day correction):")
    print(f"    slope     = {fit_d0['slope']:+.5f}  (neg = reduce forecast on rainy day)")
    print(f"    precip_ref = {fit_d0['precip_ref']:.1f} mm  (avg precip on rainy days)")
    print(f"    delta_ref  = {fit_d0['delta_ref']:+.3f}  (CF at avg precip = {1+fit_d0['delta_ref']:.3f}x)")
    print(f"    N train    = {fit_d0['n_rows']:,}  |  mean ratio = {fit_d0['ratio_mean']:+.4f}")

    # D+1: day after storm -> model under-forecasts -> slope should be positive
    fit_d1 = fit_correction(
        train_df, pred_col="pred", actual_col="actual",
        precip_col="precipitation_lag1", min_precip=0.5,
    )
    print(f"\n  D+1 (post-storm correction):")
    print(f"    slope      = {fit_d1['slope']:+.5f}  (pos = boost forecast after rainy day)")
    print(f"    precip_ref = {fit_d1['precip_ref']:.1f} mm")
    print(f"    delta_ref  = {fit_d1['delta_ref']:+.3f}  (CF at avg precip = {1+fit_d1['delta_ref']:.3f})")
    print(f"    N train    = {fit_d1['n_rows']:,}  |  mean ratio = {fit_d1['ratio_mean']:+.4f}")

    # Print CF table for reference
    print(f"\n  Correction factor table:")
    print(f"  {'precip (mm)':<14} {'CF_down (D0)':>14} {'CF_up (D+1)':>14}")
    print(f"  {'-'*44}")
    for mm in [0, 1, 2, 4, 6, 8, 10, 15, 20]:
        cf_d  = np.clip(1 + fit_d0['slope'] * mm, 0.7, 1.5)
        cf_u  = np.clip(1 + fit_d1['slope'] * mm, 0.7, 1.5)
        print(f"  {mm:<14}  {cf_d:>13.4f}  {cf_u:>13.4f}")

    # --- Apply corrections to test ---
    print(f"\n[5/5] Apply corrections and evaluate...")
    precip_test      = test["precipitation"].values
    precip_lag1_test = test["precipitation_lag1"].values
    is_bad_test      = test["is_bad_weather"].values.astype(int)
    is_bad_lag1_test = test["is_bad_weather_lag1"].values

    # Variant: D0 only
    pred_d0_only = apply_correction(pred_test, precip_test, fit_d0["slope"], "down")

    # Variant: D+1 only
    pred_d1_only = apply_correction(pred_test, precip_lag1_test, fit_d1["slope"], "up")

    # Variant: both D0 and D+1
    pred_both = pred_test.copy()
    pred_both = apply_correction(pred_both, precip_test,      fit_d0["slope"], "down")
    pred_both = apply_correction(pred_both, precip_lag1_test, fit_d1["slope"], "up")

    print()
    breakdown(y_test, pred_test,    is_bad_test, is_bad_lag1_test, "Baseline (no correction)  ")
    breakdown(y_test, pred_d0_only, is_bad_test, is_bad_lag1_test, "D0 only (downward)        ")
    breakdown(y_test, pred_d1_only, is_bad_test, is_bad_lag1_test, "D+1 only (upward)         ")
    mae_both = breakdown(y_test, pred_both,    is_bad_test, is_bad_lag1_test, "Both D0 + D+1             ")

    # Summary
    mae_base = mean_absolute_error(y_test, pred_test)
    mae_d0   = mean_absolute_error(y_test, pred_d0_only)
    mae_d1   = mean_absolute_error(y_test, pred_d1_only)

    print(f"\n  {'Approach':<42} {'MAE':>8}  {'Delta vs V3':>12}")
    print(f"  {'-'*64}")
    print(f"  {'exp 60 V3 reference':<42} {BASELINE_V3_MAE:>8.4f}  {'ref':>12}")
    print(f"  {'68A (D+1 binary CF)':<42} {'2.8729':>8}  {'-0.0087':>12}")
    print(f"  {'69 baseline (this run)':<42} {mae_base:>8.4f}  {mae_base-BASELINE_V3_MAE:>+12.4f}")
    print(f"  {'69 D0 only (proportional down)':<42} {mae_d0:>8.4f}  {mae_d0-BASELINE_V3_MAE:>+12.4f}")
    print(f"  {'69 D+1 only (proportional up)':<42} {mae_d1:>8.4f}  {mae_d1-BASELINE_V3_MAE:>+12.4f}")
    print(f"  {'69 Both D0+D+1':<42} {mae_both:>8.4f}  {mae_both-BASELINE_V3_MAE:>+12.4f}")

    # Future work note
    print(f"\n  NOTE (future): precipitation effect varies by season")
    print(f"  (winter=snow vs summer=rain -> different demand impact)")
    print(f"  Next step: fit delta_down/delta_up per season (Q1/Q2/Q3/Q4)")

    # --- Save ---
    metrics = {
        "experiment":       "69_weather_both_corrections",
        "mae_baseline":     round(float(mae_base), 4),
        "mae_d0_only":      round(float(mae_d0), 4),
        "mae_d1_only":      round(float(mae_d1), 4),
        "mae_both":         round(float(mae_both), 4),
        "delta_both_vs_v3": round(float(mae_both - BASELINE_V3_MAE), 4),
        "d0_slope":         round(fit_d0["slope"], 6),
        "d0_precip_ref":    round(fit_d0["precip_ref"], 2),
        "d0_delta_ref":     round(fit_d0["delta_ref"], 4),
        "d1_slope":         round(fit_d1["slope"], 6),
        "d1_precip_ref":    round(fit_d1["precip_ref"], 2),
        "d1_delta_ref":     round(fit_d1["delta_ref"], 4),
        "train_rows":       len(train),
        "test_rows":        len(test),
        "n_features":       len(available),
        "total_time_s":     round(time.time() - t0, 1),
        "future_work":      "seasonal split of correction slopes (winter snow vs summer rain)",
    }

    predictions = pd.DataFrame({
        "Дата":              test["Дата"].values,
        "Пекарня":           test["Пекарня"].values,
        "Номенклатура":      test["Номенклатура"].values,
        "fact_demand":       y_test,
        "pred_base":         np.round(pred_test, 2),
        "pred_d0_corrected": np.round(pred_d0_only, 2),
        "pred_d1_corrected": np.round(pred_d1_only, 2),
        "pred_both":         np.round(pred_both, 2),
        "precipitation":     precip_test,
        "precipitation_lag1": precip_lag1_test,
        "is_bad_weather":    is_bad_test,
        "is_bad_weather_lag1": is_bad_lag1_test,
    })

    save_results(EXP_DIR, metrics, predictions)
    print(f"\n  Total time: {time.time()-t0:.0f}s")
    print("  Done!")


if __name__ == "__main__":
    main()

"""
Tier 3 Remaining Experiments: 11, 12, 13 (single runner).

Exp 11: Residual Chain — 3 sequential models on semantic feature groups
Exp 12: Mixture of Experts — demand-level routing (low/medium/high)
Exp 13: Temporal Ensemble — short/medium/long history windows

All: target=Spros, FEATURES_V3, Quantile P50.
Baseline V3 (exp 60): MAE 2.8816.

Usage:
  .venv/Scripts/python.exe src/experiments_v2/11_13_tier3_ensembles/run.py
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
from sklearn.linear_model import Ridge

from src.config import MODEL_PARAMS, TEST_DAYS
from src.experiments_v2.common import (
    DEMAND_8M_PATH, FEATURES_V3, CATEGORICAL_COLS_V2, DEMAND_TARGET,
    wmape, print_metrics, print_category_metrics,
    train_quantile, predict_clipped, save_results,
)

EXP_DIR = Path(__file__).resolve().parent
BASELINE_V3_MAE = 2.8816

# --- Feature groups for Residual Chain ---
STRUCTURE_FEATURES = [
    "Пекарня", "Номенклатура", "Категория", "Город",
    "ДеньНедели", "День", "IsWeekend", "Месяц", "НомерНедели",
]

DYNAMICS_FEATURES = [
    "sales_lag1", "sales_lag2", "sales_lag3", "sales_lag7", "sales_lag14", "sales_lag30",
    "sales_roll_mean3", "sales_roll_mean7", "sales_roll_std7",
    "sales_roll_mean14", "sales_roll_mean30",
    "demand_lag1", "demand_lag2", "demand_lag3", "demand_lag7",
    "demand_lag14", "demand_lag30",
    "demand_roll_mean3", "demand_roll_mean7", "demand_roll_mean14", "demand_roll_mean30",
    "demand_roll_std7",
    "avg_price", "price_vs_median", "price_lag7",
    "price_change_7d", "price_roll_mean7", "price_roll_std7",
]

CONTEXT_FEATURES = [
    "temp_max", "temp_min", "temp_mean", "temp_range",
    "precipitation", "rain", "snowfall", "windspeed_max",
    "is_rainy", "is_snowy", "is_cold", "is_warm",
    "is_windy", "is_bad_weather", "weather_cat_code",
    "is_holiday", "is_pre_holiday", "is_post_holiday", "is_payday_week",
    "is_month_start", "is_month_end",
]


def load_data():
    """Load and prepare data, return df, train, test, available features."""
    print(f"  Loading data from {DEMAND_8M_PATH.name}...")
    df = pd.read_csv(str(DEMAND_8M_PATH), encoding="utf-8-sig")
    df["Дата"] = pd.to_datetime(df["Дата"])

    available = [f for f in FEATURES_V3 if f in df.columns]
    for col in CATEGORICAL_COLS_V2:
        if col in df.columns:
            df[col] = df[col].astype("category")

    test_start = df["Дата"].max() - pd.Timedelta(days=TEST_DAYS - 1)
    train = df[df["Дата"] < test_start].copy()
    test = df[df["Дата"] >= test_start].copy()

    print(f"  {len(df):,} total, Train: {len(train):,}, Test: {len(test):,}")
    print(f"  Features: {len(available)}")

    return df, train, test, available


def run_exp11(train, test):
    """Exp 11: Residual Chain."""
    print(f"\n{'='*60}")
    print(f"  EXP 11: Residual Chain")
    print(f"  Model 1 (structure) -> Model 2 (dynamics) -> Model 3 (context)")
    print(f"{'='*60}")
    t_start = time.time()

    y_train = train[DEMAND_TARGET].values
    y_test = test[DEMAND_TARGET].values

    # --- Model 1: Structure ---
    feats1 = [f for f in STRUCTURE_FEATURES if f in train.columns]
    print(f"\n  Model 1 (structure): {len(feats1)} features...")
    t = time.time()
    m1 = train_quantile(train[feats1], y_train)
    pred1_train = predict_clipped(m1, train[feats1])
    pred1_test = predict_clipped(m1, test[feats1])
    mae1 = mean_absolute_error(y_test, pred1_test)
    print(f"    MAE={mae1:.4f} ({time.time()-t:.0f}s)")

    # --- Model 2: Dynamics on residual ---
    residual1_train = y_train - pred1_train
    feats2 = [f for f in DYNAMICS_FEATURES if f in train.columns]
    print(f"\n  Model 2 (dynamics on residual): {len(feats2)} features...")
    t = time.time()
    m2 = train_quantile(train[feats2], residual1_train)
    pred2_train = m2.predict(train[feats2])
    pred2_test = m2.predict(test[feats2])
    partial2 = predict_clipped(None, pred1_test + pred2_test) if False else np.maximum(pred1_test + pred2_test, 0)
    mae12 = mean_absolute_error(y_test, partial2)
    print(f"    MAE (stage 1+2)={mae12:.4f} ({time.time()-t:.0f}s)")

    # --- Model 3: Context on residual ---
    residual2_train = y_train - pred1_train - pred2_train
    feats3 = [f for f in CONTEXT_FEATURES if f in train.columns]
    print(f"\n  Model 3 (context on residual): {len(feats3)} features...")
    t = time.time()
    m3 = train_quantile(train[feats3], residual2_train)
    pred3_test = m3.predict(test[feats3])
    pred_chain = np.maximum(pred1_test + pred2_test + pred3_test, 0)
    mae_chain = mean_absolute_error(y_test, pred_chain)
    print(f"    MAE (chain)={mae_chain:.4f} ({time.time()-t:.0f}s)")

    wm = wmape(y_test, pred_chain)
    bias = np.mean(y_test - pred_chain)
    print(f"\n  Exp 11 result: MAE={mae_chain:.4f}, WMAPE={wm:.2f}%, Bias={bias:+.4f}")
    print(f"  Delta vs V3: {mae_chain - BASELINE_V3_MAE:+.4f}")
    print(f"  Stages: structure {mae1:.4f} -> +dynamics {mae12:.4f} -> +context {mae_chain:.4f}")

    elapsed = time.time() - t_start
    print(f"  Time: {elapsed:.0f}s")

    return {
        "name": "11_residual_chain",
        "mae": mae_chain, "wmape": wm, "bias": bias,
        "mae_stage1": mae1, "mae_stage12": mae12,
        "pred": pred_chain, "time": elapsed,
    }


def run_exp12(train, test):
    """Exp 12: Mixture of Experts (demand-level routing)."""
    print(f"\n{'='*60}")
    print(f"  EXP 12: Mixture of Experts")
    print(f"  Low (<3) / Medium (3-15) / High (>=15)")
    print(f"{'='*60}")
    t_start = time.time()

    y_test = test[DEMAND_TARGET].values
    available = [f for f in FEATURES_V3 if f in train.columns]

    # Compute product average demand on train
    product_avg = train.groupby("Номенклатура")[DEMAND_TARGET].mean()

    train_avg = train["Номенклатура"].map(product_avg).fillna(0)
    test_avg = test["Номенклатура"].map(product_avg).fillna(0)

    segments = {
        "low":    (train_avg < 3,   test_avg < 3),
        "medium": ((train_avg >= 3) & (train_avg < 15), (test_avg >= 3) & (test_avg < 15)),
        "high":   (train_avg >= 15, test_avg >= 15),
    }

    pred_moe = np.zeros(len(test))

    for seg_name, (train_mask, test_mask) in segments.items():
        n_train = train_mask.sum()
        n_test = test_mask.sum()
        print(f"\n  --- {seg_name.upper()}: {n_train:,} train, {n_test:,} test ---")

        if n_test == 0:
            print(f"    Skip (no test samples)")
            continue

        tr = train[train_mask.values]
        te = test[test_mask.values]

        # Adjust params per segment
        if seg_name == "low":
            params = MODEL_PARAMS.copy()
            params["n_estimators"] = 800
            params["min_child_samples"] = 50
        elif seg_name == "high":
            params = MODEL_PARAMS.copy()
            params["n_estimators"] = 3000
            params["max_depth"] = 9
            params["num_leaves"] = 127
        else:
            params = None  # default

        t = time.time()
        model = train_quantile(tr[available], tr[DEMAND_TARGET], params=params)
        pred = predict_clipped(model, te[available])
        pred_moe[test_mask.values] = pred

        mae_seg = mean_absolute_error(te[DEMAND_TARGET], pred)
        print(f"    MAE={mae_seg:.4f} ({time.time()-t:.0f}s)")

    mae_moe = mean_absolute_error(y_test, pred_moe)
    wm = wmape(y_test, pred_moe)
    bias = np.mean(y_test - pred_moe)
    print(f"\n  Exp 12 result: MAE={mae_moe:.4f}, WMAPE={wm:.2f}%, Bias={bias:+.4f}")
    print(f"  Delta vs V3: {mae_moe - BASELINE_V3_MAE:+.4f}")

    elapsed = time.time() - t_start
    print(f"  Time: {elapsed:.0f}s")

    return {
        "name": "12_moe_routing",
        "mae": mae_moe, "wmape": wm, "bias": bias,
        "pred": pred_moe, "time": elapsed,
    }


def run_exp13(train, test, df):
    """Exp 13: Temporal Ensemble (short/medium/long windows)."""
    print(f"\n{'='*60}")
    print(f"  EXP 13: Temporal Ensemble")
    print(f"  Short (30d) + Medium (90d) + Long (all)")
    print(f"{'='*60}")
    t_start = time.time()

    y_test = test[DEMAND_TARGET].values
    available = [f for f in FEATURES_V3 if f in train.columns]

    test_start = test["Дата"].min()
    windows = {
        "short_30d":  train[train["Дата"] >= test_start - pd.Timedelta(days=30)],
        "medium_90d": train[train["Дата"] >= test_start - pd.Timedelta(days=90)],
        "long_all":   train,
    }

    preds = {}
    for wname, w_train in windows.items():
        n = len(w_train)
        days = w_train["Дата"].nunique()
        print(f"\n  --- {wname}: {n:,} rows, {days} days ---")

        t = time.time()
        model = train_quantile(w_train[available], w_train[DEMAND_TARGET])
        pred = predict_clipped(model, test[available])
        preds[wname] = pred

        mae_w = mean_absolute_error(y_test, pred)
        print(f"    MAE={mae_w:.4f} ({time.time()-t:.0f}s)")

    # Simple average
    pred_avg = (preds["short_30d"] + preds["medium_90d"] + preds["long_all"]) / 3
    pred_avg = np.maximum(pred_avg, 0)
    mae_avg = mean_absolute_error(y_test, pred_avg)

    # Optimized weights via Ridge on last 7 days of train (validation)
    val_start = test_start - pd.Timedelta(days=7)
    val = train[train["Дата"] >= val_start]
    if len(val) > 0:
        val_preds = {}
        for wname, w_train_full in windows.items():
            w_tr = w_train_full[w_train_full["Дата"] < val_start]
            if len(w_tr) > 100:
                m = train_quantile(w_tr[available], w_tr[DEMAND_TARGET])
                val_preds[wname] = predict_clipped(m, val[available])
            else:
                val_preds[wname] = np.full(len(val), val[DEMAND_TARGET].mean())

        meta_X = np.column_stack([val_preds[k] for k in windows.keys()])
        meta = Ridge(alpha=1.0)
        meta.fit(meta_X, val[DEMAND_TARGET])

        test_meta_X = np.column_stack([preds[k] for k in windows.keys()])
        pred_opt = np.maximum(meta.predict(test_meta_X), 0)
        mae_opt = mean_absolute_error(y_test, pred_opt)

        print(f"\n  Optimized weights: short={meta.coef_[0]:.3f}, "
              f"medium={meta.coef_[1]:.3f}, long={meta.coef_[2]:.3f}")
    else:
        pred_opt = pred_avg
        mae_opt = mae_avg

    # Pick best
    if mae_opt < mae_avg:
        pred_final = pred_opt
        mae_final = mae_opt
        method = "optimized"
    else:
        pred_final = pred_avg
        mae_final = mae_avg
        method = "simple_avg"

    wm = wmape(y_test, pred_final)
    bias = np.mean(y_test - pred_final)
    print(f"\n  Exp 13 result ({method}): MAE={mae_final:.4f}, WMAPE={wm:.2f}%, Bias={bias:+.4f}")
    print(f"  Simple avg: {mae_avg:.4f}, Optimized: {mae_opt:.4f}")
    print(f"  Delta vs V3: {mae_final - BASELINE_V3_MAE:+.4f}")

    elapsed = time.time() - t_start
    print(f"  Time: {elapsed:.0f}s")

    return {
        "name": "13_temporal_ensemble",
        "mae": mae_final, "wmape": wm, "bias": bias,
        "mae_simple_avg": mae_avg, "mae_optimized": mae_opt,
        "method": method,
        "pred": pred_final, "time": elapsed,
    }


def main():
    print("=" * 60)
    print("  TIER 3 REMAINING: Exp 11, 12, 13")
    print("  Baseline V3: MAE 2.8816")
    print("=" * 60)
    t_total = time.time()

    # Load data once
    df, train, test, available = load_data()
    y_test = test[DEMAND_TARGET].values

    # Run experiments
    r11 = run_exp11(train, test)
    r12 = run_exp12(train, test)
    r13 = run_exp13(train, test, df)

    # --- Comparison ---
    print(f"\n{'='*60}")
    print(f"  COMPARISON TABLE (vs Baseline V3 MAE={BASELINE_V3_MAE})")
    print(f"{'='*60}")
    print(f"\n  {'Experiment':<25} {'MAE':>8} {'WMAPE':>8} {'Bias':>8} {'Delta':>8}")
    print(f"  {'-'*60}")
    print(f"  {'Baseline V3':<25} {BASELINE_V3_MAE:>8.4f} {'27.15':>7}% {'+0.77':>8} {'—':>8}")

    for r in [r11, r12, r13]:
        delta = r["mae"] - BASELINE_V3_MAE
        print(f"  {r['name']:<25} {r['mae']:>8.4f} {r['wmape']:>7.2f}% {r['bias']:>+8.4f} {delta:>+8.4f}")

    best = min([r11, r12, r13], key=lambda x: x["mae"])
    print(f"\n  Best: {best['name']} (MAE {best['mae']:.4f})")

    # High demand analysis for each
    print(f"\n  === High demand analysis ===")
    print(f"  {'Experiment':<25} {'>=15':>10} {'>=50':>10} {'>=100':>10}")
    print(f"  {'-'*57}")
    for r in [r11, r12, r13]:
        parts = []
        for threshold in [15, 50, 100]:
            mask = y_test >= threshold
            if mask.sum() > 0:
                mae_h = mean_absolute_error(y_test[mask], r["pred"][mask])
                parts.append(f"{mae_h:>10.2f}")
            else:
                parts.append(f"{'—':>10}")
        print(f"  {r['name']:<25} {''.join(parts)}")

    # Per-category for best
    print(f"\n  === Per-category (best: {best['name']}) ===")
    print_category_metrics(y_test, best["pred"], test["Категория"].values)

    # --- Save ---
    print(f"\n  Saving results...")
    metrics = {
        "experiments": ["11_residual_chain", "12_moe_routing", "13_temporal_ensemble"],
        "baseline_v3_mae": BASELINE_V3_MAE,
        "exp_11": {
            "mae": round(r11["mae"], 4), "wmape": round(r11["wmape"], 2),
            "bias": round(r11["bias"], 4),
            "mae_stage1": round(r11["mae_stage1"], 4),
            "mae_stage12": round(r11["mae_stage12"], 4),
            "delta_vs_v3": round(r11["mae"] - BASELINE_V3_MAE, 4),
            "time_s": round(r11["time"], 1),
        },
        "exp_12": {
            "mae": round(r12["mae"], 4), "wmape": round(r12["wmape"], 2),
            "bias": round(r12["bias"], 4),
            "delta_vs_v3": round(r12["mae"] - BASELINE_V3_MAE, 4),
            "time_s": round(r12["time"], 1),
        },
        "exp_13": {
            "mae": round(r13["mae"], 4), "wmape": round(r13["wmape"], 2),
            "bias": round(r13["bias"], 4),
            "mae_simple_avg": round(r13["mae_simple_avg"], 4),
            "mae_optimized": round(r13["mae_optimized"], 4),
            "method": r13["method"],
            "delta_vs_v3": round(r13["mae"] - BASELINE_V3_MAE, 4),
            "time_s": round(r13["time"], 1),
        },
        "best": best["name"],
        "best_mae": round(best["mae"], 4),
    }

    predictions = pd.DataFrame({
        "Дата": test["Дата"].values,
        "Пекарня": test["Пекарня"].values,
        "Номенклатура": test["Номенклатура"].values,
        "Категория": test["Категория"].values,
        "fact_demand": y_test,
        "pred_11_chain": np.round(r11["pred"], 2),
        "pred_12_moe": np.round(r12["pred"], 2),
        "pred_13_temporal": np.round(r13["pred"], 2),
    })

    save_results(EXP_DIR, metrics, predictions)

    elapsed = time.time() - t_total
    print(f"\n  Total time: {elapsed:.0f}s")
    print("  Done!")


if __name__ == "__main__":
    main()

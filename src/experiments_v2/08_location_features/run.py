"""
Experiment 08: Location Features (demand target).

Compare two models on the SAME 138 bakeries (those with location data):
  Model A: baseline + demand features (FEATURES_V2 + demand lags)
  Model B: baseline + demand features + location features

Both train on Spros (demand), evaluate vs Spros.

Input:  data/processed/daily_sales_8m_demand.csv
        data/raw/анализ локаций по БП 02.03 ( 1) (2) (4) (1) (1).xlsx
Output: src/experiments_v2/08_location_features/metrics.json
        src/experiments_v2/08_location_features/predictions.csv

Usage:
  .venv/Scripts/python.exe src/experiments_v2/08_location_features/run.py
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

from src.config import MODEL_PARAMS, TEST_DAYS
from src.experiments_v2.common import (
    FEATURES_V2, CATEGORICAL_COLS_V2,
    wmape, print_metrics, print_category_metrics,
    train_lgbm, predict_clipped, save_results,
)

EXP_DIR = Path(__file__).resolve().parent
DEMAND_CSV = Path(ROOT) / "data" / "processed" / "daily_sales_8m_demand.csv"
LOCATION_PATH = Path(ROOT) / "data" / "raw" / "анализ локаций по БП 02.03 ( 1) (2) (4) (1) (1).xlsx"

DEMAND_TARGET = "Спрос"

# Demand features (lags/rolling on demand) -- same as exp 03
DEMAND_FEATURES = [
    "demand_lag1", "demand_lag2", "demand_lag3", "demand_lag7",
    "demand_lag14", "demand_lag30",
    "demand_roll_mean3", "demand_roll_mean7", "demand_roll_mean14", "demand_roll_mean30",
    "demand_roll_std7",
]

# --- Location feature columns to extract ---
NUMERIC_LOCATION_COLS = {
    26: "loc_area_sqm",           # Obshhaya ploshchad, kv m
    30: "loc_rent_rub",           # Tekushhaya AP, rub.
    32: "loc_rent_per_sqm",       # Tekushhaya stavka, rub/kv m
    34: "loc_traffic_nearby",     # Trafik ryadom
    35: "loc_traffic_at_point",   # Trafik v tochke po BP
    36: "loc_traffic_avg",        # Srednee znachenie (trafik)
    37: "loc_density_5min",       # plotnost naseleniya 5 min
    41: "loc_density_10min",      # plotnost naseleniya 10 min
    42: "loc_density_300m",       # plotnost naseleniya 300m
    43: "loc_income_5min",        # dokhody naseleniya 5 min
    44: "loc_income_10min",       # dokhody naseleniya 10 min
    45: "loc_income_300m",        # dokhody naseleniya 300m
    20: "loc_competitors",        # Konkurenty (chislo)
}

BINARY_LOCATION_COLS = {
    6:  "loc_has_porch",          # Krylco YEST
    17: "loc_has_vitrage",        # Vitrazh
    18: "loc_near_pyaterochka",   # pyaterochka
    19: "loc_near_canteen",       # Stolovaya
    22: "loc_near_metro",         # Stanciya METRO
    46: "loc_near_market",        # Rynok
    47: "loc_near_bc",            # nalichie BC, do 300 m
    48: "loc_near_school",        # shkola, do 300m
    49: "loc_near_college",       # Kolledzh, do 300m
}

CATEGORICAL_LOCATION_COLS = {
    2:  "loc_premise_type",       # strit / strit v tc / tc
}


def parse_binary(val):
    """Convert messy binary values to 0/1/NaN."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in ("da", "yes", "est", "yest", "est'", "1"):
        return 1
    if any(pos in s for pos in ["да", "есть", "ест"]):
        return 1
    if any(neg in s for neg in ["нет", "net", "-", "0"]):
        return 0
    try:
        v = float(s)
        return 1 if v > 0 else 0
    except ValueError:
        return np.nan


def parse_numeric(val):
    """Convert to float, handle text gracefully."""
    if pd.isna(val):
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def parse_premise_type(val):
    """Normalize premise type to category code."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if "тц" in s or "tc" in s:
        if "стрит" in s or "strit" in s:
            return 1  # strit v tc
        return 2  # tc
    return 0  # strit


def load_location_features():
    """Load and parse location features from Excel."""
    print(f"\n  Loading location data from {LOCATION_PATH.name}...")
    raw = pd.read_excel(str(LOCATION_PATH), header=None)

    data_rows = raw.iloc[9:].copy()
    data_rows = data_rows[data_rows.iloc[:, 0].notna()].copy()

    result = pd.DataFrame()
    result["Пекарня"] = data_rows.iloc[:, 0].values

    for col_idx, col_name in NUMERIC_LOCATION_COLS.items():
        result[col_name] = data_rows.iloc[:, col_idx].apply(parse_numeric).values

    for col_idx, col_name in BINARY_LOCATION_COLS.items():
        result[col_name] = data_rows.iloc[:, col_idx].apply(parse_binary).values

    for col_idx, col_name in CATEGORICAL_LOCATION_COLS.items():
        result[col_name] = data_rows.iloc[:, col_idx].apply(parse_premise_type).values

    print(f"  Loaded {len(result)} bakeries, {len(result.columns) - 1} location features")

    for col in result.columns:
        if col == "Пекарня":
            continue
        fill = result[col].notna().mean() * 100
        print(f"    {col:<30} fill: {fill:.0f}%")

    return result


def main():
    print("=" * 60)
    print("  EXPERIMENT 08: Location Features (demand target)")
    print("  Model A: baseline + demand lags")
    print("  Model B: baseline + demand lags + location features")
    print("  Target: Spros | Evaluated on 138 matched bakeries")
    print("=" * 60)
    t_start = time.time()

    # --- Load location features ---
    loc = load_location_features()

    # --- Load demand data ---
    print(f"\n[1/6] Loading demand data from {DEMAND_CSV.name}...")
    if not DEMAND_CSV.exists():
        print(f"  ERROR: {DEMAND_CSV} not found!")
        return
    df = pd.read_csv(str(DEMAND_CSV), encoding="utf-8-sig")
    df["Дата"] = pd.to_datetime(df["Дата"])
    print(f"  Full dataset: {len(df):,} rows, {df['Пекарня'].nunique()} bakeries")

    # --- Merge & filter to matched bakeries ---
    print(f"\n[2/6] Merging location features...")
    matched_bakeries = set(df["Пекарня"].unique()) & set(loc["Пекарня"].unique())
    print(f"  Matched bakeries: {len(matched_bakeries)} of {df['Пекарня'].nunique()}")

    df = df[df["Пекарня"].isin(matched_bakeries)].copy()
    print(f"  Filtered dataset: {len(df):,} rows")

    df = df.merge(loc, on="Пекарня", how="left")

    # Demand stats
    mean_demand = df[DEMAND_TARGET].mean()
    censored_pct = 100 * df["is_censored"].mean()
    print(f"  mean({DEMAND_TARGET}): {mean_demand:.4f}, censored: {censored_pct:.1f}%")

    # --- Features ---
    print(f"\n[3/6] Preparing features...")
    location_features = (
        list(NUMERIC_LOCATION_COLS.values())
        + list(BINARY_LOCATION_COLS.values())
        + list(CATEGORICAL_LOCATION_COLS.values())
    )

    # Model A: FEATURES_V2 + demand lags (same as exp 03 Model B)
    baseline_features = [f for f in FEATURES_V2 + DEMAND_FEATURES if f in df.columns]
    # Model B: + location features
    extended_features = baseline_features + [f for f in location_features if f in df.columns]

    print(f"  Model A (baseline + demand): {len(baseline_features)} features")
    print(f"  Model B (+ location):        {len(extended_features)} features "
          f"(+{len(extended_features) - len(baseline_features)} location)")

    for col in CATEGORICAL_COLS_V2:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # --- Train/test split ---
    print(f"\n[4/6] Train/test split (last {TEST_DAYS} days = test)...")
    test_start = df["Дата"].max() - pd.Timedelta(days=TEST_DAYS - 1)
    train = df[df["Дата"] < test_start].copy()
    test = df[df["Дата"] >= test_start].copy()

    print(f"  Train: {len(train):,} rows, {train['Дата'].nunique()} days")
    print(f"  Test:  {len(test):,} rows, {test['Дата'].nunique()} days")

    y_train = train[DEMAND_TARGET]
    y_test = test[DEMAND_TARGET]

    # --- Train Model A ---
    print(f"\n[5/6] Training models...")

    print(f"\n  --- Model A: baseline + demand ({len(baseline_features)} features) ---")
    t_a = time.time()
    model_a = train_lgbm(train[baseline_features], y_train)
    time_a = time.time() - t_a
    pred_a = predict_clipped(model_a, test[baseline_features])
    mae_a, wm_a, bias_a = print_metrics("Model A (baseline)", y_test, pred_a)
    print(f"    Train time: {time_a:.0f}s")

    # --- Train Model B ---
    print(f"\n  --- Model B: + location ({len(extended_features)} features) ---")
    t_b = time.time()
    model_b = train_lgbm(train[extended_features], y_train)
    time_b = time.time() - t_b
    pred_b = predict_clipped(model_b, test[extended_features])
    mae_b, wm_b, bias_b = print_metrics("Model B (+ location)", y_test, pred_b, baseline_mae=mae_a)
    print(f"    Train time: {time_b:.0f}s")

    # --- Comparison ---
    print(f"\n[6/6] Comparison (vs {DEMAND_TARGET})...")
    print(f"\n  {'Metric':<12} {'Model A':>10} {'Model B':>10} {'Delta':>10}")
    print(f"  {'-' * 44}")
    print(f"  {'MAE':<12} {mae_a:>10.4f} {mae_b:>10.4f} {mae_b - mae_a:>+10.4f}")
    print(f"  {'WMAPE':<12} {wm_a:>10.2f}% {wm_b:>10.2f}% {wm_b - wm_a:>+10.2f}%")
    print(f"  {'Bias':<12} {bias_a:>+10.4f} {bias_b:>+10.4f} {bias_b - bias_a:>+10.4f}")

    winner = "Model B (+ location)" if mae_b < mae_a else "Model A (baseline)"
    print(f"\n  Winner: {winner}")

    # High demand analysis
    print(f"\n  High demand analysis (vs {DEMAND_TARGET}):")
    for threshold in [15, 50, 100]:
        mask = y_test.values >= threshold
        if mask.sum() > 0:
            ma_a = mean_absolute_error(y_test.values[mask], pred_a[mask])
            ma_b = mean_absolute_error(y_test.values[mask], pred_b[mask])
            print(f"    >= {threshold}: N={mask.sum()}, "
                  f"A MAE={ma_a:.2f} | B MAE={ma_b:.2f} (delta {ma_b - ma_a:+.2f})")

    # Per-category
    print(f"\n  === Model A per-category ===")
    print_category_metrics(y_test, pred_a, test["Категория"].values)

    print(f"\n  === Model B per-category ===")
    print_category_metrics(y_test, pred_b, test["Категория"].values)

    # --- Feature importance (Model B -- location features) ---
    print(f"\n  === Location feature importance (Model B) ===")
    importance_b = pd.DataFrame({
        "feature": extended_features,
        "importance": model_b.feature_importances_,
    }).sort_values("importance", ascending=False)

    loc_imp = importance_b[importance_b["feature"].str.startswith("loc_")]
    print(f"    {'Feature':<30} {'Importance':>10} {'Rank':>6}")
    print(f"    {'-' * 48}")
    for rank, (_, row) in enumerate(importance_b.iterrows(), 1):
        if row["feature"].startswith("loc_"):
            print(f"    {row['feature']:<30} {row['importance']:>10.0f} {rank:>6}")

    # --- Save ---
    print(f"\n  Saving results...")
    metrics = {
        "experiment": "08_location_features",
        "target": DEMAND_TARGET,
        "model_a_baseline": {
            "mae": round(mae_a, 4),
            "wmape": round(wm_a, 2),
            "bias": round(bias_a, 4),
            "n_features": len(baseline_features),
            "train_time_s": round(time_a, 1),
        },
        "model_b_location": {
            "mae": round(mae_b, 4),
            "wmape": round(wm_b, 2),
            "bias": round(bias_b, 4),
            "n_features": len(extended_features),
            "train_time_s": round(time_b, 1),
        },
        "delta_mae": round(mae_b - mae_a, 4),
        "delta_wmape": round(wm_b - wm_a, 2),
        "winner": winner,
        "n_bakeries_matched": len(matched_bakeries),
        "train_rows": len(train),
        "test_rows": len(test),
        "censored_pct": round(censored_pct, 1),
        "location_feature_importance": loc_imp[["feature", "importance"]].to_dict("records"),
    }

    predictions = pd.DataFrame({
        "Дата": test["Дата"].values,
        "Пекарня": test["Пекарня"].values,
        "Номенклатура": test["Номенклатура"].values,
        "Категория": test["Категория"].values,
        "fact_demand": y_test.values,
        "pred_baseline": np.round(pred_a, 2),
        "pred_location": np.round(pred_b, 2),
        "is_censored": test["is_censored"].values,
        "abs_err_baseline": np.round(np.abs(y_test.values - pred_a), 2),
        "abs_err_location": np.round(np.abs(y_test.values - pred_b), 2),
    })

    save_results(EXP_DIR, metrics, predictions)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s")
    print("  Done!")


if __name__ == "__main__":
    main()

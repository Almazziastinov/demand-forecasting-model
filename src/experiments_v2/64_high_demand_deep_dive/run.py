"""
Experiment 64: High-demand item deep dive.

Proper evaluation on a specific high-demand item:
  Bakery: Баки Урманче 6 Казань
  Item:   Треугольник курица безд

Goals:
  1. Evaluate current best model (V3 + exp 61+62 features) on this item
  2. Compare global model vs local (single-item) model vs naive baselines
  3. Analyze weekly seasonality capture
  4. Understand WHERE the model fails (which days, what patterns)
  5. Visual diagnostics (plots saved to experiment folder)

Fixes vs previous notebook analysis:
  - No data leakage (no eval_set on test data)
  - Correct target: Спрос (not demand_estimated)
  - Proper temporal split (last 30 days)
  - Full feature set (FEATURES_V3 + exp 61+62)

Input:  data/processed/daily_sales_8m_demand.csv
Output: src/experiments_v2/64_high_demand_deep_dive/metrics.json
        src/experiments_v2/64_high_demand_deep_dive/plot_*.png

Usage:
  .venv\\Scripts\\python.exe src/experiments_v2/64_high_demand_deep_dive/run.py
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.config import TARGET, MODEL_PARAMS, TEST_DAYS
from src.experiments_v2.common import (
    DEMAND_8M_PATH, FEATURES_V3, CATEGORICAL_COLS_V2, DEMAND_TARGET,
    wmape, train_quantile, train_lgbm, predict_clipped, save_results,
)

EXP_DIR = Path(__file__).resolve().parent

EVAL_BAKERY = "Баки Урманче 6 Казань"
EVAL_ITEM = "Треугольник курица безд"
TEST_DAYS_LOCAL = 30

DOW_NAMES = ["Pn", "Vt", "Sr", "Ch", "Pt", "Sb", "Vs"]
DOW_NAMES_RU = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]


# -- Feature engineering (from exp 61 + 62) ------------------------------

def add_censoring_features(df):
    df = df.sort_values(["Пекарня", "Номенклатура", "Дата"]).copy()
    grp = df.groupby(["Пекарня", "Номенклатура"])
    df["is_censored_lag1"] = grp["is_censored"].shift(1).fillna(0).astype(int)
    df["lost_qty_lag1"] = grp["lost_qty"].shift(1).fillna(0)
    df["pct_censored_7d"] = (
        grp["is_censored"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean() * 100)
    ).fillna(0)
    return df

def add_dow_features(df):
    df = df.sort_values(["Пекарня", "Номенклатура", "Дата"]).copy()
    grp = df.groupby(["Пекарня", "Номенклатура", "ДеньНедели"])
    df["sales_dow_mean"] = (
        grp["Продано"].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    ).fillna(0)
    df["demand_dow_mean"] = (
        grp[DEMAND_TARGET].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    ).fillna(0)
    return df

def add_trend_features(df):
    df["demand_trend"] = df["demand_roll_mean7"] / (df["demand_roll_mean30"] + 1e-8)
    df["cv_7d"] = df["sales_roll_std7"] / (df["sales_roll_mean7"] + 1)
    return df

def add_assortment_features(df):
    df["items_in_bakery_today"] = (
        df.groupby(["Пекарня", "Дата"])["Номенклатура"].transform("nunique")
    )
    bakery_day = (
        df.groupby(["Пекарня", "Дата"])["Номенклатура"]
        .nunique().reset_index(name="__bakery_items")
        .sort_values(["Пекарня", "Дата"])
    )
    bakery_day["items_in_bakery_lag1"] = (
        bakery_day.groupby("Пекарня")["__bakery_items"].shift(1)
    )
    bakery_day["items_change"] = (
        bakery_day["__bakery_items"] - bakery_day["items_in_bakery_lag1"]
    )
    bakery_day.drop(columns=["__bakery_items"], inplace=True)
    df = df.merge(bakery_day, on=["Пекарня", "Дата"], how="left")
    for col in ["items_in_bakery_today", "items_in_bakery_lag1", "items_change"]:
        df[col] = df[col].fillna(0)
    return df

FEATURES_NEW = [
    "is_censored_lag1", "lost_qty_lag1", "pct_censored_7d",
    "sales_dow_mean", "demand_dow_mean",
    "demand_trend", "cv_7d",
    "items_in_bakery_today", "items_in_bakery_lag1", "items_change",
]

ALL_FEATURES = FEATURES_V3 + FEATURES_NEW


# -- Plotting helpers ----------------------------------------------------

def save_plot(fig, name):
    path = EXP_DIR / name
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path.name}")


def plot_predictions(dates, actual, preds_dict, title, filename):
    """Plot actual vs multiple model predictions."""
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(dates, actual, "ko-", label="Fakt", linewidth=2, markersize=5, zorder=10)

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    for i, (name, pred) in enumerate(preds_dict.items()):
        ax.plot(dates, pred, f"{colors[i % len(colors)]}", label=name,
                linewidth=1.5, marker="x", markersize=4, linestyle="--", alpha=0.8)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Data")
    ax.set_ylabel("Spros")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    fig.autofmt_xdate(rotation=45)
    save_plot(fig, filename)


def plot_errors_by_dow(test_dow, errors_dict, filename):
    """Bar chart of MAE by day of week for each model."""
    fig, ax = plt.subplots(figsize=(10, 5))
    n_models = len(errors_dict)
    width = 0.8 / n_models
    x = np.arange(7)

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    for i, (name, errors) in enumerate(errors_dict.items()):
        mae_by_dow = []
        for dow in range(7):
            mask = test_dow == dow
            if mask.sum() > 0:
                mae_by_dow.append(np.abs(errors[mask]).mean())
            else:
                mae_by_dow.append(0)
        ax.bar(x + i * width, mae_by_dow, width, label=name,
               color=colors[i % len(colors)], alpha=0.8)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(DOW_NAMES)
    ax.set_ylabel("MAE")
    ax.set_title("MAE po dnyam nedeli")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    save_plot(fig, filename)


def plot_weekly_pattern(item_df, target_col, filename):
    """Box plot of demand by day of week (full history)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    data_by_dow = [
        item_df[item_df["ДеньНедели"] == dow][target_col].values
        for dow in range(7)
    ]
    bp = ax.boxplot(data_by_dow, labels=DOW_NAMES, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#3498db")
        patch.set_alpha(0.5)
    ax.set_ylabel("Spros")
    ax.set_title(f"Nedeljnyj pattern: {EVAL_ITEM} ({EVAL_BAKERY})")
    ax.grid(axis="y", alpha=0.3)
    save_plot(fig, filename)


def plot_full_history_with_test(item_df, test_start, pred_q50, pred_dow_mean, filename):
    """Full time series with test period highlighted."""
    fig, ax = plt.subplots(figsize=(18, 6))

    train_part = item_df[item_df["Дата"] < test_start]
    test_part = item_df[item_df["Дата"] >= test_start]

    ax.plot(train_part["Дата"], train_part[DEMAND_TARGET], "b-", alpha=0.5,
            linewidth=0.8, label="Train (fakt)")
    ax.plot(test_part["Дата"], test_part[DEMAND_TARGET].values, "ko-",
            linewidth=2, markersize=4, label="Test (fakt)")
    ax.plot(test_part["Дата"], pred_q50, "r--x", linewidth=1.5,
            markersize=4, label="Global Q50", alpha=0.8)
    ax.plot(test_part["Дата"], pred_dow_mean, "g--+", linewidth=1.5,
            markersize=4, label="DOW mean 4w", alpha=0.8)

    ax.axvline(test_start, color="black", linestyle=":", linewidth=1, label="Train/Test split")
    ax.set_title(f"Polnyj ryad: {EVAL_ITEM} ({EVAL_BAKERY})", fontsize=13)
    ax.set_xlabel("Data")
    ax.set_ylabel("Spros")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate(rotation=45)
    save_plot(fig, filename)


def plot_error_distribution(errors_dict, filename):
    """Histogram of prediction errors for each model."""
    n = len(errors_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    for i, (name, errors) in enumerate(errors_dict.items()):
        axes[i].hist(errors, bins=15, color=colors[i % len(colors)], alpha=0.7, edgecolor="black")
        axes[i].axvline(0, color="black", linestyle="--", linewidth=1)
        axes[i].set_title(f"{name}\nmean={errors.mean():+.1f}, std={errors.std():.1f}")
        axes[i].set_xlabel("Oshibka (fakt - prognoz)")
    axes[0].set_ylabel("Chastota")
    fig.suptitle("Raspredelenie oshibok", fontsize=13)
    fig.tight_layout()
    save_plot(fig, filename)


# -- Main ----------------------------------------------------------------

def main():
    print("=" * 70)
    print(f"  EXPERIMENT 64: High-Demand Item Deep Dive")
    print(f"  Bakery: {EVAL_BAKERY}")
    print(f"  Item:   {EVAL_ITEM}")
    print(f"  Test:   last {TEST_DAYS_LOCAL} days")
    print("=" * 70)
    t_start = time.time()

    # [1/8] Load
    print(f"\n[1/8] Loading data...")
    df = pd.read_csv(str(DEMAND_8M_PATH), encoding="utf-8-sig")
    df["Дата"] = pd.to_datetime(df["Дата"])
    print(f"  Full dataset: {df.shape}")

    # [2/8] Build features
    print(f"\n[2/8] Building features (exp 61 + 62)...")
    df = add_censoring_features(df)
    df = add_dow_features(df)
    df = add_trend_features(df)
    df = add_assortment_features(df)

    for col in CATEGORICAL_COLS_V2:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # [3/8] Extract target item
    print(f"\n[3/8] Extracting target item...")
    item_df = df[(df["Пекарня"] == EVAL_BAKERY) & (df["Номенклатура"] == EVAL_ITEM)].copy()
    item_df = item_df.sort_values("Дата").reset_index(drop=True)
    print(f"  Item data: {len(item_df)} days")
    print(f"  Date range: {item_df['Дата'].min().date()} -- {item_df['Дата'].max().date()}")
    print(f"  Demand: mean={item_df[DEMAND_TARGET].mean():.1f}, "
          f"std={item_df[DEMAND_TARGET].std():.1f}, "
          f"CV={item_df[DEMAND_TARGET].std() / item_df[DEMAND_TARGET].mean():.3f}")

    # Weekly pattern
    print(f"\n  Nedeljnyj pattern (mean spros):")
    dow_stats = item_df.groupby("ДеньНедели")[DEMAND_TARGET].agg(["mean", "std", "count"])
    for dow in range(7):
        if dow in dow_stats.index:
            row = dow_stats.loc[dow]
            bar = "#" * int(row["mean"] / 8)
            print(f"    {DOW_NAMES[dow]}: {row['mean']:>7.1f} +/- {row['std']:>5.1f} "
                  f"(n={row['count']:>3.0f}) {bar}")

    cens_pct = item_df["is_censored"].mean() * 100
    print(f"\n  Censored days: {cens_pct:.1f}%")

    # [4/8] Temporal split
    print(f"\n[4/8] Temporal split...")
    test_start_date = item_df["Дата"].iloc[-TEST_DAYS_LOCAL]

    global_train = df[df["Дата"] < test_start_date].copy()
    item_test = item_df[item_df["Дата"] >= test_start_date].copy()

    print(f"  Global train: {len(global_train):,} rows, {global_train['Дата'].nunique()} days")
    print(f"  Item test:    {len(item_test)} rows ({item_test['Дата'].min().date()} -- "
          f"{item_test['Дата'].max().date()})")

    available = [f for f in ALL_FEATURES if f in df.columns]
    print(f"  Features: {len(available)}")

    X_global_train = global_train[available]
    y_global_train = global_train[DEMAND_TARGET]
    X_item_test = item_test[available]
    y_item_test = item_test[DEMAND_TARGET].values
    dates_test = item_test["Дата"].values

    # [5/8] Train models
    print(f"\n[5/8] Training models...")

    # Model 1: Global Quantile P50
    print(f"\n  --- Global Quantile P50 ---")
    t0 = time.time()
    model_q50 = train_quantile(X_global_train, y_global_train, alpha=0.5)
    print(f"    Time: {time.time() - t0:.0f}s")
    pred_q50 = predict_clipped(model_q50, X_item_test)

    # Model 2: Global MSE
    print(f"  --- Global MSE ---")
    t0 = time.time()
    model_mse = train_lgbm(X_global_train, y_global_train)
    print(f"    Time: {time.time() - t0:.0f}s")
    pred_mse = predict_clipped(model_mse, X_item_test)

    # Model 3: Local bakery Q50
    print(f"  --- Local bakery Q50 ---")
    bakery_train = global_train[global_train["Пекарня"] == EVAL_BAKERY].copy()
    local_feats = [f for f in available if f != "Пекарня"]
    t0 = time.time()
    model_local = train_quantile(bakery_train[local_feats], bakery_train[DEMAND_TARGET], alpha=0.5)
    print(f"    Train rows: {len(bakery_train):,}, Time: {time.time() - t0:.0f}s")
    pred_local = predict_clipped(model_local, item_test[local_feats])

    # Model 4: Item-only Q50
    print(f"  --- Item-only Q50 ---")
    item_train = item_df[item_df["Дата"] < test_start_date].copy()
    item_feats = [f for f in available if f not in ["Пекарня", "Номенклатура"]]
    t0 = time.time()
    model_item = train_quantile(item_train[item_feats], item_train[DEMAND_TARGET], alpha=0.5)
    print(f"    Train rows: {len(item_train)}, Time: {time.time() - t0:.0f}s")
    pred_item = predict_clipped(model_item, item_test[item_feats])

    # Model 5: Naive (last same DOW)
    print(f"  --- Naive baselines ---")
    pred_naive = []
    pred_dow_mean = []
    for _, row in item_test.iterrows():
        same_dow = item_df[
            (item_df["ДеньНедели"] == row["ДеньНедели"]) &
            (item_df["Дата"] < row["Дата"])
        ][DEMAND_TARGET]
        fallback = item_df[item_df["Дата"] < row["Дата"]][DEMAND_TARGET].mean()
        pred_naive.append(same_dow.iloc[-1] if len(same_dow) >= 1 else fallback)
        pred_dow_mean.append(same_dow.tail(4).mean() if len(same_dow) >= 1 else fallback)
    pred_naive = np.array(pred_naive)
    pred_dow_mean = np.array(pred_dow_mean)

    # [6/8] Evaluate
    print(f"\n[6/8] Evaluation...")

    models_preds = {
        "Global Q50": pred_q50,
        "Global MSE": pred_mse,
        "Local bakery Q50": pred_local,
        "Item-only Q50": pred_item,
        "Naive (last DOW)": pred_naive,
        "DOW mean (4w)": pred_dow_mean,
    }

    results = {}
    print(f"\n  {'Model':<22} {'MAE':>7} {'RMSE':>7} {'R2':>6} {'Bias':>8} {'WMAPE':>7}")
    print(f"  {'-' * 60}")

    for name, pred in models_preds.items():
        mae = mean_absolute_error(y_item_test, pred)
        rmse = np.sqrt(mean_squared_error(y_item_test, pred))
        r2 = r2_score(y_item_test, pred)
        bias = float(np.mean(y_item_test - pred))
        wm = np.sum(np.abs(y_item_test - pred)) / (np.sum(y_item_test) + 1e-8) * 100

        results[name] = {
            "mae": round(mae, 2), "rmse": round(rmse, 2),
            "r2": round(r2, 4), "bias": round(bias, 2), "wmape": round(wm, 2),
        }
        print(f"  {name:<22} {mae:>7.2f} {rmse:>7.2f} {r2:>6.3f} {bias:>+8.2f} {wm:>6.2f}%")

    # Day-by-day
    print(f"\n  === Den za dnem: Fakt vs Global Q50 vs DOW mean ===")
    print(f"  {'Data':<12} {'DOW':<4} {'Fakt':>7} {'Q50':>7} {'Err':>8} "
          f"{'DOW4w':>7} {'Err':>8} {'Cens':>5}")
    print(f"  {'-' * 68}")

    for i, (_, row) in enumerate(item_test.iterrows()):
        actual = y_item_test[i]
        q50 = pred_q50[i]
        dow = pred_dow_mean[i]
        cens = "DA" if row["is_censored"] == 1 else ""
        dow_name = DOW_NAMES[int(row["ДеньНедели"])]
        print(f"  {str(row['Дата'].date()):<12} {dow_name:<4} {actual:>7.1f} {q50:>7.1f} "
              f"{actual - q50:>+8.1f} {dow:>7.1f} {actual - dow:>+8.1f} {cens:>5}")

    # Error by DOW
    print(f"\n  === Oshibka po dnyam nedeli (Global Q50) ===")
    test_dow = item_test["ДеньНедели"].values
    errors_q50 = y_item_test - pred_q50

    print(f"  {'DOW':<4} {'Bias':>9} {'MAE':>7} {'N':>4}")
    print(f"  {'-' * 28}")
    for dow in range(7):
        mask = test_dow == dow
        if mask.sum() > 0:
            me = errors_q50[mask].mean()
            mae_d = np.abs(errors_q50[mask]).mean()
            print(f"  {DOW_NAMES[dow]:<4} {me:>+9.1f} {mae_d:>7.1f} {mask.sum():>4}")

    # Feature importance (item-only)
    print(f"\n  === Feature importance (item-only model, top 15) ===")
    importance = pd.DataFrame({
        "feature": item_feats,
        "importance": model_item.feature_importances_,
    }).sort_values("importance", ascending=False)
    for _, row in importance.head(15).iterrows():
        print(f"    {row['feature']:<30} {row['importance']:>6.0f}")

    # [7/8] Plots
    print(f"\n[7/8] Saving plots...")

    # Plot 1: Full history + test predictions
    plot_full_history_with_test(item_df, test_start_date, pred_q50, pred_dow_mean,
                                "plot_01_full_history.png")

    # Plot 2: Test period — all models
    plot_predictions(
        dates_test, y_item_test,
        {"Global Q50": pred_q50, "Global MSE": pred_mse,
         "Item-only Q50": pred_item, "DOW mean 4w": pred_dow_mean},
        f"Test period: {EVAL_ITEM} ({EVAL_BAKERY})",
        "plot_02_test_predictions.png",
    )

    # Plot 3: Weekly demand pattern (box plot)
    plot_weekly_pattern(item_df, DEMAND_TARGET, "plot_03_weekly_pattern.png")

    # Plot 4: Errors by DOW
    plot_errors_by_dow(
        test_dow.astype(int),
        {"Global Q50": y_item_test - pred_q50,
         "DOW mean 4w": y_item_test - pred_dow_mean,
         "Item-only Q50": y_item_test - pred_item},
        "plot_04_errors_by_dow.png",
    )

    # Plot 5: Error distributions
    plot_error_distribution(
        {"Global Q50": y_item_test - pred_q50,
         "Global MSE": y_item_test - pred_mse,
         "DOW mean 4w": y_item_test - pred_dow_mean},
        "plot_05_error_distribution.png",
    )

    # [8/8] Save
    print(f"\n[8/8] Saving results...")
    metrics = {
        "experiment": "64_high_demand_deep_dive",
        "eval_bakery": EVAL_BAKERY,
        "eval_item": EVAL_ITEM,
        "test_days": TEST_DAYS_LOCAL,
        "item_stats": {
            "n_days": len(item_df),
            "mean_demand": round(item_df[DEMAND_TARGET].mean(), 1),
            "std_demand": round(item_df[DEMAND_TARGET].std(), 1),
            "cv": round(item_df[DEMAND_TARGET].std() / item_df[DEMAND_TARGET].mean(), 3),
            "censored_pct": round(cens_pct, 1),
        },
        "results": results,
        "best_model": min(results, key=lambda k: results[k]["mae"]),
    }
    save_results(EXP_DIR, metrics)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s")
    print("  Done!")


if __name__ == "__main__":
    main()

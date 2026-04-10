"""
Experiment 23: Cluster-based models

Based on Dodo's approach: cluster products by behavior (not by name).
One model per cluster instead of 610 separate models or 1 global.

Steps:
1. Extract behavior profile for each (bakery, product) time series
2. K-Means clustering
3. Train separate LightGBM model per cluster
4. Compare with baseline (single global model)
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, silhouette_score
import json

from src.experiments_v2.common import (
    DEMAND_8M_PATH,
    FEATURES_V3,
    DEMAND_TARGET,
    CATEGORICAL_COLS_V2,
    TEST_DAYS,
    train_quantile,
    predict_clipped,
    wmape,
)

# Use demand dataset (with corrected target)
DATA_PATH = DEMAND_8M_PATH

print("=" * 60)
print("EXPERIMENT 23: Cluster-based Models (Dodo approach)")
print("=" * 60)

# ============================================================
# [1/6] Load data
# ============================================================
print("\n[1/6] Loading data...")
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
print(f"  Shape: {df.shape}")
print(f"  Date range: {df['Дата'].min()} -- {df['Дата'].max()}")

# Convert date to datetime
df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce")

# ============================================================
# [2/6] Extract behavior profile per (bakery, product)
# ============================================================
print("\n[2/6] Extracting behavior profiles for clustering...")

# Aggregate to (bakery, product, date) level for clustering
cluster_df = df.groupby(["Пекарня", "Номенклатура"]).agg(
    demand_mean=("Спрос", "mean"),
    demand_std=("Спрос", "std"),
    demand_median=("Спрос", "median"),
    sales_mean=("Продано", "mean"),
    sales_std=("Продано", "std"),
    n_days=("Дата", "count"),
    # DOW pattern: std of daily totals per day of week
    dow_std=("Продано", lambda x: x.groupby(df.loc[x.index, "ДеньНедели"]).sum().std()),
).reset_index()

# Fill NaN for products with 0 or 1 day
cluster_df["demand_std"] = cluster_df["demand_std"].fillna(0)
cluster_df["sales_std"] = cluster_df["sales_std"].fillna(0)
cluster_df["dow_std"] = cluster_df["dow_std"].fillna(0)

# Coefficient of variation
cluster_df["cv"] = cluster_df["demand_std"] / (cluster_df["demand_mean"] + 1e-8)

# Demand trend (ratio of last week vs all time)
last_week = df[df["Дата"] >= df["Дата"].max() - pd.Timedelta(days=7)]
last_week_agg = last_week.groupby(["Пекарня", "Номенклатура"])["Спрос"].mean().reset_index()
last_week_agg.columns = ["Пекарня", "Номенклатура", "last_week_mean"]

cluster_df = cluster_df.merge(last_week_agg, on=["Пекарня", "Номенклатура"], how="left")
cluster_df["last_week_mean"] = cluster_df["last_week_mean"].fillna(cluster_df["demand_mean"])
cluster_df["trend"] = cluster_df["last_week_mean"] / (cluster_df["demand_mean"] + 1e-8)

# Clip extreme trends
cluster_df["trend"] = cluster_df["trend"].clip(0.5, 2.0)

print(f"  Unique (bakery, product) pairs: {len(cluster_df)}")
print(f"  Demand mean distribution: min={cluster_df['demand_mean'].min():.1f}, "
      f"median={cluster_df['demand_mean'].median():.1f}, max={cluster_df['demand_mean'].max():.1f}")
print(f"  CV distribution: min={cluster_df['cv'].min():.2f}, "
      f"median={cluster_df['cv'].median():.2f}, max={cluster_df['cv'].max():.2f}")

# ============================================================
# [3/6] K-Means clustering
# ============================================================
print("\n[3/6] K-Means clustering...")

# Features for clustering
cluster_features = ["demand_mean", "cv", "trend", "dow_std", "n_days"]
X_cluster = cluster_df[cluster_features].values

# Log transform for skewed features
X_cluster[:, 0] = np.log1p(X_cluster[:, 0])  # demand_mean
X_cluster[:, 4] = np.log1p(X_cluster[:, 4])  # n_days

# Standardize
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

# Try different K
best_k = 5
best_score = -np.inf

for k in [3, 5, 7, 10]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_cluster_scaled)
    
    # Silhouette score (higher is better)
    from sklearn.metrics import silhouette_score
    if len(np.unique(labels)) > 1:
        score = silhouette_score(X_cluster_scaled, labels)
        print(f"  K={k}: silhouette={score:.3f}")
        if score > best_score:
            best_score = score
            best_k = k

print(f"\n  Using K={best_k} clusters")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_df["cluster"] = kmeans.fit_predict(X_cluster_scaled)

print("\n  Cluster distribution:")
cluster_sizes = cluster_df.groupby("cluster").agg(
    n_pairs=("Пекарня", "count"),
    avg_demand=("demand_mean", "mean"),
    avg_cv=("cv", "mean"),
).reset_index()
for _, row in cluster_sizes.iterrows():
    print(f"    Cluster {int(row['cluster'])}: {int(row['n_pairs'])} pairs, "
          f"avg_demand={row['avg_demand']:.1f}, avg_cv={row['avg_cv']:.2f}")

# ============================================================
# [4/6] Prepare train/test split
# ============================================================
print("\n[4/6] Preparing train/test split...")

df["Дата"] = pd.to_datetime(df["Дата"])
cutoff_date = df["Дата"].max() - pd.Timedelta(days=TEST_DAYS)

train_df = df[df["Дата"] <= cutoff_date].copy()
test_df = df[df["Дата"] > cutoff_date].copy()

print(f"  Train: {len(train_df):,} rows, {train_df['Дата'].nunique()} days")
print(f"  Test: {len(test_df):,} rows, {test_df['Дата'].nunique()} days")

# Add cluster info to train/test
bakery_product_cluster = cluster_df[["Пекарня", "Номенклатура", "cluster"]]
train_df = train_df.merge(bakery_product_cluster, on=["Пекарня", "Номенклатура"], how="left")
test_df = test_df.merge(bakery_product_cluster, on=["Пекарня", "Номенклатура"], how="left")

# ============================================================
# [5/6] Train baseline (global model) and cluster models
# ============================================================
print("\n[5/6] Training models...")

# ---- Baseline: single global model ----
print("\n  --- Baseline (global model) ---")
features = [f for f in FEATURES_V3 if f in df.columns]
cat_cols = [c for c in CATEGORICAL_COLS_V2 if c in features]

X_train = train_df[features].copy()
y_train = train_df[DEMAND_TARGET].copy()
X_test = test_df[features].copy()
y_test = test_df[DEMAND_TARGET].copy()

for col in cat_cols:
    X_train[col] = X_train[col].astype("category")
    X_test[col] = X_test[col].astype("category")

baseline_model = train_quantile(X_train, y_train, alpha=0.5)
baseline_preds = predict_clipped(baseline_model, X_test)
baseline_mae = mean_absolute_error(y_test, baseline_preds)
baseline_wmape = wmape(y_test, baseline_preds)
print(f"    Baseline MAE: {baseline_mae:.4f}, WMAPE: {baseline_wmape:.2%}")

# ---- Cluster models ----
print("\n  --- Cluster models ---")
cluster_maes = []
cluster_models = {}

for cluster_id in sorted(cluster_df["cluster"].unique()):
    cluster_train = train_df[train_df["cluster"] == cluster_id]
    cluster_test = test_df[test_df["cluster"] == cluster_id]
    
    n_train = len(cluster_train)
    n_test = len(cluster_test)
    
    if n_test < 100:
        print(f"    Cluster {cluster_id}: skipping (only {n_test} test rows)")
        continue
    
    print(f"    Cluster {cluster_id}: train={n_train:,}, test={n_test:,}")
    
    X_train_c = cluster_train[features].copy()
    y_train_c = cluster_train[DEMAND_TARGET].copy()
    X_test_c = cluster_test[features].copy()
    y_test_c = cluster_test[DEMAND_TARGET].copy()
    
    for col in cat_cols:
        X_train_c[col] = X_train_c[col].astype("category")
        X_test_c[col] = X_test_c[col].astype("category")
    
    # Train cluster model
    model_c = train_quantile(X_train_c, y_train_c, alpha=0.5)
    preds_c = predict_clipped(model_c, X_test_c)
    mae_c = mean_absolute_error(y_test_c, preds_c)
    
    cluster_maes.append({"cluster": cluster_id, "mae": mae_c, "n_test": n_test})
    cluster_models[cluster_id] = model_c
    print(f"      MAE: {mae_c:.4f}")

# ============================================================
# [6/6] Compare: global vs cluster-based
# ============================================================
print("\n[6/6] Evaluating...")

# For cluster-based: predict using cluster-specific models
test_df = test_df.copy()
test_df["cluster_pred"] = np.nan

for cluster_id, model_c in cluster_models.items():
    mask = test_df["cluster"] == cluster_id
    if mask.sum() > 0:
        X_test_c = test_df.loc[mask, features].copy()
        for col in cat_cols:
            X_test_c[col] = X_test_c[col].astype("category")
        test_df.loc[mask, "cluster_pred"] = predict_clipped(model_c, X_test_c)

# For rows without cluster (shouldn't happen), use baseline
test_df["cluster_pred"] = test_df["cluster_pred"].fillna(pd.Series(baseline_preds, index=test_df.index))

cluster_mae = mean_absolute_error(y_test, test_df["cluster_pred"])
cluster_wmape = wmape(y_test, test_df["cluster_pred"])

print(f"\n  === RESULTS ===")
print(f"  Baseline (global): MAE={baseline_mae:.4f}, WMAPE={baseline_wmape:.2%}")
print(f"  Cluster-based:      MAE={cluster_mae:.4f}, WMAPE={cluster_wmape:.2%}")
print(f"  Delta: MAE={cluster_mae - baseline_mae:+.4f}")

# Per-cluster breakdown
print(f"\n  Per-cluster MAE:")
for item in cluster_maes:
    print(f"    Cluster {item['cluster']}: MAE={item['mae']:.4f} (n_test={item['n_test']:,})")

# Save results
results = {
    "experiment": 23,
    "name": "Cluster-based models (Dodo approach)",
    "baseline_mae": baseline_mae,
    "baseline_wmape": baseline_wmape,
    "cluster_mae": cluster_mae,
    "cluster_wmape": cluster_wmape,
    "delta_mae": cluster_mae - baseline_mae,
    "k": best_k,
    "cluster_sizes": cluster_sizes.to_dict(),
    "cluster_maes": cluster_maes,
}

os.makedirs("src/experiments_v2/23_cluster_models", exist_ok=True)
with open("src/experiments_v2/23_cluster_models/metrics.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n  Results saved to metrics.json")
print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
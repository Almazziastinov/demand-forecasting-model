"""
Experiment C: Store Clustering
Compute per-store features, KMeans clustering, per-cluster LightGBM models.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error

from src.experiments.common import (
    load_data, MODEL_PARAMS, BASELINE_MAE,
    wmape, print_metrics, print_category_metrics,
    train_lgbm, predict_clipped, save_predictions,
)


def compute_store_features(df):
    """Compute per-store aggregate features for clustering."""
    store_stats = df.groupby("Пекарня").agg(
        mean_sales=("Продано", "mean"),
        std_sales=("Продано", "std"),
        median_sales=("Продано", "median"),
        n_products=("Номенклатура", "nunique"),
        n_days=("Дата", "nunique"),
        total_sales=("Продано", "sum"),
        city=("Город", "first"),
    ).reset_index()
    store_stats["std_sales"] = store_stats["std_sales"].fillna(0)
    store_stats["cv_sales"] = store_stats["std_sales"] / (store_stats["mean_sales"] + 1e-8)
    return store_stats


def main():
    print("=" * 65)
    print("  EXPERIMENT C: STORE CLUSTERING + PER-CLUSTER MODELS")
    print("=" * 65)

    df, X_train, y_train, X_test, y_test, features = load_data()

    # --- Baseline ---
    print("\n--- Baseline (global model, v6-best) ---")
    bl_model = train_lgbm(X_train, y_train)
    bl_pred = predict_clipped(bl_model, X_test)
    bl_mae, _, _ = print_metrics("Baseline", y_test, bl_pred)

    # --- Store features & clustering ---
    print("\n--- Store clustering ---")
    train_df = df[df["Дата"] < (df["Дата"].max() - pd.Timedelta(days=2))].copy()
    store_stats = compute_store_features(train_df)

    # Encode city for clustering
    le_city = LabelEncoder()
    store_stats["city_code"] = le_city.fit_transform(store_stats["city"])

    cluster_features = ["mean_sales", "std_sales", "cv_sales", "n_products", "city_code"]
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(store_stats[cluster_features])

    # Try k=5..8, pick best silhouette
    from sklearn.metrics import silhouette_score
    best_k, best_score = 5, -1
    for k in range(5, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_cluster)
        score = silhouette_score(X_cluster, labels)
        print(f"  k={k}: silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score = k, score

    print(f"  -> Luchshee k={best_k} (silhouette={best_score:.4f})")

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    store_stats["cluster"] = km_final.fit_predict(X_cluster)

    # Cluster summary
    print(f"\n  Raspredelenie po klasteram:")
    for cl in range(best_k):
        cl_stores = store_stats[store_stats["cluster"] == cl]
        cities = cl_stores["city"].value_counts().head(3)
        cities_str = ", ".join([f"{c}({n})" for c, n in cities.items()])
        print(f"    Cluster {cl}: {len(cl_stores)} pekarni, "
              f"mean_sales={cl_stores['mean_sales'].mean():.1f}, "
              f"goroda: {cities_str}")

    # Map cluster to stores
    store_to_cluster = dict(zip(store_stats["Пекарня"], store_stats["cluster"]))

    # Add cluster to train/test
    train_cluster = X_train["Пекарня"].map(store_to_cluster)
    test_cluster = X_test["Пекарня"].map(store_to_cluster)

    # Handle stores in test that weren't in train (assign to nearest cluster)
    unknown_mask = test_cluster.isna()
    if unknown_mask.any():
        print(f"  [!] {unknown_mask.sum()} strok v teste s neizvestnymi pekarnyami -> cluster 0")
        test_cluster = test_cluster.fillna(0).astype(int)

    # --- Per-cluster models ---
    print("\n--- Per-cluster models ---")
    y_pred_combined = np.zeros(len(y_test))

    for cl in range(best_k):
        train_mask = train_cluster == cl
        test_mask = test_cluster == cl

        X_tr_cl = X_train[train_mask]
        y_tr_cl = y_train[train_mask]
        X_te_cl = X_test[test_mask]
        y_te_cl = y_test[test_mask]

        if len(X_te_cl) == 0:
            print(f"  Cluster {cl}: net dannykh v teste, propuskayem")
            continue

        model = train_lgbm(X_tr_cl, y_tr_cl)
        pred = predict_clipped(model, X_te_cl)

        mae_cl = mean_absolute_error(y_te_cl, pred)
        bl_pred_cl = bl_pred[test_mask.values]
        bl_mae_cl = mean_absolute_error(y_te_cl, bl_pred_cl)
        delta = mae_cl - bl_mae_cl

        print(f"  Cluster {cl}: train={len(X_tr_cl):,} test={len(X_te_cl):,}  "
              f"MAE={mae_cl:.4f} (baseline={bl_mae_cl:.4f}, delta={delta:+.4f})")

        y_pred_combined[test_mask.values] = pred

    # --- Overall metrics ---
    print("\n--- Itogo ---")
    overall_mae, overall_wmape, overall_bias = print_metrics(
        "Store Clusters", y_test, y_pred_combined
    )

    print("\n--- Po kategoriyam ---")
    print_category_metrics(y_test, y_pred_combined, X_test["Категория"])

    print(f"\n  Baseline MAE:      {bl_mae:.4f}")
    print(f"  Store Cluster MAE: {overall_mae:.4f}")
    print(f"  Delta:             {overall_mae - bl_mae:+.4f}")

    # Save
    save_predictions(
        X_test, y_test, y_pred_combined,
        "reports/exp_c_predictions.csv",
        extra_cols={
            "pred_baseline": bl_pred,
            "cluster": test_cluster.values,
        },
    )

    summary = pd.DataFrame([{
        "experiment": "C_store_clusters",
        "mae": overall_mae,
        "wmape": overall_wmape,
        "bias": overall_bias,
        "baseline_mae": BASELINE_MAE,
        "delta": overall_mae - BASELINE_MAE,
    }])
    summary.to_csv("reports/exp_c_summary.csv", index=False)

    print("\nGotovo!")


if __name__ == "__main__":
    main()

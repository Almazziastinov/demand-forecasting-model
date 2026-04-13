"""
Experiment 66: cluster features on top of exp 63.

Variants:
  A. exp 63 combined baseline
  B. A + cluster_loc
  C. A + cluster_ts
  D. A + cluster_loc + cluster_ts
  E. Routed model by cluster_ts
"""

import os
import sys
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import TEST_DAYS
from src.experiments_v2.common import (
    CATEGORICAL_COLS_V2,
    DEMAND_8M_PATH,
    DEMAND_TARGET,
    FEATURES_V3,
    SALES_HRS_PATH,
    predict_clipped,
    print_category_metrics,
    save_results,
    train_quantile,
    wmape,
)


EXP_DIR = Path(__file__).resolve().parent
LOCATION_PATH = Path(ROOT) / "data" / "raw" / (
    "анализ локаций "
    "по БП 02.03 ( 1) (2) (4) (1) (1).xlsx"
)

DATE_COL = "Дата"
BAKERY_COL = "Пекарня"
PRODUCT_COL = "Номенклатура"
CATEGORY_COL = "Категория"
CITY_COL = "Город"
SOLD_COL = "Продано"
DOW_COL = "ДеньНедели"
FRESHNESS_COL = "Свежесть"
STALE_VALUE = "Вчерашний"

RAW_DATE_COL = "Дата продажи"
RAW_BAKERY_COL = "Касса.Торговая точка"
RAW_EVENT_COL = "Вид события по кассе"
RAW_SALES_VALUE = "Продажа"
RAW_QTY_COL = "Кол-во"

BASELINE_V3_MAE = 2.8816
EXP63_MAE = 2.8540


def add_censoring_features(df):
    df = df.sort_values([BAKERY_COL, PRODUCT_COL, DATE_COL]).copy()
    grp = df.groupby([BAKERY_COL, PRODUCT_COL])
    df["is_censored_lag1"] = grp["is_censored"].shift(1).fillna(0).astype(int)
    df["lost_qty_lag1"] = grp["lost_qty"].shift(1).fillna(0)
    df["pct_censored_7d"] = (
        grp["is_censored"].transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean() * 100)
    ).fillna(0)
    return df


def add_dow_features(df):
    df = df.sort_values([BAKERY_COL, PRODUCT_COL, DATE_COL]).copy()
    grp = df.groupby([BAKERY_COL, PRODUCT_COL, DOW_COL])
    df["sales_dow_mean"] = grp[SOLD_COL].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    ).fillna(0)
    df["demand_dow_mean"] = grp[DEMAND_TARGET].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    ).fillna(0)
    return df


def add_trend_features(df):
    df["demand_trend"] = df["demand_roll_mean7"] / (df["demand_roll_mean30"] + 1e-8)
    df["cv_7d"] = df["sales_roll_std7"] / (df["sales_roll_mean7"] + 1)
    return df


def build_stale_ratio(df):
    if not Path(SALES_HRS_PATH).exists():
        print(f"    WARNING: {SALES_HRS_PATH} not found, skipping stale_ratio")
        df["stale_ratio_lag1"] = 0.0
        return df

    print("    Parsing raw checks for stale_ratio (chunked)...")
    usecols = [RAW_DATE_COL, RAW_BAKERY_COL, PRODUCT_COL, FRESHNESS_COL, RAW_QTY_COL, RAW_EVENT_COL]
    agg_list = []

    for i, chunk in enumerate(
        pd.read_csv(str(SALES_HRS_PATH), encoding="utf-8-sig", usecols=usecols, chunksize=2_000_000)
    ):
        sales = chunk[chunk[RAW_EVENT_COL] == RAW_SALES_VALUE].copy()
        sales["is_stale"] = (sales[FRESHNESS_COL] == STALE_VALUE).astype(int)
        sales["stale_qty"] = sales["is_stale"] * sales[RAW_QTY_COL]
        agg = (
            sales.groupby([RAW_DATE_COL, RAW_BAKERY_COL, PRODUCT_COL])
            .agg(total_qty=(RAW_QTY_COL, "sum"), stale_qty=("stale_qty", "sum"))
            .reset_index()
        )
        agg_list.append(agg)
        if (i + 1) % 5 == 0:
            print(f"      chunk {i + 1}...")

    stale = pd.concat(agg_list, ignore_index=True)
    stale = (
        stale.groupby([RAW_DATE_COL, RAW_BAKERY_COL, PRODUCT_COL])
        .agg(total_qty=("total_qty", "sum"), stale_qty=("stale_qty", "sum"))
        .reset_index()
    )
    stale["stale_ratio"] = stale["stale_qty"] / (stale["total_qty"] + 1e-8)
    stale.rename(columns={RAW_DATE_COL: DATE_COL, RAW_BAKERY_COL: BAKERY_COL}, inplace=True)
    stale[DATE_COL] = pd.to_datetime(stale[DATE_COL], format="%d.%m.%Y", errors="coerce")
    stale = stale[[DATE_COL, BAKERY_COL, PRODUCT_COL, "stale_ratio"]].dropna(subset=[DATE_COL])

    # Compute lag on the compact stale table to avoid sorting/copying the full 2.6M-row dataset.
    stale = stale.sort_values([BAKERY_COL, PRODUCT_COL, DATE_COL])
    stale["stale_ratio_lag1"] = (
        stale.groupby([BAKERY_COL, PRODUCT_COL])["stale_ratio"].shift(1).fillna(0)
    )
    stale = stale[[DATE_COL, BAKERY_COL, PRODUCT_COL, "stale_ratio_lag1"]]

    df = df.merge(stale, on=[DATE_COL, BAKERY_COL, PRODUCT_COL], how="left")
    df["stale_ratio_lag1"] = df["stale_ratio_lag1"].fillna(0)
    return df


def add_assortment_features(df):
    df["items_in_bakery_today"] = df.groupby([BAKERY_COL, DATE_COL])[PRODUCT_COL].transform("nunique")
    bakery_day = (
        df.groupby([BAKERY_COL, DATE_COL])[PRODUCT_COL]
        .nunique()
        .reset_index(name="__bakery_items")
        .sort_values([BAKERY_COL, DATE_COL])
    )
    bakery_day["items_in_bakery_lag1"] = bakery_day.groupby(BAKERY_COL)["__bakery_items"].shift(1)
    bakery_day["items_change"] = bakery_day["__bakery_items"] - bakery_day["items_in_bakery_lag1"]
    bakery_day.drop(columns=["__bakery_items"], inplace=True)
    df = df.merge(bakery_day, on=[BAKERY_COL, DATE_COL], how="left")
    for col in ["items_in_bakery_today", "items_in_bakery_lag1", "items_change"]:
        df[col] = df[col].fillna(0)
    return df


EXP63_EXTRA_FEATURES = [
    "is_censored_lag1",
    "lost_qty_lag1",
    "pct_censored_7d",
    "sales_dow_mean",
    "demand_dow_mean",
    "demand_trend",
    "cv_7d",
    "stale_ratio_lag1",
    "items_in_bakery_today",
    "items_in_bakery_lag1",
    "items_change",
]


NUMERIC_LOCATION_COLS = {
    26: "loc_area_sqm",
    30: "loc_rent_rub",
    32: "loc_rent_per_sqm",
    34: "loc_traffic_nearby",
    35: "loc_traffic_at_point",
    36: "loc_traffic_avg",
    37: "loc_density_5min",
    41: "loc_density_10min",
    42: "loc_density_300m",
    43: "loc_income_5min",
    44: "loc_income_10min",
    45: "loc_income_300m",
    20: "loc_competitors",
}

BINARY_LOCATION_COLS = {
    6: "loc_has_porch",
    17: "loc_has_vitrage",
    18: "loc_near_pyaterochka",
    19: "loc_near_canteen",
    22: "loc_near_metro",
    46: "loc_near_market",
    47: "loc_near_bc",
    48: "loc_near_school",
    49: "loc_near_college",
}

CATEGORICAL_LOCATION_COLS = {2: "loc_premise_type"}


def parse_binary(val):
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
        return 1 if float(s) > 0 else 0
    except ValueError:
        return np.nan


def parse_numeric(val):
    if pd.isna(val):
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def parse_premise_type(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if "тц" in s or "tc" in s:
        if "стрит" in s or "strit" in s:
            return 1
        return 2
    return 0


def load_location_features():
    print(f"\n  Loading location data from {LOCATION_PATH.name}...")
    raw = pd.read_excel(str(LOCATION_PATH), header=None)
    data_rows = raw.iloc[9:].copy()
    data_rows = data_rows[data_rows.iloc[:, 0].notna()].copy()

    result = pd.DataFrame()
    result[BAKERY_COL] = data_rows.iloc[:, 0].values
    for col_idx, col_name in NUMERIC_LOCATION_COLS.items():
        result[col_name] = data_rows.iloc[:, col_idx].apply(parse_numeric).values
    for col_idx, col_name in BINARY_LOCATION_COLS.items():
        result[col_name] = data_rows.iloc[:, col_idx].apply(parse_binary).values
    for col_idx, col_name in CATEGORICAL_LOCATION_COLS.items():
        result[col_name] = data_rows.iloc[:, col_idx].apply(parse_premise_type).values

    print(f"  Loaded {len(result)} bakeries, {len(result.columns) - 1} location features")
    return result


def build_location_clusters(loc_df):
    feature_cols = [c for c in loc_df.columns if c != BAKERY_COL]
    work = loc_df.copy()
    for col in feature_cols:
        work[col] = work[col].fillna(work[col].median() if work[col].notna().any() else 0)

    X_scaled = StandardScaler().fit_transform(work[feature_cols].astype(float).values)
    best_k = 5
    best_score = -np.inf
    for k in [3, 4, 5, 6]:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"    location K={k}: silhouette={score:.3f}")
        if score > best_score:
            best_score = score
            best_k = k

    print(f"    using location K={best_k}")
    work["cluster_loc"] = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X_scaled).astype(int)
    summary = (
        work.groupby("cluster_loc")
        .agg(
            n_bakeries=(BAKERY_COL, "count"),
            area_mean=("loc_area_sqm", "mean"),
            traffic_mean=("loc_traffic_avg", "mean"),
            rent_mean=("loc_rent_per_sqm", "mean"),
            competitors_mean=("loc_competitors", "mean"),
        )
        .reset_index()
    )
    meta = {"k": best_k, "silhouette": round(float(best_score), 4)}
    return work[[BAKERY_COL, "cluster_loc"]], summary, meta


def build_ts_clusters(df):
    cluster_df = (
        df.groupby([BAKERY_COL, PRODUCT_COL])
        .agg(
            demand_mean=(DEMAND_TARGET, "mean"),
            demand_std=(DEMAND_TARGET, "std"),
            demand_median=(DEMAND_TARGET, "median"),
            sales_mean=(SOLD_COL, "mean"),
            sales_std=(SOLD_COL, "std"),
            n_days=(DATE_COL, "count"),
            lost_mean=("lost_qty", "mean"),
            censored_pct=("is_censored", "mean"),
            dow_std=(SOLD_COL, lambda x: x.groupby(df.loc[x.index, DOW_COL]).sum().std()),
        )
        .reset_index()
    )
    for col in ["demand_std", "sales_std", "dow_std"]:
        cluster_df[col] = cluster_df[col].fillna(0)

    cluster_df["cv"] = cluster_df["demand_std"] / (cluster_df["demand_mean"] + 1e-8)
    last_week = df[df[DATE_COL] >= df[DATE_COL].max() - pd.Timedelta(days=7)]
    last_week_agg = (
        last_week.groupby([BAKERY_COL, PRODUCT_COL])[DEMAND_TARGET].mean().reset_index(name="last_week_mean")
    )
    cluster_df = cluster_df.merge(last_week_agg, on=[BAKERY_COL, PRODUCT_COL], how="left")
    cluster_df["last_week_mean"] = cluster_df["last_week_mean"].fillna(cluster_df["demand_mean"])
    cluster_df["trend"] = (cluster_df["last_week_mean"] / (cluster_df["demand_mean"] + 1e-8)).clip(0.5, 2.0)

    feat_cols = ["demand_mean", "cv", "trend", "dow_std", "n_days", "lost_mean", "censored_pct"]
    X = cluster_df[feat_cols].copy()
    X["demand_mean"] = np.log1p(X["demand_mean"])
    X["n_days"] = np.log1p(X["n_days"])
    X_scaled = StandardScaler().fit_transform(X.values)

    best_k = 5
    best_score = -np.inf
    for k in [3, 5, 7, 9]:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"    ts K={k}: silhouette={score:.3f}")
        if score > best_score:
            best_score = score
            best_k = k

    print(f"    using ts K={best_k}")
    cluster_df["cluster_ts"] = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X_scaled).astype(int)
    summary = (
        cluster_df.groupby("cluster_ts")
        .agg(
            n_pairs=(BAKERY_COL, "count"),
            demand_mean=("demand_mean", "mean"),
            cv_mean=("cv", "mean"),
            trend_mean=("trend", "mean"),
            censored_pct=("censored_pct", "mean"),
            lost_mean=("lost_mean", "mean"),
        )
        .reset_index()
    )
    meta = {"k": best_k, "silhouette": round(float(best_score), 4)}
    return cluster_df[[BAKERY_COL, PRODUCT_COL, "cluster_ts"]], summary, meta


def evaluate_model(name, train_df, test_df, features, y_train, y_test):
    available = [f for f in features if f in train_df.columns]
    model = train_quantile(train_df[available], y_train, alpha=0.5)
    pred = predict_clipped(model, test_df[available])
    mae = mean_absolute_error(y_test, pred)
    wm = wmape(y_test, pred)
    bias = float(np.mean(y_test - pred))
    r2 = r2_score(y_test, pred)

    result = {
        "mae": round(mae, 4),
        "wmape": round(wm, 2),
        "bias": round(bias, 4),
        "r2": round(r2, 4),
        "n_features": len(available),
        "delta_vs_v3": round(mae - BASELINE_V3_MAE, 4),
        "delta_vs_exp63": round(mae - EXP63_MAE, 4),
    }

    print(f"\n  --- {name} ({len(available)} features) ---")
    print(f"    MAE={mae:.4f}, WMAPE={wm:.2f}%, Bias={bias:+.4f}, R2={r2:.4f}")
    print(f"    Delta vs V3: {mae - BASELINE_V3_MAE:+.4f}, vs exp63: {mae - EXP63_MAE:+.4f}")
    for threshold in [15, 50, 100, 200]:
        mask = y_test >= threshold
        if mask.sum() > 0:
            mae_h = mean_absolute_error(y_test[mask], pred[mask])
            bias_h = float(np.mean(y_test[mask] - pred[mask]))
            print(f"    demand>={threshold}: MAE={mae_h:.2f}, Bias={bias_h:+.2f}, N={mask.sum()}")

    return model, pred, result, available


def evaluate_routed_model(train_df, test_df, features, y_test):
    available = [f for f in features if f in train_df.columns]
    preds = np.full(len(test_df), np.nan, dtype=float)
    cluster_results = []

    print(f"\n  --- E_routed_cluster_ts ({len(available)} features) ---")
    for cluster_id in sorted(train_df["cluster_ts"].astype(int).unique()):
        cluster_train = train_df[train_df["cluster_ts"].astype(int) == cluster_id]
        cluster_test = test_df[test_df["cluster_ts"].astype(int) == cluster_id]
        if cluster_test.empty or len(cluster_train) < 500:
            continue

        model = train_quantile(cluster_train[available], cluster_train[DEMAND_TARGET], alpha=0.5)
        cluster_pred = predict_clipped(model, cluster_test[available])
        preds[test_df["cluster_ts"].astype(int) == cluster_id] = cluster_pred
        cluster_results.append(
            {
                "cluster_ts": int(cluster_id),
                "train_rows": int(len(cluster_train)),
                "test_rows": int(len(cluster_test)),
                "mae": round(float(mean_absolute_error(cluster_test[DEMAND_TARGET], cluster_pred)), 4),
            }
        )

    missing_mask = np.isnan(preds)
    if missing_mask.any():
        fallback = train_quantile(train_df[available], train_df[DEMAND_TARGET], alpha=0.5)
        preds[missing_mask] = predict_clipped(fallback, test_df.loc[missing_mask, available])

    mae = mean_absolute_error(y_test, preds)
    wm = wmape(y_test, preds)
    bias = float(np.mean(y_test - preds))
    r2 = r2_score(y_test, preds)
    result = {
        "mae": round(mae, 4),
        "wmape": round(wm, 2),
        "bias": round(bias, 4),
        "r2": round(r2, 4),
        "n_features": len(available),
        "delta_vs_v3": round(mae - BASELINE_V3_MAE, 4),
        "delta_vs_exp63": round(mae - EXP63_MAE, 4),
        "routed_clusters": cluster_results,
    }

    print(f"    MAE={mae:.4f}, WMAPE={wm:.2f}%, Bias={bias:+.4f}, R2={r2:.4f}")
    print(f"    Delta vs V3: {mae - BASELINE_V3_MAE:+.4f}, vs exp63: {mae - EXP63_MAE:+.4f}")
    for threshold in [15, 50, 100, 200]:
        mask = y_test >= threshold
        if mask.sum() > 0:
            mae_h = mean_absolute_error(y_test[mask], preds[mask])
            bias_h = float(np.mean(y_test[mask] - preds[mask]))
            print(f"    demand>={threshold}: MAE={mae_h:.2f}, Bias={bias_h:+.2f}, N={mask.sum()}")

    return preds, result, available


def main():
    print("=" * 80)
    print("  EXPERIMENT 66: Cluster Features")
    print("  exp63 features + location clusters + time-series clusters")
    print("=" * 80)
    t_start = time.time()

    print(f"\n[1/7] Loading data from {DEMAND_8M_PATH.name}...")
    df = pd.read_csv(str(DEMAND_8M_PATH), encoding="utf-8-sig")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df[DATE_COL].min().date()} -- {df[DATE_COL].max().date()}")

    print(f"\n[2/7] Building exp63 feature layer...")
    df = add_censoring_features(df)
    df = add_dow_features(df)
    df = add_trend_features(df)
    df = build_stale_ratio(df)
    df = add_assortment_features(df)

    print(f"\n[3/7] Building location clusters...")
    loc_df = load_location_features()
    loc_clusters, loc_summary, loc_meta = build_location_clusters(loc_df)
    df = df.merge(loc_clusters, on=BAKERY_COL, how="left")
    df["cluster_loc"] = df["cluster_loc"].fillna(-1).astype(int)
    print(f"  Matched bakeries with location cluster: {df.loc[df['cluster_loc'] >= 0, BAKERY_COL].nunique()}")

    print(f"\n[4/7] Building time-series clusters...")
    ts_clusters, ts_summary, ts_meta = build_ts_clusters(df)
    df = df.merge(ts_clusters, on=[BAKERY_COL, PRODUCT_COL], how="left")
    df["cluster_ts"] = df["cluster_ts"].fillna(-1).astype(int)

    for col in CATEGORICAL_COLS_V2:
        if col in df.columns:
            df[col] = df[col].astype("category")
    df["cluster_loc"] = df["cluster_loc"].astype("category")
    df["cluster_ts"] = df["cluster_ts"].astype("category")

    print(f"\n[5/7] Train/test split (last {TEST_DAYS} days = test)...")
    test_start = df[DATE_COL].max() - pd.Timedelta(days=TEST_DAYS - 1)
    train_df = df[df[DATE_COL] < test_start].copy()
    test_df = df[df[DATE_COL] >= test_start].copy()
    y_train = train_df[DEMAND_TARGET]
    y_test = test_df[DEMAND_TARGET].values
    print(f"  Train: {len(train_df):,} rows, {train_df[DATE_COL].nunique()} days")
    print(f"  Test:  {len(test_df):,} rows, {test_df[DATE_COL].nunique()} days")

    print(f"\n[6/7] Training variants...")
    base_features = [f for f in FEATURES_V3 if f in df.columns]
    exp63_features = base_features + [f for f in EXP63_EXTRA_FEATURES if f in df.columns]
    variants = {
        "A_exp63_baseline": exp63_features,
        "B_plus_cluster_loc": exp63_features + ["cluster_loc"],
        "C_plus_cluster_ts": exp63_features + ["cluster_ts"],
        "D_plus_both_clusters": exp63_features + ["cluster_loc", "cluster_ts"],
    }

    all_results = {}
    all_predictions = {}
    best_variant = None
    best_mae = float("inf")
    best_model = None
    best_pred = None
    best_available = None

    for name, feats in variants.items():
        model, pred, result, available = evaluate_model(name, train_df, test_df, feats, y_train, y_test)
        all_results[name] = result
        all_predictions[name] = pred
        if result["mae"] < best_mae:
            best_variant = name
            best_mae = result["mae"]
            best_model = model
            best_pred = pred
            best_available = available

    pred_e, result_e, available_e = evaluate_routed_model(
        train_df, test_df, exp63_features + ["cluster_loc", "cluster_ts"], y_test
    )
    all_results["E_routed_cluster_ts"] = result_e
    all_predictions["E_routed_cluster_ts"] = pred_e
    if result_e["mae"] < best_mae:
        best_variant = "E_routed_cluster_ts"
        best_mae = result_e["mae"]
        best_model = None
        best_pred = pred_e
        best_available = available_e

    print(f"\n[7/7] Reporting...")
    if best_model is not None:
        importance = pd.DataFrame(
            {"feature": best_available, "importance": best_model.feature_importances_}
        ).sort_values("importance", ascending=False)
        cluster_feats = importance[importance["feature"].isin(["cluster_loc", "cluster_ts"])]
        feature_importance_clusters = cluster_feats.to_dict("records")
        print("\n  === Cluster feature importance ===")
        if cluster_feats.empty:
            print("    cluster features not present in best single-model variant")
        else:
            for i, (_, row) in enumerate(importance.iterrows(), 1):
                if row["feature"] in {"cluster_loc", "cluster_ts"}:
                    print(f"    #{i}: {row['feature']} ({int(row['importance'])})")
    else:
        feature_importance_clusters = []
        print("\n  === Cluster feature importance ===")
        print("    best variant is routed; no single-model feature importance")

    print(f"\n  === Per-category ({best_variant}) ===")
    print_category_metrics(y_test, best_pred, test_df[CATEGORY_COL].values)

    print(f"\n  {'=' * 86}")
    print("  SUMMARY")
    print(f"  {'=' * 86}")
    print(f"  {'Variant':<25} {'Feats':>5} {'MAE':>8} {'WMAPE':>8} {'vs V3':>8} {'vs 63':>8}")
    print(f"  {'-' * 72}")
    for name, r in all_results.items():
        print(
            f"  {name:<25} {r['n_features']:>5} {r['mae']:>8.4f} "
            f"{r['wmape']:>7.2f}% {r['delta_vs_v3']:>+8.4f} {r['delta_vs_exp63']:>+8.4f}"
        )

    metrics = {
        "experiment": "66_cluster_features",
        "target": DEMAND_TARGET,
        "baseline_v3_mae": BASELINE_V3_MAE,
        "exp63_mae": EXP63_MAE,
        "best_variant": best_variant,
        "results": all_results,
        "location_cluster_meta": loc_meta,
        "time_series_cluster_meta": ts_meta,
        "location_cluster_summary": loc_summary.to_dict("records"),
        "time_series_cluster_summary": ts_summary.to_dict("records"),
        "feature_importance_clusters": feature_importance_clusters,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
    }

    predictions = pd.DataFrame(
        {
            DATE_COL: test_df[DATE_COL].values,
            BAKERY_COL: test_df[BAKERY_COL].values,
            PRODUCT_COL: test_df[PRODUCT_COL].values,
            CATEGORY_COL: test_df[CATEGORY_COL].values,
            "fact_demand": y_test,
            "cluster_loc": test_df["cluster_loc"].astype(str).values,
            "cluster_ts": test_df["cluster_ts"].astype(str).values,
            "pred_A_exp63_baseline": np.round(all_predictions["A_exp63_baseline"], 2),
            "pred_B_plus_cluster_loc": np.round(all_predictions["B_plus_cluster_loc"], 2),
            "pred_C_plus_cluster_ts": np.round(all_predictions["C_plus_cluster_ts"], 2),
            "pred_D_plus_both_clusters": np.round(all_predictions["D_plus_both_clusters"], 2),
            "pred_E_routed_cluster_ts": np.round(all_predictions["E_routed_cluster_ts"], 2),
        }
    )

    save_results(EXP_DIR, metrics, predictions)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s")
    print("  Done!")


if __name__ == "__main__":
    main()

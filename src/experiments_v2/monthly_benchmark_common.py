"""
Helpers for monthly benchmark experiments.

These utilities keep cluster construction train-safe:
- location clusters are built from static external location data
- time-series clusters are fit on train-only SKU-pair summaries
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.experiments_v2.benchmark_common import (
    BAKERY_COL,
    DATE_COL,
    DOW_COL,
    PRODUCT_COL,
    SOLD_COL,
    add_assortment_features,
    add_censoring_features,
    add_dow_features,
    add_trend_features,
)
from src.experiments_v2.common import DEMAND_TARGET


ROOT = Path(__file__).resolve().parent.parent.parent
LOCATION_PATH = ROOT / "data" / "raw" / (
    "анализ локаций "
    "по БП 02.03 ( 1) (2) (4) (1) (1).xlsx"
)

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

FEATURES_61_EXTRA = [
    "is_censored_lag1",
    "lost_qty_lag1",
    "pct_censored_7d",
    "sales_dow_mean",
    "demand_dow_mean",
    "demand_trend",
    "cv_7d",
    "stale_ratio_lag1",
]

FEATURES_62_EXTRA = [
    "items_in_bakery_today",
    "items_in_bakery_lag1",
    "items_change",
    "items_in_category_today",
    "category_items_change",
]


def parse_numeric(val):
    if pd.isna(val):
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


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


def parse_premise_type(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if "тц" in s or "tc" in s:
        if "стрит" in s or "strit" in s:
            return 1
        return 2
    return 0


def load_location_features() -> pd.DataFrame:
    if not LOCATION_PATH.exists():
        raise FileNotFoundError(f"Location feature file not found: {LOCATION_PATH}")

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

    return result


def build_location_clusters(loc_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    feature_cols = [c for c in loc_df.columns if c != BAKERY_COL]
    work = loc_df.copy()
    medians = {}
    for col in feature_cols:
        med = work[col].median() if work[col].notna().any() else 0
        medians[col] = med
        work[col] = work[col].fillna(med)

    scaler = StandardScaler().fit(work[feature_cols].astype(float).values)
    X_scaled = scaler.transform(work[feature_cols].astype(float).values)
    best_k = 5
    best_score = -np.inf
    for k in [3, 4, 5, 6]:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_score = score
            best_k = k

    kmeans_loc = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X_scaled)
    work["cluster_loc"] = kmeans_loc.labels_.astype(int)
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
    pipeline = {
        "scaler": scaler,
        "kmeans": kmeans_loc,
        "feature_cols": feature_cols,
        "medians": medians,
    }
    return work[[BAKERY_COL, "cluster_loc"]], summary, meta, pipeline


def build_exp63_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_censoring_features(df)
    df = add_dow_features(df)
    df = add_trend_features(df)
    if "stale_ratio_lag1" not in df.columns:
        df["stale_ratio_lag1"] = 0.0
    df = add_assortment_features(df)
    return df


def add_assortment_features_62(df: pd.DataFrame) -> pd.DataFrame:
    """Add bakery and category level assortment features for experiment 62."""
    df = df.sort_values([BAKERY_COL, PRODUCT_COL, DATE_COL]).copy()

    bakery_cols = ["items_in_bakery_today", "items_in_bakery_lag1", "items_change"]
    if any(col not in df.columns for col in bakery_cols):
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

    if "items_in_category_today" not in df.columns or "category_items_change" not in df.columns:
        df["items_in_category_today"] = df.groupby([BAKERY_COL, "Категория", DATE_COL])[PRODUCT_COL].transform("nunique")
        cat_day = (
            df.groupby([BAKERY_COL, "Категория", DATE_COL])[PRODUCT_COL]
            .nunique()
            .reset_index(name="__cat_items")
            .sort_values([BAKERY_COL, "Категория", DATE_COL])
        )
        cat_day["__cat_items_lag1"] = cat_day.groupby([BAKERY_COL, "Категория"])["__cat_items"].shift(1)
        cat_day["category_items_change"] = cat_day["__cat_items"] - cat_day["__cat_items_lag1"]
        cat_day.drop(columns=["__cat_items", "__cat_items_lag1"], inplace=True)
        df = df.merge(cat_day, on=[BAKERY_COL, "Категория", DATE_COL], how="left")

    for col in FEATURES_62_EXTRA:
        df[col] = df[col].fillna(0)

    return df


def build_train_ts_clusters(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    cluster_df = (
        train_df.groupby([BAKERY_COL, PRODUCT_COL])
        .agg(
            demand_mean=(DEMAND_TARGET, "mean"),
            demand_std=(DEMAND_TARGET, "std"),
            demand_median=(DEMAND_TARGET, "median"),
            sales_mean=(SOLD_COL, "mean"),
            sales_std=(SOLD_COL, "std"),
            n_days=(DATE_COL, "count"),
            lost_mean=("lost_qty", "mean"),
            censored_pct=("is_censored", "mean"),
            dow_std=(SOLD_COL, lambda x: x.groupby(train_df.loc[x.index, DOW_COL]).sum().std()),
        )
        .reset_index()
    )
    for col in ["demand_std", "sales_std", "dow_std"]:
        cluster_df[col] = cluster_df[col].fillna(0)

    cluster_df["cv"] = cluster_df["demand_std"] / (cluster_df["demand_mean"] + 1e-8)
    last_week = train_df[train_df[DATE_COL] >= train_df[DATE_COL].max() - pd.Timedelta(days=7)]
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
    scaler = StandardScaler().fit(X.values)
    X_scaled = scaler.transform(X.values)

    best_k = 5
    best_score = -np.inf
    for k in [3, 5, 7, 9]:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_score = score
            best_k = k

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X_scaled)
    cluster_df["cluster_ts"] = kmeans.labels_.astype(int)
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
    pipeline = {"scaler": scaler, "kmeans": kmeans, "feat_cols": feat_cols}
    return cluster_df[[BAKERY_COL, PRODUCT_COL, "cluster_ts"]], summary, meta, pipeline


def merge_cluster_map(df: pd.DataFrame, cluster_map: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    merged = df.merge(cluster_map, on=[BAKERY_COL, PRODUCT_COL], how="left")
    merged[cluster_col] = merged[cluster_col].fillna(-1).astype(int)
    return merged

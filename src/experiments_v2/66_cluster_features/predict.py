"""
Load best model from exp 66 and predict on new data.

Builds all required features (exp63 + clusters) from raw daily_sales CSV.

Usage:
    python src/experiments_v2/66_cluster_features/predict.py data/processed/daily_sales_8m_demand.csv
    python src/experiments_v2/66_cluster_features/predict.py input.csv -o predictions.csv
"""

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from src.experiments_v2.common import CATEGORICAL_COLS_V2, DEMAND_TARGET, predict_clipped

EXP_DIR = Path(__file__).resolve().parent
MODEL_PATH = EXP_DIR / "best_model.joblib"

DATE_COL = "Дата"
BAKERY_COL = "Пекарня"
PRODUCT_COL = "Номенклатура"
DOW_COL = "ДеньНедели"
SOLD_COL = "Продано"


# ── Feature engineering (same as run.py) ─────────────────────────────────────

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


def assign_ts_clusters(df, ts_pipeline):
    """Assign cluster_ts using saved KMeans pipeline."""
    scaler = ts_pipeline["scaler"]
    kmeans = ts_pipeline["kmeans"]
    feat_cols = ts_pipeline["feat_cols"]

    cluster_df = (
        df.groupby([BAKERY_COL, PRODUCT_COL])
        .agg(
            demand_mean=(DEMAND_TARGET, "mean"),
            demand_std=(DEMAND_TARGET, "std"),
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

    X = cluster_df[feat_cols].copy()
    X["demand_mean"] = np.log1p(X["demand_mean"])
    X["n_days"] = np.log1p(X["n_days"])
    X_scaled = scaler.transform(X.values)
    cluster_df["cluster_ts"] = kmeans.predict(X_scaled).astype(int)

    return cluster_df[[BAKERY_COL, PRODUCT_COL, "cluster_ts"]]


def build_features(df, model_data):
    """Build all exp63 + cluster features from raw data."""
    print("  Building exp63 features...")
    df = add_censoring_features(df)
    df = add_dow_features(df)
    df = add_trend_features(df)
    # stale_ratio_lag1 — skip heavy checks parsing, fill with 0
    if "stale_ratio_lag1" not in df.columns:
        df["stale_ratio_lag1"] = 0.0
    df = add_assortment_features(df)

    print("  Assigning location clusters...")
    loc_clusters = model_data["loc_clusters"]
    df = df.merge(loc_clusters, on=BAKERY_COL, how="left")
    df["cluster_loc"] = df["cluster_loc"].fillna(-1).astype(int)

    print("  Assigning time-series clusters...")
    ts_clusters = assign_ts_clusters(df, model_data["ts_pipeline"])
    df = df.merge(ts_clusters, on=[BAKERY_COL, PRODUCT_COL], how="left")
    df["cluster_ts"] = df["cluster_ts"].fillna(-1).astype(int)

    for col in CATEGORICAL_COLS_V2:
        if col in df.columns:
            df[col] = df[col].astype("category")
    df["cluster_loc"] = df["cluster_loc"].astype("category")
    df["cluster_ts"] = df["cluster_ts"].astype("category")

    return df


# ── Prediction ───────────────────────────────────────────────────────────────

def predict(df, model_data):
    variant = model_data["variant"]
    model = model_data["model"]
    features = model_data["features"]
    available = [f for f in features if f in df.columns]

    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"  WARNING: {len(missing)} features missing: {missing[:10]}")

    if variant == "E_routed_cluster_ts":
        cluster_models = model["cluster_models"]
        fallback = model["fallback"]
        preds = np.full(len(df), np.nan, dtype=float)

        for cluster_id, cluster_model in cluster_models.items():
            mask = df["cluster_ts"].astype(int) == cluster_id
            if mask.any():
                preds[mask] = predict_clipped(cluster_model, df.loc[mask, available])

        missing_mask = np.isnan(preds)
        if missing_mask.any():
            fb = fallback if fallback is not None else next(iter(cluster_models.values()))
            preds[missing_mask] = predict_clipped(fb, df.loc[missing_mask, available])
    else:
        preds = predict_clipped(model, df[available])

    return preds


def main():
    parser = argparse.ArgumentParser(description="Predict with exp 66 best model")
    parser.add_argument("input_csv", help="Path to input CSV (raw daily_sales format)")
    parser.add_argument("--output", "-o", default=None, help="Output CSV (default: exp dir/new_predictions.csv)")
    parser.add_argument("--model", "-m", default=str(MODEL_PATH), help="Path to best_model.joblib")
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model_data = joblib.load(Path(args.model))
    n_feat = model_data.get("n_features", len(model_data["features"]))
    print(f"  variant={model_data['variant']}, MAE={model_data['mae']:.4f}, features={n_feat}")

    print(f"Reading {args.input_csv}...")
    df = pd.read_csv(args.input_csv, encoding="utf-8-sig")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    print(f"  Shape: {df.shape}")

    df = build_features(df, model_data)

    print("Predicting...")
    preds = predict(df, model_data)
    df["prediction"] = np.round(preds, 2)

    output_path = args.output or str(EXP_DIR / "new_predictions.csv")
    out_cols = [DATE_COL, BAKERY_COL, PRODUCT_COL, "prediction"]
    if DEMAND_TARGET in df.columns:
        out_cols.insert(3, DEMAND_TARGET)
    df[out_cols].to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} predictions to {output_path}")


if __name__ == "__main__":
    main()

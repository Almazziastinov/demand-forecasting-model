"""
Monthly full benchmark on demand target.

This is a train-safe replacement for the short-horizon full benchmark:
 - test window is 30 days
 - `cluster_ts` is fit on train only
 - benchmark families: 60, 61, 62, 66
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = EXP_DIR / "artifacts"

from src.experiments_v2.benchmark_common import (  # noqa: E402
    BAKERY_COL,
    DATE_COL,
    PRODUCT_COL,
    CATEGORICAL_COLS_V2,
    DEMAND_8M_PATH,
    DEMAND_TARGET,
    FEATURES_V3,
    make_train_test_split,
    predict_clipped,
    regression_metrics,
    save_results,
    select_feature_columns,
    train_quantile,
    wmape,
)
from src.experiments_v2.monthly_benchmark_common import (  # noqa: E402
    FEATURES_61_EXTRA,
    FEATURES_62_EXTRA,
    add_assortment_features_62,
    build_exp63_feature_frame,
    build_location_clusters,
    build_train_ts_clusters,
    load_location_features,
    merge_cluster_map,
)


TEST_DAYS = 30

MODEL_NAMES = [
    "60_baseline_v3",
    "61_censoring_behavioral",
    "62_assortment_availability",
    "66_cluster_features",
]


def ensure_category_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in CATEGORICAL_COLS_V2:
        if col in work.columns:
            work[col] = work[col].astype("category")
    if "cluster_loc" in work.columns:
        work["cluster_loc"] = work["cluster_loc"].astype("category")
    if "cluster_ts" in work.columns:
        work["cluster_ts"] = work["cluster_ts"].astype("category")
    return work


def evaluate_single_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    available = select_feature_columns(train_df, features, drop_constants=True)
    model = train_quantile(train_df[available], train_df[DEMAND_TARGET], alpha=0.5)
    pred = predict_clipped(model, test_df[available])

    metrics = regression_metrics(test_df[DEMAND_TARGET], pred)
    result = {
        "mae": round(metrics["mae"], 4),
        "mse": round(metrics["mse"], 4),
        "r2": round(metrics["r2"], 4) if np.isfinite(metrics["r2"]) else np.nan,
        "wmape": round(metrics["wmape"], 2),
        "n_features": len(available),
    }

    print(f"\n  --- {model_name} ({len(available)} features) ---")
    print(
        f"    MAE={result['mae']:.4f}, WMAPE={result['wmape']:.2f}%, "
        f"R2={result['r2']:.4f}"
    )

    pred_df = test_df[[DATE_COL, BAKERY_COL, PRODUCT_COL]].copy()
    pred_df["fact"] = test_df[DEMAND_TARGET].values
    pred_df["pred"] = np.round(pred, 2)
    pred_df["abs_error"] = np.round(np.abs(test_df[DEMAND_TARGET].values - pred), 2)
    pred_df["model"] = model_name
    pred_df["family"] = "single"
    pred_df["status"] = "trained"

    metrics_rows = []
    for (bakery, product), group in pred_df.groupby([BAKERY_COL, PRODUCT_COL], sort=False):
        pair_metrics = regression_metrics(group["fact"], group["pred"])
        metrics_rows.append(
            {
                BAKERY_COL: bakery,
                PRODUCT_COL: product,
                "model": model_name,
                "n_test_rows": int(len(group)),
                "r2": round(pair_metrics["r2"], 4) if np.isfinite(pair_metrics["r2"]) else np.nan,
                "mse": round(pair_metrics["mse"], 4),
                "mae": round(pair_metrics["mae"], 4),
                "wmape": round(pair_metrics["wmape"], 2),
            }
        )

    return pred_df, pd.DataFrame(metrics_rows), result


def evaluate_routed_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    available = select_feature_columns(train_df, features, drop_constants=True)
    preds = np.full(len(test_df), np.nan, dtype=float)
    cluster_rows: list[dict] = []
    cluster_models = {}

    train_clusters = train_df["cluster_ts"].astype(int)
    test_clusters = test_df["cluster_ts"].astype(int)

    print(f"\n  --- {model_name} routed by cluster_ts ({len(available)} features) ---")
    for cluster_id in sorted(train_clusters.unique()):
        cluster_train = train_df[train_clusters == cluster_id]
        cluster_test = test_df[test_clusters == cluster_id]
        if cluster_test.empty or len(cluster_train) < 500:
            continue

        model = train_quantile(cluster_train[available], cluster_train[DEMAND_TARGET], alpha=0.5)
        cluster_models[int(cluster_id)] = model
        cluster_pred = predict_clipped(model, cluster_test[available])
        preds[test_clusters == cluster_id] = cluster_pred
        cluster_rows.append(
            {
                "cluster_ts": int(cluster_id),
                "train_rows": int(len(cluster_train)),
                "test_rows": int(len(cluster_test)),
                "mae": round(float(np.mean(np.abs(cluster_test[DEMAND_TARGET].values - cluster_pred))), 4),
            }
        )

    fallback_model = train_quantile(train_df[available], train_df[DEMAND_TARGET], alpha=0.5)
    missing_mask = np.isnan(preds)
    if missing_mask.any():
        preds[missing_mask] = predict_clipped(fallback_model, test_df.loc[missing_mask, available])

    metrics = regression_metrics(test_df[DEMAND_TARGET], preds)
    result = {
        "mae": round(metrics["mae"], 4),
        "mse": round(metrics["mse"], 4),
        "r2": round(metrics["r2"], 4) if np.isfinite(metrics["r2"]) else np.nan,
        "wmape": round(metrics["wmape"], 2),
        "n_features": len(available),
        "routing": "cluster_ts",
        "routed_clusters": cluster_rows,
    }

    print(
        f"    MAE={result['mae']:.4f}, WMAPE={result['wmape']:.2f}%, "
        f"R2={result['r2']:.4f}"
    )

    pred_df = test_df[[DATE_COL, BAKERY_COL, PRODUCT_COL, "cluster_ts"]].copy()
    pred_df["fact"] = test_df[DEMAND_TARGET].values
    pred_df["pred"] = np.round(preds, 2)
    pred_df["abs_error"] = np.round(np.abs(test_df[DEMAND_TARGET].values - preds), 2)
    pred_df["model"] = model_name
    pred_df["family"] = "routed"
    pred_df["status"] = "trained"

    metrics_rows = []
    for (bakery, product), group in pred_df.groupby([BAKERY_COL, PRODUCT_COL], sort=False):
        pair_metrics = regression_metrics(group["fact"], group["pred"])
        metrics_rows.append(
            {
                BAKERY_COL: bakery,
                PRODUCT_COL: product,
                "model": model_name,
                "n_test_rows": int(len(group)),
                "r2": round(pair_metrics["r2"], 4) if np.isfinite(pair_metrics["r2"]) else np.nan,
                "mse": round(pair_metrics["mse"], 4),
                "mae": round(pair_metrics["mae"], 4),
                "wmape": round(pair_metrics["wmape"], 2),
            }
        )
    metrics_df = pd.DataFrame(metrics_rows)
    return pred_df, metrics_df, result


def merge_metrics_wide(metrics_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    wide: pd.DataFrame | None = None
    for model_name, frame in metrics_frames.items():
        sub = frame[[BAKERY_COL, PRODUCT_COL, "r2", "mse", "mae", "wmape"]].copy()
        sub = sub.rename(
            columns={
                "r2": f"{model_name}_r2",
                "mse": f"{model_name}_mse",
                "mae": f"{model_name}_mae",
                "wmape": f"{model_name}_wmape",
            }
        )
        wide = sub if wide is None else wide.merge(sub, on=[BAKERY_COL, PRODUCT_COL], how="outer")

    if wide is None or wide.empty:
        return pd.DataFrame()

    wide = wide.reset_index(drop=True)
    r2_cols = [c for c in wide.columns if c.endswith("_r2")]
    r2_values = wide[r2_cols].replace([np.inf, -np.inf], np.nan).fillna(-np.inf)
    best_idx = r2_values.idxmax(axis=1)

    result = wide[[BAKERY_COL, PRODUCT_COL]].copy()
    result["best_model"] = best_idx.str.replace("_r2", "", regex=False)
    result["best_r2"] = r2_values.max(axis=1).replace(-np.inf, np.nan)
    result["best_mae"] = [
        wide.iloc[i][f"{result.iloc[i]['best_model']}_mae"] if pd.notna(result.iloc[i]["best_model"]) else np.nan
        for i in range(len(result))
    ]
    result["best_mse"] = [
        wide.iloc[i][f"{result.iloc[i]['best_model']}_mse"] if pd.notna(result.iloc[i]["best_model"]) else np.nan
        for i in range(len(result))
    ]
    result["best_wmape"] = [
        wide.iloc[i][f"{result.iloc[i]['best_model']}_wmape"] if pd.notna(result.iloc[i]["best_model"]) else np.nan
        for i in range(len(result))
    ]
    return pd.concat([result, wide.drop(columns=[BAKERY_COL, PRODUCT_COL]).reset_index(drop=True)], axis=1)


def model_summary_from_wide(best_df: pd.DataFrame) -> pd.DataFrame:
    if best_df.empty:
        return pd.DataFrame()

    model_names = sorted({col[:-3] for col in best_df.columns if col.endswith("_r2")})
    rows = []
    for model in model_names:
        r2_col = f"{model}_r2"
        mse_col = f"{model}_mse"
        mae_col = f"{model}_mae"
        wmape_col = f"{model}_wmape"
        rows.append(
            {
                "model": model,
                "n_pairs": int(best_df[r2_col].notna().sum()),
                "avg_r2": round(float(best_df[r2_col].mean(skipna=True)), 4),
                "median_r2": round(float(best_df[r2_col].median(skipna=True)), 4),
                "avg_mae": round(float(best_df[mae_col].mean(skipna=True)), 4),
                "median_mae": round(float(best_df[mae_col].median(skipna=True)), 4),
                "avg_mse": round(float(best_df[mse_col].mean(skipna=True)), 4),
                "avg_wmape": round(float(best_df[wmape_col].mean(skipna=True)), 2),
                "win_count": int((best_df["best_model"] == model).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["avg_r2", "avg_mae"], ascending=[False, True]).reset_index(drop=True)


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("MONTHLY FULL BENCHMARK")
    print("=" * 72)
    t_start = time.time()

    print(f"\n[1/6] Loading base demand data...")
    df = pd.read_csv(DEMAND_8M_PATH, encoding="utf-8-sig")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    print(
        f"  Shape: {df.shape}, date range: {df[DATE_COL].min().date()} .. {df[DATE_COL].max().date()}"
    )

    print(f"\n[2/6] Building exp63 feature frame...")
    df = build_exp63_feature_frame(df)
    df = add_assortment_features_62(df)

    print(f"\n[3/6] Building static location clusters...")
    loc_df = load_location_features()
    loc_clusters, loc_summary, loc_meta, loc_pipeline = build_location_clusters(loc_df)
    df = merge_cluster_map(df, loc_clusters, "cluster_loc")

    df = ensure_category_columns(df)

    print(f"\n[4/6] Train/test split (last 30 days = test)...")
    train_df, test_df, test_start = make_train_test_split(df, test_days=30)
    print(
        f"  Test start: {test_start.date()} | train rows: {len(train_df):,} | test rows: {len(test_df):,}"
    )
    print(f"  Train days: {train_df[DATE_COL].nunique()} | Test days: {test_df[DATE_COL].nunique()}")

    print(f"\n[5/6] Building train-only ts clusters...")
    ts_clusters, ts_summary, ts_meta, ts_pipeline = build_train_ts_clusters(train_df)
    train_df = merge_cluster_map(train_df, ts_clusters, "cluster_ts")
    test_df = merge_cluster_map(test_df, ts_clusters, "cluster_ts")
    train_df = ensure_category_columns(train_df)
    test_df = ensure_category_columns(test_df)

    print(f"\n[6/6] Training families...")
    run_rows: list[dict] = []
    artifact_rows: list[dict] = []
    metrics_by_model: dict[str, pd.DataFrame] = {}
    predictions_frames: dict[str, pd.DataFrame] = {}
    result_rows: list[dict] = []

    family_specs = [
        ("60_baseline_v3", FEATURES_V3, "single"),
        ("61_censoring_behavioral", FEATURES_V3 + FEATURES_61_EXTRA, "single"),
        ("62_assortment_availability", FEATURES_V3 + FEATURES_62_EXTRA, "single"),
        (
            "66_cluster_features",
            FEATURES_V3 + FEATURES_61_EXTRA + ["items_in_bakery_today", "items_in_bakery_lag1", "items_change"] + ["cluster_loc", "cluster_ts"],
            "routed",
        ),
    ]

    for model_name, feature_cols, mode in family_specs:
        t0 = time.time()
        if mode == "single":
            pred_df, metrics_df, result = evaluate_single_model(train_df, test_df, feature_cols, model_name)
        else:
            pred_df, metrics_df, result = evaluate_routed_model(train_df, test_df, feature_cols, model_name)
        elapsed = time.time() - t0
        run_rows.append(
            {
                "model": model_name,
                "family": "demand",
                "status": "OK",
                "elapsed_s": round(elapsed, 1),
                "script": str(Path(__file__).resolve()),
            }
        )
        artifact_rows.append(
            {
                "model": model_name,
                "artifact_dir": str(ARTIFACTS_DIR / model_name),
                "metrics_json": str(ARTIFACTS_DIR / model_name / "metrics.json"),
                "predictions_csv": str(ARTIFACTS_DIR / model_name / "predictions.csv"),
            }
        )
        metrics_by_model[model_name] = metrics_df
        predictions_frames[model_name] = pred_df
        result_rows.append({"model": model_name, **result})
        save_results(ARTIFACTS_DIR / model_name, {**result, "experiment": model_name}, pred_df)

    best_all = merge_metrics_wide(metrics_by_model)
    best_all.to_csv(EXP_DIR / "best_by_sku.csv", index=False, encoding="utf-8-sig")
    best_all.to_csv(EXP_DIR / "best_by_sku_all_models.csv", index=False, encoding="utf-8-sig")
    summary_all = model_summary_from_wide(best_all)
    summary_all.to_csv(EXP_DIR / "summary_by_model.csv", index=False, encoding="utf-8-sig")

    if not best_all.empty:
        best_summary = summary_all.sort_values(["avg_r2", "avg_mae"], ascending=[False, True]).head(1)
        best_model_name = best_summary.iloc[0]["model"]
    else:
        best_model_name = None

    overview = {
        "experiment": "full_benchmark_monthly",
        "test_days": TEST_DAYS if TEST_DAYS else 30,
        "test_start": str(test_start.date()),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "n_pairs": int(len(best_all)),
        "models": result_rows,
        "location_cluster_meta": loc_meta,
        "time_series_cluster_meta": ts_meta,
        "location_cluster_summary": loc_summary.to_dict("records"),
        "time_series_cluster_summary": ts_summary.to_dict("records"),
        "best_model_by_r2": best_model_name,
        "elapsed_s": round(time.time() - t_start, 1),
    }
    with open(EXP_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)

    pd.DataFrame(run_rows).to_csv(EXP_DIR / "run_manifest.csv", index=False, encoding="utf-8-sig")
    with open(EXP_DIR / "artifact_manifest.json", "w", encoding="utf-8") as f:
        json.dump(artifact_rows, f, ensure_ascii=False, indent=2)

    print(f"\nSaved best_by_sku: {EXP_DIR / 'best_by_sku.csv'}")
    print(f"Saved summary: {EXP_DIR / 'summary_by_model.csv'}")
    print(f"Saved metrics: {EXP_DIR / 'metrics.json'}")
    print(f"Elapsed: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()

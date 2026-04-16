"""
Experiment 67: bakery and SKU benchmark.

Compare four families on the same held-out SKU rows:
- current best global model via inference only
- per-SKU LightGBM model
- per-bakery LightGBM model
- two-week rolling average baseline

Prophet stays in experiment 68.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.config import TEST_DAYS
from src.experiments_v2.benchmark_common import (
    BAKERY_COL,
    DATE_COL,
    DEMAND_TARGET,
    PRODUCT_COL,
    build_benchmark_features,
    load_daily_demand,
    load_global_model,
    make_train_test_split,
    predict_global_from_model_data,
    regression_metrics,
    select_feature_columns,
    train_and_predict_quantile,
    two_week_average_predictions,
)


EXP_DIR = Path(__file__).resolve().parent
GLOBAL_MODEL_PATH = EXP_DIR.parent / "66_cluster_features" / "best_model.joblib"

# Fixed benchmark set selected from `sku_metrics_routed.csv`.
# Chosen by a mix of mean R2 and SKU coverage to keep the benchmark representative.
SELECTED_BAKERIES = [
    "Халтурина 8/20 Казань",
    "проспект Мусы Джалиля 20 Наб Челны",
    "Камая 1 Казань",
    "Дзержинского 47 Курск",
    "Фучика 105А Казань",
]

MODEL_NAMES = [
    "global_best",
    "sku_local",
    "bakery_local",
    "two_week_avg",
]

OUTPUT_FILES = {
    "global_best": {
        "metrics": EXP_DIR / "metrics_global_best.csv",
        "predictions": EXP_DIR / "predictions_global_best.csv",
    },
    "sku_local": {
        "metrics": EXP_DIR / "metrics_sku_local.csv",
        "predictions": EXP_DIR / "predictions_sku_local.csv",
    },
    "bakery_local": {
        "metrics": EXP_DIR / "metrics_bakery_local.csv",
        "predictions": EXP_DIR / "predictions_bakery_local.csv",
    },
    "two_week_avg": {
        "metrics": EXP_DIR / "metrics_two_week_avg.csv",
        "predictions": EXP_DIR / "predictions_two_week_avg.csv",
    },
}

SUMMARY_FILES = {
    "best_by_sku": EXP_DIR / "summary_best_by_sku.csv",
    "model_comparison": EXP_DIR / "summary_by_model.csv",
    "model_training": EXP_DIR / "training_log.csv",
    "overview": EXP_DIR / "metrics.json",
}

MIN_TRAIN_ROWS = 30


def _group_value_string(key) -> str:
    if isinstance(key, tuple):
        return " / ".join(str(x) for x in key)
    return str(key)


def _subset_by_key(df: pd.DataFrame, group_cols: list[str], key) -> pd.DataFrame:
    if len(group_cols) == 1:
        return df[df[group_cols[0]] == key].copy()

    mask = pd.Series(True, index=df.index)
    if not isinstance(key, tuple):
        key = (key,)
    for col, val in zip(group_cols, key):
        mask &= df[col] == val
    return df[mask].copy()


def evaluate_group_family(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_features: list[str],
    group_cols: list[str],
    family_name: str,
    min_train_rows: int = MIN_TRAIN_ROWS,
) -> tuple[pd.Series, list[dict]]:
    """Train one model per group and predict the corresponding test rows."""
    predictions = pd.Series(index=test_df.index, dtype=float)
    training_log: list[dict] = []

    for key, test_group in test_df.groupby(group_cols, sort=False):
        train_group = _subset_by_key(train_df, group_cols, key)
        feature_cols = select_feature_columns(train_group, base_features, drop_constants=True)

        preds, info = train_and_predict_quantile(
            train_group,
            test_group,
            feature_cols,
            target_col=DEMAND_TARGET,
            min_train_rows=min_train_rows,
        )
        predictions.loc[test_group.index] = preds

        metrics = regression_metrics(test_group[DEMAND_TARGET], preds)
        training_log.append(
            {
                "family": family_name,
                "group": _group_value_string(key),
                "train_rows": len(train_group),
                "test_rows": len(test_group),
                "train_days": int(train_group[DATE_COL].nunique()) if len(train_group) else 0,
                "test_days": int(test_group[DATE_COL].nunique()),
                "n_features": info.get("n_features", 0),
                "status": info.get("status", "unknown"),
                "mae": round(metrics["mae"], 4) if np.isfinite(metrics["mae"]) else np.nan,
                "mse": round(metrics["mse"], 4) if np.isfinite(metrics["mse"]) else np.nan,
                "r2": round(metrics["r2"], 4) if np.isfinite(metrics["r2"]) else np.nan,
                "wmape": round(metrics["wmape"], 2) if np.isfinite(metrics["wmape"]) else np.nan,
            }
        )

    fallback_value = float(train_df[DEMAND_TARGET].mean()) if len(train_df) else 0.0
    predictions = predictions.fillna(fallback_value)
    return predictions, training_log


def build_prediction_frame(
    test_df: pd.DataFrame,
    predictions: pd.Series | np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    frame = test_df[[DATE_COL, BAKERY_COL, PRODUCT_COL, DEMAND_TARGET]].copy()
    frame["model"] = model_name
    frame["prediction"] = np.asarray(predictions, dtype=float)
    frame["abs_error"] = (frame[DEMAND_TARGET] - frame["prediction"]).abs()
    return frame


def build_metrics_frame(test_df: pd.DataFrame, prediction_frame: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows: list[dict] = []
    grouped = prediction_frame.groupby([BAKERY_COL, PRODUCT_COL], sort=False)
    for (bakery, product), group in grouped:
        metrics = regression_metrics(group[DEMAND_TARGET], group["prediction"])
        rows.append(
            {
                "Пекарня": bakery,
                "Номенклатура": product,
                "model": model_name,
                "n_test_rows": int(len(group)),
                "n_test_days": int(test_df.loc[group.index, DATE_COL].nunique()),
                "r2": round(metrics["r2"], 4) if np.isfinite(metrics["r2"]) else np.nan,
                "mse": round(metrics["mse"], 4),
                "mae": round(metrics["mae"], 4),
                "wmape": round(metrics["wmape"], 2),
            }
        )
    return pd.DataFrame(rows)


def build_model_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in metrics_df.groupby("model", sort=False):
        rows.append(
            {
                "model": model_name,
                "n_pairs": int(len(group)),
                "avg_r2": round(float(group["r2"].mean(skipna=True)), 4),
                "median_r2": round(float(group["r2"].median(skipna=True)), 4),
                "avg_mae": round(float(group["mae"].mean()), 4),
                "median_mae": round(float(group["mae"].median()), 4),
                "avg_mse": round(float(group["mse"].mean()), 4),
                "avg_wmape": round(float(group["wmape"].mean()), 2),
                "win_count": 0,
            }
        )
    return pd.DataFrame(rows)


def build_best_by_sku(metrics_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
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
        return pd.DataFrame(columns=[BAKERY_COL, PRODUCT_COL, "best_model", "best_r2", "best_mae", "best_mse", "best_wmape"])

    wide = wide.reset_index(drop=True)
    r2_cols = [f"{name}_r2" for name in MODEL_NAMES]
    r2_values = wide[r2_cols].fillna(-np.inf)
    best_idx = r2_values.idxmax(axis=1)
    result = wide[[BAKERY_COL, PRODUCT_COL]].copy()
    result["best_model"] = best_idx.str.replace("_r2", "", regex=False)
    best_r2 = r2_values.max(axis=1).replace(-np.inf, np.nan)
    result["best_r2"] = best_r2.to_numpy()
    result["best_mae"] = [
        wide.iloc[i][f"{result.iloc[i]['best_model']}_mae"] for i in range(len(result))
    ]
    result["best_mse"] = [
        wide.iloc[i][f"{result.iloc[i]['best_model']}_mse"] for i in range(len(result))
    ]
    result["best_wmape"] = [
        wide.iloc[i][f"{result.iloc[i]['best_model']}_wmape"] for i in range(len(result))
    ]
    return pd.concat([result.reset_index(drop=True), wide.drop(columns=[BAKERY_COL, PRODUCT_COL]).reset_index(drop=True)], axis=1)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 67: bakery and SKU benchmark")
    parser.add_argument("--test-days", type=int, default=TEST_DAYS, help="Held-out tail length in days")
    parser.add_argument("--min-train-rows", type=int, default=MIN_TRAIN_ROWS, help="Minimum rows to fit local models")
    parser.add_argument(
        "--global-model",
        default=str(GLOBAL_MODEL_PATH),
        help="Path to the saved global model artifact from experiment 66",
    )
    args = parser.parse_args()

    start = time.time()
    print("=" * 72)
    print("EXPERIMENT 67: Bakery / SKU benchmark")
    print("=" * 72)
    print(f"Selected bakeries: {len(SELECTED_BAKERIES)}")
    for bakery in SELECTED_BAKERIES:
        print(f"  - {bakery}")

    print(f"\n[1/5] Loading daily demand data...")
    raw_df = load_daily_demand(SELECTED_BAKERIES)
    print(
        f"  Rows: {len(raw_df):,}, days: {raw_df[DATE_COL].nunique()}, "
        f"bakeries: {raw_df[BAKERY_COL].nunique()}, SKUs: {raw_df[PRODUCT_COL].nunique()}"
    )

    print(f"\n[2/5] Loading global model artifact...")
    model_data = load_global_model(args.global_model)
    base_features = model_data["features"]
    print(f"  Global variant: {model_data['variant']}")
    print(f"  Global features: {len(base_features)}")

    print(f"\n[3/5] Building benchmark features...")
    featured_df = build_benchmark_features(raw_df, model_data)
    train_df, test_df, test_start = make_train_test_split(featured_df, test_days=args.test_days)
    print(
        f"  Test start: {test_start.date()} | train rows: {len(train_df):,} | "
        f"test rows: {len(test_df):,}"
    )
    print(f"  Train days: {train_df[DATE_COL].nunique()} | Test days: {test_df[DATE_COL].nunique()}")

    metrics_frames: dict[str, pd.DataFrame] = {}
    predictions_frames: dict[str, pd.DataFrame] = {}
    training_logs: list[dict] = []

    print(f"\n[4/5] Evaluating model families...")

    global_pred = pd.Series(
        predict_global_from_model_data(test_df, model_data),
        index=test_df.index,
        name="prediction",
    )
    global_pred_frame = build_prediction_frame(test_df, global_pred, "global_best")
    metrics_frames["global_best"] = build_metrics_frame(test_df, global_pred_frame, "global_best")
    predictions_frames["global_best"] = global_pred_frame

    sku_pred, sku_log = evaluate_group_family(
        train_df,
        test_df,
        base_features,
        [BAKERY_COL, PRODUCT_COL],
        "sku_local",
        min_train_rows=args.min_train_rows,
    )
    sku_pred_frame = build_prediction_frame(test_df, sku_pred, "sku_local")
    metrics_frames["sku_local"] = build_metrics_frame(test_df, sku_pred_frame, "sku_local")
    predictions_frames["sku_local"] = sku_pred_frame
    training_logs.extend(sku_log)

    bakery_pred, bakery_log = evaluate_group_family(
        train_df,
        test_df,
        base_features,
        [BAKERY_COL],
        "bakery_local",
        min_train_rows=args.min_train_rows,
    )
    bakery_pred_frame = build_prediction_frame(test_df, bakery_pred, "bakery_local")
    metrics_frames["bakery_local"] = build_metrics_frame(test_df, bakery_pred_frame, "bakery_local")
    predictions_frames["bakery_local"] = bakery_pred_frame
    training_logs.extend(bakery_log)

    avg_series = pd.Series(
        two_week_average_predictions(featured_df, [BAKERY_COL, PRODUCT_COL], target_col=DEMAND_TARGET),
        index=featured_df.index,
    )
    avg_pred_frame = build_prediction_frame(test_df, avg_series.loc[test_df.index], "two_week_avg")
    metrics_frames["two_week_avg"] = build_metrics_frame(test_df, avg_pred_frame, "two_week_avg")
    predictions_frames["two_week_avg"] = avg_pred_frame

    print(f"\n[5/5] Saving artifacts...")
    for model_name, files in OUTPUT_FILES.items():
        save_csv(metrics_frames[model_name], files["metrics"])
        save_csv(predictions_frames[model_name], files["predictions"])
        print(f"  Saved {model_name}: {files['metrics'].name}, {files['predictions'].name}")

    metrics_all = pd.concat(metrics_frames.values(), ignore_index=True)
    model_summary = build_model_summary(metrics_all)
    best_by_sku = build_best_by_sku(metrics_frames)
    best_counts = best_by_sku["best_model"].value_counts().reindex(MODEL_NAMES, fill_value=0)
    model_summary["win_count"] = model_summary["model"].map(best_counts).fillna(0).astype(int)

    save_csv(model_summary, SUMMARY_FILES["model_comparison"])
    save_csv(best_by_sku, SUMMARY_FILES["best_by_sku"])
    save_csv(pd.DataFrame(training_logs), SUMMARY_FILES["model_training"])

    overview = {
        "experiment": "67_bakery_sku_benchmark",
        "selected_bakeries": SELECTED_BAKERIES,
        "test_days": args.test_days,
        "min_train_rows": args.min_train_rows,
        "global_model_path": str(args.global_model),
        "global_variant": model_data["variant"],
        "rows_total": int(len(raw_df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "pairs_total": int(best_by_sku[[BAKERY_COL, PRODUCT_COL]].drop_duplicates().shape[0]),
        "model_summary": model_summary.to_dict("records"),
        "best_by_sku_rows": int(len(best_by_sku)),
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    SUMMARY_FILES["overview"].write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nModel summary:")
    print(model_summary.to_string(index=False))
    print(f"\nBest-by-SKU rows: {len(best_by_sku)}")
    print(f"Saved overview JSON: {SUMMARY_FILES['overview'].name}")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

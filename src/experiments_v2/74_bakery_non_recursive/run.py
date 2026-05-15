"""
Experiment 74: non-recursive bakery forecasting on short horizon.

Use actual holdout features for each bakery-day row.
This matches the real production setup more closely when the forecast
horizon is only up to 7 days ahead and recursive multi-week rollout is
not required.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.experiments_v2.bakery_day_forecast import BASE_FEATURES
from src.experiments_v2.bakery_day_forecast import BAKERY_ID_COL
from src.experiments_v2.bakery_day_forecast import BAKERY_NAME_COL
from src.experiments_v2.bakery_day_forecast import CITY_COL
from src.experiments_v2.bakery_day_forecast import DATE_COL
from src.experiments_v2.bakery_day_forecast import TARGET_COL
from src.experiments_v2.bakery_day_forecast import build_model_frame
from src.experiments_v2.bakery_day_forecast import cast_category_columns
from src.experiments_v2.bakery_day_forecast import load_dataset
from src.experiments_v2.bakery_day_forecast import make_train_test_split
from src.experiments_v2.bakery_day_forecast import regression_metrics
from src.experiments_v2.common import predict_clipped
from src.experiments_v2.common import train_lgbm


EXP73_RUN_PATH = ROOT / "src" / "experiments_v2" / "73_weekly_total_recursive" / "run.py"
EXP73_SPEC = importlib.util.spec_from_file_location("exp73_run", EXP73_RUN_PATH)
EXP73_RUN = importlib.util.module_from_spec(EXP73_SPEC)
assert EXP73_SPEC and EXP73_SPEC.loader
EXP73_SPEC.loader.exec_module(EXP73_RUN)
compute_heuristic_blend_prediction = EXP73_RUN.compute_heuristic_blend_prediction


EXP_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "bakery_daily_sales.csv"

MODEL_NAMES = [
    "seasonal_naive_lag7_non_recursive",
    "daily_baseline_non_recursive",
    "heuristic_blend_non_recursive",
]

DEFAULT_TEST_DAYS = 7
MIN_TRAIN_ROWS = 90


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def build_output_files(output_dir: Path) -> tuple[dict[str, dict[str, Path]], dict[str, Path]]:
    output_files = {
        name: {
            "metrics": output_dir / f"metrics_{name}.csv",
            "predictions": output_dir / f"predictions_{name}.csv",
        }
        for name in MODEL_NAMES
    }
    summary_files = {
        "summary_by_model": output_dir / "summary_by_model.csv",
        "summary_best_by_bakery": output_dir / "summary_best_by_bakery.csv",
        "overview": output_dir / "metrics.json",
    }
    return output_files, summary_files


def select_feature_columns(train_df: pd.DataFrame, base_features: list[str]) -> list[str]:
    selected: list[str] = []
    for col in base_features:
        if col not in train_df.columns:
            continue
        series = train_df[col]
        if series.isna().all():
            continue
        if series.nunique(dropna=True) <= 1:
            continue
        selected.append(col)
    return selected


def build_prediction_frame(actual_df: pd.DataFrame, preds: pd.DataFrame, model_name: str) -> pd.DataFrame:
    frame = actual_df[[DATE_COL, BAKERY_ID_COL, BAKERY_NAME_COL, CITY_COL, TARGET_COL]].copy()
    merged = frame.merge(preds[[DATE_COL, BAKERY_ID_COL, "prediction"]], on=[DATE_COL, BAKERY_ID_COL], how="left")
    merged["model"] = model_name
    merged["prediction"] = pd.to_numeric(merged["prediction"], errors="coerce").fillna(0.0)
    merged["error"] = merged[TARGET_COL] - merged["prediction"]
    merged["abs_error"] = merged["error"].abs()
    return merged.sort_values([BAKERY_ID_COL, DATE_COL]).reset_index(drop=True)


def build_metrics_frame(pred_frame: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows: list[dict] = []
    for bakery_id, group in pred_frame.groupby(BAKERY_ID_COL, sort=False):
        m = regression_metrics(group[TARGET_COL], group["prediction"])
        rows.append(
            {
                BAKERY_ID_COL: bakery_id,
                BAKERY_NAME_COL: group[BAKERY_NAME_COL].iloc[0],
                CITY_COL: group[CITY_COL].iloc[0],
                "model": model_name,
                "n_test_days": int(group[DATE_COL].nunique()),
                "mae": round(m["mae"], 6),
                "mse": round(m["mse"], 6),
                "wmape": round(m["wmape"], 6),
                "bias": round(m["bias"], 6),
            }
        )
    return pd.DataFrame(rows)


def build_model_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in metrics_df.groupby("model", sort=False):
        rows.append(
            {
                "model": model_name,
                "n_bakeries": int(len(group)),
                "avg_mae": round(float(group["mae"].mean()), 6),
                "median_mae": round(float(group["mae"].median()), 6),
                "avg_mse": round(float(group["mse"].mean()), 6),
                "avg_wmape": round(float(group["wmape"].mean()), 6),
                "avg_bias": round(float(group["bias"].mean()), 6),
                "median_abs_bias": round(float(group["bias"].abs().median()), 6),
                "win_count": 0,
            }
        )
    return pd.DataFrame(rows)


def build_best_by_bakery(metrics_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    wide: pd.DataFrame | None = None
    for model_name, frame in metrics_frames.items():
        sub = frame[[BAKERY_ID_COL, BAKERY_NAME_COL, CITY_COL, "mae", "mse", "wmape", "bias"]].copy()
        sub = sub.rename(
            columns={
                "mae": f"{model_name}_mae",
                "mse": f"{model_name}_mse",
                "wmape": f"{model_name}_wmape",
                "bias": f"{model_name}_bias",
            }
        )
        merge_keys = [BAKERY_ID_COL, BAKERY_NAME_COL, CITY_COL]
        wide = sub if wide is None else wide.merge(sub, on=merge_keys, how="outer")

    if wide is None or wide.empty:
        return pd.DataFrame(columns=[BAKERY_ID_COL, "best_model", "best_mae"])

    mae_cols = [f"{name}_mae" for name in MODEL_NAMES]
    mae_values = wide[mae_cols].fillna(np.inf)
    wide["best_model"] = mae_values.idxmin(axis=1).str.replace("_mae", "", regex=False)
    wide["best_mae"] = mae_values.min(axis=1).replace(np.inf, np.nan)
    return wide


def train_daily_baseline_model(train_df: pd.DataFrame, *, min_train_rows: int) -> tuple[object | None, list[str], dict]:
    feature_cols = select_feature_columns(train_df, BASE_FEATURES)
    if len(feature_cols) == 0 or len(train_df) < min_train_rows:
        return None, feature_cols, {"status": "fallback_mean", "n_features": len(feature_cols)}

    train_x = train_df[feature_cols].copy()
    train_x, _ = cast_category_columns(train_x, train_x.copy(), feature_cols)
    model = train_lgbm(train_x, train_df[TARGET_COL])
    return model, feature_cols, {"status": "trained", "n_features": len(feature_cols)}


def seasonal_naive_lag7_non_recursive_backtest(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    fallback_mean = float(train_df[TARGET_COL].mean()) if len(train_df) else 0.0
    preds = pd.to_numeric(test_df.get("bakery_sales_lag7", 0.0), errors="coerce").fillna(fallback_mean).clip(lower=0.0)
    out = test_df[[DATE_COL, BAKERY_ID_COL]].copy()
    out["prediction"] = preds.to_numpy(dtype=float)
    return out, {"status": "non_recursive_lag7", "n_features": 1}


def daily_baseline_non_recursive_backtest(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    min_train_rows: int,
) -> tuple[pd.DataFrame, dict]:
    model, feature_cols, info = train_daily_baseline_model(train_df, min_train_rows=min_train_rows)
    fallback_mean = float(train_df[TARGET_COL].mean()) if len(train_df) else 0.0
    out = test_df[[DATE_COL, BAKERY_ID_COL]].copy()

    if model is None or len(feature_cols) == 0:
        out["prediction"] = fallback_mean
        return out, info

    predict_x = test_df[feature_cols].copy()
    _, predict_x = cast_category_columns(train_df[feature_cols].copy(), predict_x, feature_cols)
    preds = predict_clipped(model, predict_x)
    out["prediction"] = preds
    return out, info


def heuristic_blend_non_recursive_backtest(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    min_train_rows: int,
) -> tuple[pd.DataFrame, dict]:
    model, feature_cols, info = train_daily_baseline_model(train_df, min_train_rows=min_train_rows)
    fallback_mean = float(train_df[TARGET_COL].mean()) if len(train_df) else 0.0
    out = test_df[[DATE_COL, BAKERY_ID_COL]].copy()

    if model is None or len(feature_cols) == 0:
        base_pred = np.full(len(test_df), fallback_mean, dtype=float)
    else:
        predict_x = test_df[feature_cols].copy()
        _, predict_x = cast_category_columns(train_df[feature_cols].copy(), predict_x, feature_cols)
        base_pred = predict_clipped(model, predict_x)

    out["prediction"] = compute_heuristic_blend_prediction(test_df, base_pred, fallback_mean=fallback_mean)
    out_info = dict(info)
    out_info["status"] = f"{info.get('status', 'unknown')}_plus_non_recursive_heuristic_blend"
    return out, out_info


def evaluate_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    min_train_rows: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], list[dict]]:
    prediction_frames: dict[str, pd.DataFrame] = {}
    metrics_frames: dict[str, pd.DataFrame] = {}
    training_log: list[dict] = []

    model_runs = [
        (
            "seasonal_naive_lag7_non_recursive",
            seasonal_naive_lag7_non_recursive_backtest(train_df, test_df),
        ),
        (
            "daily_baseline_non_recursive",
            daily_baseline_non_recursive_backtest(train_df, test_df, min_train_rows=min_train_rows),
        ),
        (
            "heuristic_blend_non_recursive",
            heuristic_blend_non_recursive_backtest(train_df, test_df, min_train_rows=min_train_rows),
        ),
    ]

    for model_name, (preds, info) in model_runs:
        frame = build_prediction_frame(test_df, preds, model_name)
        prediction_frames[model_name] = frame
        metrics_frames[model_name] = build_metrics_frame(frame, model_name)
        gm = regression_metrics(frame[TARGET_COL], frame["prediction"])
        training_log.append(
            {
                "model": model_name,
                "status": info.get("status", "unknown"),
                "n_features": info.get("n_features", 0),
                "rows_train": len(train_df),
                "rows_test": len(test_df),
                "mae": round(gm["mae"], 6),
                "mse": round(gm["mse"], 6),
                "wmape": round(gm["wmape"], 6),
                "bias": round(gm["bias"], 6),
            }
        )

    return prediction_frames, metrics_frames, training_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 74: non-recursive bakery backtest")
    parser.add_argument("--dataset-path", default=str(DATA_PATH))
    parser.add_argument("--test-days", type=int, default=DEFAULT_TEST_DAYS)
    parser.add_argument("--min-train-rows", type=int, default=MIN_TRAIN_ROWS)
    parser.add_argument("--output-dir", default=str(EXP_DIR))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_files, summary_files = build_output_files(output_dir)

    start = time.time()
    print("=" * 72)
    print("EXPERIMENT 74: Non-recursive bakery backtest")
    print("=" * 72)

    print("\n[1/4] Loading bakery-day frame...")
    df = build_model_frame(load_dataset(args.dataset_path))
    print(
        f"  rows={len(df):,} | dates={df[DATE_COL].nunique()} | bakeries={df[BAKERY_ID_COL].nunique()} | "
        f"range={df[DATE_COL].min().date()}..{df[DATE_COL].max().date()}"
    )

    print("\n[2/4] Building holdout split...")
    train_df, test_df, test_start = make_train_test_split(df, args.test_days)
    print(
        f"  test_start={test_start.date()} | rows_train={len(train_df):,} | rows_test={len(test_df):,} | "
        f"train_days={train_df[DATE_COL].nunique()} | test_days={test_df[DATE_COL].nunique()}"
    )

    print("\n[3/4] Running non-recursive backtests...")
    prediction_frames, metrics_frames, training_log = evaluate_models(
        train_df,
        test_df,
        min_train_rows=args.min_train_rows,
    )

    print("\n[4/4] Saving artifacts...")
    for model_name in MODEL_NAMES:
        save_csv(metrics_frames[model_name], output_files[model_name]["metrics"])
        save_csv(prediction_frames[model_name], output_files[model_name]["predictions"])
        print(f"  saved {model_name}")

    metrics_all = pd.concat(metrics_frames.values(), ignore_index=True)
    summary_by_model = build_model_summary(metrics_all)
    best_by_bakery = build_best_by_bakery(metrics_frames)
    best_counts = best_by_bakery["best_model"].value_counts().reindex(MODEL_NAMES, fill_value=0)
    summary_by_model["win_count"] = summary_by_model["model"].map(best_counts).fillna(0).astype(int)

    save_csv(summary_by_model, summary_files["summary_by_model"])
    save_csv(best_by_bakery, summary_files["summary_best_by_bakery"])

    overview = {
        "experiment": "74_bakery_non_recursive",
        "dataset_path": str(args.dataset_path),
        "test_days": args.test_days,
        "min_train_rows": args.min_train_rows,
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "bakeries_total": int(df[BAKERY_ID_COL].nunique()),
        "test_start": str(test_start.date()),
        "summary_by_model": summary_by_model.to_dict("records"),
        "training_log": training_log,
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    summary_files["overview"].write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nModel summary:")
    print(summary_by_model.to_string(index=False))
    print(f"\nDone in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()

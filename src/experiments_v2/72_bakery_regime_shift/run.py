"""
Experiment 72: bakery regime-shift backtest.

Compare four bakery-level approaches intended to reduce amplitude collapse:
1) normalized target model
2) fast seasonal features
3) weekly total -> daily share
4) global + local seasonal override

Baseline global LightGBM is kept as a reference.
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

from src.experiments_v2.bakery_regime_shift_common import BASE_FEATURES
from src.experiments_v2.bakery_regime_shift_common import BAKERY_ID_COL
from src.experiments_v2.bakery_regime_shift_common import BAKERY_NAME_COL
from src.experiments_v2.bakery_regime_shift_common import CITY_COL
from src.experiments_v2.bakery_regime_shift_common import DATE_COL
from src.experiments_v2.bakery_regime_shift_common import FAST_SEASONAL_FEATURES
from src.experiments_v2.bakery_regime_shift_common import TARGET_COL
from src.experiments_v2.bakery_regime_shift_common import WEEKLY_FEATURES
from src.experiments_v2.bakery_regime_shift_common import add_fast_seasonal_features
from src.experiments_v2.bakery_regime_shift_common import add_normalized_target
from src.experiments_v2.bakery_regime_shift_common import bakery_predictability_table
from src.experiments_v2.bakery_regime_shift_common import build_bakery_weekly_frame
from src.experiments_v2.bakery_regime_shift_common import cast_category_columns
from src.experiments_v2.bakery_regime_shift_common import compute_recent_weekday_share_lookup
from src.experiments_v2.bakery_regime_shift_common import load_bakery_frame
from src.experiments_v2.bakery_regime_shift_common import local_seasonal_scaled_prediction
from src.experiments_v2.bakery_regime_shift_common import make_train_test_split
from src.experiments_v2.bakery_regime_shift_common import regression_metrics
from src.experiments_v2.common import predict_clipped
from src.experiments_v2.common import train_lgbm


EXP_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "bakery_daily_sales.csv"

MODEL_NAMES = [
    "baseline_global_lgbm",
    "normalized_target_lgbm",
    "fast_seasonal_lgbm",
    "weekly_total_daily_share",
    "global_local_hybrid",
]

OUTPUT_FILES = {
    name: {
        "metrics": EXP_DIR / f"metrics_{name}.csv",
        "predictions": EXP_DIR / f"predictions_{name}.csv",
    }
    for name in MODEL_NAMES
}

SUMMARY_FILES = {
    "best_by_bakery": EXP_DIR / "summary_best_by_bakery.csv",
    "model_comparison": EXP_DIR / "summary_by_model.csv",
    "training_log": EXP_DIR / "training_log.csv",
    "overview": EXP_DIR / "metrics.json",
}

DEFAULT_TEST_DAYS = 30
MIN_TRAIN_ROWS = 60


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


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


def build_prediction_frame(test_df: pd.DataFrame, preds: np.ndarray, model_name: str) -> pd.DataFrame:
    frame = test_df[[DATE_COL, BAKERY_ID_COL, BAKERY_NAME_COL, CITY_COL, TARGET_COL]].copy()
    frame["model"] = model_name
    frame["prediction"] = np.asarray(preds, dtype=float)
    frame["error"] = frame[TARGET_COL] - frame["prediction"]
    frame["abs_error"] = frame["error"].abs()
    return frame


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
    best_idx = mae_values.idxmin(axis=1)
    wide["best_model"] = best_idx.str.replace("_mae", "", regex=False)
    wide["best_mae"] = mae_values.min(axis=1).replace(np.inf, np.nan)
    return wide


def fit_predict_global_lgbm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    min_train_rows: int,
) -> tuple[np.ndarray, dict]:
    feature_cols = select_feature_columns(train_df, feature_cols)
    if len(feature_cols) == 0 or len(train_df) < min_train_rows:
        fallback = float(train_df[TARGET_COL].mean()) if len(train_df) else 0.0
        return np.full(len(test_df), fallback), {"status": "fallback_mean", "n_features": len(feature_cols)}

    train_x = train_df[feature_cols].copy()
    test_x = test_df[feature_cols].copy()
    train_x, test_x = cast_category_columns(train_x, test_x, feature_cols)
    model = train_lgbm(train_x, train_df[TARGET_COL])
    preds = predict_clipped(model, test_x)
    return preds, {"status": "trained", "n_features": len(feature_cols)}


def fit_predict_normalized_target(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    min_train_rows: int,
) -> tuple[np.ndarray, dict]:
    train_work = add_normalized_target(train_df)
    test_work = add_normalized_target(test_df)
    feature_cols = select_feature_columns(train_work, BASE_FEATURES)
    if len(feature_cols) == 0 or len(train_work) < min_train_rows:
        fallback = float(train_df[TARGET_COL].mean()) if len(train_df) else 0.0
        return np.full(len(test_df), fallback), {"status": "fallback_mean", "n_features": len(feature_cols)}

    train_x = train_work[feature_cols].copy()
    test_x = test_work[feature_cols].copy()
    train_x, test_x = cast_category_columns(train_x, test_x, feature_cols)
    model = train_lgbm(train_x, train_work["target_norm_roll7"])
    pred_ratio = np.clip(np.asarray(model.predict(test_x), dtype=float), 0.0, None)
    preds = pred_ratio * pd.to_numeric(test_work["bakery_sales_roll_mean7"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return preds, {"status": "trained", "n_features": len(feature_cols)}


def fit_predict_fast_seasonal(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    min_train_rows: int,
) -> tuple[np.ndarray, dict]:
    train_work = add_fast_seasonal_features(train_df)
    test_work = add_fast_seasonal_features(test_df)
    feature_cols = select_feature_columns(train_work, FAST_SEASONAL_FEATURES)
    if len(feature_cols) == 0 or len(train_work) < min_train_rows:
        fallback = float(train_df[TARGET_COL].mean()) if len(train_df) else 0.0
        return np.full(len(test_df), fallback), {"status": "fallback_mean", "n_features": len(feature_cols)}

    train_x = train_work[feature_cols].copy()
    test_x = test_work[feature_cols].copy()
    train_x, test_x = cast_category_columns(train_x, test_x, feature_cols)
    model = train_lgbm(train_x, train_work[TARGET_COL])
    preds = predict_clipped(model, test_x)
    return preds, {"status": "trained", "n_features": len(feature_cols)}


def fit_predict_weekly_total_daily_share(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    min_train_rows: int,
) -> tuple[np.ndarray, dict]:
    train_daily = train_df.sort_values([BAKERY_ID_COL, DATE_COL]).copy()
    test_daily = test_df.sort_values([BAKERY_ID_COL, DATE_COL]).copy()
    weekly_all = build_bakery_weekly_frame(pd.concat([train_daily, test_daily], ignore_index=True))
    train_weekly = weekly_all[weekly_all["week_start"] < test_daily[DATE_COL].min()].copy()
    feature_cols = select_feature_columns(train_weekly, WEEKLY_FEATURES)
    if len(feature_cols) == 0 or len(train_weekly) < min_train_rows:
        fallback = float(train_df[TARGET_COL].mean()) if len(train_df) else 0.0
        return np.full(len(test_df), fallback), {"status": "fallback_mean", "n_features": len(feature_cols)}

    preds_by_key: dict[tuple[int, pd.Timestamp], float] = {}
    week_starts = sorted((test_daily[DATE_COL] - pd.to_timedelta(test_daily[DATE_COL].dt.dayofweek, unit="D")).unique())

    for week_start in week_starts:
        hist_daily = pd.concat(
            [
                train_daily,
                test_daily[(test_daily[DATE_COL] < week_start)].assign(**{TARGET_COL: test_daily.loc[test_daily[DATE_COL] < week_start, TARGET_COL]}),
            ],
            ignore_index=True,
        )
        hist_daily = hist_daily[hist_daily[DATE_COL] < week_start].copy()
        hist_weekly = build_bakery_weekly_frame(hist_daily)
        pred_week = build_bakery_weekly_frame(
            pd.concat([hist_daily, test_daily[(test_daily[DATE_COL] >= week_start) & (test_daily[DATE_COL] < week_start + pd.Timedelta(days=7))]], ignore_index=True)
        )
        pred_week = pred_week[pred_week["week_start"] == week_start].copy()
        usable_features = select_feature_columns(hist_weekly, WEEKLY_FEATURES)
        if len(usable_features) == 0 or len(hist_weekly) < min_train_rows:
            fallback_value = float(hist_weekly["week_sales"].mean()) if len(hist_weekly) else 0.0
            pred_week["pred_week_sales"] = fallback_value
        else:
            train_x = hist_weekly[usable_features].copy()
            pred_x = pred_week[usable_features].copy()
            train_x, pred_x = cast_category_columns(train_x, pred_x, usable_features)
            model = train_lgbm(train_x, hist_weekly["week_sales"])
            pred_week["pred_week_sales"] = predict_clipped(model, pred_x)

        share_lookup = compute_recent_weekday_share_lookup(hist_daily, recent_weeks=4)
        pred_days = test_daily[(test_daily[DATE_COL] >= week_start) & (test_daily[DATE_COL] < week_start + pd.Timedelta(days=7))].copy()
        pred_days = pred_days.merge(pred_week[[BAKERY_ID_COL, "pred_week_sales"]], on=BAKERY_ID_COL, how="left")
        pred_days = pred_days.merge(share_lookup, on=[BAKERY_ID_COL, "dow"], how="left")
        pred_days["weekday_share"] = pred_days["weekday_share"].fillna(1.0 / 7.0)
        pred_days["pred"] = pred_days["pred_week_sales"] * pred_days["weekday_share"]
        for row in pred_days.itertuples(index=False):
            preds_by_key[(getattr(row, BAKERY_ID_COL), getattr(row, DATE_COL))] = float(getattr(row, "pred"))

    preds = np.array([preds_by_key.get((row[BAKERY_ID_COL], row[DATE_COL]), 0.0) for _, row in test_daily.iterrows()], dtype=float)
    return preds, {"status": "trained", "n_features": len(feature_cols)}


def fit_predict_global_local_hybrid(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    min_train_rows: int,
) -> tuple[np.ndarray, dict]:
    base_preds, base_info = fit_predict_fast_seasonal(train_df, test_df, min_train_rows=min_train_rows)
    predictability = bakery_predictability_table(train_df)
    pred_map = pd.DataFrame({"_idx": test_df.index, "base_pred": base_preds, BAKERY_ID_COL: test_df[BAKERY_ID_COL].values, DATE_COL: test_df[DATE_COL].values})
    pred_map = pred_map.merge(predictability, on=BAKERY_ID_COL, how="left")

    out_preds = pred_map["base_pred"].to_numpy(dtype=float)
    for date_value in sorted(test_df[DATE_COL].unique()):
        hist = pd.concat([train_df, test_df[test_df[DATE_COL] < date_value]], ignore_index=True)
        local_df = local_seasonal_scaled_prediction(hist, pd.Timestamp(date_value))
        local_df = local_df.merge(predictability, on=BAKERY_ID_COL, how="left")
        local_df["use_local_override"] = local_df["use_local_override"].fillna(False)
        mask = (pred_map[DATE_COL] == date_value)
        current = pred_map.loc[mask, [BAKERY_ID_COL]].merge(local_df[[BAKERY_ID_COL, "local_scaled_pred", "use_local_override"]], on=BAKERY_ID_COL, how="left")
        replace_mask = current["use_local_override"].fillna(False).to_numpy(dtype=bool)
        out_preds[np.where(mask)[0][replace_mask]] = current.loc[replace_mask, "local_scaled_pred"].to_numpy(dtype=float)

    return out_preds, {"status": base_info["status"], "n_features": base_info["n_features"]}


def evaluate_models(train_df: pd.DataFrame, test_df: pd.DataFrame, *, min_train_rows: int) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], list[dict]]:
    predictions: dict[str, pd.DataFrame] = {}
    metrics: dict[str, pd.DataFrame] = {}
    training_log: list[dict] = []

    evaluators = {
        "baseline_global_lgbm": lambda: fit_predict_global_lgbm(train_df, test_df, BASE_FEATURES, min_train_rows=min_train_rows),
        "normalized_target_lgbm": lambda: fit_predict_normalized_target(train_df, test_df, min_train_rows=min_train_rows),
        "fast_seasonal_lgbm": lambda: fit_predict_fast_seasonal(train_df, test_df, min_train_rows=min_train_rows),
        "weekly_total_daily_share": lambda: fit_predict_weekly_total_daily_share(train_df, test_df, min_train_rows=min_train_rows),
        "global_local_hybrid": lambda: fit_predict_global_local_hybrid(train_df, test_df, min_train_rows=min_train_rows),
    }

    for model_name in MODEL_NAMES:
        preds, info = evaluators[model_name]()
        pred_frame = build_prediction_frame(test_df, preds, model_name)
        predictions[model_name] = pred_frame
        metrics[model_name] = build_metrics_frame(pred_frame, model_name)
        global_metrics = regression_metrics(test_df[TARGET_COL], preds)
        training_log.append(
            {
                "model": model_name,
                "status": info.get("status", "unknown"),
                "n_features": info.get("n_features", 0),
                "rows_train": len(train_df),
                "rows_test": len(test_df),
                "mae": round(global_metrics["mae"], 6),
                "mse": round(global_metrics["mse"], 6),
                "wmape": round(global_metrics["wmape"], 6),
                "bias": round(global_metrics["bias"], 6),
            }
        )

    return predictions, metrics, training_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 72: bakery regime shift")
    parser.add_argument("--dataset-path", default=str(DATA_PATH))
    parser.add_argument("--test-days", type=int, default=DEFAULT_TEST_DAYS)
    parser.add_argument("--min-train-rows", type=int, default=MIN_TRAIN_ROWS)
    args = parser.parse_args()

    start = time.time()
    print("=" * 72)
    print("EXPERIMENT 72: Bakery regime shift")
    print("=" * 72)

    print("\n[1/4] Loading bakery-day frame...")
    df = load_bakery_frame(args.dataset_path)
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

    print("\n[3/4] Evaluating models...")
    prediction_frames, metrics_frames, training_log = evaluate_models(
        train_df,
        test_df,
        min_train_rows=args.min_train_rows,
    )

    print("\n[4/4] Saving artifacts...")
    for model_name in MODEL_NAMES:
        save_csv(metrics_frames[model_name], OUTPUT_FILES[model_name]["metrics"])
        save_csv(prediction_frames[model_name], OUTPUT_FILES[model_name]["predictions"])
        print(f"  saved {model_name}")

    metrics_all = pd.concat(metrics_frames.values(), ignore_index=True)
    model_summary = build_model_summary(metrics_all)
    best_by_bakery = build_best_by_bakery(metrics_frames)
    best_counts = best_by_bakery["best_model"].value_counts().reindex(MODEL_NAMES, fill_value=0)
    model_summary["win_count"] = model_summary["model"].map(best_counts).fillna(0).astype(int)

    save_csv(model_summary, SUMMARY_FILES["model_comparison"])
    save_csv(best_by_bakery, SUMMARY_FILES["best_by_bakery"])
    save_csv(pd.DataFrame(training_log), SUMMARY_FILES["training_log"])

    overview = {
        "experiment": "72_bakery_regime_shift",
        "dataset_path": str(args.dataset_path),
        "test_days": args.test_days,
        "min_train_rows": args.min_train_rows,
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "bakeries_total": int(df[BAKERY_ID_COL].nunique()),
        "test_start": str(test_start.date()),
        "model_summary": model_summary.to_dict("records"),
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    SUMMARY_FILES["overview"].write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nModel summary:")
    print(model_summary.to_string(index=False))
    print(f"\nDone in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()

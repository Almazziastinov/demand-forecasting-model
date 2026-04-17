"""
Monthly local benchmark on the full SKU space.

Compares:
- prophet_local: one Prophet model per [Пекарня, Номенклатура]
- lgbm_local: one LightGBM model per [Пекарня, Номенклатура]
- two_week_avg: rolling 14-day baseline

This runner uses the same 30-day holdout as the monthly global benchmark.
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

from src.experiments_v2.benchmark_common import (  # noqa: E402
    BAKERY_COL,
    DATE_COL,
    DEMAND_TARGET,
    PRODUCT_COL,
    load_daily_demand,
    make_train_test_split,
    regression_metrics,
    select_feature_columns,
    train_and_predict_quantile,
    two_week_average_predictions,
)
from src.experiments_v2.common import FEATURES_V3  # noqa: E402
from src.experiments_v2.monthly_benchmark_common import (  # noqa: E402
    FEATURES_61_EXTRA,
    FEATURES_62_EXTRA,
    add_assortment_features_62,
    build_exp63_feature_frame,
)


EXP_DIR = Path(__file__).resolve().parent

MODEL_NAMES = [
    "prophet_local",
    "lgbm_local",
    "two_week_avg",
]

OUTPUT_FILES = {
    "prophet_local": {
        "metrics": EXP_DIR / "metrics_prophet_local.csv",
        "predictions": EXP_DIR / "predictions_prophet_local.csv",
    },
    "lgbm_local": {
        "metrics": EXP_DIR / "metrics_lgbm_local.csv",
        "predictions": EXP_DIR / "predictions_lgbm_local.csv",
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

RU_HOLIDAYS = {
    (1, 1): "new_year",
    (1, 2): "new_year",
    (1, 3): "new_year",
    (1, 4): "new_year",
    (1, 5): "new_year",
    (1, 6): "new_year",
    (1, 7): "christmas",
    (1, 8): "new_year",
    (2, 23): "defender_day",
    (3, 8): "women_day",
    (5, 1): "spring_day",
    (5, 9): "victory_day",
    (6, 12): "russia_day",
    (11, 4): "unity_day",
}

PROPHET_REGRESSORS = (
    "is_holiday",
    "is_pre_holiday",
    "is_post_holiday",
    "is_payday_week",
    "is_month_start",
    "is_month_end",
)

MIN_TRAIN_ROWS = 30
BASE_FEATURES = FEATURES_V3 + FEATURES_61_EXTRA + FEATURES_62_EXTRA


def prophet_available() -> bool:
    try:
        from prophet import Prophet  # noqa: F401
    except ImportError:
        return False
    return True


def _import_prophet():
    try:
        from prophet import Prophet
    except ImportError as exc:  # pragma: no cover - handled by fallback path
        raise RuntimeError(
            "Prophet is not installed. Install it before running the local monthly benchmark."
        ) from exc
    return Prophet


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _key_to_label(key) -> str:
    if isinstance(key, tuple):
        return " / ".join(str(x) for x in key)
    return str(key)


def _empty_group_like(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[0:0].copy()


def build_calendar_frame(dates: pd.Series | pd.Index | list[pd.Timestamp]) -> pd.DataFrame:
    ds = pd.to_datetime(pd.Series(dates), errors="coerce").dropna().drop_duplicates().sort_values()
    frame = pd.DataFrame({"ds": ds.to_list()})
    frame["day_of_week"] = frame["ds"].dt.dayofweek
    frame["day"] = frame["ds"].dt.day
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype(int)
    frame["is_holiday"] = frame["ds"].apply(lambda d: int((d.month, d.day) in RU_HOLIDAYS))
    frame["is_pre_holiday"] = frame["ds"].apply(
        lambda d: int(((d + pd.Timedelta(days=1)).month, (d + pd.Timedelta(days=1)).day) in RU_HOLIDAYS)
    )
    frame["is_post_holiday"] = frame["ds"].apply(
        lambda d: int(((d - pd.Timedelta(days=1)).month, (d - pd.Timedelta(days=1)).day) in RU_HOLIDAYS)
    )
    frame["is_payday_week"] = frame["day"].isin([4, 5, 6, 19, 20, 21]).astype(int)
    frame["is_month_start"] = (frame["day"] <= 5).astype(int)
    frame["is_month_end"] = (frame["day"] >= 25).astype(int)
    return frame[[
        "ds",
        "is_weekend",
        "is_holiday",
        "is_pre_holiday",
        "is_post_holiday",
        "is_payday_week",
        "is_month_start",
        "is_month_end",
    ]]


def build_holidays_frame(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    rows: list[dict] = []
    for year in range(int(start_date.year), int(end_date.year) + 1):
        for (month, day), holiday_name in RU_HOLIDAYS.items():
            try:
                ds = pd.Timestamp(year=year, month=month, day=day)
            except ValueError:
                continue
            if start_date <= ds <= end_date:
                rows.append({"holiday": holiday_name, "ds": ds, "lower_window": 0, "upper_window": 0})
    if not rows:
        return pd.DataFrame(columns=["holiday", "ds", "lower_window", "upper_window"])
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def subset_by_key(df: pd.DataFrame, group_cols: list[str], key) -> pd.DataFrame:
    if len(group_cols) == 1:
        return df[df[group_cols[0]] == key].copy()

    mask = pd.Series(True, index=df.index)
    if not isinstance(key, tuple):
        key = (key,)
    for col, val in zip(group_cols, key):
        mask &= df[col] == val
    return df.loc[mask].copy()


def build_prediction_frame(test_df: pd.DataFrame, predictions: pd.Series | np.ndarray, model_name: str) -> pd.DataFrame:
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
                BAKERY_COL: bakery,
                PRODUCT_COL: product,
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
        columns = [BAKERY_COL, PRODUCT_COL, "best_model", "best_r2", "best_mae", "best_mse", "best_wmape"]
        return pd.DataFrame(columns=columns)

    wide = wide.reset_index(drop=True)
    r2_cols = [f"{name}_r2" for name in MODEL_NAMES]
    r2_values = wide[r2_cols].apply(pd.to_numeric, errors="coerce").fillna(-np.inf)
    best_idx = r2_values.idxmax(axis=1)
    best_r2 = r2_values.max(axis=1)

    result = wide[[BAKERY_COL, PRODUCT_COL]].copy()
    result["best_model"] = best_idx.str.replace("_r2", "", regex=False)
    result["best_r2"] = best_r2.replace(-np.inf, np.nan).to_numpy()
    result["best_mae"] = [
        wide.iloc[i][f"{result.iloc[i]['best_model']}_mae"] if result.iloc[i]["best_model"] in MODEL_NAMES else np.nan
        for i in range(len(result))
    ]
    result["best_mse"] = [
        wide.iloc[i][f"{result.iloc[i]['best_model']}_mse"] if result.iloc[i]["best_model"] in MODEL_NAMES else np.nan
        for i in range(len(result))
    ]
    result["best_wmape"] = [
        wide.iloc[i][f"{result.iloc[i]['best_model']}_wmape"] if result.iloc[i]["best_model"] in MODEL_NAMES else np.nan
        for i in range(len(result))
    ]
    return pd.concat([result.reset_index(drop=True), wide.drop(columns=[BAKERY_COL, PRODUCT_COL]).reset_index(drop=True)], axis=1)


def fit_predict_prophet_group(
    train_group: pd.DataFrame,
    test_group: pd.DataFrame,
    min_train_rows: int,
    prophet_ok: bool,
) -> tuple[np.ndarray, dict]:
    train_group = train_group.sort_values(DATE_COL).copy()
    test_group = test_group.sort_values(DATE_COL).copy()

    if not prophet_ok:
        fallback = float(train_group[DEMAND_TARGET].mean()) if len(train_group) else 0.0
        return np.full(len(test_group), fallback, dtype=float), {
            "status": "prophet_not_installed",
            "n_train": len(train_group),
            "n_test": len(test_group),
            "n_regressors": 0,
            "holiday_rows": 0,
        }

    if len(train_group) < min_train_rows or train_group[DEMAND_TARGET].nunique(dropna=True) <= 1:
        fallback = float(train_group[DEMAND_TARGET].mean()) if len(train_group) else 0.0
        return np.full(len(test_group), fallback, dtype=float), {
            "status": "fallback_mean",
            "n_train": len(train_group),
            "n_test": len(test_group),
            "n_regressors": 0,
            "holiday_rows": 0,
        }

    Prophet = _import_prophet()

    calendar = build_calendar_frame(pd.concat([train_group[DATE_COL], test_group[DATE_COL]], ignore_index=True))
    train_frame = train_group[[DATE_COL, DEMAND_TARGET]].rename(columns={DATE_COL: "ds", DEMAND_TARGET: "y"})
    test_frame = test_group[[DATE_COL, DEMAND_TARGET]].rename(columns={DATE_COL: "ds", DEMAND_TARGET: "y"})
    train_frame = train_frame.merge(calendar, on="ds", how="left")
    test_frame = test_frame.merge(calendar, on="ds", how="left")

    usable_regressors = [col for col in PROPHET_REGRESSORS if train_frame[col].nunique(dropna=True) > 1]

    holidays_df = build_holidays_frame(train_frame["ds"].min(), test_frame["ds"].max())
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        holidays=holidays_df if not holidays_df.empty else None,
    )

    for regressor in usable_regressors:
        model.add_regressor(regressor, standardize=False)

    model.fit(train_frame[["ds", "y", *usable_regressors]].copy())
    forecast = model.predict(test_frame[["ds", *usable_regressors]].copy())
    preds = np.asarray(forecast["yhat"], dtype=float)
    preds = np.clip(preds, 0.0, None)
    info = {
        "status": "trained",
        "n_train": len(train_group),
        "n_test": len(test_group),
        "n_regressors": len(usable_regressors),
        "holiday_rows": len(holidays_df),
    }
    return preds, info


def evaluate_prophet_family(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    min_train_rows: int,
) -> tuple[pd.Series, list[dict]]:
    prophet_ok = prophet_available()
    predictions = pd.Series(index=test_df.index, dtype=float)
    training_log: list[dict] = []
    train_groups = {key: group for key, group in train_df.groupby([BAKERY_COL, PRODUCT_COL], sort=False)}
    grouped_test = list(test_df.groupby([BAKERY_COL, PRODUCT_COL], sort=False))

    total_groups = len(grouped_test)
    for idx, (key, test_group) in enumerate(grouped_test, start=1):
        train_group = train_groups.get(key, _empty_group_like(train_df))
        if idx == 1 or idx % 250 == 0 or idx == total_groups:
            print(f"[prophet_local] {idx}/{total_groups}: {_key_to_label(key)}", flush=True)

        preds, info = fit_predict_prophet_group(train_group, test_group, min_train_rows=min_train_rows, prophet_ok=prophet_ok)
        predictions.loc[test_group.index] = preds
        metrics = regression_metrics(test_group[DEMAND_TARGET], preds)
        training_log.append(
            {
                "family": "prophet_local",
                "group": _key_to_label(key),
                "train_rows": len(train_group),
                "test_rows": len(test_group),
                "train_days": int(train_group[DATE_COL].nunique()) if len(train_group) else 0,
                "test_days": int(test_group[DATE_COL].nunique()),
                "n_regressors": info.get("n_regressors", 0),
                "holiday_rows": info.get("holiday_rows", 0),
                "status": info.get("status", "unknown"),
                "mae": round(metrics["mae"], 4) if np.isfinite(metrics["mae"]) else np.nan,
                "mse": round(metrics["mse"], 4) if np.isfinite(metrics["mse"]) else np.nan,
                "r2": round(metrics["r2"], 4) if np.isfinite(metrics["r2"]) else np.nan,
                "wmape": round(metrics["wmape"], 2) if np.isfinite(metrics["wmape"]) else np.nan,
            }
        )

    fallback_value = float(train_df[DEMAND_TARGET].mean()) if len(train_df) else 0.0
    return predictions.fillna(fallback_value), training_log


def evaluate_lgbm_family(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_features: list[str],
    min_train_rows: int,
) -> tuple[pd.Series, list[dict]]:
    predictions = pd.Series(index=test_df.index, dtype=float)
    training_log: list[dict] = []
    train_groups = {key: group for key, group in train_df.groupby([BAKERY_COL, PRODUCT_COL], sort=False)}
    grouped_test = list(test_df.groupby([BAKERY_COL, PRODUCT_COL], sort=False))

    total_groups = len(grouped_test)
    for idx, (key, test_group) in enumerate(grouped_test, start=1):
        train_group = train_groups.get(key, _empty_group_like(train_df))
        if idx == 1 or idx % 250 == 0 or idx == total_groups:
            print(f"[lgbm_local] {idx}/{total_groups}: {_key_to_label(key)}", flush=True)

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
                "family": "lgbm_local",
                "group": _key_to_label(key),
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
    return predictions.fillna(fallback_value), training_log


def evaluate_two_week_avg(train_df: pd.DataFrame, featured_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.Series, list[dict]]:
    series = pd.Series(
        two_week_average_predictions(featured_df, [BAKERY_COL, PRODUCT_COL], target_col=DEMAND_TARGET),
        index=featured_df.index,
        name="prediction",
    )
    pred_test = series.loc[test_df.index].copy()
    metrics = regression_metrics(test_df[DEMAND_TARGET], pred_test)
    training_log = [
        {
            "family": "two_week_avg",
            "group": "all",
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_days": int(train_df[DATE_COL].nunique()) if len(train_df) else 0,
            "test_days": int(test_df[DATE_COL].nunique()),
            "n_features": 0,
            "status": "deterministic",
            "mae": round(float(metrics["mae"]), 4),
            "mse": round(float(metrics["mse"]), 4),
            "r2": round(float(metrics["r2"]), 4)
            if np.isfinite(metrics["r2"])
            else np.nan,
            "wmape": round(float(metrics["wmape"]), 2),
        }
    ]
    return pred_test, training_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Monthly local benchmark on all SKU pairs")
    parser.add_argument("--test-days", type=int, default=30, help="Held-out tail length in days")
    parser.add_argument("--min-train-rows", type=int, default=MIN_TRAIN_ROWS, help="Minimum rows for local fitting")
    args = parser.parse_args()

    start = time.time()
    print("=" * 72)
    print("MONTHLY LOCAL BENCHMARK")
    print("=" * 72)

    print("[1/5] Loading demand data...")
    raw_df = load_daily_demand()
    print(
        f"  Rows: {len(raw_df):,}, days: {raw_df[DATE_COL].nunique()}, "
        f"bakeries: {raw_df[BAKERY_COL].nunique()}, SKUs: {raw_df[PRODUCT_COL].nunique()}"
    )

    print("[2/5] Building feature frame...")
    featured_df = build_exp63_feature_frame(raw_df)
    featured_df = add_assortment_features_62(featured_df)
    featured_df = featured_df.sort_values([BAKERY_COL, PRODUCT_COL, DATE_COL]).reset_index(drop=True)

    train_df, test_df, test_start = make_train_test_split(featured_df, test_days=args.test_days)
    print(f"  Test start: {test_start.date()} | train rows: {len(train_df):,} | test rows: {len(test_df):,}")
    print(f"  Train days: {train_df[DATE_COL].nunique()} | Test days: {test_df[DATE_COL].nunique()}")

    print("[3/5] Evaluating model families...")
    metrics_frames: dict[str, pd.DataFrame] = {}
    prediction_frames: dict[str, pd.DataFrame] = {}
    training_logs: list[dict] = []

    prophet_pred, prophet_log = evaluate_prophet_family(train_df, test_df, min_train_rows=args.min_train_rows)
    prophet_frame = build_prediction_frame(test_df, prophet_pred, "prophet_local")
    metrics_frames["prophet_local"] = build_metrics_frame(test_df, prophet_frame, "prophet_local")
    prediction_frames["prophet_local"] = prophet_frame
    training_logs.extend(prophet_log)

    lgbm_pred, lgbm_log = evaluate_lgbm_family(
        train_df,
        test_df,
        BASE_FEATURES,
        min_train_rows=args.min_train_rows,
    )
    lgbm_frame = build_prediction_frame(test_df, lgbm_pred, "lgbm_local")
    metrics_frames["lgbm_local"] = build_metrics_frame(test_df, lgbm_frame, "lgbm_local")
    prediction_frames["lgbm_local"] = lgbm_frame
    training_logs.extend(lgbm_log)

    avg_pred, avg_log = evaluate_two_week_avg(train_df, featured_df, test_df)
    avg_frame = build_prediction_frame(test_df, avg_pred, "two_week_avg")
    metrics_frames["two_week_avg"] = build_metrics_frame(test_df, avg_frame, "two_week_avg")
    prediction_frames["two_week_avg"] = avg_frame
    training_logs.extend(avg_log)

    print("[4/5] Saving artifacts...")
    for model_name, files in OUTPUT_FILES.items():
        save_csv(metrics_frames[model_name], files["metrics"])
        save_csv(prediction_frames[model_name], files["predictions"])
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
        "experiment": "sku_local_monthly",
        "status": "completed",
        "prophet_available": prophet_available(),
        "test_days": args.test_days,
        "min_train_rows": args.min_train_rows,
        "rows_total": int(len(raw_df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "pairs_total": int(best_by_sku[[BAKERY_COL, PRODUCT_COL]].drop_duplicates().shape[0]),
        "model_summary": model_summary.to_dict("records"),
        "best_by_sku_rows": int(len(best_by_sku)),
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    SUMMARY_FILES["overview"].write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[5/5] Done")
    print("Model summary:")
    print(model_summary.to_string(index=False))
    print(f"Best-by-SKU rows: {len(best_by_sku)}")
    print(f"Saved overview JSON: {SUMMARY_FILES['overview'].name}")
    print(f"Elapsed: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()

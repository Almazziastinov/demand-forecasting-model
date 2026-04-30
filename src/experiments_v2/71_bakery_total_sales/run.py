"""
Experiment 71: total sales by bakery per day.

Task:
- aggregate sales to bakery-day level
- target = Продано
- compare:
  1) one global LightGBM model across all bakeries
  2) one LightGBM model per bakery
  3) one Prophet model per bakery using the best config from experiment 68

Prophet config source:
- best summary in exp 68 -> prophet_holidays
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

from src.experiments_v2.benchmark_common import (
    BAKERY_COL,
    DATE_COL,
    regression_metrics,
    select_feature_columns,
)
from src.experiments_v2.common import predict_clipped, train_lgbm


EXP_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "daily_sales_8m_demand.csv"
CITY_COL = "Город"
TARGET_COL = "Продано"

WEATHER_COLS = [
    "temp_max",
    "temp_min",
    "temp_mean",
    "temp_range",
    "precipitation",
    "rain",
    "snowfall",
    "windspeed_max",
    "is_rainy",
    "is_snowy",
    "is_cold",
    "is_warm",
    "is_windy",
    "is_bad_weather",
    "weather_cat_code",
]

BASE_FEATURES = [
    BAKERY_COL,
    CITY_COL,
    "ДеньНедели",
    "День",
    "IsWeekend",
    "Месяц",
    "НомерНедели",
    "is_holiday",
    "is_pre_holiday",
    "is_post_holiday",
    "is_payday_week",
    "is_month_start",
    "is_month_end",
    *WEATHER_COLS,
    "sales_lag1",
    "sales_lag2",
    "sales_lag3",
    "sales_lag7",
    "sales_lag14",
    "sales_lag30",
    "sales_roll_mean3",
    "sales_roll_mean7",
    "sales_roll_mean14",
    "sales_roll_mean30",
    "sales_roll_std7",
    "sales_dow_mean",
    "sales_trend",
    "cv_7d",
]

MODEL_NAMES = [
    "global_lgbm",
    "per_bakery_lgbm",
    "per_bakery_prophet_holidays",
]

OUTPUT_FILES = {
    "global_lgbm": {
        "metrics": EXP_DIR / "metrics_global_lgbm.csv",
        "predictions": EXP_DIR / "predictions_global_lgbm.csv",
    },
    "per_bakery_lgbm": {
        "metrics": EXP_DIR / "metrics_per_bakery_lgbm.csv",
        "predictions": EXP_DIR / "predictions_per_bakery_lgbm.csv",
    },
    "per_bakery_prophet_holidays": {
        "metrics": EXP_DIR / "metrics_per_bakery_prophet_holidays.csv",
        "predictions": EXP_DIR / "predictions_per_bakery_prophet_holidays.csv",
    },
}

SUMMARY_FILES = {
    "best_by_bakery": EXP_DIR / "summary_best_by_bakery.csv",
    "model_comparison": EXP_DIR / "summary_by_model.csv",
    "training_log": EXP_DIR / "training_log.csv",
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

MIN_TRAIN_ROWS = 30
DEFAULT_TEST_DAYS = 30


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def prophet_available() -> bool:
    try:
        from prophet import Prophet  # noqa: F401
    except ImportError:
        return False
    return True


def _import_prophet():
    try:
        from prophet import Prophet
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError("Prophet is not installed.") from exc
    return Prophet


def load_bakery_daily_sales() -> pd.DataFrame:
    usecols = [DATE_COL, BAKERY_COL, CITY_COL, TARGET_COL, *WEATHER_COLS]
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", usecols=usecols)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, BAKERY_COL, TARGET_COL]).copy()

    agg_map = {TARGET_COL: "sum", CITY_COL: "first"}
    for col in WEATHER_COLS:
        agg_map[col] = "first"

    daily = (
        df.groupby([DATE_COL, BAKERY_COL], as_index=False)
        .agg(agg_map)
        .sort_values([BAKERY_COL, DATE_COL])
        .reset_index(drop=True)
    )
    return daily


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ДеньНедели"] = df[DATE_COL].dt.dayofweek
    df["День"] = df[DATE_COL].dt.day
    df["IsWeekend"] = (df["ДеньНедели"] >= 5).astype(int)
    df["Месяц"] = df[DATE_COL].dt.month
    df["НомерНедели"] = df[DATE_COL].dt.isocalendar().week.astype(int)
    df["is_holiday"] = df[DATE_COL].apply(lambda d: int((d.month, d.day) in RU_HOLIDAYS))
    df["is_pre_holiday"] = df[DATE_COL].apply(
        lambda d: int(((d + pd.Timedelta(days=1)).month, (d + pd.Timedelta(days=1)).day) in RU_HOLIDAYS)
    )
    df["is_post_holiday"] = df[DATE_COL].apply(
        lambda d: int(((d - pd.Timedelta(days=1)).month, (d - pd.Timedelta(days=1)).day) in RU_HOLIDAYS)
    )
    df["is_payday_week"] = df["День"].isin([4, 5, 6, 19, 20, 21]).astype(int)
    df["is_month_start"] = (df["День"] <= 5).astype(int)
    df["is_month_end"] = (df["День"] >= 25).astype(int)
    return df


def add_sales_history_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values([BAKERY_COL, DATE_COL]).copy()
    grp = df.groupby(BAKERY_COL)[TARGET_COL]

    for lag in [1, 2, 3, 7, 14, 30]:
        df[f"sales_lag{lag}"] = grp.shift(lag)

    for window, min_periods in [(3, 1), (7, 1), (14, 7), (30, 14)]:
        df[f"sales_roll_mean{window}"] = grp.transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=min_periods).mean()
        )

    df["sales_roll_std7"] = grp.transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=2).std()
    )

    dow_grp = df.groupby([BAKERY_COL, "ДеньНедели"])[TARGET_COL]
    df["sales_dow_mean"] = dow_grp.transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )

    df["sales_trend"] = df["sales_roll_mean7"] / (df["sales_roll_mean30"] + 1e-8)
    df["cv_7d"] = df["sales_roll_std7"] / (df["sales_roll_mean7"] + 1.0)

    lag_cols = [c for c in df.columns if c.startswith("sales_lag") or c.startswith("sales_roll")]
    lag_cols.extend(["sales_dow_mean", "sales_trend", "cv_7d"])
    for col in lag_cols:
        df[col] = df[col].fillna(0)

    return df


def build_dataset() -> pd.DataFrame:
    df = load_bakery_daily_sales()
    df = add_calendar_features(df)
    df = add_sales_history_features(df)
    return df


def cast_categorical_columns(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    categorical_cols = [col for col in [BAKERY_COL, CITY_COL, "Месяц"] if col in feature_cols]
    for col in categorical_cols:
        if col not in train_x.columns or col not in test_x.columns:
            continue
        train_x[col] = train_x[col].astype("category")
        test_x[col] = pd.Categorical(test_x[col], categories=train_x[col].cat.categories)
    return train_x, test_x


def train_predict_lgbm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    min_train_rows: int,
) -> tuple[np.ndarray, dict]:
    train_df = train_df.dropna(subset=[TARGET_COL]).copy()
    feature_cols = [f for f in feature_cols if f in train_df.columns and f in test_df.columns]
    feature_cols = select_feature_columns(train_df, feature_cols, drop_constants=True)

    if len(feature_cols) == 0 or len(train_df) < min_train_rows:
        fallback = float(train_df[TARGET_COL].mean()) if len(train_df) else 0.0
        return np.full(len(test_df), fallback, dtype=float), {
            "status": "fallback_mean",
            "n_train": len(train_df),
            "n_features": len(feature_cols),
        }

    train_x = train_df[feature_cols].copy()
    test_x = test_df[feature_cols].copy()
    train_x, test_x = cast_categorical_columns(train_x, test_x, feature_cols)

    model = train_lgbm(train_x, train_df[TARGET_COL])
    preds = predict_clipped(model, test_x)
    return preds, {
        "status": "trained",
        "n_train": len(train_df),
        "n_features": len(feature_cols),
    }


def build_holidays_frame(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    rows: list[dict] = []
    for year in range(int(start_date.year), int(end_date.year) + 1):
        for (month, day), holiday_name in RU_HOLIDAYS.items():
            try:
                ds = pd.Timestamp(year=year, month=month, day=day)
            except ValueError:
                continue
            if start_date <= ds <= end_date:
                rows.append(
                    {
                        "holiday": holiday_name,
                        "ds": ds,
                        "lower_window": 0,
                        "upper_window": 0,
                    }
                )
    return pd.DataFrame(rows, columns=["holiday", "ds", "lower_window", "upper_window"])


def fit_predict_prophet(
    train_group: pd.DataFrame,
    test_group: pd.DataFrame,
    min_train_rows: int,
) -> tuple[np.ndarray, dict]:
    if not prophet_available():
        fallback = float(train_group[TARGET_COL].mean()) if len(train_group) else 0.0
        return np.full(len(test_group), fallback, dtype=float), {
            "status": "prophet_not_installed",
            "n_train": len(train_group),
            "holiday_rows": 0,
        }

    if len(train_group) < min_train_rows or train_group[TARGET_COL].nunique(dropna=True) <= 1:
        fallback = float(train_group[TARGET_COL].mean()) if len(train_group) else 0.0
        return np.full(len(test_group), fallback, dtype=float), {
            "status": "fallback_mean",
            "n_train": len(train_group),
            "holiday_rows": 0,
        }

    Prophet = _import_prophet()
    train_frame = train_group[[DATE_COL, TARGET_COL]].rename(columns={DATE_COL: "ds", TARGET_COL: "y"}).copy()
    test_frame = test_group[[DATE_COL]].rename(columns={DATE_COL: "ds"}).copy()
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
    model.fit(train_frame)
    forecast = model.predict(test_frame)
    preds = np.clip(np.asarray(forecast["yhat"], dtype=float), 0.0, None)
    return preds, {
        "status": "trained",
        "n_train": len(train_group),
        "holiday_rows": len(holidays_df),
    }


def make_train_test_split(df: pd.DataFrame, test_days: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    max_date = df[DATE_COL].max()
    test_start = max_date - pd.Timedelta(days=test_days - 1)
    train_df = df[df[DATE_COL] < test_start].copy()
    test_df = df[df[DATE_COL] >= test_start].copy()
    return train_df, test_df, test_start


def evaluate_per_bakery_family(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    family_name: str,
    min_train_rows: int,
) -> tuple[pd.Series, list[dict]]:
    predictions = pd.Series(index=test_df.index, dtype=float)
    training_log: list[dict] = []

    for bakery, test_group in test_df.groupby(BAKERY_COL, sort=False):
        train_group = train_df[train_df[BAKERY_COL] == bakery].copy()
        if family_name == "per_bakery_lgbm":
            preds, info = train_predict_lgbm(train_group, test_group, feature_cols, min_train_rows=min_train_rows)
        else:
            preds, info = fit_predict_prophet(train_group, test_group, min_train_rows=min_train_rows)

        predictions.loc[test_group.index] = preds
        metrics = regression_metrics(test_group[TARGET_COL], preds)
        training_log.append(
            {
                "family": family_name,
                "bakery": bakery,
                "train_rows": len(train_group),
                "test_rows": len(test_group),
                "train_days": int(train_group[DATE_COL].nunique()) if len(train_group) else 0,
                "test_days": int(test_group[DATE_COL].nunique()),
                "status": info.get("status", "unknown"),
                "n_features": info.get("n_features", 0),
                "holiday_rows": info.get("holiday_rows", 0),
                "mae": round(metrics["mae"], 4) if np.isfinite(metrics["mae"]) else np.nan,
                "mse": round(metrics["mse"], 4) if np.isfinite(metrics["mse"]) else np.nan,
                "r2": round(metrics["r2"], 4) if np.isfinite(metrics["r2"]) else np.nan,
                "wmape": round(metrics["wmape"], 2) if np.isfinite(metrics["wmape"]) else np.nan,
            }
        )

    fallback_value = float(train_df[TARGET_COL].mean()) if len(train_df) else 0.0
    return predictions.fillna(fallback_value), training_log


def build_prediction_frame(
    test_df: pd.DataFrame,
    predictions: pd.Series | np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    frame = test_df[[DATE_COL, BAKERY_COL, TARGET_COL]].copy()
    frame["model"] = model_name
    frame["prediction"] = np.asarray(predictions, dtype=float)
    frame["abs_error"] = (frame[TARGET_COL] - frame["prediction"]).abs()
    return frame


def build_metrics_frame(prediction_frame: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows: list[dict] = []
    for bakery, group in prediction_frame.groupby(BAKERY_COL, sort=False):
        metrics = regression_metrics(group[TARGET_COL], group["prediction"])
        rows.append(
            {
                BAKERY_COL: bakery,
                "model": model_name,
                "n_test_rows": int(len(group)),
                "n_test_days": int(group[DATE_COL].nunique()),
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
                "n_bakeries": int(len(group)),
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


def build_best_by_bakery(metrics_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    wide: pd.DataFrame | None = None
    for model_name, frame in metrics_frames.items():
        sub = frame[[BAKERY_COL, "r2", "mse", "mae", "wmape"]].copy()
        sub = sub.rename(
            columns={
                "r2": f"{model_name}_r2",
                "mse": f"{model_name}_mse",
                "mae": f"{model_name}_mae",
                "wmape": f"{model_name}_wmape",
            }
        )
        wide = sub if wide is None else wide.merge(sub, on=[BAKERY_COL], how="outer")

    if wide is None or wide.empty:
        return pd.DataFrame(columns=[BAKERY_COL, "best_model", "best_r2", "best_mae", "best_mse", "best_wmape"])

    r2_cols = [f"{name}_r2" for name in MODEL_NAMES]
    r2_values = wide[r2_cols].fillna(-np.inf)
    best_idx = r2_values.idxmax(axis=1)
    result = wide[[BAKERY_COL]].copy()
    result["best_model"] = best_idx.str.replace("_r2", "", regex=False)
    result["best_r2"] = r2_values.max(axis=1).replace(-np.inf, np.nan).to_numpy()
    result["best_mae"] = [wide.iloc[i][f"{result.iloc[i]['best_model']}_mae"] for i in range(len(result))]
    result["best_mse"] = [wide.iloc[i][f"{result.iloc[i]['best_model']}_mse"] for i in range(len(result))]
    result["best_wmape"] = [wide.iloc[i][f"{result.iloc[i]['best_model']}_wmape"] for i in range(len(result))]
    return pd.concat([result.reset_index(drop=True), wide.drop(columns=[BAKERY_COL]).reset_index(drop=True)], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 71: bakery total sales")
    parser.add_argument("--test-days", type=int, default=DEFAULT_TEST_DAYS, help="Held-out tail length in days")
    parser.add_argument("--min-train-rows", type=int, default=MIN_TRAIN_ROWS, help="Minimum rows to fit local models")
    args = parser.parse_args()

    start = time.time()
    print("=" * 72)
    print("EXPERIMENT 71: Bakery total sales")
    print("=" * 72)

    print(f"\n[1/4] Building bakery-day dataset...")
    df = build_dataset()
    print(
        f"  Rows: {len(df):,}, days: {df[DATE_COL].nunique()}, "
        f"bakeries: {df[BAKERY_COL].nunique()}"
    )
    print(
        f"  Date range: {df[DATE_COL].min().date()} -- {df[DATE_COL].max().date()} | "
        f"mean sales/day: {df[TARGET_COL].mean():.2f}"
    )

    print(f"\n[2/4] Train/test split...")
    train_df, test_df, test_start = make_train_test_split(df, test_days=args.test_days)
    print(
        f"  Test start: {test_start.date()} | train rows: {len(train_df):,} | "
        f"test rows: {len(test_df):,}"
    )
    print(f"  Train days: {train_df[DATE_COL].nunique()} | Test days: {test_df[DATE_COL].nunique()}")

    feature_cols = [f for f in BASE_FEATURES if f in df.columns]
    metrics_frames: dict[str, pd.DataFrame] = {}
    prediction_frames: dict[str, pd.DataFrame] = {}
    training_logs: list[dict] = []

    print(f"\n[3/4] Evaluating models...")

    global_preds, global_info = train_predict_lgbm(
        train_df,
        test_df,
        feature_cols,
        min_train_rows=args.min_train_rows,
    )
    global_pred_frame = build_prediction_frame(test_df, global_preds, "global_lgbm")
    metrics_frames["global_lgbm"] = build_metrics_frame(global_pred_frame, "global_lgbm")
    prediction_frames["global_lgbm"] = global_pred_frame
    training_logs.append(
        {
            "family": "global_lgbm",
            "bakery": "__all__",
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_days": int(train_df[DATE_COL].nunique()),
            "test_days": int(test_df[DATE_COL].nunique()),
            "status": global_info.get("status", "unknown"),
            "n_features": global_info.get("n_features", 0),
            "holiday_rows": 0,
            **{
                k: (round(v, 4) if isinstance(v, float) and np.isfinite(v) else v)
                for k, v in regression_metrics(test_df[TARGET_COL], global_preds).items()
            },
        }
    )

    bakery_lgbm_preds, bakery_lgbm_log = evaluate_per_bakery_family(
        train_df,
        test_df,
        feature_cols,
        "per_bakery_lgbm",
        min_train_rows=args.min_train_rows,
    )
    bakery_lgbm_frame = build_prediction_frame(test_df, bakery_lgbm_preds, "per_bakery_lgbm")
    metrics_frames["per_bakery_lgbm"] = build_metrics_frame(bakery_lgbm_frame, "per_bakery_lgbm")
    prediction_frames["per_bakery_lgbm"] = bakery_lgbm_frame
    training_logs.extend(bakery_lgbm_log)

    bakery_prophet_preds, bakery_prophet_log = evaluate_per_bakery_family(
        train_df,
        test_df,
        feature_cols,
        "per_bakery_prophet_holidays",
        min_train_rows=args.min_train_rows,
    )
    bakery_prophet_frame = build_prediction_frame(test_df, bakery_prophet_preds, "per_bakery_prophet_holidays")
    metrics_frames["per_bakery_prophet_holidays"] = build_metrics_frame(
        bakery_prophet_frame,
        "per_bakery_prophet_holidays",
    )
    prediction_frames["per_bakery_prophet_holidays"] = bakery_prophet_frame
    training_logs.extend(bakery_prophet_log)

    print(f"\n[4/4] Saving artifacts...")
    for model_name, files in OUTPUT_FILES.items():
        save_csv(metrics_frames[model_name], files["metrics"])
        save_csv(prediction_frames[model_name], files["predictions"])
        print(f"  Saved {model_name}: {files['metrics'].name}, {files['predictions'].name}")

    metrics_all = pd.concat(metrics_frames.values(), ignore_index=True)
    model_summary = build_model_summary(metrics_all)
    best_by_bakery = build_best_by_bakery(metrics_frames)
    best_counts = best_by_bakery["best_model"].value_counts().reindex(MODEL_NAMES, fill_value=0)
    model_summary["win_count"] = model_summary["model"].map(best_counts).fillna(0).astype(int)

    save_csv(model_summary, SUMMARY_FILES["model_comparison"])
    save_csv(best_by_bakery, SUMMARY_FILES["best_by_bakery"])
    save_csv(pd.DataFrame(training_logs), SUMMARY_FILES["training_log"])

    overview = {
        "experiment": "71_bakery_total_sales",
        "target": TARGET_COL,
        "prophet_variant": "prophet_holidays",
        "prophet_available": prophet_available(),
        "test_days": args.test_days,
        "min_train_rows": args.min_train_rows,
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "bakeries_total": int(df[BAKERY_COL].nunique()),
        "model_summary": model_summary.to_dict("records"),
        "best_by_bakery_rows": int(len(best_by_bakery)),
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    SUMMARY_FILES["overview"].write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nModel summary:")
    print(model_summary.to_string(index=False))
    print(f"\nBest-by-bakery rows: {len(best_by_bakery)}")
    print(f"Prophet available: {prophet_available()}")
    print(f"\nDone in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()

"""
Experiment 68: Prophet benchmark.

This runner benchmarks Prophet variants on the same selected bakery/SKU contract
used by experiment 67.
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
    DEMAND_TARGET,
    PRODUCT_COL,
    build_r2_filtered_subset,
    load_best_by_sku,
    load_daily_demand,
    make_train_test_split,
    regression_metrics,
)


EXP_DIR = Path(__file__).resolve().parent
BASE_SUMMARY_PATH = EXP_DIR.parent / "67_bakery_sku_benchmark" / "summary_best_by_sku.csv"

# Shared bakery contract from experiment 67.
SELECTED_BAKERIES = [
    "Халтурина 8/20 Казань",
    "проспект Мусы Джалиля 20 Наб Челны",
    "Камая 1 Казань",
    "Дзержинского 47 Курск",
    "Фучика 105А Казань",
]

PROPHET_VARIANTS = [
    "prophet_base",
    "prophet_holidays",
    "prophet_calendar",
    "prophet_calendar_holidays",
    "prophet_safe_regressors",
]

OUTPUT_FILES = {
    variant: {
        "metrics": EXP_DIR / f"metrics_{variant}.csv",
        "predictions": EXP_DIR / f"predictions_{variant}.csv",
    }
    for variant in PROPHET_VARIANTS
}

SUMMARY_FILES = {
    "best_by_sku": EXP_DIR / "summary_best_by_sku.csv",
    "model_comparison": EXP_DIR / "summary_by_model.csv",
    "model_training": EXP_DIR / "training_log.csv",
    "overview": EXP_DIR / "metrics.json",
    "selected_subset": EXP_DIR / "selected_sku_subset.csv",
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

CALENDAR_REGRESSORS = ("is_holiday", "is_pre_holiday", "is_post_holiday")
SAFE_REGRESSORS = ("is_payday_week", "is_month_start", "is_month_end")

MIN_TRAIN_ROWS = 30


def prophet_available() -> bool:
    try:
        from prophet import Prophet  # noqa: F401
    except ImportError:
        return False
    return True


def _import_prophet():
    try:
        from prophet import Prophet
    except ImportError as exc:  # pragma: no cover - handled in main
        raise RuntimeError(
            "Prophet is not installed. Install it before running the full benchmark."
        ) from exc
    return Prophet


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def build_calendar_frame(dates: pd.Series | pd.Index | list[pd.Timestamp]) -> pd.DataFrame:
    ds = pd.to_datetime(pd.Series(dates), errors="coerce").dropna().drop_duplicates().sort_values()
    frame = pd.DataFrame({"ds": ds.to_list()})
    frame["day_of_week"] = frame["ds"].dt.dayofweek
    frame["day"] = frame["ds"].dt.day
    frame["month"] = frame["ds"].dt.month
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
    start_year = int(start_date.year)
    end_year = int(end_date.year)
    for year in range(start_year, end_year + 1):
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
    if not rows:
        return pd.DataFrame(columns=["holiday", "ds", "lower_window", "upper_window"])
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def variant_spec(variant: str) -> dict:
    if variant == "prophet_base":
        return {"holidays": False, "regressors": (), "changepoint_prior_scale": 0.05, "seasonality_prior_scale": 10.0}
    if variant == "prophet_holidays":
        return {"holidays": True, "regressors": (), "changepoint_prior_scale": 0.05, "seasonality_prior_scale": 10.0}
    if variant == "prophet_calendar":
        return {
            "holidays": False,
            "regressors": CALENDAR_REGRESSORS,
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 10.0,
        }
    if variant == "prophet_calendar_holidays":
        return {
            "holidays": True,
            "regressors": CALENDAR_REGRESSORS + SAFE_REGRESSORS,
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 10.0,
        }
    if variant == "prophet_safe_regressors":
        return {
            "holidays": False,
            "regressors": SAFE_REGRESSORS,
            "changepoint_prior_scale": 0.08,
            "seasonality_prior_scale": 8.0,
        }
    raise KeyError(f"Unknown Prophet variant: {variant}")


def subset_by_key(df: pd.DataFrame, group_cols: list[str], key) -> pd.DataFrame:
    if len(group_cols) == 1:
        return df[df[group_cols[0]] == key].copy()

    mask = pd.Series(True, index=df.index)
    if not isinstance(key, tuple):
        key = (key,)
    for col, val in zip(group_cols, key):
        mask &= df[col] == val
    return df[mask].copy()


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


def build_best_by_sku(metrics_frames: dict[str, pd.DataFrame], summary_context: pd.DataFrame) -> pd.DataFrame:
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
    r2_cols = [f"{name}_r2" for name in PROPHET_VARIANTS]
    r2_values = wide[r2_cols].fillna(-np.inf)
    best_idx = r2_values.idxmax(axis=1)
    result = wide[[BAKERY_COL, PRODUCT_COL]].copy()
    result["best_model"] = best_idx.str.replace("_r2", "", regex=False)
    best_r2 = r2_values.max(axis=1).replace(-np.inf, np.nan)
    result["best_r2"] = best_r2.to_numpy()
    result["best_mae"] = [wide.iloc[i][f"{result.iloc[i]['best_model']}_mae"] for i in range(len(result))]
    result["best_mse"] = [wide.iloc[i][f"{result.iloc[i]['best_model']}_mse"] for i in range(len(result))]
    result["best_wmape"] = [wide.iloc[i][f"{result.iloc[i]['best_model']}_wmape"] for i in range(len(result))]

    if not summary_context.empty:
        result = result.merge(summary_context, on=[BAKERY_COL, PRODUCT_COL], how="left")

    return pd.concat(
        [result.reset_index(drop=True), wide.drop(columns=[BAKERY_COL, PRODUCT_COL]).reset_index(drop=True)],
        axis=1,
    )


def fit_predict_prophet_group(
    train_group: pd.DataFrame,
    test_group: pd.DataFrame,
    variant: str,
    min_train_rows: int,
) -> tuple[np.ndarray, dict]:
    Prophet = _import_prophet()
    spec = variant_spec(variant)

    train_group = train_group.sort_values(DATE_COL).copy()
    test_group = test_group.sort_values(DATE_COL).copy()
    if len(train_group) < min_train_rows or train_group[DEMAND_TARGET].nunique(dropna=True) <= 1:
        fallback = float(train_group[DEMAND_TARGET].mean()) if len(train_group) else 0.0
        return np.full(len(test_group), fallback, dtype=float), {
            "status": "fallback_mean",
            "n_train": len(train_group),
            "n_test": len(test_group),
            "n_regressors": len(spec["regressors"]),
            "holiday_rows": 0,
        }

    calendar = build_calendar_frame(pd.concat([train_group[DATE_COL], test_group[DATE_COL]], ignore_index=True))
    train_frame = train_group[[DATE_COL, DEMAND_TARGET]].rename(columns={DATE_COL: "ds", DEMAND_TARGET: "y"})
    test_frame = test_group[[DATE_COL, DEMAND_TARGET]].rename(columns={DATE_COL: "ds", DEMAND_TARGET: "y"})
    train_frame = train_frame.merge(calendar, on="ds", how="left")
    test_frame = test_frame.merge(calendar, on="ds", how="left")

    holidays_df = None
    if spec["holidays"]:
        holidays_df = build_holidays_frame(train_frame["ds"].min(), test_frame["ds"].max())

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=spec["changepoint_prior_scale"],
        seasonality_prior_scale=spec["seasonality_prior_scale"],
        holidays_prior_scale=10.0,
        holidays=holidays_df if holidays_df is not None and not holidays_df.empty else None,
    )

    for regressor in spec["regressors"]:
        model.add_regressor(regressor, standardize=False)

    train_cols = ["ds", "y", *spec["regressors"]]
    future_cols = ["ds", *spec["regressors"]]
    model.fit(train_frame[train_cols].copy())
    forecast = model.predict(test_frame[future_cols].copy())
    preds = np.asarray(forecast["yhat"], dtype=float)
    preds = np.clip(preds, 0.0, None)

    info = {
        "status": "trained",
        "n_train": len(train_group),
        "n_test": len(test_group),
        "n_regressors": len(spec["regressors"]),
        "holiday_rows": 0 if holidays_df is None else len(holidays_df),
    }
    return preds, info


def evaluate_prophet_family(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    variant: str,
    min_train_rows: int,
) -> tuple[pd.Series, list[dict]]:
    predictions = pd.Series(index=test_df.index, dtype=float)
    training_log: list[dict] = []
    grouped_test = list(test_df.groupby([BAKERY_COL, PRODUCT_COL], sort=False))
    total_groups = len(grouped_test)

    for idx, (key, test_group) in enumerate(grouped_test, start=1):
        train_group = subset_by_key(train_df, [BAKERY_COL, PRODUCT_COL], key)
        if idx == 1 or idx % 10 == 0 or idx == total_groups:
            label = " / ".join(str(x) for x in key) if isinstance(key, tuple) else str(key)
            print(f"[{variant}] {idx}/{total_groups}: {label}", flush=True)
        preds, info = fit_predict_prophet_group(train_group, test_group, variant, min_train_rows=min_train_rows)
        predictions.loc[test_group.index] = preds

        metrics = regression_metrics(test_group[DEMAND_TARGET], preds)
        training_log.append(
            {
                "family": variant,
                "group": " / ".join(str(x) for x in key) if isinstance(key, tuple) else str(key),
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
    predictions = predictions.fillna(fallback_value)
    return predictions, training_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 68: Prophet benchmark")
    parser.add_argument("--test-days", type=int, default=7, help="Held-out tail length in days")
    parser.add_argument("--min-train-rows", type=int, default=MIN_TRAIN_ROWS, help="Minimum rows to fit local Prophet models")
    parser.add_argument(
        "--r2-threshold",
        type=float,
        default=0.30,
        help="Select SKU pairs where global_best_r2 is below this threshold",
    )
    parser.add_argument(
        "--summary-path",
        default=str(BASE_SUMMARY_PATH),
        help="Path to 67 summary_best_by_sku.csv used for filtering",
    )
    args = parser.parse_args()

    start = time.time()
    print("=" * 72)
    print("EXPERIMENT 68: Prophet benchmark")
    print("=" * 72)

    best_by_sku = load_best_by_sku(args.summary_path)
    selected = build_r2_filtered_subset(best_by_sku, args.r2_threshold)
    save_csv(selected, SUMMARY_FILES["selected_subset"])
    print(f"Summary source: {args.summary_path}")
    print(f"R2 threshold: {args.r2_threshold}")
    print(f"Selected rows: {len(selected)}")
    print(f"Saved shortlist: {SUMMARY_FILES['selected_subset']}")

    if not prophet_available():
        overview = {
            "experiment": "68_prophet_benchmark",
            "status": "prophet_not_installed",
            "summary_path": str(args.summary_path),
            "r2_threshold": args.r2_threshold,
            "selected_rows": int(len(selected)),
            "selected_bakeries": SELECTED_BAKERIES,
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        SUMMARY_FILES["overview"].write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Prophet is not installed in this environment.")
        print("Shortlist generation completed; full benchmark will run after installing Prophet.")
        return

    raw_df = load_daily_demand(SELECTED_BAKERIES)
    print(f"Rows: {len(raw_df):,}, days: {raw_df[DATE_COL].nunique()}, bakeries: {raw_df[BAKERY_COL].nunique()}, SKUs: {raw_df[PRODUCT_COL].nunique()}")

    selected_pairs = selected[[BAKERY_COL, PRODUCT_COL]].drop_duplicates()
    benchmark_df = raw_df.merge(selected_pairs, on=[BAKERY_COL, PRODUCT_COL], how="inner")
    train_df, test_df, test_start = make_train_test_split(benchmark_df, test_days=args.test_days)
    print(f"Test start: {test_start.date()} | train rows: {len(train_df):,} | test rows: {len(test_df):,}")
    print(f"Train days: {train_df[DATE_COL].nunique()} | Test days: {test_df[DATE_COL].nunique()}")

    metrics_frames: dict[str, pd.DataFrame] = {}
    prediction_frames: dict[str, pd.DataFrame] = {}
    training_logs: list[dict] = []

    for variant in PROPHET_VARIANTS:
        spec = variant_spec(variant)
        reg_text = ", ".join(spec["regressors"]) if spec["regressors"] else "none"
        print(f"Evaluating {variant} | holidays={spec['holidays']} | regressors={reg_text}", flush=True)
        pred_series, train_log = evaluate_prophet_family(
            train_df,
            test_df,
            variant,
            min_train_rows=args.min_train_rows,
        )
        pred_frame = build_prediction_frame(test_df, pred_series, variant)
        metrics_frame = build_metrics_frame(test_df, pred_frame, variant)
        metrics_frames[variant] = metrics_frame
        prediction_frames[variant] = pred_frame
        training_logs.extend(train_log)

    print("Saving artifacts...")
    for variant, files in OUTPUT_FILES.items():
        save_csv(metrics_frames[variant], files["metrics"])
        save_csv(prediction_frames[variant], files["predictions"])
        print(f"  Saved {variant}: {files['metrics'].name}, {files['predictions'].name}")

    selected_context = selected[[BAKERY_COL, PRODUCT_COL, "global_best_r2", "sku_local_r2", "two_week_avg_r2"]].copy()
    metrics_all = pd.concat(metrics_frames.values(), ignore_index=True)
    model_summary = build_model_summary(metrics_all)
    best_by_sku = build_best_by_sku(metrics_frames, selected_context)
    best_counts = best_by_sku["best_model"].value_counts().reindex(PROPHET_VARIANTS, fill_value=0)
    model_summary["win_count"] = model_summary["model"].map(best_counts).fillna(0).astype(int)

    save_csv(model_summary, SUMMARY_FILES["model_comparison"])
    save_csv(best_by_sku, SUMMARY_FILES["best_by_sku"])
    save_csv(pd.DataFrame(training_logs), SUMMARY_FILES["model_training"])

    overview = {
        "experiment": "68_prophet_benchmark",
        "status": "completed" if prophet_available() else "prophet_not_installed",
        "summary_path": str(args.summary_path),
        "r2_threshold": args.r2_threshold,
        "selected_rows": int(len(selected)),
        "rows_total": int(len(raw_df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "pairs_total": int(best_by_sku[[BAKERY_COL, PRODUCT_COL]].drop_duplicates().shape[0]),
        "model_summary": model_summary.to_dict("records"),
        "best_by_sku_rows": int(len(best_by_sku)),
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    SUMMARY_FILES["overview"].write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Model summary:")
    print(model_summary.to_string(index=False))
    print(f"Best-by-SKU rows: {len(best_by_sku)}")
    print(f"Saved overview JSON: {SUMMARY_FILES['overview'].name}")
    print(f"Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()

from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.config import MODEL_PARAMS
from src.experiments_v2.bakery_day_forecast import BASE_FEATURES
from src.experiments_v2.bakery_day_forecast import BAKERY_ID_COL
from src.experiments_v2.bakery_day_forecast import BAKERY_NAME_COL
from src.experiments_v2.bakery_day_forecast import CITY_COL
from src.experiments_v2.bakery_day_forecast import DATE_COL
from src.experiments_v2.bakery_day_forecast import TARGET_COL
from src.experiments_v2.bakery_day_forecast import build_model_frame
from src.experiments_v2.bakery_day_forecast import load_dataset
from src.experiments_v2.bakery_day_forecast import make_train_test_split
from src.experiments_v2.common import predict_clipped
from src.experiments_v2.planning_metrics import aggregate_planning_metrics
from src.experiments_v2.planning_metrics import summarize_models_by_planning_metrics


EXP_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = ROOT / "data" / "processed" / "bakery_daily_sales.csv"

PAYDAY_DAYS = (5, 20)
EVENT_WINDOW_DAYS = 7

EVENT_FEATURES = [
    "holiday_name_feature",
    "event_window_type",
    "event_distance_bin",
    "current_event_city",
    "nearest_event_city",
    "event_dow_interaction",
    "is_pre_event_1d",
    "is_pre_event_2d",
    "is_pre_event_3d",
    "is_post_event_1d",
    "is_post_event_2d",
    "is_post_event_3d",
    "is_event_window_7d",
    "days_to_payday",
    "days_since_payday",
    "payday_distance",
    "is_payday",
    "is_pre_payday_1d",
    "is_pre_payday_2d",
    "is_post_payday_1d",
    "is_post_payday_2d",
    "payday_window_type",
    "payday_dow_interaction",
]

CATEGORICAL_EVENT_FEATURES = {
    BAKERY_ID_COL,
    CITY_COL,
    "month",
    "current_event_cluster",
    "prev_event_cluster",
    "next_event_cluster",
    "holiday_name_feature",
    "event_window_type",
    "event_distance_bin",
    "current_event_city",
    "nearest_event_city",
    "event_dow_interaction",
    "payday_window_type",
    "payday_dow_interaction",
}


def add_enriched_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add explicit event/payday features for top-down LGBM experiments."""
    work = df.copy()
    days_to_next = pd.to_numeric(
        work.get("days_to_next_event", 999),
        errors="coerce",
    ).fillna(999)
    days_since_prev = pd.to_numeric(
        work.get("days_since_prev_event", 999),
        errors="coerce",
    ).fillna(999)
    is_holiday = pd.to_numeric(
        work.get("is_holiday", 0),
        errors="coerce",
    ).fillna(0).astype(int)

    holiday_name = work.get("holiday_name", "")
    work["holiday_name_feature"] = pd.Series(holiday_name, index=work.index).fillna("")
    work.loc[work["holiday_name_feature"].eq(""), "holiday_name_feature"] = (
        "no_holiday"
    )

    work["is_pre_event_1d"] = (days_to_next == 1).astype(int)
    work["is_pre_event_2d"] = (days_to_next == 2).astype(int)
    work["is_pre_event_3d"] = (days_to_next == 3).astype(int)
    work["is_post_event_1d"] = (days_since_prev == 1).astype(int)
    work["is_post_event_2d"] = (days_since_prev == 2).astype(int)
    work["is_post_event_3d"] = (days_since_prev == 3).astype(int)
    nearest_event_distance = np.minimum(days_to_next, days_since_prev)
    work["is_event_window_7d"] = (
        (nearest_event_distance <= EVENT_WINDOW_DAYS) | (is_holiday == 1)
    ).astype(int)

    work["event_window_type"] = "regular"
    work.loc[is_holiday == 1, "event_window_type"] = "event_day"
    work.loc[(is_holiday == 0) & (days_to_next <= 3), "event_window_type"] = (
        "pre_event_1_3"
    )
    work.loc[
        (is_holiday == 0) & (days_to_next > 3) & (days_to_next <= 7),
        "event_window_type",
    ] = "pre_event_4_7"
    work.loc[(is_holiday == 0) & (days_since_prev <= 3), "event_window_type"] = (
        "post_event_1_3"
    )
    work.loc[
        (is_holiday == 0) & (days_since_prev > 3) & (days_since_prev <= 7),
        "event_window_type",
    ] = "post_event_4_7"

    work["event_distance_bin"] = pd.cut(
        nearest_event_distance.clip(upper=999),
        bins=[-1, 0, 1, 3, 7, 14, 999],
        labels=["0", "1", "2_3", "4_7", "8_14", "far"],
    ).astype(str)

    current_cluster = work.get("current_event_cluster", "cluster_none").astype(str)
    next_cluster = work.get("next_event_cluster", "cluster_none").astype(str)
    prev_cluster = work.get("prev_event_cluster", "cluster_none").astype(str)
    nearest_cluster = np.where(
        days_to_next <= days_since_prev,
        next_cluster,
        prev_cluster,
    )
    nearest_cluster = pd.Series(nearest_cluster, index=work.index).fillna(
        "cluster_none"
    )

    work["current_event_city"] = current_cluster + "|" + work[CITY_COL].astype(str)
    work["nearest_event_city"] = nearest_cluster + "|" + work[CITY_COL].astype(str)
    work["event_dow_interaction"] = (
        nearest_cluster + "|dow_" + work["dow"].astype(str)
    )

    return add_payday_distance_features(work)


def add_payday_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    dates = pd.to_datetime(work[DATE_COL], errors="coerce")
    days_to = []
    days_since = []
    for date_value in dates:
        if pd.isna(date_value):
            days_to.append(999)
            days_since.append(999)
            continue
        candidates = []
        for month_offset in [-1, 0, 1]:
            shifted = date_value + pd.DateOffset(months=month_offset)
            for day in PAYDAY_DAYS:
                try:
                    candidates.append(pd.Timestamp(shifted.year, shifted.month, day))
                except ValueError:
                    continue
        future = [(c - date_value).days for c in candidates if c >= date_value]
        past = [(date_value - c).days for c in candidates if c <= date_value]
        days_to.append(min(future) if future else 999)
        days_since.append(min(past) if past else 999)

    work["days_to_payday"] = pd.Series(days_to, index=work.index).astype(int)
    work["days_since_payday"] = pd.Series(days_since, index=work.index).astype(int)
    work["payday_distance"] = np.minimum(
        work["days_to_payday"],
        work["days_since_payday"],
    )
    work["is_payday"] = (work["payday_distance"] == 0).astype(int)
    work["is_pre_payday_1d"] = (work["days_to_payday"] == 1).astype(int)
    work["is_pre_payday_2d"] = (work["days_to_payday"] == 2).astype(int)
    work["is_post_payday_1d"] = (work["days_since_payday"] == 1).astype(int)
    work["is_post_payday_2d"] = (work["days_since_payday"] == 2).astype(int)

    work["payday_window_type"] = "regular"
    work.loc[work["is_payday"] == 1, "payday_window_type"] = "payday"
    work.loc[work["is_pre_payday_1d"] == 1, "payday_window_type"] = "pre_payday_1"
    work.loc[work["is_pre_payday_2d"] == 1, "payday_window_type"] = "pre_payday_2"
    work.loc[work["is_post_payday_1d"] == 1, "payday_window_type"] = (
        "post_payday_1"
    )
    work.loc[work["is_post_payday_2d"] == 1, "payday_window_type"] = (
        "post_payday_2"
    )
    work["payday_dow_interaction"] = (
        work["payday_window_type"].astype(str) + "|dow_" + work["dow"].astype(str)
    )
    return work


def select_feature_columns(
    frame: pd.DataFrame,
    feature_candidates: list[str],
) -> list[str]:
    selected: list[str] = []
    for col in feature_candidates:
        if col not in frame.columns:
            continue
        series = frame[col]
        if series.isna().all() or series.nunique(dropna=True) <= 1:
            continue
        selected.append(col)
    return selected


def cast_categorical_columns(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for col in [c for c in CATEGORICAL_EVENT_FEATURES if c in feature_cols]:
        train_x[col] = train_x[col].astype("category")
        test_x[col] = pd.Categorical(
            test_x[col],
            categories=train_x[col].cat.categories,
        )
    return train_x, test_x


def train_predict_lgbm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    train_x = train_df[feature_cols].copy()
    test_x = test_df[feature_cols].copy()
    train_x, test_x = cast_categorical_columns(train_x, test_x, feature_cols)
    params = MODEL_PARAMS.copy()
    params["verbosity"] = -1
    model = LGBMRegressor(**params)
    model.fit(train_x, train_df[TARGET_COL])
    return predict_clipped(model, test_x)


def build_predictions(
    test_df: pd.DataFrame,
    model_name: str,
    prediction: np.ndarray,
) -> pd.DataFrame:
    cols = [DATE_COL, BAKERY_ID_COL, BAKERY_NAME_COL, CITY_COL, TARGET_COL]
    out = test_df[cols].copy()
    out["model"] = model_name
    out["prediction"] = prediction
    return out


def run_experiment(
    dataset_path: str | Path,
    *,
    output_dir: str | Path,
    test_days: int,
) -> dict[str, Path]:
    df = add_enriched_event_features(build_model_frame(load_dataset(dataset_path)))
    train_df, test_df, test_start = make_train_test_split(df, test_days)

    legacy_base_candidates = [f for f in BASE_FEATURES if f not in EVENT_FEATURES]
    base_features = select_feature_columns(train_df, legacy_base_candidates)
    enriched_features = select_feature_columns(
        train_df,
        [*BASE_FEATURES, *EVENT_FEATURES],
    )

    base_pred = train_predict_lgbm(train_df, test_df, base_features)
    enriched_pred = train_predict_lgbm(train_df, test_df, enriched_features)
    predictions = pd.concat(
        [
            build_predictions(test_df, "lgbm_top_down_base_events", base_pred),
            build_predictions(
                test_df,
                "lgbm_top_down_enriched_events",
                enriched_pred,
            ),
        ],
        ignore_index=True,
    )

    summary_by_model = summarize_models_by_planning_metrics(
        predictions,
        model_col="model",
        actual_col=TARGET_COL,
        prediction_col="prediction",
    )
    city_summary = aggregate_planning_metrics(
        predictions,
        group_cols=["model", CITY_COL],
        actual_col=TARGET_COL,
        prediction_col="prediction",
    )
    event_summary = aggregate_planning_metrics(
        predictions.merge(
            test_df[[DATE_COL, BAKERY_ID_COL, "event_window_type"]],
            on=[DATE_COL, BAKERY_ID_COL],
            how="left",
        ),
        group_cols=["model", "event_window_type"],
        actual_col=TARGET_COL,
        prediction_col="prediction",
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions": out_dir / "predictions.csv",
        "summary_by_model": out_dir / "summary_by_model.csv",
        "city_summary": out_dir / "city_summary.csv",
        "event_summary": out_dir / "event_summary.csv",
        "metrics_json": out_dir / "metrics.json",
    }
    predictions.to_csv(paths["predictions"], index=False, encoding="utf-8-sig")
    summary_by_model.to_csv(
        paths["summary_by_model"],
        index=False,
        encoding="utf-8-sig",
    )
    city_summary.to_csv(paths["city_summary"], index=False, encoding="utf-8-sig")
    event_summary.to_csv(paths["event_summary"], index=False, encoding="utf-8-sig")
    paths["metrics_json"].write_text(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "test_days": test_days,
                "test_start": str(test_start.date()),
                "rows_train": int(len(train_df)),
                "rows_test": int(len(test_df)),
                "base_feature_count": len(base_features),
                "enriched_feature_count": len(enriched_features),
                "base_features": base_features,
                "enriched_features": enriched_features,
                "models": summary_by_model.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare base vs enriched event features for bakery LGBM"
    )
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output-dir", default=str(EXP_DIR))
    parser.add_argument("--test-days", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_experiment(
        args.dataset_path,
        output_dir=args.output_dir,
        test_days=args.test_days,
    )
    print("=" * 72)
    print("EXP80 EVENT FEATURES LGBM TOP-DOWN")
    print("=" * 72)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

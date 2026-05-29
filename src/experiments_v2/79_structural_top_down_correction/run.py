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
from src.experiments_v2.bakery_day_forecast import cast_category_columns
from src.experiments_v2.bakery_day_forecast import load_dataset
from src.experiments_v2.bakery_day_forecast import make_train_test_split
from src.experiments_v2.planning_metrics import aggregate_planning_metrics
from src.experiments_v2.planning_metrics import planning_metrics
from src.experiments_v2.planning_metrics import summarize_models_by_planning_metrics


EXP_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = ROOT / "data" / "processed" / "bakery_daily_sales.csv"

BASELINE_COL = "structural_top_down_baseline"
LGBM_BASELINE_COL = "lgbm_top_down_baseline"
CORRECTION_COL = "correction_prediction"
RESIDUAL_COL = "structural_residual"
LGBM_RESIDUAL_COL = "lgbm_residual"
BASE_TARGET_COL = "bakery_sales_base_rolling_quantile_capped"


def select_feature_columns(frame: pd.DataFrame) -> list[str]:
    extra = [
        BASELINE_COL,
        LGBM_BASELINE_COL,
        "structural_baseline_level",
        "structural_baseline_shape",
        "structural_baseline_recent_weight",
    ]
    selected: list[str] = []
    for col in [*BASE_FEATURES, *extra]:
        if col not in frame.columns:
            continue
        series = frame[col]
        if series.isna().all() or series.nunique(dropna=True) <= 1:
            continue
        selected.append(col)
    return selected


def _target_series(df: pd.DataFrame) -> pd.Series:
    col = BASE_TARGET_COL if BASE_TARGET_COL in df.columns else TARGET_COL
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(lower=0.0)


def build_structural_profile(
    train_df: pd.DataFrame,
    *,
    recent_days: int = 28,
    long_days: int = 180,
    recent_shape_weight: float = 0.35,
) -> pd.DataFrame:
    """Build bakery x dow structural profile from train history only."""
    rows: list[dict[str, object]] = []
    max_date = train_df[DATE_COL].max()
    for bakery_id, group in train_df.groupby(BAKERY_ID_COL, sort=False):
        work = group.sort_values(DATE_COL).copy()
        work["_target"] = _target_series(work)
        long = work[work[DATE_COL] >= max_date - pd.Timedelta(days=long_days - 1)]
        recent = work[work[DATE_COL] >= max_date - pd.Timedelta(days=recent_days - 1)]
        if long.empty:
            long = work
        if recent.empty:
            recent = long

        long_level = float(long["_target"].mean()) if len(long) else 0.0
        recent_level = float(recent["_target"].mean()) if len(recent) else long_level
        level = 0.65 * recent_level + 0.35 * long_level

        long_dow = long.groupby("dow")["_target"].median()
        recent_dow = recent.groupby("dow")["_target"].median()
        long_shape = _normalize_dow_shape(long_dow, fallback_level=long_level)
        recent_shape = _normalize_dow_shape(recent_dow, fallback_level=recent_level)
        weight = recent_shape_weight if len(recent) >= 14 else 0.0
        blended_shape = (1.0 - weight) * long_shape + weight * recent_shape
        blended_shape = blended_shape / max(float(blended_shape.mean()), 1e-8)

        attrs = {
            BAKERY_ID_COL: bakery_id,
            BAKERY_NAME_COL: work[BAKERY_NAME_COL].iloc[-1],
            CITY_COL: work[CITY_COL].iloc[-1]
            if CITY_COL in work.columns
            else "unknown",
            "structural_baseline_level": level,
            "structural_baseline_recent_weight": weight,
            "structural_long_level": long_level,
            "structural_recent_level": recent_level,
        }
        for dow in range(7):
            rows.append(
                {
                    **attrs,
                    "dow": dow,
                    "structural_baseline_shape": float(blended_shape.loc[dow]),
                    BASELINE_COL: max(level * float(blended_shape.loc[dow]), 0.0),
                }
            )
    return pd.DataFrame(rows)


def _normalize_dow_shape(dow_values: pd.Series, *, fallback_level: float) -> pd.Series:
    values = pd.Series(index=range(7), dtype=float)
    values.loc[dow_values.index.astype(int)] = pd.to_numeric(
        dow_values, errors="coerce"
    )
    fallback = (
        fallback_level
        if np.isfinite(fallback_level) and fallback_level > 0
        else 0.0
    )
    values = values.fillna(fallback).clip(lower=0.0)
    mean_value = float(values.mean())
    if mean_value <= 1e-8:
        return pd.Series(1.0, index=range(7), dtype=float)
    return values / mean_value


def attach_structural_baseline(
    df: pd.DataFrame,
    profile: pd.DataFrame,
) -> pd.DataFrame:
    work = df.merge(
        profile[
            [
                BAKERY_ID_COL,
                "dow",
                BASELINE_COL,
                "structural_baseline_level",
                "structural_baseline_shape",
                "structural_baseline_recent_weight",
            ]
        ],
        on=[BAKERY_ID_COL, "dow"],
        how="left",
        validate="many_to_one",
    )
    global_level = float(_target_series(df).mean()) if len(df) else 0.0
    work[BASELINE_COL] = pd.to_numeric(work[BASELINE_COL], errors="coerce").fillna(
        global_level
    )
    work["structural_baseline_level"] = pd.to_numeric(
        work["structural_baseline_level"], errors="coerce"
    ).fillna(global_level)
    work["structural_baseline_shape"] = pd.to_numeric(
        work["structural_baseline_shape"], errors="coerce"
    ).fillna(1.0)
    work["structural_baseline_recent_weight"] = pd.to_numeric(
        work["structural_baseline_recent_weight"], errors="coerce"
    ).fillna(0.0)
    return work


def train_correction_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    residual_col: str,
) -> np.ndarray:
    train_x = train_df[feature_cols].copy()
    test_x = test_df[feature_cols].copy()
    train_x, test_x = cast_category_columns(train_x, test_x, feature_cols)

    params = MODEL_PARAMS.copy()
    params["verbosity"] = -1
    model = LGBMRegressor(**params)
    model.fit(train_x, train_df[residual_col])
    return model.predict(test_x)


def train_lgbm_top_down_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_x = train_df[feature_cols].copy()
    test_x = test_df[feature_cols].copy()
    train_x, test_x = cast_category_columns(train_x, test_x, feature_cols)

    params = MODEL_PARAMS.copy()
    params["verbosity"] = -1
    model = LGBMRegressor(**params)
    model.fit(train_x, _target_series(train_df))
    train_pred = model.predict(train_x)
    test_pred = model.predict(test_x)
    return np.clip(train_pred, a_min=0.0, a_max=None), np.clip(
        test_pred,
        a_min=0.0,
        a_max=None,
    )


def build_predictions(
    test_df: pd.DataFrame,
    structural_correction: np.ndarray,
    lgbm_correction: np.ndarray,
) -> pd.DataFrame:
    output_cols = [DATE_COL, BAKERY_ID_COL, BAKERY_NAME_COL, CITY_COL, TARGET_COL]
    frames = []

    specs = [
        ("structural_top_down", BASELINE_COL, None),
        ("structural_plus_correction_lgbm", BASELINE_COL, structural_correction),
        ("lgbm_top_down", LGBM_BASELINE_COL, None),
        ("lgbm_top_down_plus_correction_lgbm", LGBM_BASELINE_COL, lgbm_correction),
    ]
    for model_name, baseline_col, correction in specs:
        frame = test_df[output_cols].copy()
        frame["model"] = model_name
        baseline = test_df[baseline_col].to_numpy(dtype=float)
        if correction is None:
            frame["prediction"] = np.clip(baseline, a_min=0.0, a_max=None)
        else:
            frame[CORRECTION_COL] = correction
            frame["prediction"] = np.clip(
                baseline + correction,
                a_min=0.0,
                a_max=None,
            )
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def build_aggregate_reports(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "summary_by_model": summarize_models_by_planning_metrics(
            predictions,
            model_col="model",
            actual_col=TARGET_COL,
            prediction_col="prediction",
        ),
        "city_summary": aggregate_planning_metrics(
            predictions,
            group_cols=["model", CITY_COL],
            actual_col=TARGET_COL,
            prediction_col="prediction",
        ),
        "bakery_summary": aggregate_planning_metrics(
            predictions,
            group_cols=["model", BAKERY_ID_COL, BAKERY_NAME_COL, CITY_COL],
            actual_col=TARGET_COL,
            prediction_col="prediction",
        ),
    }


def run_experiment(
    dataset_path: str | Path,
    *,
    output_dir: str | Path,
    test_days: int,
) -> dict[str, Path]:
    df = build_model_frame(load_dataset(dataset_path))
    train_df, test_df, test_start = make_train_test_split(df, test_days)
    profile = build_structural_profile(train_df)
    train_df = attach_structural_baseline(train_df, profile)
    test_df = attach_structural_baseline(test_df, profile)
    train_df[RESIDUAL_COL] = _target_series(train_df) - train_df[BASELINE_COL]

    lgbm_feature_cols = select_feature_columns(
        train_df.drop(columns=[BASELINE_COL], errors="ignore")
    )
    train_lgbm_pred, test_lgbm_pred = train_lgbm_top_down_baseline(
        train_df,
        test_df,
        lgbm_feature_cols,
    )
    train_df[LGBM_BASELINE_COL] = train_lgbm_pred
    test_df[LGBM_BASELINE_COL] = test_lgbm_pred
    train_df[LGBM_RESIDUAL_COL] = _target_series(train_df) - train_df[LGBM_BASELINE_COL]

    feature_cols = select_feature_columns(train_df)
    structural_correction = train_correction_model(
        train_df,
        test_df,
        feature_cols,
        residual_col=RESIDUAL_COL,
    )
    lgbm_correction = train_correction_model(
        train_df,
        test_df,
        feature_cols,
        residual_col=LGBM_RESIDUAL_COL,
    )
    predictions = build_predictions(test_df, structural_correction, lgbm_correction)
    reports = build_aggregate_reports(predictions)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "profile": out_dir / "structural_profile.csv",
        "predictions": out_dir / "predictions.csv",
        "summary_by_model": out_dir / "summary_by_model.csv",
        "city_summary": out_dir / "city_summary.csv",
        "bakery_summary": out_dir / "bakery_summary.csv",
        "metrics_json": out_dir / "metrics.json",
    }
    profile.to_csv(paths["profile"], index=False, encoding="utf-8-sig")
    predictions.to_csv(paths["predictions"], index=False, encoding="utf-8-sig")
    for name, report in reports.items():
        report.to_csv(paths[name], index=False, encoding="utf-8-sig")

    metrics = {
        "dataset_path": str(dataset_path),
        "test_days": test_days,
        "test_start": str(test_start.date()),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "feature_count": len(feature_cols),
        "feature_cols": feature_cols,
        "lgbm_feature_count": len(lgbm_feature_cols),
        "lgbm_feature_cols": lgbm_feature_cols,
        "models": reports["summary_by_model"].to_dict("records"),
        "structural_train_fit": planning_metrics(
            train_df.assign(prediction=train_df[BASELINE_COL]),
            actual_col=TARGET_COL,
            prediction_col="prediction",
        ),
        "lgbm_train_fit": planning_metrics(
            train_df.assign(prediction=train_df[LGBM_BASELINE_COL]),
            actual_col=TARGET_COL,
            prediction_col="prediction",
        ),
    }
    paths["metrics_json"].write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Structural top-down bakery baseline with residual correction"
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
    print("EXP79 STRUCTURAL TOP-DOWN CORRECTION")
    print("=" * 72)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

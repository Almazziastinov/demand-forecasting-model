from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from lightgbm import LGBMRegressor

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
from src.experiments_v2.bakery_day_forecast import regression_metrics
from src.experiments_v2.common import predict_clipped


ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = ROOT / "data" / "processed" / "bakery_daily_sales.csv"

ROLLING_TARGET_COL = "bakery_sales_base_rolling_capped"
QUANTILE_TARGET_COL = "bakery_sales_base_quantile_capped"
ROLLING_QUANTILE_TARGET_COL = "bakery_sales_base_rolling_quantile_capped"
SAMPLE_WEIGHT_COL = "base_model_sample_weight"
OBSERVED_TARGET_COL = "bakery_sales_observed"
SALES_MISSING_FLAG_COL = "sales_missing_flag"

MODEL_SPECS = [
    {
        "model": "raw_target_lgbm",
        "target_col": TARGET_COL,
        "sample_weight_col": None,
    },
    {
        "model": "rolling_capped_target_lgbm",
        "target_col": ROLLING_TARGET_COL,
        "sample_weight_col": None,
    },
    {
        "model": "rolling_capped_weighted_lgbm",
        "target_col": ROLLING_TARGET_COL,
        "sample_weight_col": SAMPLE_WEIGHT_COL,
    },
    {
        "model": "raw_target_weighted_lgbm",
        "target_col": TARGET_COL,
        "sample_weight_col": SAMPLE_WEIGHT_COL,
    },
    {
        "model": "quantile_capped_target_lgbm_benchmark",
        "target_col": QUANTILE_TARGET_COL,
        "sample_weight_col": None,
    },
    {
        "model": "rolling_quantile_capped_target_lgbm",
        "target_col": ROLLING_QUANTILE_TARGET_COL,
        "sample_weight_col": None,
    },
    {
        "model": "rolling_quantile_capped_weighted_lgbm",
        "target_col": ROLLING_QUANTILE_TARGET_COL,
        "sample_weight_col": SAMPLE_WEIGHT_COL,
    },
]


def select_feature_columns(frame: pd.DataFrame) -> list[str]:
    selected: list[str] = []
    for col in BASE_FEATURES:
        if col not in frame.columns:
            continue
        series = frame[col]
        if series.isna().all() or series.nunique(dropna=True) <= 1:
            continue
        selected.append(col)
    return selected


def train_lgbm_with_optional_weights(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    sample_weight: pd.Series | None,
) -> LGBMRegressor:
    params = MODEL_PARAMS.copy()
    params["verbosity"] = -1
    model = LGBMRegressor(**params)
    if sample_weight is None:
        model.fit(train_x, train_y)
    else:
        model.fit(train_x, train_y, sample_weight=sample_weight)
    return model


def evaluate_model_spec(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    spec: dict[str, str | None],
) -> tuple[pd.DataFrame, dict[str, object]]:
    target_col = str(spec["target_col"])
    sample_weight_col = spec.get("sample_weight_col")
    if target_col not in train_df.columns:
        raise KeyError(f"Missing target column for {spec['model']}: {target_col}")

    train_x = train_df[feature_cols].copy()
    test_x = test_df[feature_cols].copy()
    train_x, test_x = cast_category_columns(train_x, test_x, feature_cols)

    train_y = (
        pd.to_numeric(train_df[target_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    sample_weight = None
    if sample_weight_col:
        sample_weight = (
            pd.to_numeric(train_df[str(sample_weight_col)], errors="coerce")
            .fillna(1.0)
            .clip(lower=0.05, upper=1.0)
        )

    model = train_lgbm_with_optional_weights(train_x, train_y, sample_weight)
    preds = predict_clipped(model, test_x)

    output_cols = [DATE_COL, BAKERY_ID_COL, BAKERY_NAME_COL, CITY_COL, TARGET_COL]
    for col in [OBSERVED_TARGET_COL, SALES_MISSING_FLAG_COL]:
        if col in test_df.columns:
            output_cols.append(col)
    pred_frame = test_df[output_cols].copy()
    pred_frame["model"] = spec["model"]
    pred_frame["train_target_col"] = target_col
    pred_frame["prediction"] = preds
    if OBSERVED_TARGET_COL in pred_frame.columns:
        pred_frame["actual_sales"] = pd.to_numeric(
            pred_frame[OBSERVED_TARGET_COL],
            errors="coerce",
        )
    else:
        pred_frame["actual_sales"] = pd.to_numeric(
            pred_frame[TARGET_COL],
            errors="coerce",
        )
    if SALES_MISSING_FLAG_COL in pred_frame.columns:
        pred_frame["eval_flag"] = (
            pd.to_numeric(pred_frame[SALES_MISSING_FLAG_COL], errors="coerce")
            .fillna(0)
            .astype(int)
            == 0
        ).astype(int)
    else:
        pred_frame["eval_flag"] = 1

    pred_frame["error"] = pred_frame["actual_sales"] - pred_frame["prediction"]
    pred_frame["abs_error"] = pred_frame["error"].abs()

    eval_frame = pred_frame[pred_frame["eval_flag"] == 1]
    metrics = regression_metrics(eval_frame["actual_sales"], eval_frame["prediction"])
    info = {
        "model": spec["model"],
        "target_col": target_col,
        "sample_weight_col": sample_weight_col,
        "feature_count": len(feature_cols),
        "eval_rows": int(len(eval_frame)),
        "sample_weight_mean": float(sample_weight.mean())
        if sample_weight is not None
        else 1.0,
        **metrics,
    }
    return pred_frame, info


def build_bakery_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if "eval_flag" in predictions.columns:
        work = predictions[predictions["eval_flag"] == 1].copy()
    else:
        work = predictions.copy()
    for (model_name, bakery_id), group in work.groupby(
        ["model", BAKERY_ID_COL], sort=False
    ):
        if "actual_sales" in group.columns:
            actual = group["actual_sales"]
        else:
            actual = group[TARGET_COL]
        metrics = regression_metrics(actual, group["prediction"])
        rows.append(
            {
                "model": model_name,
                BAKERY_ID_COL: bakery_id,
                BAKERY_NAME_COL: group[BAKERY_NAME_COL].iloc[0],
                CITY_COL: group[CITY_COL].iloc[0],
                "n_days": int(group[DATE_COL].nunique()),
                "mae": round(metrics["mae"], 6),
                "wmape": round(metrics["wmape"], 6),
                "bias": round(metrics["bias"], 6),
            }
        )
    return pd.DataFrame(rows)


def build_model_summary(
    model_metrics: list[dict[str, object]], bakery_metrics: pd.DataFrame
) -> pd.DataFrame:
    summary = pd.DataFrame(model_metrics).copy()
    for col in ["mae", "mse", "wmape", "bias", "sample_weight_mean"]:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce").round(6)

    win_counts = (
        bakery_metrics.loc[bakery_metrics.groupby(BAKERY_ID_COL)["mae"].idxmin()]
        .groupby("model")
        .size()
        .rename("win_count")
        .reset_index()
    )
    summary = summary.merge(win_counts, on="model", how="left")
    summary["win_count"] = summary["win_count"].fillna(0).astype(int)
    return summary.sort_values("mae").reset_index(drop=True)


def run_experiment(
    dataset_path: str | Path,
    *,
    output_dir: str | Path,
    test_days: int,
) -> dict[str, Path]:
    df = load_dataset(dataset_path)
    frame = build_model_frame(df)
    train_df, test_df, test_start = make_train_test_split(frame, test_days)
    feature_cols = select_feature_columns(frame)
    if not feature_cols:
        raise ValueError("No usable feature columns")

    prediction_frames = []
    model_metrics = []
    for spec in MODEL_SPECS:
        pred_frame, metrics = evaluate_model_spec(train_df, test_df, feature_cols, spec)
        prediction_frames.append(pred_frame)
        model_metrics.append(metrics)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    bakery_metrics = build_bakery_metrics(predictions)
    summary = build_model_summary(model_metrics, bakery_metrics)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_by_model": out_dir / "summary_by_model.csv",
        "bakery_metrics": out_dir / "bakery_metrics.csv",
        "predictions": out_dir / "predictions.csv",
        "metrics_json": out_dir / "metrics.json",
    }
    summary.to_csv(paths["summary_by_model"], index=False, encoding="utf-8-sig")
    bakery_metrics.to_csv(paths["bakery_metrics"], index=False, encoding="utf-8-sig")
    predictions.to_csv(paths["predictions"], index=False, encoding="utf-8-sig")
    paths["metrics_json"].write_text(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "test_days": test_days,
                "test_start": str(test_start.date()),
                "rows_train": int(len(train_df)),
                "rows_test": int(len(test_df)),
                "rows_test_eval": int(
                    test_df[SALES_MISSING_FLAG_COL].eq(0).sum()
                    if SALES_MISSING_FLAG_COL in test_df.columns
                    else len(test_df)
                ),
                "feature_count": len(feature_cols),
                "models": summary.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare bakery target cleaning strategies"
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
    print("EXP78 BAKERY TARGET CLEANING")
    print("=" * 72)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

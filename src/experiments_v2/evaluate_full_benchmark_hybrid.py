"""
Evaluate a hybrid full-benchmark model that picks the best model per SKU.

The best model is taken from `full_benchmark/best_by_sku.csv`. For each
bakery/product pair, the corresponding model's predictions are used for every
date in the common benchmark window, and aggregate `r2`, `mae`, and `mse`
metrics are computed on the resulting hybrid prediction set.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
FULL_DIR = ROOT / "src" / "experiments_v2" / "full_benchmark"
ARTIFACTS_DIR = FULL_DIR / "artifacts"
REPORTS_DIR = ROOT / "reports"

DATE_COL = "Дата"
BAKERY_COL = "Пекарня"
PRODUCT_COL = "Номенклатура"

MODEL_SPECS = [
    {
        "name": "60_baseline_v3",
        "path": ARTIFACTS_DIR / "60_baseline_v3" / "predictions.csv",
        "fact_col": "fact_demand",
        "pred_col": "pred_P50",
    },
    {
        "name": "61_censoring_behavioral",
        "path": ARTIFACTS_DIR / "61_censoring_behavioral" / "predictions.csv",
        "fact_col": "fact",
        "pred_col": "pred",
    },
    {
        "name": "62_assortment_availability",
        "path": ARTIFACTS_DIR / "62_assortment_availability" / "predictions.csv",
        "fact_col": "fact",
        "pred_col": "pred",
    },
    {
        "name": "66_cluster_features",
        "path": ARTIFACTS_DIR / "66_cluster_features" / "new_predictions.csv",
        "fact_col": "Спрос",
        "pred_col": "prediction",
    },
]

ALLOWED_MODELS = [spec["name"] for spec in MODEL_SPECS]


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    r2 = np.nan
    if len(y_true_arr) >= 2 and not np.allclose(y_true_arr, y_true_arr[0]):
        r2 = float(r2_score(y_true_arr, y_pred_arr))
    return {
        "r2": r2,
        "mse": float(mean_squared_error(y_true_arr, y_pred_arr)),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "wmape": float(np.sum(np.abs(y_true_arr - y_pred_arr)) / (np.sum(y_true_arr) + 1e-8) * 100),
    }


def build_common_dates() -> list[str]:
    date_sets: list[set[str]] = []
    for spec in MODEL_SPECS:
        if spec["name"] == "66_cluster_features":
            continue
        df = pd.read_csv(spec["path"], encoding="utf-8-sig", usecols=[DATE_COL])
        dates = pd.to_datetime(df[DATE_COL]).dt.strftime("%Y-%m-%d")
        date_sets.append(set(dates.tolist()))

    common = set.intersection(*date_sets)
    if not common:
        raise ValueError("No common date window found")
    return sorted(common)


def build_restricted_best_map(best_by_sku: pd.DataFrame) -> dict[tuple[str, str], str]:
    r2_cols = [f"{name}_r2" for name in ALLOWED_MODELS if f"{name}_r2" in best_by_sku.columns]
    if not r2_cols:
        raise ValueError("No allowed model columns found in best_by_sku")

    work = best_by_sku[[BAKERY_COL, PRODUCT_COL, *r2_cols]].copy()
    r2_values = work[r2_cols].replace([np.inf, -np.inf], np.nan).fillna(-np.inf)
    best_idx = r2_values.idxmax(axis=1)
    best_model = best_idx.str.replace("_r2", "", regex=False)
    return dict(zip(zip(work[BAKERY_COL], work[PRODUCT_COL]), best_model.tolist()))


def load_model_predictions(spec: dict, common_dates: list[str]) -> pd.DataFrame:
    df = pd.read_csv(spec["path"], encoding="utf-8-sig")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df[df[DATE_COL].dt.strftime("%Y-%m-%d").isin(common_dates)].copy()
    df = df[[DATE_COL, BAKERY_COL, PRODUCT_COL, spec["fact_col"], spec["pred_col"]]].copy()
    df = df.rename(columns={spec["fact_col"]: "fact", spec["pred_col"]: "pred"})
    df["model"] = spec["name"]
    return df


def main() -> None:
    common_dates = build_common_dates()
    best_path = FULL_DIR / "best_by_sku.csv"
    if not best_path.exists():
        raise FileNotFoundError(f"Missing best-by-SKU table: {best_path}")

    best_by_sku = pd.read_csv(best_path, encoding="utf-8-sig")
    best_map = build_restricted_best_map(best_by_sku)

    frames = []
    for spec in MODEL_SPECS:
        if not spec["path"].exists():
            continue
        frames.append(load_model_predictions(spec, common_dates))

    if not frames:
        raise RuntimeError("No prediction files were loaded")

    all_preds = pd.concat(frames, ignore_index=True)
    all_preds["best_model"] = [
        best_map.get((bakery, product)) for bakery, product in zip(all_preds[BAKERY_COL], all_preds[PRODUCT_COL])
    ]
    hybrid = all_preds[all_preds["model"] == all_preds["best_model"]].copy()

    if hybrid.empty:
        raise RuntimeError("Hybrid prediction set is empty")

    metrics = regression_metrics(hybrid["fact"], hybrid["pred"])
    model_usage = hybrid["model"].value_counts().to_dict()

    out_csv = REPORTS_DIR / "hybrid_full_benchmark_predictions.csv"
    out_json = REPORTS_DIR / "hybrid_full_benchmark_metrics.json"
    hybrid.to_csv(out_csv, index=False, encoding="utf-8-sig")
    out_json.write_text(
        json.dumps(
            {
                "common_dates": common_dates,
                "rows": int(len(hybrid)),
                "sku_pairs": int(hybrid[[BAKERY_COL, PRODUCT_COL]].drop_duplicates().shape[0]),
                "metrics": {
                    "r2": round(metrics["r2"], 4) if np.isfinite(metrics["r2"]) else None,
                    "mae": round(metrics["mae"], 4),
                    "mse": round(metrics["mse"], 4),
                    "wmape": round(metrics["wmape"], 2),
                },
                "model_usage": model_usage,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Hybrid rows: {len(hybrid):,}")
    print(f"Hybrid SKU pairs: {hybrid[[BAKERY_COL, PRODUCT_COL]].drop_duplicates().shape[0]:,}")
    print(f"r2: {metrics['r2']:.4f}")
    print(f"mae: {metrics['mae']:.4f}")
    print(f"mse: {metrics['mse']:.4f}")
    print(f"wmape: {metrics['wmape']:.2f}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()

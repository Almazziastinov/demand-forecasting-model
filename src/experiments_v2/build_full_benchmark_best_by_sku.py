"""
Build `best_by_sku.csv` for the full benchmark suite.

The benchmark mixes different model families, but all models are evaluated on
the same short held-out window except experiment 66, which also has a longer
prediction file. For experiment 66 we keep only the rows that fall inside the
common window used by the other full-benchmark models.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "src" / "experiments_v2"
FULL_DIR = EXP_DIR / "full_benchmark"
ARTIFACTS_DIR = FULL_DIR / "artifacts"

DATE_COL = "Дата"
BAKERY_COL = "Пекарня"
PRODUCT_COL = "Номенклатура"


MODEL_SPECS = [
    {
        "name": "60_baseline_v3",
        "path": ARTIFACTS_DIR / "60_baseline_v3" / "predictions.csv",
        "fact_col": "fact_demand",
        "pred_col": "pred_P50",
        "date_col": DATE_COL,
    },
    {
        "name": "43_quantile",
        "path": ARTIFACTS_DIR / "43_quantile" / "predictions.csv",
        "fact_col": "fact",
        "pred_col": "pred_P50",
        "date_col": DATE_COL,
    },
    {
        "name": "45_log_residual",
        "path": ARTIFACTS_DIR / "45_log_residual" / "predictions.csv",
        "fact_col": "fact",
        "pred_col": "pred",
        "date_col": DATE_COL,
    },
    {
        "name": "46_variance_weighted",
        "path": ARTIFACTS_DIR / "46_variance_weighted" / "predictions.csv",
        "fact_col": "fact",
        "pred_col": "pred",
        "date_col": DATE_COL,
    },
    {
        "name": "61_censoring_behavioral",
        "path": ARTIFACTS_DIR / "61_censoring_behavioral" / "predictions.csv",
        "fact_col": "fact",
        "pred_col": "pred",
        "date_col": DATE_COL,
    },
    {
        "name": "62_assortment_availability",
        "path": ARTIFACTS_DIR / "62_assortment_availability" / "predictions.csv",
        "fact_col": "fact",
        "pred_col": "pred",
        "date_col": DATE_COL,
    },
    {
        "name": "63_combined_61_62",
        "path": ARTIFACTS_DIR / "63_combined_61_62" / "predictions.csv",
        "fact_col": "fact",
        "pred_col": "pred",
        "date_col": DATE_COL,
    },
    {
        "name": "66_cluster_features",
        "path": ARTIFACTS_DIR / "66_cluster_features" / "new_predictions.csv",
        "fact_col": "Спрос",
        "pred_col": "prediction",
        "date_col": DATE_COL,
    },
]


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


def load_prediction_table(spec: dict, date_filter: set[str] | None = None) -> tuple[pd.DataFrame, dict]:
    path = spec["path"]
    if not path.exists():
        return pd.DataFrame(), {"model": spec["name"], "status": "missing", "path": str(path)}

    df = pd.read_csv(path, encoding="utf-8-sig")
    if spec["date_col"] not in df.columns:
        raise KeyError(f"{path} is missing date column {spec['date_col']}")
    if spec["fact_col"] not in df.columns:
        raise KeyError(f"{path} is missing fact column {spec['fact_col']}")
    if spec["pred_col"] not in df.columns:
        raise KeyError(f"{path} is missing prediction column {spec['pred_col']}")

    df[spec["date_col"]] = pd.to_datetime(df[spec["date_col"]])
    if date_filter is not None:
        df = df[df[spec["date_col"]].dt.strftime("%Y-%m-%d").isin(date_filter)].copy()

    work = df[[BAKERY_COL, PRODUCT_COL, spec["fact_col"], spec["pred_col"]]].copy()
    work = work.rename(columns={spec["fact_col"]: "fact", spec["pred_col"]: "pred"})

    rows: list[dict] = []
    for (bakery, product), group in work.groupby([BAKERY_COL, PRODUCT_COL], sort=False):
        metrics = regression_metrics(group["fact"], group["pred"])
        rows.append(
            {
                BAKERY_COL: bakery,
                PRODUCT_COL: product,
                "model": spec["name"],
                "n_test_rows": int(len(group)),
                "r2": round(metrics["r2"], 4) if np.isfinite(metrics["r2"]) else np.nan,
                "mse": round(metrics["mse"], 4),
                "mae": round(metrics["mae"], 4),
                "wmape": round(metrics["wmape"], 2),
            }
        )

    out = pd.DataFrame(rows)
    meta = {
        "model": spec["name"],
        "status": "loaded" if len(df) else "empty",
        "path": str(path),
        "rows": int(len(df)),
        "dates_min": df[spec["date_col"]].min().strftime("%Y-%m-%d") if len(df) else None,
        "dates_max": df[spec["date_col"]].max().strftime("%Y-%m-%d") if len(df) else None,
        "dates_nunique": int(df[spec["date_col"]].dt.normalize().nunique()) if len(df) else 0,
    }
    return out, meta


def build_common_date_filter() -> set[str]:
    date_sets: list[set[str]] = []
    for spec in MODEL_SPECS:
        if spec["name"] == "66_cluster_features":
            continue
        path = spec["path"]
        if not path.exists():
            continue
        df = pd.read_csv(path, encoding="utf-8-sig", usecols=[DATE_COL])
        dates = pd.to_datetime(df[DATE_COL]).dt.strftime("%Y-%m-%d")
        date_sets.append(set(dates.tolist()))

    if not date_sets:
        raise FileNotFoundError("No full_benchmark prediction files were found")

    common = set.intersection(*date_sets)
    if not common:
        raise ValueError("No common date window found across full_benchmark models")
    return common


def merge_metrics(metrics_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
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
        return pd.DataFrame()

    wide = wide.reset_index(drop=True)
    r2_cols = [c for c in wide.columns if c.endswith("_r2")]
    r2_values = wide[r2_cols].replace([np.inf, -np.inf], np.nan).fillna(-np.inf)
    best_idx = r2_values.idxmax(axis=1)

    result = wide[[BAKERY_COL, PRODUCT_COL]].copy()
    result["best_model"] = best_idx.str.replace("_r2", "", regex=False)
    result["best_r2"] = r2_values.max(axis=1).replace(-np.inf, np.nan)
    result["best_mae"] = [
        wide.iloc[i][f"{result.iloc[i]['best_model']}_mae"] if pd.notna(result.iloc[i]["best_model"]) else np.nan
        for i in range(len(result))
    ]
    result["best_mse"] = [
        wide.iloc[i][f"{result.iloc[i]['best_model']}_mse"] if pd.notna(result.iloc[i]["best_model"]) else np.nan
        for i in range(len(result))
    ]
    result["best_wmape"] = [
        wide.iloc[i][f"{result.iloc[i]['best_model']}_wmape"] if pd.notna(result.iloc[i]["best_model"]) else np.nan
        for i in range(len(result))
    ]

    return pd.concat([result, wide.drop(columns=[BAKERY_COL, PRODUCT_COL]).reset_index(drop=True)], axis=1)


def main() -> None:
    FULL_DIR.mkdir(parents=True, exist_ok=True)

    common_dates = build_common_date_filter()
    common_dates_sorted = sorted(common_dates)
    print(f"Common benchmark dates: {common_dates_sorted[0]} .. {common_dates_sorted[-1]} ({len(common_dates)} days)")

    metrics_frames: dict[str, pd.DataFrame] = {}
    manifest: list[dict] = []

    for spec in MODEL_SPECS:
        frame, meta = load_prediction_table(spec, date_filter=common_dates)
        manifest.append(meta)
        if frame.empty:
            continue
        metrics_frames[spec["name"]] = frame
        print(f"Loaded {spec['name']}: {len(frame):,} sku-pairs, {meta['rows']:,} rows")

    best_by_sku = merge_metrics(metrics_frames)
    if best_by_sku.empty:
        raise RuntimeError("No metrics could be built for full_benchmark")

    output_csv = FULL_DIR / "best_by_sku.csv"
    output_json = FULL_DIR / "best_by_sku_manifest.json"
    best_by_sku.to_csv(output_csv, index=False, encoding="utf-8-sig")
    output_json.write_text(
        json.dumps(
            {
                "common_dates": common_dates_sorted,
                "models": manifest,
                "rows": int(len(best_by_sku)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved: {output_csv}")
    print(f"Saved: {output_json}")
    print(f"Rows: {len(best_by_sku):,}")


if __name__ == "__main__":
    main()

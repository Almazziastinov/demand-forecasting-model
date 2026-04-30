"""
Experiment 70: Bad SKU filtering / downweighting / target smoothing.

Step 2 from HYBRID_RESEARCH_PLAN:
try to improve the global model by changing how weak SKU pairs
participate in training.

Variants:
  1. baseline_all           - train on all rows
  2. exclude_weak           - drop weak SKU rows from training
  3. downweight_weak_025    - weak SKU rows get sample_weight=0.25
  4. downweight_weak_050    - weak SKU rows get sample_weight=0.50
  5. smooth_weak_roll7      - weak SKU target is blended with demand_roll_mean7
  6. global66_weak_only     - train only on rows where best_model is 66_cluster_features
                               and best_r2 < 0

Weak SKU definition:
  best_r2 < 0 from reports/hybrid_research/sku_r2_summary.csv

Input:
  data/processed/daily_sales_8m_demand.csv
  reports/hybrid_research/sku_r2_summary.csv

Output:
  src/experiments_v2/70_bad_sku_filtering/metrics.json
  src/experiments_v2/70_bad_sku_filtering/predictions.csv
  src/experiments_v2/70_bad_sku_filtering/variant_summary.csv
  src/experiments_v2/70_bad_sku_filtering/variant_predictions.csv
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from src.config import MODEL_PARAMS  # noqa: E402
from src.experiments_v2.common import (  # noqa: E402
    CATEGORICAL_COLS_V2,
    DEMAND_8M_PATH,
    DEMAND_TARGET,
    FEATURES_V3,
    predict_clipped,
    print_category_metrics,
    print_metrics,
    save_results,
    wmape,
)


EXP_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = Path(ROOT) / "reports" / "hybrid_research"
WEAK_SKU_PATH = RESEARCH_DIR / "sku_r2_summary.csv"

BASELINE_V3_MAE = 2.8816
GLOBAL_MODEL_NAME = "66_cluster_features"
WEAK_R2_THRESHOLD = 0.0
SMOOTH_ALPHA = 0.7
WEAK_WEIGHT_025 = 0.25
WEAK_WEIGHT_050 = 0.50
TEST_DAYS_MONTH = 30


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    if np.allclose(y_true, y_true[0]):
        return float("nan")
    return float(r2_score(y_true, y_pred))


def load_daily_data() -> pd.DataFrame:
    df = pd.read_csv(str(DEMAND_8M_PATH), encoding="utf-8-sig")
    df["Дата"] = pd.to_datetime(df["Дата"])
    for col in CATEGORICAL_COLS_V2:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def load_weak_skus() -> pd.DataFrame:
    if not WEAK_SKU_PATH.exists():
        raise FileNotFoundError(f"Weak SKU summary not found: {WEAK_SKU_PATH}")

    weak = pd.read_csv(WEAK_SKU_PATH, encoding="utf-8-sig")
    required = {"Пекарня", "Номенклатура", "best_model", "best_r2"}
    missing = required.difference(weak.columns)
    if missing:
        raise KeyError(f"Missing columns in {WEAK_SKU_PATH}: {sorted(missing)}")

    weak = weak[["Пекарня", "Номенклатура", "best_model", "best_r2"]].copy()
    weak["best_model"] = weak["best_model"].astype(str)
    weak["best_r2"] = pd.to_numeric(weak["best_r2"], errors="coerce")
    weak["is_weak_sku"] = weak["best_r2"] < WEAK_R2_THRESHOLD
    weak["is_global66_weak"] = (weak["best_model"] == GLOBAL_MODEL_NAME) & weak["is_weak_sku"]
    return weak


def add_weak_sku_flag(df: pd.DataFrame, weak_map: pd.DataFrame) -> pd.DataFrame:
    work = df.merge(
        weak_map[["Пекарня", "Номенклатура", "best_model", "best_r2", "is_weak_sku", "is_global66_weak"]],
        on=["Пекарня", "Номенклатура"],
        how="left",
    )
    work["best_model"] = work["best_model"].fillna("")
    work["is_weak_sku"] = work["is_weak_sku"].fillna(False)
    work["is_global66_weak"] = work["is_global66_weak"].fillna(False)
    work["best_r2"] = work["best_r2"].fillna(np.nan)
    for col in CATEGORICAL_COLS_V2:
        if col in work.columns:
            work[col] = work[col].astype("category")
    return work


def make_split(df: pd.DataFrame, test_days: int = TEST_DAYS_MONTH) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_start = df["Дата"].max() - pd.Timedelta(days=test_days - 1)
    train = df[df["Дата"] < test_start].copy()
    test = df[df["Дата"] >= test_start].copy()
    return train, test


def train_quantile_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> LGBMRegressor:
    params = MODEL_PARAMS.copy()
    params["objective"] = "quantile"
    params["alpha"] = 0.5
    params.pop("metric", None)
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def blended_target(y: np.ndarray, weak_mask: np.ndarray, roll7: np.ndarray, alpha: float = SMOOTH_ALPHA) -> np.ndarray:
    y_adj = y.astype(float).copy()
    roll7_safe = np.where(np.isfinite(roll7), roll7, y_adj)
    y_adj[weak_mask] = alpha * y_adj[weak_mask] + (1.0 - alpha) * roll7_safe[weak_mask]
    return y_adj


def evaluate_predictions(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_weak_mask: np.ndarray,
) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    wm = float(wmape(y_true, y_pred))
    bias = float(np.mean(y_true - y_pred))
    r2 = safe_r2(y_true, y_pred)

    weak_true = y_true[test_weak_mask]
    weak_pred = y_pred[test_weak_mask]
    strong_true = y_true[~test_weak_mask]
    strong_pred = y_pred[~test_weak_mask]

    weak_mae = float(mean_absolute_error(weak_true, weak_pred)) if len(weak_true) else float("nan")
    weak_wmape = float(wmape(weak_true, weak_pred)) if len(weak_true) else float("nan")
    weak_r2 = safe_r2(weak_true, weak_pred) if len(weak_true) else float("nan")
    strong_mae = float(mean_absolute_error(strong_true, strong_pred)) if len(strong_true) else float("nan")
    strong_wmape = float(wmape(strong_true, strong_pred)) if len(strong_true) else float("nan")
    strong_r2 = safe_r2(strong_true, strong_pred) if len(strong_true) else float("nan")

    return {
        "variant": name,
        "mae": mae,
        "wmape": wm,
        "bias": bias,
        "r2": r2,
        "weak_mae": weak_mae,
        "weak_wmape": weak_wmape,
        "weak_r2": weak_r2,
        "strong_mae": strong_mae,
        "strong_wmape": strong_wmape,
        "strong_r2": strong_r2,
    }


def fit_variant(
    name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    weak_col: str = "is_weak_sku",
    sample_weight: np.ndarray | None = None,
    target: np.ndarray | None = None,
    train_mask: np.ndarray | None = None,
) -> tuple[dict, np.ndarray]:
    work = train.copy()

    if train_mask is not None:
        work = work.loc[train_mask].copy()
        if sample_weight is not None:
            sample_weight = sample_weight[train_mask]
        if target is not None:
            target = target[train_mask]

    X_train = work[features].copy()
    y_train = target if target is not None else work[DEMAND_TARGET].to_numpy(dtype=float)
    X_test = test[features].copy()
    y_test = test[DEMAND_TARGET].to_numpy(dtype=float)
    test_weak_mask = test[weak_col].to_numpy(dtype=bool)

    for col in CATEGORICAL_COLS_V2:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")
        if col in X_test.columns:
            X_test[col] = X_test[col].astype("category")

    t0 = time.time()
    model = train_quantile_model(X_train, y_train, sample_weight=sample_weight)
    train_time = time.time() - t0
    y_pred = predict_clipped(model, X_test)

    metrics = evaluate_predictions(name, y_test, y_pred, test_weak_mask)
    metrics["train_rows"] = int(len(work))
    metrics["train_weak_rows"] = int(work[weak_col].sum())
    metrics["train_weak_share"] = float(work[weak_col].mean()) if len(work) else 0.0
    metrics["test_rows"] = int(len(test))
    metrics["test_weak_rows"] = int(test_weak_mask.sum())
    metrics["test_weak_share"] = float(test_weak_mask.mean()) if len(test_weak_mask) else 0.0
    metrics["train_time_s"] = round(train_time, 1)

    return metrics, y_pred


def main() -> None:
    print("=" * 72)
    print("  EXPERIMENT 70: Bad SKU filtering / downweighting / smoothing")
    print("  Step 2 from HYBRID_RESEARCH_PLAN")
    print("=" * 72)
    print(f"  Holdout: last {TEST_DAYS_MONTH} days")
    print(f"  Global model reference: {GLOBAL_MODEL_NAME}")

    if not WEAK_SKU_PATH.exists():
        raise FileNotFoundError(f"Required summary file not found: {WEAK_SKU_PATH}")

    df = load_daily_data()
    weak_map = load_weak_skus()
    df = add_weak_sku_flag(df, weak_map)

    available = [f for f in FEATURES_V3 if f in df.columns]
    missing = [f for f in FEATURES_V3 if f not in df.columns]

    print(f"\n[1/5] Data")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['Дата'].min().date()} -- {df['Дата'].max().date()}")
    print(f"  Using {len(available)} of {len(FEATURES_V3)} features")
    if missing:
        print(f"  Missing features: {missing}")

    weak_share = df["is_weak_sku"].mean() * 100
    print(f"  Weak SKU share (best_r2 < {WEAK_R2_THRESHOLD:+.1f}): {weak_share:.1f}%")

    train, test = make_split(df, TEST_DAYS_MONTH)
    print(f"\n[2/5] Split")
    print(f"  Train rows: {len(train):,} ({train['Дата'].nunique()} days)")
    print(f"  Test rows:  {len(test):,} ({test['Дата'].nunique()} days)")
    print(f"  Train weak share: {train['is_weak_sku'].mean() * 100:.1f}%")
    print(f"  Test weak share:  {test['is_weak_sku'].mean() * 100:.1f}%")
    print(f"  Train global66-weak rows: {train['is_global66_weak'].sum():,}")

    y_test = test[DEMAND_TARGET].to_numpy(dtype=float)
    test_weak_mask = test["is_weak_sku"].to_numpy(dtype=bool)
    roll7_train = train["demand_roll_mean7"].to_numpy(dtype=float) if "demand_roll_mean7" in train.columns else train[DEMAND_TARGET].to_numpy(dtype=float)
    roll7_train = np.where(np.isfinite(roll7_train), roll7_train, train[DEMAND_TARGET].to_numpy(dtype=float))

    print(f"\n[3/5] Train variants")
    variants: list[tuple[str, pd.DataFrame, dict, np.ndarray]] = []

    baseline_metrics, baseline_pred = fit_variant("baseline_all", train, test, available)
    variants.append(("baseline_all", train, baseline_metrics, baseline_pred))

    exclude_mask = ~train["is_weak_sku"].to_numpy(dtype=bool)
    exclude_metrics, exclude_pred = fit_variant("exclude_weak", train, test, available, train_mask=exclude_mask)
    variants.append(("exclude_weak", train.loc[exclude_mask].copy(), exclude_metrics, exclude_pred))

    weak_mask_train = train["is_weak_sku"].to_numpy(dtype=bool)
    weights_025 = np.where(weak_mask_train, WEAK_WEIGHT_025, 1.0)
    weights_025 = weights_025 / weights_025.mean()
    down_025_metrics, down_025_pred = fit_variant(
        "downweight_weak_025",
        train,
        test,
        available,
        sample_weight=weights_025,
    )
    variants.append(("downweight_weak_025", train, down_025_metrics, down_025_pred))

    weights_050 = np.where(weak_mask_train, WEAK_WEIGHT_050, 1.0)
    weights_050 = weights_050 / weights_050.mean()
    down_050_metrics, down_050_pred = fit_variant(
        "downweight_weak_050",
        train,
        test,
        available,
        sample_weight=weights_050,
    )
    variants.append(("downweight_weak_050", train, down_050_metrics, down_050_pred))

    y_smooth = blended_target(
        train[DEMAND_TARGET].to_numpy(dtype=float),
        weak_mask_train,
        roll7_train,
        alpha=SMOOTH_ALPHA,
    )
    smooth_metrics, smooth_pred = fit_variant(
        "smooth_weak_roll7",
        train,
        test,
        available,
        target=y_smooth,
    )
    variants.append(("smooth_weak_roll7", train, smooth_metrics, smooth_pred))

    global66_mask_train = train["is_global66_weak"].to_numpy(dtype=bool)
    if global66_mask_train.sum() > 0:
        global66_metrics, global66_pred = fit_variant(
            "global66_weak_only",
            train,
            test,
            available,
            train_mask=global66_mask_train,
        )
        variants.append(("global66_weak_only", train.loc[global66_mask_train].copy(), global66_metrics, global66_pred))
    else:
        print("  [warn] No train rows matched best_model=66_cluster_features and best_r2<0")

    print(f"\n[4/5] Comparison")
    summary_rows = []
    predictions = pd.DataFrame(
        {
            "Дата": test["Дата"].values,
            "Пекарня": test["Пекарня"].values,
            "Номенклатура": test["Номенклатура"].values,
            "Категория": test["Категория"].values if "Категория" in test.columns else "",
            "fact": y_test,
            "is_weak_sku": test_weak_mask,
            "is_global66_weak": test["is_global66_weak"].values,
        }
    )

    for name, _, metrics, pred in variants:
        summary_rows.append(metrics)
        predictions[f"pred_{name}"] = np.round(pred, 2)
        print(
            f"  {name:<22} MAE={metrics['mae']:.4f}  "
            f"WMAPE={metrics['wmape']:.2f}%  R2={metrics['r2']:.4f}  "
            f"weak_MAE={metrics['weak_mae']:.4f}"
        )

    summary = pd.DataFrame(summary_rows).sort_values("mae").reset_index(drop=True)
    best_row = summary.iloc[0].to_dict()
    best_name = str(best_row["variant"])
    best_pred = predictions[f"pred_{best_name}"].to_numpy(dtype=float)

    print(f"\n  Best variant: {best_name}")
    print_metrics(best_name, y_test, best_pred, baseline_mae=BASELINE_V3_MAE)
    print(f"  Delta vs baseline v3 MAE: {best_row['mae'] - BASELINE_V3_MAE:+.4f}")
    print("\n  Per-category metrics for best variant")
    print_category_metrics(y_test, best_pred, test["Категория"].values)

    print(f"\n[5/5] Saving")
    summary_path = EXP_DIR / "variant_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    predictions_path = EXP_DIR / "variant_predictions.csv"
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    metrics = {
        "experiment": "70_bad_sku_filtering",
        "target": DEMAND_TARGET,
        "weak_r2_threshold": WEAK_R2_THRESHOLD,
        "global_model_name": GLOBAL_MODEL_NAME,
        "test_days": TEST_DAYS_MONTH,
        "baseline_v3_mae": BASELINE_V3_MAE,
        "best_variant": best_name,
        "best_mae": round(float(best_row["mae"]), 4),
        "best_wmape": round(float(best_row["wmape"]), 2),
        "best_bias": round(float(best_row["bias"]), 4),
        "best_r2": round(float(best_row["r2"]), 4) if np.isfinite(best_row["r2"]) else None,
        "weak_share_pct": round(float(weak_share), 2),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_weak_rows": int(train["is_weak_sku"].sum()),
        "train_global66_weak_rows": int(train["is_global66_weak"].sum()),
        "test_weak_rows": int(test["is_weak_sku"].sum()),
        "summary_csv": str(summary_path.name),
        "predictions_csv": str(predictions_path.name),
        "variants": summary[["variant", "mae", "wmape", "bias", "r2", "weak_mae", "weak_wmape", "train_time_s"]].to_dict("records"),
    }

    save_results(
        EXP_DIR,
        metrics,
        predictions=pd.DataFrame(
            {
                "Дата": test["Дата"].values,
                "Пекарня": test["Пекарня"].values,
                "Номенклатура": test["Номенклатура"].values,
                "Категория": test["Категория"].values if "Категория" in test.columns else "",
                "fact": y_test,
                "pred": np.round(best_pred, 2),
                "is_weak_sku": test_weak_mask,
                "is_global66_weak": test["is_global66_weak"].values,
            }
        ),
    )

    with open(EXP_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"  Summary CSV: {summary_path}")
    print(f"  Variant predictions: {predictions_path}")
    print("  Done!")


if __name__ == "__main__":
    main()

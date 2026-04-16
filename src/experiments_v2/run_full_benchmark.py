"""
Run the full benchmark suite sequentially and aggregate results.

This runner executes the selected experiment families one by one, copies the
important per-model artifacts into a central directory, and builds a single
`best_by_sku.csv` across all models.

Outputs are written to `src/experiments_v2/full_benchmark/`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")
EXP_DIR = Path(__file__).resolve().parent
OUT_DIR = EXP_DIR / "full_benchmark"
ARTIFACTS_DIR = OUT_DIR / "artifacts"

BAKERY_COL = "Пекарня"
PRODUCT_COL = "Номенклатура"


EXPERIMENTS = [
    {
        "name": "60_baseline_v3",
        "script": "60_baseline_v3/run.py",
        "family": "demand",
        "fact_col": "fact_demand",
        "pred_col": "pred_P50",
    },
    {
        "name": "43_quantile",
        "script": "43_quantile/run.py",
        "family": "sales",
        "fact_col": "fact",
        "pred_col": "pred_P50",
    },
    {
        "name": "45_log_residual",
        "script": "45_log_residual/run.py",
        "family": "sales",
        "fact_col": "fact",
        "pred_col": "pred",
    },
    {
        "name": "46_variance_weighted",
        "script": "46_variance_weighted/run.py",
        "family": "sales",
        "fact_col": "fact",
        "pred_col": "pred",
    },
    {
        "name": "61_censoring_behavioral",
        "script": "61_censoring_behavioral/run.py",
        "family": "demand",
        "fact_col": "fact",
        "pred_col": "pred",
    },
    {
        "name": "62_assortment_availability",
        "script": "62_assortment_availability/run.py",
        "family": "demand",
        "fact_col": "fact",
        "pred_col": "pred",
    },
    {
        "name": "63_combined_61_62",
        "script": "63_combined_61_62/run.py",
        "family": "demand",
        "fact_col": "fact",
        "pred_col": "pred",
    },
    {
        "name": "53_quantile_demand",
        "script": "53_quantile_demand/run.py",
        "family": "demand",
        "fact_col": "fact",
        "pred_col": "pred_P50",
    },
    {
        "name": "55_log_residual_demand",
        "script": "55_log_residual_demand/run.py",
        "family": "demand",
        "fact_col": "fact",
        "pred_col": "pred",
    },
    {
        "name": "56_variance_weighted_demand",
        "script": "56_variance_weighted_demand/run.py",
        "family": "demand",
        "fact_col": "fact",
        "pred_col": "pred",
    },
    {
        "name": "66_cluster_features",
        "script": "66_cluster_features/run.py",
        "family": "demand",
        "fact_col": "fact_demand",
        "pred_col": "pred_E_routed_cluster_ts",
    },
]

FAMILY_ORDER = ["sales", "demand"]


def run_command(script_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run([PYTHON, str(script_path)], cwd=str(ROOT))


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


def summarize_predictions(
    df: pd.DataFrame,
    fact_col: str,
    pred_col: str,
    model_name: str,
    family: str,
) -> pd.DataFrame:
    work = df[[BAKERY_COL, PRODUCT_COL, fact_col, pred_col]].copy()
    work = work.rename(columns={fact_col: "fact", pred_col: "pred"})
    rows: list[dict] = []
    for (bakery, product), group in work.groupby([BAKERY_COL, PRODUCT_COL], sort=False):
        metrics = regression_metrics(group["fact"], group["pred"])
        rows.append(
            {
                BAKERY_COL: bakery,
                PRODUCT_COL: product,
                "model": model_name,
                "family": family,
                "n_test_rows": int(len(group)),
                "r2": round(metrics["r2"], 4) if np.isfinite(metrics["r2"]) else np.nan,
                "mse": round(metrics["mse"], 4),
                "mae": round(metrics["mae"], 4),
                "wmape": round(metrics["wmape"], 2),
            }
        )
    return pd.DataFrame(rows)


def merge_model_frames(metrics_by_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    wide: pd.DataFrame | None = None
    for model_name, frame in metrics_by_model.items():
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
    best_r2_col = r2_values.idxmax(axis=1)

    result = wide[[BAKERY_COL, PRODUCT_COL]].copy()
    result["best_model"] = best_r2_col.str.replace("_r2", "", regex=False)
    result["best_r2"] = r2_values.max(axis=1).replace(-np.inf, np.nan)

    best_models = result["best_model"].tolist()
    result["best_mae"] = [
        wide.iloc[i][f"{best_models[i]}_mae"] if pd.notna(best_models[i]) else np.nan for i in range(len(result))
    ]
    result["best_mse"] = [
        wide.iloc[i][f"{best_models[i]}_mse"] if pd.notna(best_models[i]) else np.nan for i in range(len(result))
    ]
    result["best_wmape"] = [
        wide.iloc[i][f"{best_models[i]}_wmape"] if pd.notna(best_models[i]) else np.nan for i in range(len(result))
    ]

    return pd.concat([result, wide.drop(columns=[BAKERY_COL, PRODUCT_COL]).reset_index(drop=True)], axis=1)


def build_model_summary_from_wide(best_df: pd.DataFrame) -> pd.DataFrame:
    if best_df.empty:
        return pd.DataFrame()

    model_names = sorted({col[:-3] for col in best_df.columns if col.endswith("_r2")})
    rows = []
    for model in model_names:
        r2_col = f"{model}_r2"
        mse_col = f"{model}_mse"
        mae_col = f"{model}_mae"
        wmape_col = f"{model}_wmape"
        if r2_col not in best_df.columns:
            continue
        rows.append(
            {
                "model": model,
                "n_pairs": int(best_df[r2_col].notna().sum()),
                "avg_r2": round(float(best_df[r2_col].mean(skipna=True)), 4),
                "median_r2": round(float(best_df[r2_col].median(skipna=True)), 4),
                "avg_mae": round(float(best_df[mae_col].mean(skipna=True)), 4),
                "median_mae": round(float(best_df[mae_col].median(skipna=True)), 4),
                "avg_mse": round(float(best_df[mse_col].mean(skipna=True)), 4),
                "avg_wmape": round(float(best_df[wmape_col].mean(skipna=True)), 2),
                "win_count": int((best_df["best_model"] == model).sum()),
            }
        )

    return pd.DataFrame(rows).sort_values(["avg_r2", "avg_mae"], ascending=[False, True]).reset_index(drop=True)


def copy_model_artifacts(exp_name: str) -> dict:
    exp_dir = EXP_DIR / exp_name
    dest_dir = ARTIFACTS_DIR / exp_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for filename in ["metrics.json", "predictions.csv"]:
        src = exp_dir / filename
        if src.exists():
            dst = dest_dir / filename
            shutil.copy2(src, dst)
            copied.append({"model": exp_name, "source": str(src), "copied_to": str(dst)})

    return {
        "model": exp_name,
        "artifact_dir": str(dest_dir),
        "metrics_json": str(dest_dir / "metrics.json"),
        "predictions_csv": str(dest_dir / "predictions.csv"),
        "copied_files": copied,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("FULL BENCHMARK RUNNER")
    print("=" * 72)
    t_start = time.time()

    run_rows: list[dict] = []
    artifact_rows: list[dict] = []

    for idx, exp in enumerate(EXPERIMENTS, start=1):
        script_path = EXP_DIR / exp["script"]
        print(f"\n{'#' * 72}")
        print(f"[{idx}/{len(EXPERIMENTS)}] {exp['name']} ({exp['family']})")
        print(f"{'#' * 72}")
        t0 = time.time()
        ret = run_command(script_path)
        elapsed = time.time() - t0
        status = "OK" if ret.returncode == 0 else f"FAIL({ret.returncode})"
        run_rows.append(
            {
                "model": exp["name"],
                "family": exp["family"],
                "status": status,
                "elapsed_s": round(elapsed, 1),
                "script": str(script_path),
            }
        )
        print(f"  >>> {exp['name']}: {status} ({elapsed:.1f}s)")
        artifact_rows.append(copy_model_artifacts(exp["name"]))
        if ret.returncode != 0:
            pd.DataFrame(run_rows).to_csv(OUT_DIR / "run_manifest.csv", index=False, encoding="utf-8-sig")
            with open(OUT_DIR / "artifact_manifest.json", "w", encoding="utf-8") as f:
                json.dump(artifact_rows, f, ensure_ascii=False, indent=2)
            raise SystemExit(ret.returncode)

    manifest = pd.DataFrame(run_rows)
    manifest.to_csv(OUT_DIR / "run_manifest.csv", index=False, encoding="utf-8-sig")
    with open(OUT_DIR / "artifact_manifest.json", "w", encoding="utf-8") as f:
        json.dump(artifact_rows, f, ensure_ascii=False, indent=2)

    model_metrics_rows: list[pd.DataFrame] = []
    family_tables: dict[str, list[pd.DataFrame]] = {family: [] for family in FAMILY_ORDER}
    family_model_frames: dict[str, dict[str, pd.DataFrame]] = {family: {} for family in FAMILY_ORDER}
    overall_model_frames: dict[str, pd.DataFrame] = {}

    for exp in EXPERIMENTS:
        exp_dir = EXP_DIR / exp["name"]
        pred_path = exp_dir / "predictions.csv"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing predictions file: {pred_path}")

        df = pd.read_csv(pred_path, encoding="utf-8-sig")
        metrics_df = summarize_predictions(
            df=df,
            fact_col=exp["fact_col"],
            pred_col=exp["pred_col"],
            model_name=exp["name"],
            family=exp["family"],
        )
        model_metrics_rows.append(metrics_df)
        family_tables[exp["family"]].append(metrics_df)
        family_model_frames[exp["family"]][exp["name"]] = metrics_df
        overall_model_frames[exp["name"]] = metrics_df

    long_all = pd.concat(model_metrics_rows, ignore_index=True)
    long_all.to_csv(OUT_DIR / "benchmark_all_models_long.csv", index=False, encoding="utf-8-sig")

    summary_all = build_model_summary_from_wide(best_all)
    summary_all.to_csv(OUT_DIR / "summary_by_model_all.csv", index=False, encoding="utf-8-sig")

    best_all = merge_model_frames(overall_model_frames)
    best_all.to_csv(OUT_DIR / "best_by_sku.csv", index=False, encoding="utf-8-sig")
    best_all.to_csv(OUT_DIR / "best_by_sku_all_models.csv", index=False, encoding="utf-8-sig")

    for family in FAMILY_ORDER:
        family_long = pd.concat(family_tables[family], ignore_index=True) if family_tables[family] else pd.DataFrame()
        if family_long.empty:
            continue
        family_long.to_csv(OUT_DIR / f"benchmark_{family}_models_long.csv", index=False, encoding="utf-8-sig")
        family_summary = build_model_summary_from_wide(family_best)
        family_summary.to_csv(OUT_DIR / f"summary_by_model_{family}.csv", index=False, encoding="utf-8-sig")

        family_best = merge_model_frames(family_model_frames[family])
        family_best.to_csv(OUT_DIR / f"best_by_sku_{family}.csv", index=False, encoding="utf-8-sig")

    overview = {
        "experiments": [exp["name"] for exp in EXPERIMENTS],
        "n_models": len(EXPERIMENTS),
        "n_pairs": int(len(best_all)),
        "elapsed_s": round(time.time() - t_start, 1),
    }
    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)

    elapsed_total = time.time() - t_start
    print(f"\n{'=' * 72}")
    print(f"Done in {elapsed_total:.1f}s")
    print(f"Outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display


PREDICTION_FILES = {
    "lag7": "predictions_seasonal_naive_lag7_non_recursive.csv",
    "baseline": "predictions_daily_baseline_non_recursive.csv",
    "blend": "predictions_heuristic_blend_non_recursive.csv",
}


def resolve_root(root: Path | None = None) -> Path:
    base = (root or Path.cwd()).resolve()
    if base.name == "notebooks":
        return base.parent
    return base


def resolve_exp74_dir(root: Path | None = None, run_name: str = "default") -> Path:
    base_root = resolve_root(root)
    exp_root = base_root / "src" / "experiments_v2" / "74_bakery_non_recursive"
    return exp_root if run_name == "default" else exp_root / run_name


def _metrics(actual: pd.Series, pred: pd.Series) -> dict[str, float]:
    actual_arr = pd.to_numeric(actual, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    pred_arr = pd.to_numeric(pred, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    err = actual_arr - pred_arr
    return {
        "mae": float(np.mean(np.abs(err))),
        "mse": float(np.mean(err**2)),
        "wmape": float(np.sum(np.abs(err)) / (np.sum(actual_arr) + 1e-8) * 100.0),
        "bias": float(np.mean(err)),
    }


def load_exp74_backtest(root: Path | None = None, run_name: str = "default") -> pd.DataFrame:
    exp_dir = resolve_exp74_dir(root, run_name=run_name)
    merged: pd.DataFrame | None = None

    for model_name, filename in PREDICTION_FILES.items():
        frame = pd.read_csv(exp_dir / filename, encoding="utf-8-sig")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["bakery_sales"] = pd.to_numeric(frame["bakery_sales"], errors="coerce").fillna(0.0)
        frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce").fillna(0.0)
        frame = frame.rename(columns={"prediction": f"pred_{model_name}"})
        keep_cols = ["date", "bakery_id", "bakery_name", "city", "bakery_sales", f"pred_{model_name}"]
        merged = frame[keep_cols] if merged is None else merged.merge(
            frame[["date", "bakery_id", f"pred_{model_name}"]],
            on=["date", "bakery_id"],
            how="inner",
        )

    if merged is None:
        raise ValueError("No prediction files loaded")

    for model_name in PREDICTION_FILES:
        merged[f"err_{model_name}"] = merged["bakery_sales"] - merged[f"pred_{model_name}"]
    return merged.sort_values(["bakery_id", "date"]).reset_index(drop=True)


def build_model_summary(backtest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in PREDICTION_FILES:
        rows.append({"model": model_name, **_metrics(backtest["bakery_sales"], backtest[f"pred_{model_name}"])})
    return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)


def load_exp74_run_summary(root: Path | None = None, run_name: str = "default") -> pd.DataFrame:
    exp_dir = resolve_exp74_dir(root, run_name=run_name)
    summary = pd.read_csv(exp_dir / "summary_by_model.csv", encoding="utf-8-sig")
    summary["run_name"] = run_name
    return summary


def compare_exp74_runs(root: Path | None = None, run_names: list[str] | None = None) -> pd.DataFrame:
    run_names = run_names or ["30d", "30d_targeted_holiday_overrides", "30d_cluster_features"]
    frames = [load_exp74_run_summary(root, run_name=run_name) for run_name in run_names]
    combined = pd.concat(frames, ignore_index=True)
    metric_cols = ["avg_mae", "median_mae", "avg_mse", "avg_wmape", "avg_bias", "median_abs_bias", "win_count"]
    pivot = combined.pivot(index="model", columns="run_name", values=metric_cols)
    pivot = pivot.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)
    return pivot.reset_index()


def build_bakery_model_summary(backtest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bakery_id, group in backtest.groupby("bakery_id", sort=False):
        row = {
            "bakery_id": bakery_id,
            "bakery_name": group["bakery_name"].iloc[0],
            "city": group["city"].iloc[0],
        }
        for model_name in PREDICTION_FILES:
            metrics = _metrics(group["bakery_sales"], group[f"pred_{model_name}"])
            row[f"{model_name}_mae"] = metrics["mae"]
            row[f"{model_name}_wmape"] = metrics["wmape"]
            row[f"{model_name}_bias"] = metrics["bias"]
        mae_cols = [f"{model_name}_mae" for model_name in PREDICTION_FILES]
        mae_series = pd.Series({col: row[col] for col in mae_cols})
        row["best_model"] = mae_series.idxmin().replace("_mae", "")
        row["best_mae"] = float(mae_series.min())
        row["blend_minus_baseline_mae"] = row["blend_mae"] - row["baseline_mae"]
        row["lag7_minus_blend_mae"] = row["lag7_mae"] - row["blend_mae"]
        rows.append(row)
    return pd.DataFrame(rows)


def compare_exp74_bakery_runs(
    root: Path | None = None,
    *,
    baseline_run: str = "30d",
    candidate_run: str = "30d_cluster_features",
    model_name: str = "baseline",
) -> pd.DataFrame:
    base = build_bakery_model_summary(load_exp74_backtest(root, run_name=baseline_run)).rename(
        columns={
            f"{model_name}_mae": f"{baseline_run}_{model_name}_mae",
            f"{model_name}_wmape": f"{baseline_run}_{model_name}_wmape",
            f"{model_name}_bias": f"{baseline_run}_{model_name}_bias",
        }
    )
    cand = build_bakery_model_summary(load_exp74_backtest(root, run_name=candidate_run)).rename(
        columns={
            f"{model_name}_mae": f"{candidate_run}_{model_name}_mae",
            f"{model_name}_wmape": f"{candidate_run}_{model_name}_wmape",
            f"{model_name}_bias": f"{candidate_run}_{model_name}_bias",
        }
    )
    merged = base[
        ["bakery_id", "bakery_name", "city", f"{baseline_run}_{model_name}_mae", f"{baseline_run}_{model_name}_wmape", f"{baseline_run}_{model_name}_bias"]
    ].merge(
        cand[
            ["bakery_id", f"{candidate_run}_{model_name}_mae", f"{candidate_run}_{model_name}_wmape", f"{candidate_run}_{model_name}_bias"]
        ],
        on="bakery_id",
        how="inner",
    )
    merged[f"{candidate_run}_minus_{baseline_run}_{model_name}_mae"] = (
        merged[f"{candidate_run}_{model_name}_mae"] - merged[f"{baseline_run}_{model_name}_mae"]
    )
    merged[f"{candidate_run}_minus_{baseline_run}_{model_name}_wmape"] = (
        merged[f"{candidate_run}_{model_name}_wmape"] - merged[f"{baseline_run}_{model_name}_wmape"]
    )
    merged[f"{candidate_run}_minus_{baseline_run}_{model_name}_bias"] = (
        merged[f"{candidate_run}_{model_name}_bias"] - merged[f"{baseline_run}_{model_name}_bias"]
    )
    return merged.sort_values(f"{candidate_run}_minus_{baseline_run}_{model_name}_mae").reset_index(drop=True)


def build_exp74_network_daily(backtest: pd.DataFrame) -> pd.DataFrame:
    daily = (
        backtest.groupby("date", as_index=False)
        .agg(
            actual=("bakery_sales", "sum"),
            pred_lag7=("pred_lag7", "sum"),
            pred_baseline=("pred_baseline", "sum"),
            pred_blend=("pred_blend", "sum"),
        )
        .sort_values("date")
    )
    daily["err_lag7"] = daily["actual"] - daily["pred_lag7"]
    daily["err_baseline"] = daily["actual"] - daily["pred_baseline"]
    daily["err_blend"] = daily["actual"] - daily["pred_blend"]
    return daily


def plot_exp74_run_comparison(
    root: Path | None = None,
    *,
    run_names: list[str] | None = None,
    model_name: str = "baseline",
) -> None:
    run_names = run_names or ["30d", "30d_targeted_holiday_overrides", "30d_cluster_features"]
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    for run_name in run_names:
        backtest = load_exp74_backtest(root, run_name=run_name)
        daily = build_exp74_network_daily(backtest)
        axes[0].plot(daily["date"], daily["actual"], color="black", linewidth=2, alpha=0.30)
        axes[0].plot(daily["date"], daily[f"pred_{model_name}"], marker="o", label=run_name)
        axes[1].plot(daily["date"], daily[f"err_{model_name}"], marker="o", label=run_name)

    axes[0].set_title(f"Exp74 {model_name} comparison across runs")
    axes[0].set_ylabel("Sales")
    axes[0].legend()

    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_ylabel("Error")
    axes[1].set_xlabel("Date")
    axes[1].legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_network_forecast(backtest: pd.DataFrame, *, title_suffix: str = "") -> None:
    daily = (
        backtest.groupby("date", as_index=False)
        .agg(
            actual=("bakery_sales", "sum"),
            pred_lag7=("pred_lag7", "sum"),
            pred_baseline=("pred_baseline", "sum"),
            pred_blend=("pred_blend", "sum"),
        )
        .sort_values("date")
    )
    daily["err_lag7"] = daily["actual"] - daily["pred_lag7"]
    daily["err_baseline"] = daily["actual"] - daily["pred_baseline"]
    daily["err_blend"] = daily["actual"] - daily["pred_blend"]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    axes[0].plot(daily["date"], daily["actual"], marker="o", linewidth=2, label="Actual")
    axes[0].plot(daily["date"], daily["pred_baseline"], marker="o", label="Baseline")
    axes[0].plot(daily["date"], daily["pred_blend"], marker="o", label="Heuristic blend")
    axes[0].plot(daily["date"], daily["pred_lag7"], marker="o", label="Lag7 naive")
    axes[0].set_title(f"Exp74 non-recursive bakery sales{title_suffix}")
    axes[0].set_ylabel("Sales")
    axes[0].legend()

    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].plot(daily["date"], daily["err_baseline"], marker="o", label="Baseline error")
    axes[1].plot(daily["date"], daily["err_blend"], marker="o", label="Blend error")
    axes[1].plot(daily["date"], daily["err_lag7"], marker="o", label="Lag7 error")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Error")
    axes[1].legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_bakery_forecast(
    backtest: pd.DataFrame,
    *,
    bakery_id: int | None = None,
    bakery_name_contains: str | None = None,
) -> pd.DataFrame:
    work = backtest.copy()
    if bakery_id is not None:
        work = work[work["bakery_id"] == bakery_id].copy()
    if bakery_name_contains is not None:
        work = work[work["bakery_name"].str.contains(bakery_name_contains, case=False, na=False)].copy()
    if work.empty:
        raise ValueError("No rows found for the requested bakery filter")

    bakery_name = work["bakery_name"].iloc[0]
    bakery_id_value = work["bakery_id"].iloc[0]
    city = work["city"].iloc[0]

    metrics_rows = []
    for model_name in PREDICTION_FILES:
        metrics_rows.append({"model": model_name, **_metrics(work["bakery_sales"], work[f"pred_{model_name}"])})
    metrics = pd.DataFrame(metrics_rows).sort_values("mae").reset_index(drop=True)
    display(metrics)

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    axes[0].plot(work["date"], work["bakery_sales"], marker="o", linewidth=2, label="Actual")
    axes[0].plot(work["date"], work["pred_baseline"], marker="o", label="Baseline")
    axes[0].plot(work["date"], work["pred_blend"], marker="o", label="Heuristic blend")
    axes[0].plot(work["date"], work["pred_lag7"], marker="o", label="Lag7 naive")
    axes[0].set_title(f"Bakery {bakery_id_value} | {bakery_name} | {city}")
    axes[0].set_ylabel("Sales")
    axes[0].legend()

    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].plot(work["date"], work["err_baseline"], marker="o", label="Baseline error")
    axes[1].plot(work["date"], work["err_blend"], marker="o", label="Blend error")
    axes[1].plot(work["date"], work["err_lag7"], marker="o", label="Lag7 error")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Error")
    axes[1].legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return metrics


def show_exp74_overview(root: Path | None = None, run_name: str = "default", *, title_suffix: str = "") -> pd.DataFrame:
    backtest = load_exp74_backtest(root, run_name=run_name)
    summary = build_model_summary(backtest)
    bakery_summary = build_bakery_model_summary(backtest)
    display(summary)
    display(bakery_summary.sort_values("blend_minus_baseline_mae").head(15))
    display(bakery_summary.sort_values("lag7_minus_blend_mae").head(15))
    plot_network_forecast(backtest, title_suffix=title_suffix)
    return backtest

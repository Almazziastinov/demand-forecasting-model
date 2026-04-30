from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from src.experiments_v2.benchmark_common import (  # noqa: E402
    BAKERY_COL as HIST_BAKERY_COL,
    DATE_COL as HIST_DATE_COL,
    DEMAND_8M_PATH,
    DEMAND_TARGET as HIST_DEMAND_TARGET,
    PRODUCT_COL as HIST_PRODUCT_COL,
)


ROOT = Path(__file__).resolve().parent.parent
MONTHLY_ROOT = ROOT / "src" / "experiments_v2" / "Full_benchmark_mounth_results" / "src" / "experiments_v2"
MERGED_BEST_BY_SKU_PATH = MONTHLY_ROOT / "merged_best_by_sku.csv"
FULL_BENCHMARK_DIR = MONTHLY_ROOT / "full_benchmark_monthly"
SKU_LOCAL_DIR = MONTHLY_ROOT / "sku_local_monthly"
MONTHLY_METRICS_PATH = FULL_BENCHMARK_DIR / "metrics.json"
AVG_LAG_METRICS_PATH = FULL_BENCHMARK_DIR / "artifacts" / "avg_lag_7_14_core" / "sku_metrics.csv"


MODEL_SPECS = [
    {
        "name": "60_baseline_v3",
        "path": FULL_BENCHMARK_DIR / "artifacts" / "60_baseline_v3" / "predictions.csv",
        "fact_col": "fact",
        "pred_col": "pred",
    },
    {
        "name": "61_censoring_behavioral",
        "path": FULL_BENCHMARK_DIR / "artifacts" / "61_censoring_behavioral" / "predictions.csv",
        "fact_col": "fact",
        "pred_col": "pred",
    },
    {
        "name": "62_assortment_availability",
        "path": FULL_BENCHMARK_DIR / "artifacts" / "62_assortment_availability" / "predictions.csv",
        "fact_col": "fact",
        "pred_col": "pred",
    },
    {
        "name": "66_cluster_features",
        "path": FULL_BENCHMARK_DIR / "artifacts" / "66_cluster_features" / "predictions.csv",
        "fact_col": "fact",
        "pred_col": "pred",
    },
    {
        "name": "lgbm_local",
        "path": SKU_LOCAL_DIR / "predictions_lgbm_local.csv",
        "fact_col": "Спрос",
        "pred_col": "prediction",
    },
    {
        "name": "prophet_local",
        "path": SKU_LOCAL_DIR / "predictions_prophet_local.csv",
        "fact_col": "Спрос",
        "pred_col": "prediction",
    },
    {
        "name": "two_week_avg",
        "path": SKU_LOCAL_DIR / "predictions_two_week_avg.csv",
        "fact_col": "Спрос",
        "pred_col": "prediction",
    },
    {
        "name": "avg_lag_7_14_core",
        "path": FULL_BENCHMARK_DIR / "artifacts" / "avg_lag_7_14_core" / "predictions.csv",
        "fact_col": "fact",
        "pred_col": "pred",
    },
]


def r2_bucket(value: float) -> str:
    if pd.isna(value):
        return "unknown"
    if value <= 0:
        return "R2 <= 0"
    if value <= 0.1:
        return "0 < R2 <= 0.1"
    if value <= 0.2:
        return "0.1 < R2 <= 0.2"
    if value <= 0.4:
        return "0.2 < R2 <= 0.4"
    return "R2 > 0.4"


def _normalize_best_by_sku(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing SKU summary: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    if len(df.columns) < 2:
        raise KeyError("best_by_sku file must have at least bakery and product columns")

    rename_map = {
        df.columns[0]: "bakery",
        df.columns[1]: "product",
    }
    df = df.rename(columns=rename_map).copy()
    if "best_r2" not in df.columns:
        raise KeyError("best_by_sku file must contain best_r2")

    df["best_r2"] = pd.to_numeric(df["best_r2"], errors="coerce")
    if "best_mae" in df.columns:
        df["best_mae"] = pd.to_numeric(df["best_mae"], errors="coerce")
    if "best_wmape" in df.columns:
        df["best_wmape"] = pd.to_numeric(df["best_wmape"], errors="coerce")
    df["r2_bucket"] = df["best_r2"].apply(r2_bucket)
    return df


def _normalize_model_frame(spec: dict) -> pd.DataFrame:
    if not spec["path"].exists():
        raise FileNotFoundError(f"Missing benchmark artifact: {spec['path']}")

    df = pd.read_csv(spec["path"], encoding="utf-8-sig")
    rename_map = {}
    if len(df.columns) >= 1:
        rename_map[df.columns[0]] = "date"
    if len(df.columns) >= 2:
        rename_map[df.columns[1]] = "bakery"
    if len(df.columns) >= 3:
        rename_map[df.columns[2]] = "product"
    if spec["fact_col"] in df.columns:
        rename_map[spec["fact_col"]] = "fact"
    if spec["pred_col"] in df.columns:
        rename_map[spec["pred_col"]] = f"pred_{spec['name']}"

    keep = [col for col in rename_map if col in df.columns]
    df = df[keep].rename(columns=rename_map).copy()
    if "date" not in df.columns or "bakery" not in df.columns or "product" not in df.columns:
        raise KeyError(f"Unexpected schema for {spec['path']}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["fact"] = pd.to_numeric(df["fact"], errors="coerce")
    df[f"pred_{spec['name']}"] = pd.to_numeric(df[f"pred_{spec['name']}"], errors="coerce")
    df = df.dropna(subset=["date", "bakery", "product"]).copy()
    return df


def _apply_demo_winner(best_by_sku: pd.DataFrame) -> pd.DataFrame:
    work = best_by_sku.copy()
    rename_map = {col: f"source_{col}" for col in ["best_model", "best_r2", "best_mae", "best_mse", "best_wmape", "winner_source"] if col in work.columns}
    if rename_map:
        work = work.rename(columns=rename_map)

    for target_col, source_col in [
        ("best_model", "source_best_model"),
        ("best_r2", "source_best_r2"),
        ("best_mae", "source_best_mae"),
        ("best_mse", "source_best_mse"),
        ("best_wmape", "source_best_wmape"),
    ]:
        if source_col in work.columns:
            work[target_col] = work[source_col]

    work["source_best_r2"] = pd.to_numeric(work.get("source_best_r2"), errors="coerce")
    if "avg_lag_7_14_core_r2" in work.columns:
        work["avg_lag_7_14_core_r2"] = pd.to_numeric(work["avg_lag_7_14_core_r2"], errors="coerce")
        demo_wins = work["avg_lag_7_14_core_r2"].notna() & (
            work["source_best_r2"].isna() | (work["avg_lag_7_14_core_r2"] > work["source_best_r2"])
        )
        work.loc[demo_wins, "best_model"] = "avg_lag_7_14_core"
        work.loc[demo_wins, "best_r2"] = work.loc[demo_wins, "avg_lag_7_14_core_r2"]
        if "avg_lag_7_14_core_mae" in work.columns:
            work.loc[demo_wins, "best_mae"] = work.loc[demo_wins, "avg_lag_7_14_core_mae"]
        if "avg_lag_7_14_core_mse" in work.columns:
            work.loc[demo_wins, "best_mse"] = work.loc[demo_wins, "avg_lag_7_14_core_mse"]
        if "avg_lag_7_14_core_wmape" in work.columns:
            work.loc[demo_wins, "best_wmape"] = work.loc[demo_wins, "avg_lag_7_14_core_wmape"]
        if "source_winner_source" in work.columns:
            work["winner_source"] = work["source_winner_source"]
            work.loc[demo_wins, "winner_source"] = "avg_lag_7_14_core"
    return work


def _coalesce_fact_columns(df: pd.DataFrame) -> pd.Series:
    fact_cols = [c for c in df.columns if c.startswith("fact_")]
    if not fact_cols:
        return pd.Series(np.nan, index=df.index)
    return df[fact_cols].bfill(axis=1).iloc[:, 0]


def _load_common_dates() -> list[str]:
    if MONTHLY_METRICS_PATH.exists():
        try:
            payload = json.loads(MONTHLY_METRICS_PATH.read_text(encoding="utf-8"))
            start = pd.to_datetime(payload.get("test_start"), errors="coerce")
            test_days = int(payload.get("test_days", 0))
            if not pd.isna(start) and test_days > 0:
                return [day.strftime("%Y-%m-%d") for day in pd.date_range(start, periods=test_days, freq="D")]
        except Exception:
            pass

    manifest_path = ROOT / "src" / "experiments_v2" / "full_benchmark" / "best_by_sku_manifest.json"
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            common_dates = payload.get("common_dates") or []
            if common_dates:
                return [str(day) for day in common_dates]
        except Exception:
            pass

    return []


def _load_history(date_filter: list[str] | None = None) -> pd.DataFrame:
    history = pd.read_csv(
        DEMAND_8M_PATH,
        encoding="utf-8-sig",
        usecols=[HIST_DATE_COL, HIST_BAKERY_COL, HIST_PRODUCT_COL, HIST_DEMAND_TARGET],
    )
    history = history.rename(
        columns={
            HIST_DATE_COL: "hist_date",
            HIST_BAKERY_COL: "bakery",
            HIST_PRODUCT_COL: "product",
            HIST_DEMAND_TARGET: "hist_fact",
        }
    )
    history["hist_date"] = pd.to_datetime(history["hist_date"], errors="coerce")
    history["hist_fact"] = pd.to_numeric(history["hist_fact"], errors="coerce")
    history = history.dropna(subset=["hist_date", "bakery", "product"]).copy()
    if date_filter:
        history = history[history["hist_date"].dt.strftime("%Y-%m-%d").isin(date_filter)].copy()
    history = history.sort_values(["bakery", "product", "hist_date"]).reset_index(drop=True)
    return history


def _add_avg_lag_core_model(
    wide: pd.DataFrame,
    history: pd.DataFrame,
    name: str = "avg_lag_7_14_core",
) -> pd.DataFrame:
    work = wide[["date", "bakery", "product"]].copy().reset_index(drop=True)
    work["row_id"] = np.arange(len(work), dtype=int)

    history_lookup = history.rename(columns={"hist_date": "lag_date", "hist_fact": "lag_fact"})[
        ["bakery", "product", "lag_date", "lag_fact"]
    ].drop_duplicates(subset=["bakery", "product", "lag_date"])

    def _merge_lag(offset_days: int) -> pd.Series:
        target = work[["row_id", "date", "bakery", "product"]].copy()
        target["lag_date"] = target["date"] - pd.Timedelta(days=offset_days)
        merged = target.merge(history_lookup, on=["bakery", "product", "lag_date"], how="left")
        merged = merged.sort_values("row_id")
        return merged["lag_fact"].reset_index(drop=True)

    work["lag_7"] = _merge_lag(7)
    work["lag_14"] = _merge_lag(14)

    lag_7 = pd.to_numeric(work["lag_7"], errors="coerce")
    lag_14 = pd.to_numeric(work["lag_14"], errors="coerce")
    pred = pd.concat([lag_7, lag_14], axis=1).mean(axis=1, skipna=True)
    wide[f"pred_{name}"] = pred.to_numpy()
    return wide


def _add_avg_lag_filled_model(wide: pd.DataFrame, name: str = "avg_lag_7_14_filled") -> pd.DataFrame:
    work = wide[["date", "bakery", "product", "fact", "pred_two_week_avg"]].copy()
    work = work.sort_values(["bakery", "product", "date"]).copy()
    work["date_lag_7"] = work["date"] - pd.Timedelta(days=7)
    work["date_lag_14"] = work["date"] - pd.Timedelta(days=14)

    lag_lookup = work[["bakery", "product", "date", "fact"]].rename(columns={"date": "lag_date", "fact": "lag_fact"})
    work = work.merge(
        lag_lookup,
        left_on=["bakery", "product", "date_lag_7"],
        right_on=["bakery", "product", "lag_date"],
        how="left",
    ).rename(columns={"lag_fact": "lag_7"})
    work = work.merge(
        lag_lookup,
        left_on=["bakery", "product", "date_lag_14"],
        right_on=["bakery", "product", "lag_date"],
        how="left",
        suffixes=("", "_14"),
    ).rename(columns={"lag_fact": "lag_14"})

    lag_7 = pd.to_numeric(work["lag_7"], errors="coerce")
    lag_14 = pd.to_numeric(work["lag_14"], errors="coerce")
    pred = pd.concat([lag_7, lag_14], axis=1).mean(axis=1, skipna=True)
    pred = pred.fillna(lag_7).fillna(lag_14)
    pred = pred.fillna(pd.to_numeric(work["pred_two_week_avg"], errors="coerce"))
    wide[f"pred_{name}"] = pred.to_numpy()
    return wide


@lru_cache(maxsize=1)
def load_store() -> dict:
    best_by_sku = _normalize_best_by_sku(MERGED_BEST_BY_SKU_PATH)
    if AVG_LAG_METRICS_PATH.exists():
        avg_lag_metrics = pd.read_csv(AVG_LAG_METRICS_PATH, encoding="utf-8-sig")
        avg_lag_metrics = avg_lag_metrics.rename(
            columns={
                "r2": "avg_lag_7_14_core_r2",
                "mse": "avg_lag_7_14_core_mse",
                "mae": "avg_lag_7_14_core_mae",
                "wmape": "avg_lag_7_14_core_wmape",
                "n_test_rows": "avg_lag_7_14_core_n_test_rows",
            }
        )
        best_by_sku = best_by_sku.merge(avg_lag_metrics, on=["bakery", "product"], how="left")
    best_by_sku = _apply_demo_winner(best_by_sku)
    common_dates = _load_common_dates()
    lag_history = _load_history(None)
    if common_dates:
        base_history = lag_history[lag_history["hist_date"].dt.strftime("%Y-%m-%d").isin(common_dates)].copy()
    else:
        base_history = lag_history.copy()

    frames = []
    available_models: list[str] = []
    model_status: dict[str, str] = {}
    for spec in MODEL_SPECS:
        if spec["path"].exists():
            frames.append(_normalize_model_frame(spec))
            available_models.append(spec["name"])
            model_status[spec["name"]] = "loaded"
        else:
            model_status[spec["name"]] = "missing"

    key_cols = ["date", "bakery", "product"]
    wide = base_history.rename(columns={"hist_date": "date", "hist_fact": "fact"}).copy()
    wide = wide[key_cols + ["fact"]].copy()

    if frames:
        for frame in frames:
            pred_cols = [c for c in frame.columns if c.startswith("pred_")]
            wide = wide.merge(frame[key_cols + pred_cols], on=key_cols, how="left")

        pred_cols = [c for c in wide.columns if c.startswith("pred_")]
        for col in pred_cols:
            wide[col] = pd.to_numeric(wide[col], errors="coerce")
        wide["fact"] = pd.to_numeric(wide["fact"], errors="coerce")
        wide["date"] = pd.to_datetime(wide["date"], errors="coerce")
        if "pred_avg_lag_7_14_core" not in wide.columns:
            wide = _add_avg_lag_core_model(wide, lag_history, "avg_lag_7_14_core")
        if "pred_avg_lag_7_14_core" in wide.columns:
            wide["pred_avg_lag_7_14_core"] = pd.to_numeric(wide["pred_avg_lag_7_14_core"], errors="coerce")
    else:
        wide["fact"] = pd.to_numeric(wide["fact"], errors="coerce")
        wide["date"] = pd.to_datetime(wide["date"], errors="coerce")

    wide = wide.merge(best_by_sku, on=["bakery", "product"], how="left", suffixes=("", "_sku"))
    wide["best_r2"] = pd.to_numeric(wide["best_r2"], errors="coerce")
    wide["r2_bucket"] = wide["best_r2"].apply(r2_bucket)

    min_date = pd.to_datetime(wide["date"]).min()
    max_date = pd.to_datetime(wide["date"]).max()

    return {
        "best_by_sku": best_by_sku,
        "wide": wide,
        "history": lag_history,
        "models": available_models,
        "available_models": available_models,
        "model_status": model_status,
        "bakeries": sorted(best_by_sku["bakery"].dropna().unique().tolist()),
        "buckets": ["all", "R2 <= 0", "0 < R2 <= 0.1", "0.1 < R2 <= 0.2", "0.2 < R2 <= 0.4", "R2 > 0.4"],
        "date_min": None if pd.isna(min_date) else str(min_date.date()),
        "date_max": None if pd.isna(max_date) else str(max_date.date()),
        "date_range_label": None
        if pd.isna(min_date) or pd.isna(max_date)
        else f"{min_date.strftime('%Y-%m')} to {max_date.strftime('%Y-%m')}",
    }


def filter_products(bakery: str, bucket: str = "all") -> pd.DataFrame:
    store = load_store()
    df = store["best_by_sku"]
    sub = df[df["bakery"] == bakery].copy()
    if bucket != "all":
        sub = sub[sub["r2_bucket"] == bucket].copy()
    sort_cols = [col for col in ["best_r2", "product"] if col in sub.columns]
    return sub.sort_values(sort_cols, ascending=[False, True] if len(sort_cols) == 2 else [True]).reset_index(drop=True)


def resolve_compare_model(main_model: str, compare_model: str) -> str:
    store = load_store()
    models = store["models"]
    if compare_model != main_model:
        return compare_model
    for model in models:
        if model != main_model:
            return model
    return compare_model


def _metric_frame(group: pd.DataFrame, pred_col: str) -> dict:
    total_rows = int(len(group))
    work = group[["fact", pred_col]].dropna()
    if work.empty:
        return {"n_rows": 0, "coverage": 0.0, "mae": np.nan, "wmape": np.nan, "r2": np.nan, "bias": np.nan}

    y_true = work["fact"].to_numpy(dtype=float)
    y_pred = work[pred_col].to_numpy(dtype=float)
    r2 = np.nan
    if len(y_true) >= 2 and not np.allclose(y_true, y_true[0]):
        r2 = float(r2_score(y_true, y_pred))

    return {
        "n_rows": int(len(work)),
        "coverage": float(len(work) / total_rows) if total_rows else 0.0,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "wmape": float(np.sum(np.abs(y_true - y_pred)) / (np.sum(y_true) + 1e-8) * 100),
        "r2": r2,
        "bias": float(np.mean(y_true - y_pred)),
    }


def _resolve_prediction_column(subset: pd.DataFrame, model_name: str) -> str | None:
    model_col = f"pred_{model_name}"
    if model_col in subset.columns:
        return model_col

    fallback_col = "pred_avg_lag_7_14_core"
    if fallback_col in subset.columns:
        return fallback_col

    pred_cols = [c for c in subset.columns if c.startswith("pred_")]
    return pred_cols[0] if pred_cols else None


def _build_avg_lag_series(history: pd.DataFrame, bakery: str, product: str, dates: pd.Series) -> pd.DataFrame:
    work = pd.DataFrame(
        {
            "date": pd.to_datetime(dates, errors="coerce"),
            "bakery": bakery,
            "product": product,
        }
    ).dropna(subset=["date"]).copy()
    if work.empty:
        return pd.DataFrame(columns=["date", "bakery", "product", "pred_avg_lag_7_14_core"])

    history_group = history[(history["bakery"] == bakery) & (history["product"] == product)].copy()
    if history_group.empty:
        work["pred_avg_lag_7_14_core"] = np.nan
        return work

    history_group = history_group.rename(columns={"hist_date": "lag_date", "hist_fact": "lag_fact"})[
        ["lag_date", "lag_fact"]
    ].drop_duplicates(subset=["lag_date"]).sort_values("lag_date")

    def _lag_for_offset(offset_days: int) -> pd.Series:
        target = work[["date"]].copy()
        target["lag_date"] = target["date"] - pd.Timedelta(days=offset_days)
        target = target.sort_values("lag_date")
        merged = pd.merge_asof(target, history_group, on="lag_date", direction="backward", allow_exact_matches=True)
        return merged.sort_index()["lag_fact"].reset_index(drop=True)

    work["lag_7"] = _lag_for_offset(7)
    work["lag_14"] = _lag_for_offset(14)
    work["pred_avg_lag_7_14_core"] = pd.concat([work["lag_7"], work["lag_14"]], axis=1).mean(axis=1, skipna=True)
    return work[["date", "bakery", "product", "pred_avg_lag_7_14_core"]]


def build_series(bakery: str, product: str, main_model: str, compare_model: str) -> dict:
    store = load_store()
    wide = store["wide"]
    history = store["history"]
    compare_model = resolve_compare_model(main_model, compare_model)

    subset = wide[(wide["bakery"] == bakery) & (wide["product"] == product)].copy()
    if subset.empty:
        return {"series": [], "metrics": {}, "sku_info": None}

    if "pred_avg_lag_7_14_core" not in subset.columns:
        avg_lag = _build_avg_lag_series(history, bakery, product, subset["date"])
        subset = subset.merge(avg_lag, on=["date", "bakery", "product"], how="left")

    main_col = _resolve_prediction_column(subset, main_model)
    compare_col = _resolve_prediction_column(subset, compare_model)
    required = ["date", "fact"]
    for col in [main_col, compare_col]:
        if col and col not in required:
            required.append(col)
    subset = subset[required].dropna(subset=["fact"]).copy()
    if main_col is None or compare_col is None:
        return {"series": [], "metrics": {}, "sku_info": None}
    subset = subset.dropna(subset=[main_col, compare_col], how="all")
    if subset.empty:
        return {"series": [], "metrics": {}, "sku_info": None}

    subset = subset.sort_values("date").copy()

    metrics = {
        "main": _metric_frame(subset, main_col),
        "compare": _metric_frame(subset, compare_col),
    }
    sku_row = store["best_by_sku"][(store["best_by_sku"]["bakery"] == bakery) & (store["best_by_sku"]["product"] == product)].head(1)
    sku_info = sku_row.iloc[0].to_dict() if not sku_row.empty else None

    return {
        "series": [
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "fact": round(float(row["fact"]), 2),
                "main_pred": round(float(row[main_col]), 2),
                "compare_pred": round(float(row[compare_col]), 2),
            }
            for _, row in subset.iterrows()
        ],
        "metrics": metrics,
        "sku_info": sku_info,
    }


def bakery_comparison_table(bakery: str, main_model: str, compare_model: str, bucket: str = "all") -> list[dict]:
    store = load_store()
    wide = store["wide"]
    history = store["history"]
    compare_model = resolve_compare_model(main_model, compare_model)
    sku_meta = filter_products(bakery, bucket)
    if sku_meta.empty:
        return []

    sub = wide[wide["bakery"] == bakery].copy()
    rows = []
    for _, sku_row in sku_meta.iterrows():
        product = sku_row["product"]
        grp = sub[sub["product"] == product].copy()
        if grp.empty:
            continue

        if "pred_avg_lag_7_14_core" not in grp.columns:
            avg_lag = _build_avg_lag_series(history, bakery, product, grp["date"])
            grp = grp.merge(avg_lag, on=["date", "bakery", "product"], how="left")

        main_col = _resolve_prediction_column(grp, main_model)
        compare_col = _resolve_prediction_column(grp, compare_model)
        main_metrics = _metric_frame(grp, main_col) if main_col in grp.columns else {"mae": np.nan, "wmape": np.nan, "r2": np.nan}
        compare_metrics = _metric_frame(grp, compare_col) if compare_col in grp.columns else {"mae": np.nan, "wmape": np.nan, "r2": np.nan}

        rows.append(
            {
                "bakery": bakery,
                "product": product,
                "best_model": sku_row.get("best_model"),
                "best_r2": None if pd.isna(sku_row.get("best_r2")) else round(float(sku_row.get("best_r2")), 4),
                "bucket": sku_row.get("r2_bucket"),
                "main_mae": None if pd.isna(main_metrics["mae"]) else round(float(main_metrics["mae"]), 4),
                "main_wmape": None if pd.isna(main_metrics["wmape"]) else round(float(main_metrics["wmape"]), 2),
                "compare_mae": None if pd.isna(compare_metrics["mae"]) else round(float(compare_metrics["mae"]), 4),
                "compare_wmape": None if pd.isna(compare_metrics["wmape"]) else round(float(compare_metrics["wmape"]), 2),
                "delta_mae": None
                if pd.isna(main_metrics["mae"]) or pd.isna(compare_metrics["mae"])
                else round(float(main_metrics["mae"] - compare_metrics["mae"]), 4),
                "n_rows": int(main_metrics.get("n_rows", 0)),
            }
        )
    return rows

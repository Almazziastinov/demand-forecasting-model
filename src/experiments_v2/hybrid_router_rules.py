"""
Utilities for the hybrid SKU router research.

This module keeps the routing rules and comparison logic outside the notebook so
the experiment can be rerun from a normal Python entry point.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

BAKERY_COL = "Пекарня"
PRODUCT_COL = "Номенклатура"

CANDIDATE_MODELS = [
    "60_baseline_v3",
    "61_censoring_behavioral",
    "62_assortment_availability",
    "66_cluster_features",
    "prophet_local",
    "lgbm_local",
    "two_week_avg",
]

ROUTER_META_FEATURES = [
    "n_days",
    "mean_demand",
    "median_demand",
    "std_demand",
    "share_zero",
    "share_censored",
    "lost_mean",
    "demand_trend_mean",
    "cv_7d_mean",
    "cv_demand",
]


def load_router_candidates(path: str | Path) -> pd.DataFrame:
    """Load the benchmark candidate table and keep only routing-eligible rows."""
    df = pd.read_csv(Path(path), encoding="utf-8-sig")

    work = df.copy()

    def _coerce_bool(value: object) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {"true", "1", "yes", "y"}

    for col in ["has_model_metrics", "all_model_metrics_missing", "is_short_history"]:
        if col in work.columns:
            work[col] = work[col].map(_coerce_bool)

    if "has_model_metrics" in work.columns:
        work = work[work["has_model_metrics"]].copy()
    if "all_model_metrics_missing" in work.columns:
        work = work[~work["all_model_metrics_missing"]].copy()
    if "is_short_history" in work.columns:
        work = work[~work["is_short_history"]].copy()

    return work.reset_index(drop=True)


def metric_col(model: str, metric: str) -> str:
    return f"{model}_{metric}"


def _build_router_estimator():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
    )


def build_oracle_best(
    df: pd.DataFrame,
    candidate_models: list[str] | None = None,
) -> pd.DataFrame:
    """Return the per-SKU oracle winner among the available benchmark models."""
    models = candidate_models or CANDIDATE_MODELS
    available_models = [m for m in models if metric_col(m, "r2") in df.columns]

    if not available_models:
        columns = [
            BAKERY_COL,
            PRODUCT_COL,
            "oracle_model",
            "oracle_r2",
            "oracle_mae",
            "oracle_mse",
            "oracle_wmape",
        ]
        return pd.DataFrame(columns=columns)

    r2_cols = [metric_col(m, "r2") for m in available_models]
    r2_values = df[r2_cols].apply(pd.to_numeric, errors="coerce")
    best_idx = r2_values.idxmax(axis=1)

    oracle = df[[BAKERY_COL, PRODUCT_COL]].copy()
    oracle["oracle_model"] = best_idx.str.replace("_r2", "", regex=False)
    oracle["oracle_r2"] = r2_values.max(axis=1).to_numpy()
    oracle["oracle_mae"] = [
        df.iloc[i][metric_col(oracle.iloc[i]["oracle_model"], "mae")]
        for i in range(len(df))
    ]
    oracle["oracle_mse"] = [
        df.iloc[i][metric_col(oracle.iloc[i]["oracle_model"], "mse")]
        for i in range(len(df))
    ]
    oracle["oracle_wmape"] = [
        df.iloc[i][metric_col(oracle.iloc[i]["oracle_model"], "wmape")]
        for i in range(len(df))
    ]
    return oracle


def pick_rule_based_v0(row: pd.Series) -> str:
    """Initial hand-written router from the first research pass."""
    if row["cv_demand"] > 1.0:
        return "two_week_avg"
    if row["mean_demand"] < 5:
        return "two_week_avg"
    if row["share_censored"] > 0.5:
        return "61_censoring_behavioral"
    if row["cv_demand"] < 0.6 and row["mean_demand"] > 10:
        return "66_cluster_features"
    if row["share_censored"] > 0.35:
        return "62_assortment_availability"
    if row["mean_demand"] > 8 and row["cv_demand"] < 0.8:
        return "lgbm_local"
    return "66_cluster_features"


def pick_rule_based_v1(row: pd.Series) -> str:
    """Relaxed router that reduces overuse of the simple fallback."""
    if row["cv_demand"] > 1.25 and row["mean_demand"] < 8:
        return "two_week_avg"
    if row["mean_demand"] < 4 and row["cv_demand"] > 0.8:
        return "two_week_avg"
    if row["share_censored"] > 0.65 and row["mean_demand"] < 15:
        return "61_censoring_behavioral"
    if row["share_censored"] > 0.45 and row["mean_demand"] < 12:
        return "62_assortment_availability"
    if row["cv_demand"] < 0.7 and row["mean_demand"] > 8:
        return "66_cluster_features"
    if row["mean_demand"] > 12 and row["cv_demand"] < 0.85:
        return "66_cluster_features"
    if row["mean_demand"] > 6 and row["cv_demand"] < 1.0:
        return "lgbm_local"
    return "66_cluster_features"


def _difficulty_score(row: pd.Series) -> float:
    score = 0.0

    mean_demand = float(row.get("mean_demand", 0.0) or 0.0)
    cv_demand = float(row.get("cv_demand", 0.0) or 0.0)
    share_censored = float(row.get("share_censored", 0.0) or 0.0)
    n_days = float(row.get("n_days", 0.0) or 0.0)
    demand_trend = float(row.get("demand_trend_mean", 1.0) or 1.0)
    cv_7d_mean = float(row.get("cv_7d_mean", 0.0) or 0.0)

    if mean_demand < 4:
        score += 1.2
    elif mean_demand < 8:
        score += 0.6

    if cv_demand > 1.2:
        score += 1.2
    elif cv_demand > 0.9:
        score += 0.7

    if share_censored > 0.7:
        score += 1.1
    elif share_censored > 0.5:
        score += 0.6

    if n_days < 30:
        score += 1.0
    elif n_days < 60:
        score += 0.4

    if demand_trend < 0.85 or demand_trend > 1.15:
        score += 0.25

    if cv_7d_mean > 0.45:
        score += 0.2

    return score


def pick_rule_based_v2(row: pd.Series) -> str:
    """More conservative routing policy for the next production-safe pass."""
    mean_demand = float(row.get("mean_demand", 0.0) or 0.0)
    cv_demand = float(row.get("cv_demand", 0.0) or 0.0)
    share_censored = float(row.get("share_censored", 0.0) or 0.0)
    n_days = float(row.get("n_days", 0.0) or 0.0)
    difficulty = _difficulty_score(row)

    if n_days < 14:
        return "two_week_avg"
    if cv_demand > 1.35 and mean_demand < 6:
        return "two_week_avg"
    if difficulty >= 3.0 and mean_demand < 8:
        return "two_week_avg"

    if share_censored > 0.7 and mean_demand < 18:
        return "61_censoring_behavioral"
    if share_censored > 0.5 and mean_demand < 12:
        return "62_assortment_availability"

    if mean_demand > 14 and cv_demand < 0.75 and share_censored < 0.6:
        return "66_cluster_features"
    if mean_demand > 8 and cv_demand < 0.9 and share_censored < 0.55:
        return "66_cluster_features"
    if mean_demand > 6 and cv_demand < 0.8 and n_days > 90:
        return "lgbm_local"

    return "66_cluster_features"


def attach_choice_metrics(
    df: pd.DataFrame,
    choice_col: str,
    prefix: str,
) -> pd.DataFrame:
    """Attach per-row metrics for the model chosen by a routing column."""
    result = df[[BAKERY_COL, PRODUCT_COL, choice_col]].copy()
    chosen = result[choice_col].astype(str)
    result[f"{prefix}_model"] = chosen
    for metric in ["r2", "mae", "mse", "wmape"]:
        result[f"{prefix}_{metric}"] = [
            df.iloc[i][metric_col(chosen.iloc[i], metric)]
            for i in range(len(df))
        ]
    return result


def build_rule_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the rule-augmented table and the comparison table."""
    work = df.copy()
    work["rule_model_v0"] = work.apply(pick_rule_based_v0, axis=1)
    work["rule_model_v1"] = work.apply(pick_rule_based_v1, axis=1)
    work["rule_model_v2"] = work.apply(pick_rule_based_v2, axis=1)
    work["single_model_66"] = "66_cluster_features"

    oracle = build_oracle_best(work)
    rule_v0_metrics = attach_choice_metrics(work, "rule_model_v0", "rule_v0")
    rule_v1_metrics = attach_choice_metrics(work, "rule_model_v1", "rule_v1")
    rule_v2_metrics = attach_choice_metrics(work, "rule_model_v2", "rule_v2")
    single_metrics = attach_choice_metrics(work, "single_model_66", "single")

    compare_df = (
        oracle.merge(rule_v0_metrics, on=[BAKERY_COL, PRODUCT_COL], how="left")
        .merge(rule_v1_metrics, on=[BAKERY_COL, PRODUCT_COL], how="left")
        .merge(rule_v2_metrics, on=[BAKERY_COL, PRODUCT_COL], how="left")
        .merge(single_metrics, on=[BAKERY_COL, PRODUCT_COL], how="left")
    )

    return work, compare_df


def build_router_training_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build the feature table used to train a learned router."""
    oracle = build_oracle_best(df)
    table = df[[BAKERY_COL, PRODUCT_COL, *ROUTER_META_FEATURES]].copy()
    table = table.merge(oracle, on=[BAKERY_COL, PRODUCT_COL], how="left")
    for col in ROUTER_META_FEATURES:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    return table


def build_gated_router_predictions(
    work_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    fallback_model: str,
    confidence_threshold: float,
) -> pd.DataFrame:
    """Apply a confidence gate to the learned router predictions."""
    scored = work_df.copy()
    if "learned_router_confidence" not in scored.columns:
        scored = scored.merge(scores_df, on=[BAKERY_COL, PRODUCT_COL], how="left")
    gated_col = "learned_router_gated_model"
    scored[gated_col] = np.where(
        scored["learned_router_confidence"] >= confidence_threshold,
        scored["learned_router_model"],
        fallback_model,
    )
    return scored


def evaluate_gated_router_grid(
    work_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    thresholds: list[float] | None = None,
    fallback_models: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Search confidence thresholds and fallback models for the best gated router."""
    thresholds = thresholds or [round(x, 2) for x in np.arange(0.3, 0.81, 0.05)]
    fallback_models = fallback_models or ["66_cluster_features", "two_week_avg"]

    rows: list[dict] = []
    best_payload: dict | None = None

    for fallback_model in fallback_models:
        for threshold in thresholds:
            gated_df = build_gated_router_predictions(
                work_df,
                scores_df,
                fallback_model=fallback_model,
                confidence_threshold=threshold,
            )
            gated_metrics = attach_choice_metrics(
                gated_df,
                "learned_router_gated_model",
                "learned_gate",
            )
            summary = summarize_scheme(
                gated_metrics,
                "learned_router_gated",
                "learned_gate_r2",
                "learned_gate_mae",
                "learned_gate_mse",
                "learned_gate_wmape",
            )
            summary["threshold"] = threshold
            summary["fallback_model"] = fallback_model
            summary["gate_coverage"] = round(
                float(
                    (
                        gated_df["learned_router_confidence"] >= threshold
                    ).mean()
                ),
                4,
            )
            rows.append(summary)

            candidate_key = (
                summary["avg_wmape"],
                -summary["avg_r2"],
                summary["avg_mae"],
            )
            if best_payload is None or candidate_key < best_payload["key"]:
                best_payload = {
                    "key": candidate_key,
                    "summary": summary,
                    "gated_df": gated_df,
                    "gated_metrics": gated_metrics,
                }

    sweep_df = pd.DataFrame(rows).sort_values(
        ["avg_wmape", "avg_r2", "avg_mae"],
        ascending=[True, False, True],
    )
    if best_payload is None:
        best_payload = {
            "summary": {},
            "gated_df": pd.DataFrame(),
            "gated_metrics": pd.DataFrame(),
        }
    return sweep_df, best_payload


def train_learned_router_scores(
    training_table: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "oracle_model",
) -> tuple[pd.DataFrame, dict]:
    """Train a lightweight learned router with out-of-fold predictions."""
    features = feature_cols or ROUTER_META_FEATURES
    work = training_table.dropna(subset=[target_col]).copy()
    valid_feature_cols = [col for col in features if col in work.columns]
    work = work.dropna(subset=valid_feature_cols, how="all").copy()
    if work.empty:
        empty = pd.DataFrame(
            columns=[
                BAKERY_COL,
                PRODUCT_COL,
                "learned_router_model",
                "learned_router_confidence",
                "learned_router_margin",
            ]
        )
        return empty, {"status": "empty"}

    feature_cols = [col for col in features if col in work.columns]
    X = work[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = work[target_col].astype(str)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_counts = pd.Series(y_encoded).value_counts()
    n_splits = int(min(5, class_counts.min()))

    oof_model = pd.Series(index=work.index, dtype=object)
    oof_confidence = pd.Series(index=work.index, dtype=float)
    oof_margin = pd.Series(index=work.index, dtype=float)
    info = {
        "status": "trained",
        "n_rows": int(len(work)),
        "n_features": int(len(feature_cols)),
        "classes": label_encoder.classes_.tolist(),
        "n_splits": n_splits,
    }

    if n_splits < 2:
        model = _build_router_estimator()
        model.fit(X, y_encoded)
        proba = model.predict_proba(X)
        sorted_proba = np.sort(proba, axis=1)
        oof_confidence.loc[work.index] = sorted_proba[:, -1]
        oof_margin.loc[work.index] = sorted_proba[:, -1] - sorted_proba[:, -2]
        oof_model.loc[work.index] = label_encoder.inverse_transform(model.predict(X))
        info["status"] = "trained_in_sample"
        scores = work[[BAKERY_COL, PRODUCT_COL]].copy()
        scores["learned_router_model"] = oof_model
        scores["learned_router_confidence"] = oof_confidence
        scores["learned_router_margin"] = oof_margin
        return scores.reset_index(drop=True), info

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, valid_idx in skf.split(X, y_encoded):
        model = _build_router_estimator()
        model.fit(X.iloc[train_idx], y_encoded[train_idx])
        proba = model.predict_proba(X.iloc[valid_idx])
        pred_idx = np.argmax(proba, axis=1)
        sorted_proba = np.sort(proba, axis=1)
        oof_model.iloc[valid_idx] = label_encoder.inverse_transform(pred_idx)
        oof_confidence.iloc[valid_idx] = sorted_proba[:, -1]
        oof_margin.iloc[valid_idx] = sorted_proba[:, -1] - sorted_proba[:, -2]

    scores = work[[BAKERY_COL, PRODUCT_COL]].copy()
    scores["learned_router_model"] = oof_model
    scores["learned_router_confidence"] = oof_confidence
    scores["learned_router_margin"] = oof_margin
    return scores.reset_index(drop=True), info


def train_learned_router_oof(
    training_table: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "oracle_model",
) -> tuple[pd.Series, dict]:
    """Backward-compatible wrapper that returns only the OOF model labels."""
    scores, info = train_learned_router_scores(
        training_table,
        feature_cols=feature_cols,
        target_col=target_col,
    )
    if scores.empty:
        return pd.Series(dtype=object), info
    return scores["learned_router_model"].reset_index(drop=True), info


def summarize_scheme(
    df: pd.DataFrame,
    name: str,
    r2_col: str,
    mae_col: str,
    mse_col: str,
    wmape_col: str,
) -> dict:
    return {
        "scheme": name,
        "avg_r2": round(float(df[r2_col].mean()), 4),
        "median_r2": round(float(df[r2_col].median()), 4),
        "avg_mae": round(float(df[mae_col].mean()), 4),
        "avg_mse": round(float(df[mse_col].mean()), 4),
        "avg_wmape": round(float(df[wmape_col].mean()), 2),
        "positive_r2_share": round(float((df[r2_col] > 0).mean()), 4),
        "n_rows": int(len(df)),
    }


def build_summary(compare_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize oracle, both routing rules, and the single-model baseline."""
    oracle_summary = summarize_scheme(
        compare_df,
        "oracle_best",
        "oracle_r2",
        "oracle_mae",
        "oracle_mse",
        "oracle_wmape",
    )
    rule_v0_summary = summarize_scheme(
        compare_df,
        "rule_based_v0",
        "rule_v0_r2",
        "rule_v0_mae",
        "rule_v0_mse",
        "rule_v0_wmape",
    )
    rule_v1_summary = summarize_scheme(
        compare_df,
        "rule_based_v1",
        "rule_v1_r2",
        "rule_v1_mae",
        "rule_v1_mse",
        "rule_v1_wmape",
    )
    rule_v2_summary = summarize_scheme(
        compare_df,
        "rule_based_v2",
        "rule_v2_r2",
        "rule_v2_mae",
        "rule_v2_mse",
        "rule_v2_wmape",
    )
    learned_summary = summarize_scheme(
        compare_df,
        "learned_router",
        "learned_r2",
        "learned_mae",
        "learned_mse",
        "learned_wmape",
    )
    learned_gated_summary = summarize_scheme(
        compare_df,
        "learned_router_gated",
        "learned_gate_r2",
        "learned_gate_mae",
        "learned_gate_mse",
        "learned_gate_wmape",
    )
    single_summary = summarize_scheme(
        compare_df,
        "single_model_66",
        "single_r2",
        "single_mae",
        "single_mse",
        "single_wmape",
    )
    return pd.DataFrame(
        [
            oracle_summary,
            rule_v0_summary,
            rule_v1_summary,
            rule_v2_summary,
            learned_summary,
            learned_gated_summary,
            single_summary,
        ]
    )


def build_win_counts(df: pd.DataFrame, choice_col: str) -> pd.DataFrame:
    """Count how often each model is chosen by a routing rule."""
    return (
        df[choice_col]
        .value_counts()
        .rename_axis("model")
        .reset_index(name="win_count")
    )


def write_artifacts(
    router_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    prefix: str = "",
) -> dict[str, Path]:
    """Write the routing outputs to CSV files.

    Returns a mapping of artifact name to path.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    work_df, compare_df = build_rule_tables(router_df)
    v0_counts = build_win_counts(work_df, "rule_model_v0")
    v1_counts = build_win_counts(work_df, "rule_model_v1")
    v2_counts = build_win_counts(work_df, "rule_model_v2")

    router_training_table = build_router_training_table(work_df)
    learned_scores, learned_info = train_learned_router_scores(router_training_table)
    learned_scores_name = f"{prefix}learned_router_scores.csv"
    learned_oof_name = f"{prefix}learned_router_oof.csv"
    gate_sweep_name = f"{prefix}learned_router_gate_sweep.csv"
    gated_best_name = f"{prefix}learned_router_gated_best.csv"
    work_df = work_df.merge(learned_scores, on=[BAKERY_COL, PRODUCT_COL], how="left")
    learned_metrics = attach_choice_metrics(work_df, "learned_router_model", "learned")
    compare_df = compare_df.merge(
        learned_metrics,
        on=[BAKERY_COL, PRODUCT_COL],
        how="left",
    )
    gate_sweep_df, best_gate = evaluate_gated_router_grid(work_df, learned_scores)
    gated_df = best_gate["gated_df"]
    gated_metrics = best_gate["gated_metrics"]
    if not gated_metrics.empty:
        compare_df = compare_df.merge(
            gated_metrics,
            on=[BAKERY_COL, PRODUCT_COL],
            how="left",
        )
    summary_df = build_summary(compare_df)

    router_candidates_name = f"{prefix}router_candidates_with_rules_v3.csv"
    oracle_name = f"{prefix}oracle_best_by_sku_v3.csv"
    comparison_name = f"{prefix}routing_comparison_rows_v3.csv"
    summary_name = f"{prefix}routing_summary_v3.csv"
    v0_counts_name = f"{prefix}routing_rule_v0_win_counts_v2.csv"
    v1_counts_name = f"{prefix}routing_rule_v1_win_counts_v2.csv"
    v2_counts_name = f"{prefix}routing_rule_v2_win_counts_v2.csv"
    training_table_name = f"{prefix}router_training_table.csv"
    learned_name = learned_oof_name

    artifacts = {
        "router_candidates_with_rules": out_dir / router_candidates_name,
        "oracle_best_by_sku": out_dir / oracle_name,
        "routing_comparison_rows": out_dir / comparison_name,
        "routing_summary": out_dir / summary_name,
        "routing_rule_v0_win_counts": out_dir / v0_counts_name,
        "routing_rule_v1_win_counts": out_dir / v1_counts_name,
        "routing_rule_v2_win_counts": out_dir / v2_counts_name,
        "router_training_table": out_dir / training_table_name,
        "learned_router_oof": out_dir / learned_name,
        "learned_router_scores": out_dir / learned_scores_name,
        "learned_router_gate_sweep": out_dir / gate_sweep_name,
        "learned_router_gated_best": out_dir / gated_best_name,
    }

    work_df.to_csv(
        artifacts["router_candidates_with_rules"],
        index=False,
        encoding="utf-8-sig",
    )
    build_oracle_best(work_df).to_csv(
        artifacts["oracle_best_by_sku"],
        index=False,
        encoding="utf-8-sig",
    )
    compare_df.to_csv(
        artifacts["routing_comparison_rows"],
        index=False,
        encoding="utf-8-sig",
    )
    summary_df.to_csv(
        artifacts["routing_summary"],
        index=False,
        encoding="utf-8-sig",
    )
    v0_counts.to_csv(
        artifacts["routing_rule_v0_win_counts"],
        index=False,
        encoding="utf-8-sig",
    )
    v1_counts.to_csv(
        artifacts["routing_rule_v1_win_counts"],
        index=False,
        encoding="utf-8-sig",
    )
    v2_counts.to_csv(
        artifacts["routing_rule_v2_win_counts"],
        index=False,
        encoding="utf-8-sig",
    )
    router_training_table.to_csv(
        artifacts["router_training_table"],
        index=False,
        encoding="utf-8-sig",
    )
    learned_scores.to_csv(
        artifacts["learned_router_scores"],
        index=False,
        encoding="utf-8-sig",
    )
    learned_metrics.to_csv(
        artifacts["learned_router_oof"],
        index=False,
        encoding="utf-8-sig",
    )
    gate_sweep_df.to_csv(
        artifacts["learned_router_gate_sweep"],
        index=False,
        encoding="utf-8-sig",
    )
    if not gated_df.empty:
        gated_df.to_csv(
            artifacts["learned_router_gated_best"],
            index=False,
            encoding="utf-8-sig",
        )

    return artifacts

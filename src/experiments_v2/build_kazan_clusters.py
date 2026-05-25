from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
RANDOM_STATE = 42

BAKERY_ASSIGNMENTS_OUTPUT = "kazan_bakery_clusters.csv"
SKU_ASSIGNMENTS_OUTPUT = "kazan_sitnaya_sku_clusters.csv"
BAKERY_SUMMARY_OUTPUT = "kazan_bakery_cluster_summary.csv"
SKU_SUMMARY_OUTPUT = "kazan_sitnaya_sku_cluster_summary.csv"
METRICS_OUTPUT = "kazan_clusters_summary.json"


def load_profile_map(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def _prepare_feature_frame(df: pd.DataFrame, feature_cols: list[str], log_cols: set[str]) -> pd.DataFrame:
    work = df[feature_cols].copy()
    for col in feature_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    for col in log_cols:
        if col in work.columns:
            work[col] = np.log1p(work[col].clip(lower=0.0))
    for col in feature_cols:
        median = work[col].median()
        work[col] = work[col].fillna(median if pd.notna(median) else 0.0)
    return work


def _pick_best_k(
    X_scaled: np.ndarray,
    *,
    k_candidates: list[int],
) -> tuple[int, float, dict[int, float]]:
    valid_candidates = [k for k in k_candidates if 2 <= k < len(X_scaled)]
    if not valid_candidates:
        fallback_k = 2 if len(X_scaled) >= 3 else 1
        return fallback_k, np.nan, {}

    scores: dict[int, float] = {}
    best_k = valid_candidates[0]
    best_score = -np.inf
    for k in valid_candidates:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(X_scaled)
        if len(np.unique(labels)) < 2:
            continue
        score = float(silhouette_score(X_scaled, labels))
        scores[k] = score
        if score > best_score:
            best_score = score
            best_k = k

    if not scores:
        return valid_candidates[0], np.nan, {}
    return best_k, best_score, scores


def cluster_entities(
    df: pd.DataFrame,
    *,
    id_cols: list[str],
    feature_cols: list[str],
    log_cols: set[str],
    k_candidates: list[int],
    cluster_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if df.empty:
        return df.copy(), pd.DataFrame(), {"rows": 0, "selected_k": 0, "silhouette_scores": {}}

    feature_frame = _prepare_feature_frame(df, feature_cols, log_cols)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_frame)

    if len(df) < 3:
        assigned = df[id_cols + feature_cols].copy()
        assigned[cluster_col] = 0
        summary = (
            assigned.groupby(cluster_col, as_index=False)[feature_cols]
            .mean()
            .assign(cluster_size=len(assigned))
        )
        metrics = {"rows": int(len(df)), "selected_k": 1, "silhouette_scores": {}}
        return assigned, summary, metrics

    best_k, best_score, score_map = _pick_best_k(X_scaled, k_candidates=k_candidates)
    model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    labels = model.fit_predict(X_scaled)

    assigned = df[id_cols + feature_cols].copy()
    assigned[cluster_col] = labels.astype(int)

    summary = (
        assigned.groupby(cluster_col, as_index=False)
        .agg({**{col: "mean" for col in feature_cols}, **{id_cols[0]: "size"}})
        .rename(columns={id_cols[0]: "cluster_size"})
        .sort_values("cluster_size", ascending=False)
        .reset_index(drop=True)
    )

    metrics = {
        "rows": int(len(df)),
        "selected_k": int(best_k),
        "best_silhouette": None if pd.isna(best_score) else round(float(best_score), 6),
        "silhouette_scores": {str(k): round(float(v), 6) for k, v in score_map.items()},
    }
    return assigned, summary, metrics


def build_kazan_clusters(
    *,
    bakery_profile_path: str | Path,
    sku_profile_path: str | Path,
    output_dir: str | Path,
    bakery_k_candidates: list[int],
    sku_k_candidates: list[int],
) -> dict[str, Path]:
    bakery_profiles = load_profile_map(bakery_profile_path)
    sku_profiles = load_profile_map(sku_profile_path)

    bakery_feature_cols = [
        "mean_bakery_sales",
        "cv_bakery_sales",
        "weekday_profile_stability",
        "weekly_amplitude_cv",
        "trend_slope_ratio",
        "category_share_mean",
        "category_share_std",
        "active_sku_mean",
    ]
    sku_feature_cols = [
        "mean_sales",
        "cv_sales",
        "zero_share",
        "weekday_profile_stability",
        "weekly_amplitude_cv",
        "bakery_total_sales_corr",
        "category_total_sales_corr",
        "sku_share_in_bakery_total_mean",
        "hour_profile_stability",
        "active_hours_mean",
        "release_present_share",
    ]

    bakery_assignments, bakery_summary, bakery_metrics = cluster_entities(
        bakery_profiles,
        id_cols=["bakery_id", "bakery_name", "city"],
        feature_cols=bakery_feature_cols,
        log_cols={"mean_bakery_sales", "active_sku_mean"},
        k_candidates=bakery_k_candidates,
        cluster_col="bakery_cluster",
    )
    sku_assignments, sku_summary, sku_metrics = cluster_entities(
        sku_profiles,
        id_cols=["bakery_id", "bakery_name", "city", "product_id", "product_name", "category_name"],
        feature_cols=sku_feature_cols,
        log_cols={"mean_sales", "active_hours_mean"},
        k_candidates=sku_k_candidates,
        cluster_col="sku_cluster",
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bakery_assignments_path = out_dir / BAKERY_ASSIGNMENTS_OUTPUT
    sku_assignments_path = out_dir / SKU_ASSIGNMENTS_OUTPUT
    bakery_summary_path = out_dir / BAKERY_SUMMARY_OUTPUT
    sku_summary_path = out_dir / SKU_SUMMARY_OUTPUT
    metrics_path = out_dir / METRICS_OUTPUT

    bakery_assignments.to_csv(bakery_assignments_path, index=False, encoding="utf-8-sig")
    sku_assignments.to_csv(sku_assignments_path, index=False, encoding="utf-8-sig")
    bakery_summary.to_csv(bakery_summary_path, index=False, encoding="utf-8-sig")
    sku_summary.to_csv(sku_summary_path, index=False, encoding="utf-8-sig")
    metrics = {
        "bakery_clusters": bakery_metrics,
        "sku_clusters": sku_metrics,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "bakery_assignments": bakery_assignments_path,
        "sku_assignments": sku_assignments_path,
        "bakery_summary": bakery_summary_path,
        "sku_summary": sku_summary_path,
        "metrics": metrics_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build bakery and SKU clusters for Kazan sitnaya sample")
    parser.add_argument("--bakery-profile-path", default=str(ROOT / "data" / "processed" / "kazan_bakery_profile_map.csv"))
    parser.add_argument("--sku-profile-path", default=str(ROOT / "data" / "processed" / "kazan_sitnaya_sku_profile_map.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--bakery-k-candidates", nargs="+", type=int, default=[3, 4, 5, 6])
    parser.add_argument("--sku-k-candidates", nargs="+", type=int, default=[3, 4, 5, 6])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_kazan_clusters(
        bakery_profile_path=args.bakery_profile_path,
        sku_profile_path=args.sku_profile_path,
        output_dir=args.output_dir,
        bakery_k_candidates=args.bakery_k_candidates,
        sku_k_candidates=args.sku_k_candidates,
    )
    print("=" * 72)
    print("KAZAN CLUSTERS")
    print("=" * 72)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

BAKERY_COL = "bakery_id"
BAKERY_NAME_COL = "bakery_name"
CITY_COL = "city"
CATEGORY_COL = "category_name"
SKU_COL = "product_id"
SKU_NAME_COL = "product_name"

OUTPUT_NAME = "kazan_decomposition_path_scores.csv"
SUMMARY_OUTPUT = "kazan_decomposition_path_scores_summary.json"


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def _component_score(stability: pd.Series) -> pd.Series:
    weekday = pd.to_numeric(stability.get("weekday_share_stability"), errors="coerce")
    cv = pd.to_numeric(stability.get("cv_share"), errors="coerce")
    obs = pd.to_numeric(stability.get("observed_days"), errors="coerce")

    cv_score = 1.0 / (1.0 + cv.clip(lower=0.0))
    obs_score = (obs / obs.max()).clip(lower=0.0, upper=1.0) if pd.notna(obs.max()) and obs.max() > 0 else pd.Series(np.nan, index=stability.index)

    return (
        0.50 * weekday
        + 0.35 * cv_score
        + 0.15 * obs_score
    )


def build_path_scores(
    *,
    sku_clusters: pd.DataFrame,
    bakery_clusters: pd.DataFrame,
    city_sku_stability: pd.DataFrame,
    sku_cluster_in_category_stability: pd.DataFrame,
    sku_in_cluster_stability: pd.DataFrame,
    bakery_cluster_sku_stability: pd.DataFrame,
) -> pd.DataFrame:
    city_sku = city_sku_stability.copy()
    city_sku["city_sku_component_score"] = _component_score(city_sku)

    sku_cluster_cat = sku_cluster_in_category_stability.copy()
    sku_cluster_cat["sku_cluster_in_category_component_score"] = _component_score(sku_cluster_cat)

    sku_in_cluster = sku_in_cluster_stability.copy()
    sku_in_cluster["sku_in_cluster_component_score"] = _component_score(sku_in_cluster)

    bakery_cluster_sku = bakery_cluster_sku_stability.copy()
    bakery_cluster_sku["bakery_cluster_sku_component_score"] = _component_score(bakery_cluster_sku)

    base = sku_clusters[
        [BAKERY_COL, BAKERY_NAME_COL, CITY_COL, CATEGORY_COL, SKU_COL, SKU_NAME_COL, "sku_cluster"]
    ].drop_duplicates()
    base = base.merge(
        bakery_clusters[[BAKERY_COL, "bakery_cluster"]].drop_duplicates(),
        on=BAKERY_COL,
        how="left",
        validate="many_to_one",
    )

    base = base.merge(
        city_sku[
            [BAKERY_COL, BAKERY_NAME_COL, CITY_COL, CATEGORY_COL, SKU_COL, SKU_NAME_COL, "city_sku_component_score"]
        ],
        on=[BAKERY_COL, BAKERY_NAME_COL, CITY_COL, CATEGORY_COL, SKU_COL, SKU_NAME_COL],
        how="left",
        validate="one_to_one",
    )
    base = base.merge(
        sku_cluster_cat[
            [BAKERY_COL, BAKERY_NAME_COL, CITY_COL, CATEGORY_COL, "sku_cluster", "sku_cluster_in_category_component_score"]
        ],
        on=[BAKERY_COL, BAKERY_NAME_COL, CITY_COL, CATEGORY_COL, "sku_cluster"],
        how="left",
        validate="many_to_one",
    )
    base = base.merge(
        sku_in_cluster[
            [BAKERY_COL, BAKERY_NAME_COL, CITY_COL, CATEGORY_COL, "sku_cluster", SKU_COL, SKU_NAME_COL, "sku_in_cluster_component_score"]
        ],
        on=[BAKERY_COL, BAKERY_NAME_COL, CITY_COL, CATEGORY_COL, "sku_cluster", SKU_COL, SKU_NAME_COL],
        how="left",
        validate="one_to_one",
    )
    base = base.merge(
        bakery_cluster_sku[
            ["bakery_cluster", BAKERY_COL, BAKERY_NAME_COL, CITY_COL, CATEGORY_COL, SKU_COL, SKU_NAME_COL, "bakery_cluster_sku_component_score"]
        ],
        on=["bakery_cluster", BAKERY_COL, BAKERY_NAME_COL, CITY_COL, CATEGORY_COL, SKU_COL, SKU_NAME_COL],
        how="left",
        validate="one_to_one",
    )

    # We do not yet have a separate stability map for bakery_category_share_in_total.
    # For now use the strongly supported category layer as a constant prior in category-based paths.
    base["bakery_category_prior_score"] = 0.90

    base["path_score_bakery_total_to_category_to_sku_cluster_to_sku"] = (
        0.15 * base["bakery_category_prior_score"]
        + 0.40 * base["sku_cluster_in_category_component_score"]
        + 0.45 * base["sku_in_cluster_component_score"]
    )
    base["path_score_bakery_category_to_sku_cluster_to_sku"] = (
        0.45 * base["sku_cluster_in_category_component_score"]
        + 0.55 * base["sku_in_cluster_component_score"]
    )
    base["path_score_city_sku_to_bakery"] = base["city_sku_component_score"]
    base["path_score_bakery_cluster_sku_to_bakery"] = base["bakery_cluster_sku_component_score"]

    path_cols = [
        "path_score_bakery_total_to_category_to_sku_cluster_to_sku",
        "path_score_bakery_category_to_sku_cluster_to_sku",
        "path_score_city_sku_to_bakery",
        "path_score_bakery_cluster_sku_to_bakery",
    ]
    renamed = {
        "path_score_bakery_total_to_category_to_sku_cluster_to_sku": "bakery_total_to_category_to_sku_cluster_to_sku",
        "path_score_bakery_category_to_sku_cluster_to_sku": "bakery_category_to_sku_cluster_to_sku",
        "path_score_city_sku_to_bakery": "city_sku_to_bakery",
        "path_score_bakery_cluster_sku_to_bakery": "bakery_cluster_sku_to_bakery",
    }

    def _rank_paths(row: pd.Series) -> tuple[str | float, float | float, float | float]:
        values = [(renamed[col], row[col]) for col in path_cols if pd.notna(row[col])]
        if not values:
            return np.nan, np.nan, np.nan
        values.sort(key=lambda item: item[1], reverse=True)
        best_name, best_score = values[0]
        second_score = values[1][1] if len(values) > 1 else np.nan
        confidence = best_score - second_score if pd.notna(second_score) else np.nan
        return best_name, best_score, confidence

    ranked = base.apply(_rank_paths, axis=1, result_type="expand")
    ranked.columns = ["best_decomposition_path", "best_path_score", "path_confidence"]
    base = pd.concat([base, ranked], axis=1)

    return base.sort_values(["best_decomposition_path", "best_path_score"], ascending=[True, False]).reset_index(drop=True)


def build_summary(path_scores: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(path_scores)),
        "best_path_counts": path_scores["best_decomposition_path"].value_counts(dropna=False).to_dict(),
        "mean_best_path_score": round(float(pd.to_numeric(path_scores["best_path_score"], errors="coerce").mean()), 6) if not path_scores.empty else 0.0,
        "mean_path_confidence": round(float(pd.to_numeric(path_scores["path_confidence"], errors="coerce").mean()), 6) if not path_scores.empty else 0.0,
    }


def save_outputs(output_dir: str | Path, path_scores: pd.DataFrame, summary: dict[str, object]) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / OUTPUT_NAME
    summary_path = out_dir / SUMMARY_OUTPUT
    path_scores.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"path_scores": csv_path, "summary": summary_path}


def build_kazan_decomposition_path_scores(
    *,
    sku_clusters_path: str | Path,
    bakery_clusters_path: str | Path,
    city_sku_stability_path: str | Path,
    sku_cluster_in_category_stability_path: str | Path,
    sku_in_cluster_stability_path: str | Path,
    bakery_cluster_sku_stability_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    sku_clusters = load_csv(sku_clusters_path)
    bakery_clusters = load_csv(bakery_clusters_path)
    city_sku_stability = load_csv(city_sku_stability_path)
    sku_cluster_in_category_stability = load_csv(sku_cluster_in_category_stability_path)
    sku_in_cluster_stability = load_csv(sku_in_cluster_stability_path)
    bakery_cluster_sku_stability = load_csv(bakery_cluster_sku_stability_path)

    path_scores = build_path_scores(
        sku_clusters=sku_clusters,
        bakery_clusters=bakery_clusters,
        city_sku_stability=city_sku_stability,
        sku_cluster_in_category_stability=sku_cluster_in_category_stability,
        sku_in_cluster_stability=sku_in_cluster_stability,
        bakery_cluster_sku_stability=bakery_cluster_sku_stability,
    )
    summary = build_summary(path_scores)
    return save_outputs(output_dir, path_scores, summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build decomposition path scores for Kazan sample")
    parser.add_argument("--sku-clusters-path", default=str(ROOT / "data" / "processed" / "kazan_sitnaya_sku_clusters.csv"))
    parser.add_argument("--bakery-clusters-path", default=str(ROOT / "data" / "processed" / "kazan_bakery_clusters.csv"))
    parser.add_argument("--city-sku-stability-path", default=str(ROOT / "data" / "processed" / "kazan_bakery_share_in_city_sku_stability.csv"))
    parser.add_argument("--sku-cluster-in-category-stability-path", default=str(ROOT / "data" / "processed" / "kazan_sku_cluster_share_in_bakery_category_stability.csv"))
    parser.add_argument("--sku-in-cluster-stability-path", default=str(ROOT / "data" / "processed" / "kazan_sku_share_in_bakery_sku_cluster_stability.csv"))
    parser.add_argument("--bakery-cluster-sku-stability-path", default=str(ROOT / "data" / "processed" / "kazan_bakery_share_in_bakery_cluster_sku_stability.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "processed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_kazan_decomposition_path_scores(
        sku_clusters_path=args.sku_clusters_path,
        bakery_clusters_path=args.bakery_clusters_path,
        city_sku_stability_path=args.city_sku_stability_path,
        sku_cluster_in_category_stability_path=args.sku_cluster_in_category_stability_path,
        sku_in_cluster_stability_path=args.sku_in_cluster_stability_path,
        bakery_cluster_sku_stability_path=args.bakery_cluster_sku_stability_path,
        output_dir=args.output_dir,
    )
    print("=" * 72)
    print("KAZAN DECOMPOSITION PATH SCORES")
    print("=" * 72)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

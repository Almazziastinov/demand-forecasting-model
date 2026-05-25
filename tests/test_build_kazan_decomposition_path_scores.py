import importlib.util
import json
from pathlib import Path
import shutil
import uuid

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "experiments_v2"
    / "build_kazan_decomposition_path_scores.py"
)
SPEC = importlib.util.spec_from_file_location("build_kazan_decomposition_path_scores", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_build_kazan_decomposition_path_scores_outputs_best_path() -> None:
    tmp_path = Path.cwd() / ".pytest_local" / f"build_kazan_path_scores_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    sku_clusters_path = tmp_path / "sku_clusters.csv"
    bakery_clusters_path = tmp_path / "bakery_clusters.csv"
    city_sku_stability_path = tmp_path / "city_sku_stability.csv"
    sku_cluster_in_category_stability_path = tmp_path / "sku_cluster_cat_stability.csv"
    sku_in_cluster_stability_path = tmp_path / "sku_in_cluster_stability.csv"
    bakery_cluster_sku_stability_path = tmp_path / "bakery_cluster_sku_stability.csv"
    output_dir = tmp_path / "processed"

    sku_clusters = pd.DataFrame(
        {
            "bakery_id": ["1"],
            "bakery_name": ["B1"],
            "city": ["Казань"],
            "category_name": ["Выпечка сытная"],
            "product_id": ["101"],
            "product_name": ["S1"],
            "sku_cluster": [2],
        }
    )
    sku_clusters.to_csv(sku_clusters_path, index=False, encoding="utf-8-sig")

    bakery_clusters = pd.DataFrame({"bakery_id": ["1"], "bakery_cluster": [0]})
    bakery_clusters.to_csv(bakery_clusters_path, index=False, encoding="utf-8-sig")

    city_sku_stability = pd.DataFrame(
        {
            "bakery_id": ["1"],
            "bakery_name": ["B1"],
            "city": ["Казань"],
            "category_name": ["Выпечка сытная"],
            "product_id": ["101"],
            "product_name": ["S1"],
            "weekday_share_stability": [0.90],
            "cv_share": [0.60],
            "observed_days": [100],
        }
    )
    city_sku_stability.to_csv(city_sku_stability_path, index=False, encoding="utf-8-sig")

    sku_cluster_cat_stability = pd.DataFrame(
        {
            "bakery_id": ["1"],
            "bakery_name": ["B1"],
            "city": ["Казань"],
            "category_name": ["Выпечка сытная"],
            "sku_cluster": [2],
            "weekday_share_stability": [0.98],
            "cv_share": [0.10],
            "observed_days": [100],
        }
    )
    sku_cluster_cat_stability.to_csv(sku_cluster_in_category_stability_path, index=False, encoding="utf-8-sig")

    sku_in_cluster_stability = pd.DataFrame(
        {
            "bakery_id": ["1"],
            "bakery_name": ["B1"],
            "city": ["Казань"],
            "category_name": ["Выпечка сытная"],
            "sku_cluster": [2],
            "product_id": ["101"],
            "product_name": ["S1"],
            "weekday_share_stability": [0.97],
            "cv_share": [0.12],
            "observed_days": [100],
        }
    )
    sku_in_cluster_stability.to_csv(sku_in_cluster_stability_path, index=False, encoding="utf-8-sig")

    bakery_cluster_sku_stability = pd.DataFrame(
        {
            "bakery_cluster": [0],
            "bakery_id": ["1"],
            "bakery_name": ["B1"],
            "city": ["Казань"],
            "category_name": ["Выпечка сытная"],
            "product_id": ["101"],
            "product_name": ["S1"],
            "weekday_share_stability": [0.92],
            "cv_share": [0.30],
            "observed_days": [100],
        }
    )
    bakery_cluster_sku_stability.to_csv(bakery_cluster_sku_stability_path, index=False, encoding="utf-8-sig")

    try:
        paths = MODULE.build_kazan_decomposition_path_scores(
            sku_clusters_path=sku_clusters_path,
            bakery_clusters_path=bakery_clusters_path,
            city_sku_stability_path=city_sku_stability_path,
            sku_cluster_in_category_stability_path=sku_cluster_in_category_stability_path,
            sku_in_cluster_stability_path=sku_in_cluster_stability_path,
            bakery_cluster_sku_stability_path=bakery_cluster_sku_stability_path,
            output_dir=output_dir,
        )

        path_scores = pd.read_csv(paths["path_scores"], encoding="utf-8-sig")
        summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))

        assert len(path_scores) == 1
        assert path_scores.iloc[0]["best_decomposition_path"] == "bakery_category_to_sku_cluster_to_sku"
        assert summary["rows"] == 1
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

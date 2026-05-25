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
    / "build_kazan_share_stability_maps.py"
)
SPEC = importlib.util.spec_from_file_location("build_kazan_share_stability_maps", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_build_kazan_share_stability_maps_outputs_expected_files() -> None:
    tmp_path = Path.cwd() / ".pytest_local" / f"build_kazan_share_stability_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    city_sku_path = tmp_path / "city_sku_share.csv"
    sku_cluster_path = tmp_path / "sku_cluster_share.csv"
    sku_share_path = tmp_path / "sku_share.csv"
    bakery_cluster_sku_path = tmp_path / "bakery_cluster_sku_share.csv"
    output_dir = tmp_path / "processed"

    dates = pd.date_range("2026-05-01", periods=7, freq="D")

    city_sku_df = pd.DataFrame(
        {
            "date": dates,
            "bakery_id": ["1"] * 7,
            "bakery_name": ["B1"] * 7,
            "city": ["Казань"] * 7,
            "category_name": ["Выпечка сытная"] * 7,
            "product_id": ["101"] * 7,
            "product_name": ["S1"] * 7,
            "bakery_share_in_city_sku": [0.30, 0.31, 0.29, 0.30, 0.32, 0.31, 0.30],
        }
    )
    city_sku_df.to_csv(city_sku_path, index=False, encoding="utf-8-sig")

    sku_cluster_df = pd.DataFrame(
        {
            "date": dates,
            "bakery_id": ["1"] * 7,
            "bakery_name": ["B1"] * 7,
            "city": ["Казань"] * 7,
            "category_name": ["Выпечка сытная"] * 7,
            "sku_cluster": [2] * 7,
            "sku_cluster_share_in_bakery_category": [0.8, 0.82, 0.79, 0.81, 0.8, 0.83, 0.79],
        }
    )
    sku_cluster_df.to_csv(sku_cluster_path, index=False, encoding="utf-8-sig")

    sku_share_df = pd.DataFrame(
        {
            "date": dates,
            "bakery_id": ["1"] * 7,
            "bakery_name": ["B1"] * 7,
            "city": ["Казань"] * 7,
            "category_name": ["Выпечка сытная"] * 7,
            "sku_cluster": [2] * 7,
            "product_id": ["101"] * 7,
            "product_name": ["S1"] * 7,
            "sku_share_in_bakery_sku_cluster": [0.20, 0.22, 0.19, 0.21, 0.20, 0.18, 0.20],
        }
    )
    sku_share_df.to_csv(sku_share_path, index=False, encoding="utf-8-sig")

    bakery_cluster_sku_df = pd.DataFrame(
        {
            "date": dates,
            "bakery_cluster": [0] * 7,
            "bakery_id": ["1"] * 7,
            "bakery_name": ["B1"] * 7,
            "city": ["Казань"] * 7,
            "category_name": ["Выпечка сытная"] * 7,
            "product_id": ["101"] * 7,
            "product_name": ["S1"] * 7,
            "bakery_share_in_bakery_cluster_sku": [0.4, 0.42, 0.38, 0.41, 0.39, 0.40, 0.41],
        }
    )
    bakery_cluster_sku_df.to_csv(bakery_cluster_sku_path, index=False, encoding="utf-8-sig")

    try:
        paths = MODULE.build_kazan_share_stability_maps(
            bakery_share_in_city_sku_path=city_sku_path,
            sku_cluster_share_in_bakery_category_path=sku_cluster_path,
            sku_share_in_bakery_sku_cluster_path=sku_share_path,
            bakery_share_in_bakery_cluster_sku_path=bakery_cluster_sku_path,
            output_dir=output_dir,
        )

        city_sku_stability = pd.read_csv(paths["bakery_share_in_city_sku"], encoding="utf-8-sig")
        summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))

        assert len(city_sku_stability) == 1
        assert round(float(city_sku_stability.iloc[0]["mean_share"]), 6) == 0.304286
        assert "bakery_share_in_city_sku" in summary
        assert summary["bakery_share_in_city_sku"]["rows"] == 1
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

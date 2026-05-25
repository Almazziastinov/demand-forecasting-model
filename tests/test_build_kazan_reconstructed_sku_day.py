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
    / "build_kazan_reconstructed_sku_day.py"
)
SPEC = importlib.util.spec_from_file_location("build_kazan_reconstructed_sku_day", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_build_kazan_reconstructed_sku_day_outputs_reconstructed_series() -> None:
    tmp_path = Path.cwd() / ".pytest_local" / f"build_kazan_reconstructed_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)

    sku_daily_path = tmp_path / "sku_daily.csv"
    bakery_category_daily_path = tmp_path / "bakery_category_daily.csv"
    bakery_category_share_path = tmp_path / "bakery_category_share.csv"
    city_sku_day_path = tmp_path / "city_sku_day.csv"
    bakery_share_city_path = tmp_path / "bakery_share_city.csv"
    bakery_sku_cluster_day_path = tmp_path / "bakery_sku_cluster_day.csv"
    sku_cluster_share_path = tmp_path / "sku_cluster_share.csv"
    sku_share_cluster_path = tmp_path / "sku_share_cluster.csv"
    bakery_cluster_sku_day_path = tmp_path / "bakery_cluster_sku_day.csv"
    bakery_share_cluster_sku_path = tmp_path / "bakery_share_cluster_sku.csv"
    path_scores_path = tmp_path / "path_scores.csv"
    output_dir = tmp_path / "processed"

    sku_daily = pd.DataFrame(
        [
            {
                "date": "2026-05-01",
                "bakery_id": "1",
                "bakery_name": "B1",
                "city": "Казань",
                "category_name": "Выпечка сытная",
                "product_id": "101",
                "product_name": "S1",
                "observed_sales_qty": 10.0,
                "release_qty": 12.0,
                "row_quality_score": 0.9,
                "bakery_total_sales_qty": 100.0,
            }
        ]
    )
    sku_daily.to_csv(sku_daily_path, index=False, encoding="utf-8-sig")

    bakery_category_daily = pd.DataFrame(
        [{"date": "2026-05-01", "bakery_id": "1", "category_name": "Выпечка сытная", "category_sales_qty": 50.0}]
    )
    bakery_category_daily.to_csv(bakery_category_daily_path, index=False, encoding="utf-8-sig")

    bakery_category_share = pd.DataFrame(
        [{"date": "2026-05-01", "bakery_id": "1", "category_name": "Выпечка сытная", "bakery_category_share_in_total": 0.5}]
    )
    bakery_category_share.to_csv(bakery_category_share_path, index=False, encoding="utf-8-sig")

    city_sku_day = pd.DataFrame(
        [{"date": "2026-05-01", "city": "Казань", "category_name": "Выпечка сытная", "product_id": "101", "product_name": "S1", "city_sku_sales_qty": 40.0}]
    )
    city_sku_day.to_csv(city_sku_day_path, index=False, encoding="utf-8-sig")

    bakery_share_city = pd.DataFrame(
        [{"date": "2026-05-01", "bakery_id": "1", "bakery_name": "B1", "city": "Казань", "category_name": "Выпечка сытная", "product_id": "101", "product_name": "S1", "bakery_share_in_city_sku": 0.25}]
    )
    bakery_share_city.to_csv(bakery_share_city_path, index=False, encoding="utf-8-sig")

    bakery_sku_cluster_day = pd.DataFrame(
        [{"date": "2026-05-01", "bakery_id": "1", "category_name": "Выпечка сытная", "sku_cluster": 2, "bakery_sku_cluster_sales_qty": 30.0}]
    )
    bakery_sku_cluster_day.to_csv(bakery_sku_cluster_day_path, index=False, encoding="utf-8-sig")

    sku_cluster_share = pd.DataFrame(
        [{"date": "2026-05-01", "bakery_id": "1", "bakery_name": "B1", "city": "Казань", "category_name": "Выпечка сытная", "sku_cluster": 2, "sku_cluster_share_in_bakery_category": 0.6}]
    )
    sku_cluster_share.to_csv(sku_cluster_share_path, index=False, encoding="utf-8-sig")

    sku_share_cluster = pd.DataFrame(
        [{"date": "2026-05-01", "bakery_id": "1", "bakery_name": "B1", "city": "Казань", "category_name": "Выпечка сытная", "sku_cluster": 2, "product_id": "101", "product_name": "S1", "sku_share_in_bakery_sku_cluster": 0.5}]
    )
    sku_share_cluster.to_csv(sku_share_cluster_path, index=False, encoding="utf-8-sig")

    bakery_cluster_sku_day = pd.DataFrame(
        [{"date": "2026-05-01", "bakery_cluster": 0, "city": "Казань", "category_name": "Выпечка сытная", "product_id": "101", "product_name": "S1", "bakery_cluster_sku_sales_qty": 20.0}]
    )
    bakery_cluster_sku_day.to_csv(bakery_cluster_sku_day_path, index=False, encoding="utf-8-sig")

    bakery_share_cluster_sku = pd.DataFrame(
        [{"date": "2026-05-01", "bakery_cluster": 0, "bakery_id": "1", "bakery_name": "B1", "city": "Казань", "category_name": "Выпечка сытная", "product_id": "101", "product_name": "S1", "bakery_share_in_bakery_cluster_sku": 0.4}]
    )
    bakery_share_cluster_sku.to_csv(bakery_share_cluster_sku_path, index=False, encoding="utf-8-sig")

    path_scores = pd.DataFrame(
        [
            {
                "bakery_id": "1",
                "bakery_name": "B1",
                "city": "Казань",
                "category_name": "Выпечка сытная",
                "product_id": "101",
                "product_name": "S1",
                "sku_cluster": 2,
                "bakery_cluster": 0,
                "best_decomposition_path": "bakery_category_to_sku_cluster_to_sku",
                "best_path_score": 0.95,
                "path_confidence": 0.05,
            }
        ]
    )
    path_scores.to_csv(path_scores_path, index=False, encoding="utf-8-sig")

    try:
        paths = MODULE.build_kazan_reconstructed_sku_day(
            sku_daily_path=sku_daily_path,
            bakery_category_daily_path=bakery_category_daily_path,
            bakery_category_share_in_total_daily_path=bakery_category_share_path,
            city_sku_day_path=city_sku_day_path,
            bakery_share_in_city_sku_daily_path=bakery_share_city_path,
            bakery_sku_cluster_day_path=bakery_sku_cluster_day_path,
            sku_cluster_share_in_bakery_category_daily_path=sku_cluster_share_path,
            sku_share_in_bakery_sku_cluster_daily_path=sku_share_cluster_path,
            bakery_cluster_sku_day_path=bakery_cluster_sku_day_path,
            bakery_share_in_bakery_cluster_sku_daily_path=bakery_share_cluster_sku_path,
            path_scores_path=path_scores_path,
            output_dir=output_dir,
        )

        reconstructed = pd.read_csv(paths["reconstructed"], encoding="utf-8-sig")
        summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))

        assert len(reconstructed) == 1
        assert round(float(reconstructed.iloc[0]["reconstructed_sales_qty"]), 6) == 15.0
        assert summary["rows"] == 1
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

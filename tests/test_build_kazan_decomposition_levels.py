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
    / "build_kazan_decomposition_levels.py"
)
SPEC = importlib.util.spec_from_file_location("build_kazan_decomposition_levels", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_build_kazan_decomposition_levels_outputs_expected_shares() -> None:
    tmp_path = Path.cwd() / ".pytest_local" / f"build_kazan_decomp_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    bakery_daily_path = tmp_path / "bakery_daily.csv"
    bakery_category_daily_path = tmp_path / "bakery_category_daily.csv"
    sku_daily_path = tmp_path / "sku_daily.csv"
    bakery_clusters_path = tmp_path / "bakery_clusters.csv"
    sku_clusters_path = tmp_path / "sku_clusters.csv"
    output_dir = tmp_path / "processed"

    bakery_daily = pd.DataFrame(
        [
            {"date": "2026-05-01", "bakery_id": "1", "bakery_name": "B1", "city": "Казань", "bakery_sales": 100},
            {"date": "2026-05-01", "bakery_id": "2", "bakery_name": "B2", "city": "Казань", "bakery_sales": 80},
        ]
    )
    bakery_daily.to_csv(bakery_daily_path, index=False, encoding="utf-8-sig")

    bakery_category_daily = pd.DataFrame(
        [
            {"date": "2026-05-01", "bakery_id": "1", "bakery_name": "B1", "city": "Казань", "category_name": "Выпечка сытная", "category_sales_qty": 50, "bakery_total_sales_qty": 100},
            {"date": "2026-05-01", "bakery_id": "2", "bakery_name": "B2", "city": "Казань", "category_name": "Выпечка сытная", "category_sales_qty": 40, "bakery_total_sales_qty": 80},
        ]
    )
    bakery_category_daily.to_csv(bakery_category_daily_path, index=False, encoding="utf-8-sig")

    sku_daily = pd.DataFrame(
        [
            {"date": "2026-05-01", "bakery_id": "1", "bakery_name": "B1", "city": "Казань", "product_id": "101", "product_name": "S1", "category_name": "Выпечка сытная", "observed_sales_qty": 30},
            {"date": "2026-05-01", "bakery_id": "1", "bakery_name": "B1", "city": "Казань", "product_id": "102", "product_name": "S2", "category_name": "Выпечка сытная", "observed_sales_qty": 20},
            {"date": "2026-05-01", "bakery_id": "2", "bakery_name": "B2", "city": "Казань", "product_id": "101", "product_name": "S1", "category_name": "Выпечка сытная", "observed_sales_qty": 10},
            {"date": "2026-05-01", "bakery_id": "2", "bakery_name": "B2", "city": "Казань", "product_id": "102", "product_name": "S2", "category_name": "Выпечка сытная", "observed_sales_qty": 30},
        ]
    )
    sku_daily.to_csv(sku_daily_path, index=False, encoding="utf-8-sig")

    bakery_clusters = pd.DataFrame({"bakery_id": ["1", "2"], "bakery_cluster": [0, 0]})
    bakery_clusters.to_csv(bakery_clusters_path, index=False, encoding="utf-8-sig")

    sku_clusters = pd.DataFrame(
        {
            "bakery_id": ["1", "1", "2", "2"],
            "product_id": ["101", "102", "101", "102"],
            "sku_cluster": [1, 1, 1, 1],
        }
    )
    sku_clusters.to_csv(sku_clusters_path, index=False, encoding="utf-8-sig")

    try:
        paths = MODULE.build_kazan_decomposition_levels(
            bakery_daily_path=bakery_daily_path,
            bakery_category_daily_path=bakery_category_daily_path,
            sku_daily_path=sku_daily_path,
            bakery_clusters_path=bakery_clusters_path,
            sku_clusters_path=sku_clusters_path,
            output_dir=output_dir,
        )

        city_sku = pd.read_csv(paths["city_sku_day"], encoding="utf-8-sig")
        bakery_share_in_city_sku = pd.read_csv(paths["bakery_share_in_city_sku_daily"], encoding="utf-8-sig")
        sku_share_in_cluster = pd.read_csv(paths["sku_share_in_bakery_sku_cluster_daily"], encoding="utf-8-sig")
        summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))

        s1_city = city_sku.loc[city_sku["product_id"] == 101, "city_sku_sales_qty"].iloc[0]
        assert s1_city == 40

        share_b1_s1 = bakery_share_in_city_sku.loc[
            (bakery_share_in_city_sku["bakery_id"] == 1) & (bakery_share_in_city_sku["product_id"] == 101),
            "bakery_share_in_city_sku",
        ].iloc[0]
        assert round(float(share_b1_s1), 6) == 0.75

        share_cluster_b1_s1 = sku_share_in_cluster.loc[
            (sku_share_in_cluster["bakery_id"] == 1) & (sku_share_in_cluster["product_id"] == 101),
            "sku_share_in_bakery_sku_cluster",
        ].iloc[0]
        assert round(float(share_cluster_b1_s1), 6) == 0.6

        assert "city_sku_day" in summary
        assert summary["city_sku_day"]["rows"] == 2
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

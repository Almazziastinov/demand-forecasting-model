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
    / "build_kazan_anchor_suitability.py"
)
SPEC = importlib.util.spec_from_file_location("build_kazan_anchor_suitability", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_build_kazan_anchor_suitability_outputs_best_anchor_levels() -> None:
    tmp_path = Path.cwd() / ".pytest_local" / f"build_kazan_anchor_suitability_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    bakery_daily_path = tmp_path / "bakery_daily.csv"
    bakery_category_daily_path = tmp_path / "bakery_category_daily.csv"
    sku_daily_path = tmp_path / "sku_daily.csv"
    sku_hourly_path = tmp_path / "sku_hourly.csv"
    bakery_clusters_path = tmp_path / "bakery_clusters.csv"
    sku_clusters_path = tmp_path / "sku_clusters.csv"
    sku_profile_map_path = tmp_path / "sku_profile_map.csv"
    output_dir = tmp_path / "processed"

    dates = pd.date_range("2026-05-01", periods=7, freq="D")

    bakery_daily = pd.DataFrame(
        {
            "date": dates.tolist() * 2,
            "bakery_id": ["1"] * 7 + ["2"] * 7,
            "bakery_name": ["B1"] * 7 + ["B2"] * 7,
            "city": ["Казань"] * 14,
            "bakery_sales": [100, 110, 120, 130, 140, 150, 160, 90, 95, 100, 105, 110, 115, 120],
            "dow": [d.weekday() for d in dates] * 2,
        }
    )
    bakery_daily.to_csv(bakery_daily_path, index=False, encoding="utf-8-sig")

    bakery_category_daily = pd.DataFrame(
        {
            "date": dates.tolist() * 2,
            "bakery_id": ["1"] * 7 + ["2"] * 7,
            "bakery_name": ["B1"] * 7 + ["B2"] * 7,
            "city": ["Казань"] * 14,
            "category_name": ["Выпечка сытная"] * 14,
            "category_sales_qty": [50, 55, 60, 65, 70, 75, 80, 30, 32, 34, 36, 38, 40, 42],
            "bakery_total_sales_qty": [100, 110, 120, 130, 140, 150, 160, 90, 95, 100, 105, 110, 115, 120],
            "dow": [d.weekday() for d in dates] * 2,
        }
    )
    bakery_category_daily.to_csv(bakery_category_daily_path, index=False, encoding="utf-8-sig")

    sku_daily = pd.DataFrame(
        {
            "date": dates.tolist() * 2,
            "bakery_id": ["1"] * 7 + ["2"] * 7,
            "bakery_name": ["B1"] * 7 + ["B2"] * 7,
            "city": ["Казань"] * 14,
            "product_id": ["101"] * 7 + ["102"] * 7,
            "product_name": ["S1"] * 7 + ["S2"] * 7,
            "category_name": ["Выпечка сытная"] * 14,
            "observed_sales_qty": [25, 27, 30, 33, 35, 38, 40, 5, 8, 4, 9, 3, 10, 2],
            "bakery_total_sales_qty": [100, 110, 120, 130, 140, 150, 160, 90, 95, 100, 105, 110, 115, 120],
            "category_sales_qty": [50, 55, 60, 65, 70, 75, 80, 30, 32, 34, 36, 38, 40, 42],
            "sku_sales_share_in_bakery_total": [0.25] * 7 + [0.055, 0.084, 0.04, 0.086, 0.027, 0.087, 0.017],
            "dow": [d.weekday() for d in dates] * 2,
        }
    )
    sku_daily.to_csv(sku_daily_path, index=False, encoding="utf-8-sig")

    sku_hourly = pd.DataFrame(
        [
            {"date": d, "dow": d.weekday(), "hour": 9, "bakery_id": "1", "bakery_name": "B1", "product_id": "101", "product_name": "S1", "category_name": "Выпечка сытная", "sku_hour_sales": 10.0}
            for d in dates
        ]
        + [
            {"date": d, "dow": d.weekday(), "hour": 10, "bakery_id": "1", "bakery_name": "B1", "product_id": "101", "product_name": "S1", "category_name": "Выпечка сытная", "sku_hour_sales": 15.0}
            for d in dates
        ]
        + [
            {"date": d, "dow": d.weekday(), "hour": 9, "bakery_id": "2", "bakery_name": "B2", "product_id": "102", "product_name": "S2", "category_name": "Выпечка сытная", "sku_hour_sales": 5.0 if d.day % 2 else 0.0}
            for d in dates
        ]
    )
    sku_hourly.to_csv(sku_hourly_path, index=False, encoding="utf-8-sig")

    bakery_clusters = pd.DataFrame({"bakery_id": ["1", "2"], "bakery_cluster": [0, 0]})
    bakery_clusters.to_csv(bakery_clusters_path, index=False, encoding="utf-8-sig")

    sku_clusters = pd.DataFrame(
        {
            "bakery_id": ["1", "2"],
            "product_id": ["101", "102"],
            "sku_cluster": [1, 2],
        }
    )
    sku_clusters.to_csv(sku_clusters_path, index=False, encoding="utf-8-sig")

    sku_profile_map = pd.DataFrame(
        {
            "bakery_id": ["1", "2"],
            "product_id": ["101", "102"],
            "weekday_profile_stability": [0.95, 0.40],
            "hour_profile_stability": [0.90, 0.20],
        }
    )
    sku_profile_map.to_csv(sku_profile_map_path, index=False, encoding="utf-8-sig")

    try:
        paths = MODULE.build_kazan_anchor_suitability(
            bakery_daily_path=bakery_daily_path,
            bakery_category_daily_path=bakery_category_daily_path,
            sku_daily_path=sku_daily_path,
            sku_hourly_path=sku_hourly_path,
            bakery_clusters_path=bakery_clusters_path,
            sku_clusters_path=sku_clusters_path,
            sku_profile_map_path=sku_profile_map_path,
            output_dir=output_dir,
        )

        anchor_map = pd.read_csv(paths["anchor_map"], encoding="utf-8-sig")
        summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))

        assert len(anchor_map) == 2
        assert "best_anchor_level" in anchor_map.columns
        assert "bakery_category_anchor_score" in anchor_map.columns
        assert summary["rows"] == 2
        assert summary["best_anchor_counts"]
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

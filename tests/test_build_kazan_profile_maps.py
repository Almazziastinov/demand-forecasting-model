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
    / "build_kazan_profile_maps.py"
)
SPEC = importlib.util.spec_from_file_location("build_kazan_profile_maps", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_build_kazan_profile_maps_outputs_expected_artifacts() -> None:
    tmp_path = Path.cwd() / ".pytest_local" / f"build_kazan_profile_maps_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    bakery_daily_path = tmp_path / "bakery_daily.csv"
    bakery_category_daily_path = tmp_path / "bakery_category_daily.csv"
    sku_daily_path = tmp_path / "sku_daily.csv"
    sku_hourly_path = tmp_path / "sku_hourly.csv"
    output_dir = tmp_path / "processed"

    dates = pd.date_range("2026-05-01", periods=7, freq="D")

    bakery_daily = pd.DataFrame(
        {
            "date": dates,
            "bakery_id": ["1"] * 7,
            "bakery_name": ["Bakery 1"] * 7,
            "city": ["Казань"] * 7,
            "bakery_sales": [100, 110, 120, 130, 140, 150, 160],
            "avg_price": [90, 91, 92, 93, 94, 95, 96],
            "dow": [d.weekday() for d in dates],
        }
    )
    bakery_daily.to_csv(bakery_daily_path, index=False, encoding="utf-8-sig")

    bakery_category_daily = pd.DataFrame(
        {
            "date": dates,
            "bakery_id": ["1"] * 7,
            "bakery_name": ["Bakery 1"] * 7,
            "city": ["Казань"] * 7,
            "category_name": ["Выпечка сытная"] * 7,
            "category_sales_qty": [40, 44, 48, 52, 56, 60, 64],
            "category_release_qty": [45, 49, 53, 57, 61, 65, 69],
            "active_sku_count": [2] * 7,
            "selling_sku_count": [2] * 7,
            "mean_row_quality_score": [0.9] * 7,
            "bakery_total_sales_qty": [100, 110, 120, 130, 140, 150, 160],
            "category_share_in_bakery_total": [0.4] * 7,
        }
    )
    bakery_category_daily.to_csv(bakery_category_daily_path, index=False, encoding="utf-8-sig")

    sku_daily = pd.DataFrame(
        {
            "date": dates.tolist() * 2,
            "bakery_id": ["1"] * 14,
            "bakery_name": ["Bakery 1"] * 14,
            "city": ["Казань"] * 14,
            "product_id": ["101"] * 7 + ["102"] * 7,
            "product_name": ["SKU A"] * 7 + ["SKU B"] * 7,
            "category_name": ["Выпечка сытная"] * 14,
            "observed_sales_qty": [20, 22, 24, 26, 28, 30, 32, 5, 0, 5, 0, 5, 0, 5],
            "sales_hours_count": [3] * 7 + [1] * 7,
            "sales_present_flag": [1] * 7 + [1, 0, 1, 0, 1, 0, 1],
            "release_qty": [22, 24, 26, 28, 30, 32, 34, 5, 0, 5, 0, 5, 0, 5],
            "release_present_flag": [1] * 14,
            "row_quality_score": [0.95] * 14,
            "bakery_total_sales_qty": [100, 110, 120, 130, 140, 150, 160] * 2,
            "bakery_avg_price_all_categories": [90] * 14,
            "sku_sales_share_in_bakery_total": [0.2] * 7 + [0.05, 0.0, 0.041667, 0.0, 0.035714, 0.0, 0.03125],
            "dow": [d.weekday() for d in dates] * 2,
        }
    )
    sku_daily.to_csv(sku_daily_path, index=False, encoding="utf-8-sig")

    hourly_rows = []
    for date in dates:
        hourly_rows.append(
            {
                "date": date,
                "dow": date.weekday(),
                "hour": 9,
                "bakery_id": "1",
                "bakery_name": "Bakery 1",
                "product_id": "101",
                "product_name": "SKU A",
                "category_name": "Выпечка сытная",
                "sku_hour_sales": 10.0,
            }
        )
        hourly_rows.append(
            {
                "date": date,
                "dow": date.weekday(),
                "hour": 10,
                "bakery_id": "1",
                "bakery_name": "Bakery 1",
                "product_id": "101",
                "product_name": "SKU A",
                "category_name": "Выпечка сытная",
                "sku_hour_sales": 10.0,
            }
        )
        hourly_rows.append(
            {
                "date": date,
                "dow": date.weekday(),
                "hour": 11,
                "bakery_id": "1",
                "bakery_name": "Bakery 1",
                "product_id": "102",
                "product_name": "SKU B",
                "category_name": "Выпечка сытная",
                "sku_hour_sales": 5.0 if date.day % 2 == 1 else 0.0,
            }
        )
    sku_hourly = pd.DataFrame(hourly_rows)
    sku_hourly.to_csv(sku_hourly_path, index=False, encoding="utf-8-sig")

    try:
        paths = MODULE.build_kazan_profile_maps(
            bakery_daily_path=bakery_daily_path,
            bakery_category_daily_path=bakery_category_daily_path,
            sku_daily_path=sku_daily_path,
            sku_hourly_path=sku_hourly_path,
            output_dir=output_dir,
        )

        bakery_profile = pd.read_csv(paths["bakery_profile_map"], encoding="utf-8-sig")
        sku_profile = pd.read_csv(paths["sku_profile_map"], encoding="utf-8-sig")
        summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))

        assert len(bakery_profile) == 1
        assert len(sku_profile) == 2
        assert "category_share_mean" in bakery_profile.columns
        assert "hour_profile_stability" in sku_profile.columns
        assert round(float(bakery_profile.iloc[0]["category_share_mean"]), 6) == 0.4

        sku_a = sku_profile.loc[sku_profile["product_name"] == "SKU A"].iloc[0]
        sku_b = sku_profile.loc[sku_profile["product_name"] == "SKU B"].iloc[0]
        assert sku_a["mean_sales"] > sku_b["mean_sales"]
        assert summary["bakery_profiles"] == 1
        assert summary["sku_profiles"] == 2
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

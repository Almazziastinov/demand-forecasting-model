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
    / "build_kazan_sitnaya_sample.py"
)
SPEC = importlib.util.spec_from_file_location("build_kazan_sitnaya_sample", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_build_kazan_sitnaya_sample_filters_city_category_and_selects_top_bakeries() -> None:
    tmp_path = Path.cwd() / ".pytest_local" / f"build_kazan_sitnaya_sample_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    daily_path = tmp_path / "daily.csv"
    bakery_daily_path = tmp_path / "bakery_daily.csv"
    hourly_path = tmp_path / "hourly.csv"
    output_dir = tmp_path / "processed"

    daily_df = pd.DataFrame(
        [
            {
                "date": "2026-05-01",
                "bakery_id": "1",
                "bakery_name": "Bakery 1",
                "city": "Казань",
                "product_id": "101",
                "product_name": "SKU A",
                "category_name": "Выпечка сытная",
                "observed_sales_qty": 5.0,
                "sales_hours_count": 3,
                "sales_present_flag": 1,
                "release_qty": 6.0,
                "release_present_flag": 1,
                "row_quality_score": 0.9,
            },
            {
                "date": "2026-05-02",
                "bakery_id": "1",
                "bakery_name": "Bakery 1",
                "city": "Казань",
                "product_id": "101",
                "product_name": "SKU A",
                "category_name": "Выпечка сытная",
                "observed_sales_qty": 3.0,
                "sales_hours_count": 2,
                "sales_present_flag": 1,
                "release_qty": 0.0,
                "release_present_flag": 0,
                "row_quality_score": 0.8,
            },
            {
                "date": "2026-05-01",
                "bakery_id": "2",
                "bakery_name": "Bakery 2",
                "city": "Казань",
                "product_id": "102",
                "product_name": "SKU B",
                "category_name": "Выпечка сытная",
                "observed_sales_qty": 1.0,
                "sales_hours_count": 1,
                "sales_present_flag": 1,
                "release_qty": 0.0,
                "release_present_flag": 0,
                "row_quality_score": 0.7,
            },
            {
                "date": "2026-05-01",
                "bakery_id": "3",
                "bakery_name": "Bakery 3",
                "city": "Самара",
                "product_id": "103",
                "product_name": "SKU C",
                "category_name": "Выпечка сытная",
                "observed_sales_qty": 9.0,
                "sales_hours_count": 4,
                "sales_present_flag": 1,
                "release_qty": 8.0,
                "release_present_flag": 1,
                "row_quality_score": 1.0,
            },
            {
                "date": "2026-05-01",
                "bakery_id": "4",
                "bakery_name": "Bakery 4",
                "city": "Казань",
                "product_id": "104",
                "product_name": "SKU D",
                "category_name": "Хлеб",
                "observed_sales_qty": 10.0,
                "sales_hours_count": 5,
                "sales_present_flag": 1,
                "release_qty": 10.0,
                "release_present_flag": 1,
                "row_quality_score": 1.0,
            },
        ]
    )
    daily_df.to_csv(daily_path, index=False, encoding="utf-8-sig")

    bakery_daily_df = pd.DataFrame(
        [
            {
                "date": "2026-05-01",
                "bakery_id": "1",
                "bakery_name": "Bakery 1",
                "city": "Казань",
                "bakery_sales": 20.0,
                "avg_price": 100.0,
            },
            {
                "date": "2026-05-02",
                "bakery_id": "1",
                "bakery_name": "Bakery 1",
                "city": "Казань",
                "bakery_sales": 12.0,
                "avg_price": 90.0,
            },
            {
                "date": "2026-05-01",
                "bakery_id": "2",
                "bakery_name": "Bakery 2",
                "city": "Казань",
                "bakery_sales": 5.0,
                "avg_price": 80.0,
            },
            {
                "date": "2026-05-01",
                "bakery_id": "3",
                "bakery_name": "Bakery 3",
                "city": "Самара",
                "bakery_sales": 30.0,
                "avg_price": 110.0,
            },
        ]
    )
    bakery_daily_df.to_csv(bakery_daily_path, index=False, encoding="utf-8-sig")

    hourly_df = pd.DataFrame(
        [
            {
                "date": "2026-05-01",
                "dow": 4,
                "hour": 9,
                "bakery_id": "1",
                "bakery_name": "Bakery 1",
                "product_id": "101",
                "product_name": "SKU A",
                "category_name": "Выпечка сытная",
                "sku_hour_sales": 2.0,
                "bakery_hour_sales": 10.0,
                "sku_share_in_hour": 0.2,
            },
            {
                "date": "2026-05-01",
                "dow": 4,
                "hour": 10,
                "bakery_id": "2",
                "bakery_name": "Bakery 2",
                "product_id": "102",
                "product_name": "SKU B",
                "category_name": "Выпечка сытная",
                "sku_hour_sales": 1.0,
                "bakery_hour_sales": 3.0,
                "sku_share_in_hour": 0.333333,
            },
            {
                "date": "2026-05-01",
                "dow": 4,
                "hour": 11,
                "bakery_id": "3",
                "bakery_name": "Bakery 3",
                "product_id": "103",
                "product_name": "SKU C",
                "category_name": "Выпечка сытная",
                "sku_hour_sales": 4.0,
                "bakery_hour_sales": 11.0,
                "sku_share_in_hour": 0.363636,
            },
        ]
    )
    hourly_df.to_csv(hourly_path, index=False, encoding="utf-8-sig")

    try:
        paths = MODULE.build_kazan_sitnaya_sample(
            daily_path=daily_path,
            bakery_daily_path=bakery_daily_path,
            hourly_path=hourly_path,
            output_dir=output_dir,
            top_n_bakeries=1,
            chunk_size=2,
        )

        daily_sample = pd.read_csv(paths["daily_sample"], encoding="utf-8-sig")
        bakery_daily_sample = pd.read_csv(paths["bakery_daily_sample"], encoding="utf-8-sig")
        bakery_category_daily_sample = pd.read_csv(paths["bakery_category_daily_sample"], encoding="utf-8-sig")
        hourly_sample = pd.read_csv(paths["hourly_sample"], encoding="utf-8-sig")
        bakery_selection = pd.read_csv(paths["bakery_selection"], encoding="utf-8-sig")
        summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))

        assert daily_sample["bakery_id"].astype(str).nunique() == 1
        assert set(daily_sample["bakery_id"].astype(str)) == {"1"}
        assert daily_sample["city"].eq("Казань").all()
        assert daily_sample["category_name"].eq("Выпечка сытная").all()
        assert "bakery_total_sales_qty" in daily_sample.columns
        assert daily_sample["bakery_total_sales_qty"].tolist() == [20.0, 12.0]
        assert daily_sample["sku_sales_share_in_bakery_total"].round(6).tolist() == [0.25, 0.25]

        assert bakery_daily_sample["bakery_id"].astype(str).nunique() == 1
        assert set(bakery_daily_sample["bakery_id"].astype(str)) == {"1"}
        assert bakery_daily_sample["city"].eq("Казань").all()
        assert len(bakery_category_daily_sample) == 2
        assert bakery_category_daily_sample["category_sales_qty"].tolist() == [5.0, 3.0]
        assert bakery_category_daily_sample["category_share_in_bakery_total"].round(6).tolist() == [0.25, 0.25]

        assert hourly_sample["bakery_id"].astype(str).nunique() == 1
        assert set(hourly_sample["bakery_id"].astype(str)) == {"1"}
        assert hourly_sample["category_name"].eq("Выпечка сытная").all()

        assert len(bakery_selection) == 1
        assert str(bakery_selection.iloc[0]["bakery_id"]) == "1"
        assert summary["selected_bakeries"] == 1
        assert summary["candidate_bakeries_in_scope"] == 2
        assert summary["bakery_daily_rows"] == 2
        assert summary["bakery_category_daily_rows"] == 2
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

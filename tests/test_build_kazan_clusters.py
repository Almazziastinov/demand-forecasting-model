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
    / "build_kazan_clusters.py"
)
SPEC = importlib.util.spec_from_file_location("build_kazan_clusters", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_build_kazan_clusters_outputs_assignments_and_metrics() -> None:
    tmp_path = Path.cwd() / ".pytest_local" / f"build_kazan_clusters_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    bakery_profile_path = tmp_path / "bakery_profile.csv"
    sku_profile_path = tmp_path / "sku_profile.csv"
    output_dir = tmp_path / "processed"

    bakery_profile = pd.DataFrame(
        [
            {"bakery_id": "1", "bakery_name": "B1", "city": "Казань", "mean_bakery_sales": 2000, "cv_bakery_sales": 0.15, "weekday_profile_stability": 0.95, "weekly_amplitude_cv": 0.12, "trend_slope_ratio": 0.01, "category_share_mean": 0.50, "category_share_std": 0.05, "active_sku_mean": 20},
            {"bakery_id": "2", "bakery_name": "B2", "city": "Казань", "mean_bakery_sales": 1900, "cv_bakery_sales": 0.17, "weekday_profile_stability": 0.94, "weekly_amplitude_cv": 0.13, "trend_slope_ratio": 0.02, "category_share_mean": 0.48, "category_share_std": 0.06, "active_sku_mean": 19},
            {"bakery_id": "3", "bakery_name": "B3", "city": "Казань", "mean_bakery_sales": 700, "cv_bakery_sales": 0.40, "weekday_profile_stability": 0.80, "weekly_amplitude_cv": 0.35, "trend_slope_ratio": -0.03, "category_share_mean": 0.25, "category_share_std": 0.11, "active_sku_mean": 8},
            {"bakery_id": "4", "bakery_name": "B4", "city": "Казань", "mean_bakery_sales": 650, "cv_bakery_sales": 0.45, "weekday_profile_stability": 0.78, "weekly_amplitude_cv": 0.38, "trend_slope_ratio": -0.02, "category_share_mean": 0.22, "category_share_std": 0.10, "active_sku_mean": 7},
        ]
    )
    bakery_profile.to_csv(bakery_profile_path, index=False, encoding="utf-8-sig")

    sku_profile = pd.DataFrame(
        [
            {"bakery_id": "1", "bakery_name": "B1", "city": "Казань", "product_id": "101", "product_name": "S1", "category_name": "Выпечка сытная", "mean_sales": 80, "cv_sales": 0.20, "zero_share": 0.0, "weekday_profile_stability": 0.90, "weekly_amplitude_cv": 0.15, "bakery_total_sales_corr": 0.80, "category_total_sales_corr": 0.85, "sku_share_in_bakery_total_mean": 0.06, "hour_profile_stability": 0.70, "active_hours_mean": 6, "release_present_share": 0.90},
            {"bakery_id": "2", "bakery_name": "B2", "city": "Казань", "product_id": "102", "product_name": "S2", "category_name": "Выпечка сытная", "mean_sales": 75, "cv_sales": 0.25, "zero_share": 0.0, "weekday_profile_stability": 0.88, "weekly_amplitude_cv": 0.18, "bakery_total_sales_corr": 0.78, "category_total_sales_corr": 0.82, "sku_share_in_bakery_total_mean": 0.05, "hour_profile_stability": 0.68, "active_hours_mean": 6, "release_present_share": 0.92},
            {"bakery_id": "3", "bakery_name": "B3", "city": "Казань", "product_id": "103", "product_name": "S3", "category_name": "Выпечка сытная", "mean_sales": 8, "cv_sales": 1.20, "zero_share": 0.55, "weekday_profile_stability": 0.40, "weekly_amplitude_cv": 0.60, "bakery_total_sales_corr": 0.25, "category_total_sales_corr": 0.20, "sku_share_in_bakery_total_mean": 0.01, "hour_profile_stability": 0.20, "active_hours_mean": 2, "release_present_share": 0.30},
            {"bakery_id": "4", "bakery_name": "B4", "city": "Казань", "product_id": "104", "product_name": "S4", "category_name": "Выпечка сытная", "mean_sales": 7, "cv_sales": 1.30, "zero_share": 0.60, "weekday_profile_stability": 0.35, "weekly_amplitude_cv": 0.65, "bakery_total_sales_corr": 0.20, "category_total_sales_corr": 0.18, "sku_share_in_bakery_total_mean": 0.01, "hour_profile_stability": 0.18, "active_hours_mean": 2, "release_present_share": 0.28},
        ]
    )
    sku_profile.to_csv(sku_profile_path, index=False, encoding="utf-8-sig")

    try:
        paths = MODULE.build_kazan_clusters(
            bakery_profile_path=bakery_profile_path,
            sku_profile_path=sku_profile_path,
            output_dir=output_dir,
            bakery_k_candidates=[2, 3],
            sku_k_candidates=[2, 3],
        )

        bakery_assignments = pd.read_csv(paths["bakery_assignments"], encoding="utf-8-sig")
        sku_assignments = pd.read_csv(paths["sku_assignments"], encoding="utf-8-sig")
        metrics = json.loads(Path(paths["metrics"]).read_text(encoding="utf-8"))

        assert "bakery_cluster" in bakery_assignments.columns
        assert "sku_cluster" in sku_assignments.columns
        assert bakery_assignments["bakery_cluster"].nunique() >= 2
        assert sku_assignments["sku_cluster"].nunique() >= 2
        assert metrics["bakery_clusters"]["selected_k"] >= 2
        assert metrics["sku_clusters"]["selected_k"] >= 2
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

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
    / "build_kazan_temporal_normative_sku_day.py"
)
SPEC = importlib.util.spec_from_file_location("build_kazan_temporal_normative_sku_day", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_build_kazan_temporal_normative_sku_day_outputs_temporal_series() -> None:
    tmp_path = Path.cwd() / ".pytest_local" / f"build_kazan_temporal_normative_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    reconstructed_path = tmp_path / "reconstructed.csv"
    output_dir = tmp_path / "processed"

    dates = pd.date_range("2026-05-01", periods=14, freq="D")
    reconstructed = pd.DataFrame(
        {
            "date": dates,
            "bakery_id": ["1"] * 14,
            "bakery_name": ["B1"] * 14,
            "city": ["Казань"] * 14,
            "category_name": ["Выпечка сытная"] * 14,
            "product_id": ["101"] * 14,
            "product_name": ["S1"] * 14,
            "observed_sales_qty": [10, 12, 11, 13, 12, 8, 7, 11, 13, 12, 14, 13, 9, 8],
            "reconstructed_sales_qty": [10, 12, 11, 13, 12, 8, 7, 10, 12, 11, 13, 12, 8, 7],
            "best_decomposition_path": ["bakery_category_to_sku_cluster_to_sku"] * 14,
        }
    )
    reconstructed.to_csv(reconstructed_path, index=False, encoding="utf-8-sig")

    try:
        paths = MODULE.build_kazan_temporal_normative_sku_day(
            reconstructed_path=reconstructed_path,
            output_dir=output_dir,
            recent_weeks=4,
            ewma_alpha=0.5,
        )

        temporal = pd.read_csv(paths["temporal_normative"], encoding="utf-8-sig")
        summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))

        assert len(temporal) == 14
        assert "temporal_normative_qty" in temporal.columns
        assert temporal["temporal_normative_qty"].notna().all()
        assert summary["rows"] == 14
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

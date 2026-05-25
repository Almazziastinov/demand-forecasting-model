import importlib.util
from pathlib import Path
import shutil
import uuid

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "analysis"
    / "plot_kazan_reconstruction.py"
)
SPEC = importlib.util.spec_from_file_location("plot_kazan_reconstruction", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_build_plots_generates_png_for_specific_pair() -> None:
    tmp_path = Path.cwd() / ".pytest_local" / f"plot_kazan_reconstruction_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    input_path = tmp_path / "reconstructed.csv"
    output_dir = tmp_path / "plots"

    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-05-01", periods=5, freq="D"),
            "bakery_id": ["1"] * 5,
            "bakery_name": ["B1"] * 5,
            "product_id": ["101"] * 5,
            "product_name": ["S1"] * 5,
            "best_decomposition_path": ["bakery_category_to_sku_cluster_to_sku"] * 5,
            "observed_sales_qty": [10, 11, 9, 12, 10],
            "reconstructed_sales_qty": [9, 10, 10, 11, 10],
            "reconstruction_abs_gap": [1, 1, 1, 1, 0],
        }
    )
    df.to_csv(input_path, index=False, encoding="utf-8-sig")

    try:
        generated = MODULE.build_plots(
            input_path=input_path,
            output_dir=output_dir,
            bakery_id="1",
            product_id="101",
            examples_per_path=1,
        )
        assert len(generated) == 1
        assert generated[0].exists()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

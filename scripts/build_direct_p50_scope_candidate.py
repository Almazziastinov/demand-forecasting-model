"""Build a local Direct artifact candidate with versioned-scope P50 factors."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "models/direct_alpha_025_v1"
DEFAULT_FACTORS = (
    ROOT
    / "reports/versioned_direct_scope_p50_loss_0.75_august_20260909/p50_factors.csv"
)
DEFAULT_OUTPUT = ROOT / "models/direct_alpha_025_p50_scope075_v2"
MODEL_FILES = (
    "direct_model.joblib",
    "stockout_classifier.joblib",
    "lost_severity_model.joblib",
    "floor_history.csv.gz",
    "floor_history.parquet",
)


def build_candidate(base: Path, factors_path: Path, output: Path) -> dict[str, object]:
    metadata = json.loads((base / "metadata.json").read_text(encoding="utf-8"))
    factors = pd.read_csv(factors_path)
    required = {"bakery_id", "p50_factor_new"}
    missing = required.difference(factors.columns)
    if missing:
        raise ValueError(f"P50 factor input is missing columns: {sorted(missing)}")
    selected = factors.dropna(subset=["bakery_id", "p50_factor_new"])
    by_bakery = selected.groupby("bakery_id")["p50_factor_new"].median()
    if by_bakery.empty:
        raise ValueError("P50 factor input produced no bakery factors")

    output.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        shutil.copy2(base / name, output / name)
    metadata.update(
        {
            "version": "direct_alpha_025_p50_scope075_v2",
            "p50_factors": {
                str(int(bakery_id)): float(value)
                for bakery_id, value in by_bakery.items()
            },
            "p50_fallback": float(selected["p50_factor_new"].median()),
            "p50_target_scope": "versioned_direct_sku",
            "p50_lost_demand_weight": 0.75,
            "p50_selection_period": "2026-07-27..2026-08-23",
            "p50_validation_period": "2026-09-01..2026-09-08",
            "production_write": False,
        }
    )
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "version": metadata["version"],
        "bakery_factors": int(len(by_bakery)),
        "p50_fallback": metadata["p50_fallback"],
        "output": str(output),
        "production_write": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--factors", type=Path, default=DEFAULT_FACTORS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build_candidate(args.base, args.factors, args.output), indent=2))


if __name__ == "__main__":
    main()

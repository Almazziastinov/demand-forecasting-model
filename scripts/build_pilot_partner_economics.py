"""Precompute guarded partner economics for the pilot management dev report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pilot_economics import simulate_partner_economics  # noqa: E402


DEFAULT_REPORT = ROOT / "reports" / "pilot_management_summary"
DEFAULT_MAPPING = (
    ROOT / "reports" / "markup_price_mapping_20260826" / "mapped_products.csv"
)


def build(report_dir: Path, mapping_path: Path) -> dict:
    detail = pd.read_csv(report_dir / "detail.csv")
    mapping = pd.read_csv(mapping_path, encoding="utf-8-sig")
    if "valid_economics" in mapping.columns:
        mapping = mapping[mapping["valid_economics"].fillna(False).astype(bool)]
    mapping = (
        mapping[["product_id", "unit_price", "unit_cost"]]
        .sort_values("unit_price")
        .drop_duplicates("product_id", keep="last")
    )
    detail = detail.merge(mapping, on="product_id", how="left", validate="many_to_one")
    rows, coverage = simulate_partner_economics(detail)
    rows.to_csv(report_dir / "economics_daily.csv", index=False, encoding="utf-8-sig")
    mapping.to_csv(
        report_dir / "economics_mapping.csv", index=False, encoding="utf-8-sig"
    )
    metadata = {
        **coverage,
        "method": "two_day_fifo_v1",
        "yesterday_discount": 0.30,
        "mapping_source": str(mapping_path),
    }
    (report_dir / "economics_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    args = parser.parse_args()
    print(
        json.dumps(build(args.report_dir, args.mapping), ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()

"""Build a snapshot-free research panel with exact 56-calendar-day production scope."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_tournament.scope import SCOPE_KEYS, calendar_prior_release_56  # noqa: E402


SOURCE_COLUMNS = [
    *SCOPE_KEYS,
    "product_name",
    "category_name",
    "observed_sales_qty",
    "demand",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_panel(source: pd.DataFrame, flows: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(SOURCE_COLUMNS) - set(source.columns))
    if missing:
        raise ValueError(f"Source panel is missing columns: {missing}")
    work = source[SOURCE_COLUMNS].copy()
    work["date"] = pd.to_datetime(work["date"], errors="raise").dt.normalize()
    if work.duplicated(SCOPE_KEYS).any():
        raise ValueError("Source panel has duplicate SKU-day keys")
    history = calendar_prior_release_56(flows, targets=work)
    work = work.merge(history, on=SCOPE_KEYS, how="left", validate="one_to_one")
    work = work[work["prior_release_56_calendar"].gt(0)].copy()
    work["scope_source"] = "calendar_prior_production_56d_no_snapshot"
    return work.sort_values(SCOPE_KEYS).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--flows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = pd.read_parquet(args.source, columns=SOURCE_COLUMNS)
    flows = pd.read_parquet(args.flows, columns=[*SCOPE_KEYS, "release_qty"])
    panel = build_panel(source, flows)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    panel.to_parquet(args.output_dir / "panel.parquet", index=False)
    monthly = (
        panel.groupby(panel["date"].dt.to_period("M"))
        .agg(
            rows=("product_id", "size"),
            dates=("date", "nunique"),
            bakeries=("bakery_id", "nunique"),
            products=("product_id", "nunique"),
            sales=("observed_sales_qty", "sum"),
        )
        .reset_index()
    )
    monthly["date"] = monthly["date"].astype(str)
    monthly.to_csv(args.output_dir / "monthly_scope.csv", index=False)
    metadata = {
        "production_write": False,
        "scope": (
            "Source's prior-activity universe intersected with exact "
            "D-56..D-1 positive production; no snapshots"
        ),
        "source_rows": int(len(source)),
        "panel_rows": int(len(panel)),
        "source_path": str(args.source.resolve()),
        "source_sha256": _sha256(args.source),
        "flows_path": str(args.flows.resolve()),
        "flows_sha256": _sha256(args.flows),
        "point_in_time_verified": False,
        "remaining_risks": [
            "Source's activity-based SKU universe and upstream fact "
            "arrival times are not independently verified",
            "Demand labels are reconstructed after the target date",
            "Source prior-activity rule may omit pairs included in the "
            "actual production assortment",
        ],
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(monthly.to_string(index=False))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

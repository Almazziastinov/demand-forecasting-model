"""Audit a frozen produced-pair panel against exact calendar-time history."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_tournament.scope import (  # noqa: E402
    SCOPE_KEYS,
    calendar_prior_release_56,
    snapshot_timing_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--flows", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def audit_scope(
    predictions: pd.DataFrame, flows: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {*SCOPE_KEYS, "scope_source"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction panel is missing columns: {missing}")
    rows = predictions[[*SCOPE_KEYS, "scope_source"]].copy()
    rows["date"] = pd.to_datetime(rows["date"], errors="raise").dt.normalize()
    if rows.duplicated(SCOPE_KEYS).any():
        raise ValueError("Prediction panel contains duplicate SKU-day keys")
    flows = flows.copy()
    flows["date"] = pd.to_datetime(flows["date"], errors="raise").dt.normalize()
    history = calendar_prior_release_56(flows, targets=rows)
    rows = rows.merge(
        history, on=SCOPE_KEYS, how="left", validate="one_to_one", indicator=True
    )
    probe_indices = np.linspace(
        0, len(rows) - 1, num=min(25, len(rows)), dtype=int
    )
    for index in probe_indices:
        probe = rows.iloc[index]
        direct = flows[
            flows["bakery_id"].eq(probe["bakery_id"])
            & flows["product_id"].eq(probe["product_id"])
            & flows["date"].between(
                probe["date"] - pd.Timedelta(days=56),
                probe["date"] - pd.Timedelta(days=1),
            )
        ]["release_qty"].sum()
        calculated = probe["prior_release_56_calendar"]
        if not np.isclose(0.0 if pd.isna(calculated) else calculated, direct):
            raise ValueError("Calendar rolling result failed direct spot-check")
    rows["month"] = rows["date"].dt.to_period("M").astype(str)
    rows["calendar_scope_valid"] = rows["prior_release_56_calendar"].gt(0)
    summary = (
        rows.groupby(["month", "scope_source"], dropna=False)
        .agg(
            rows=("calendar_scope_valid", "size"),
            missing_flow_key=("_merge", lambda value: int(value.eq("left_only").sum())),
            valid_calendar_56=("calendar_scope_valid", "sum"),
        )
        .reset_index()
    )
    summary["invalid_calendar_56"] = summary["rows"] - summary["valid_calendar_56"]
    metadata = {
        "production_write": False,
        "source_rows": int(len(rows)),
        "invalid_calendar_56": int((~rows["calendar_scope_valid"]).sum()),
        "missing_flow_key": int(rows["_merge"].eq("left_only").sum()),
        "independent_spot_checks": int(len(probe_indices)),
        "scope_point_in_time_verified": False,
        "reason": (
            "Calendar release check only; archived snapshot timing and "
            "upstream source provenance remain unaudited."
        ),
    }
    return summary, metadata


def main() -> None:
    args = parse_args()
    predictions = pd.read_parquet(args.predictions)
    flows = pd.read_parquet(args.flows, columns=[*SCOPE_KEYS, "release_qty"])
    summary, metadata = audit_scope(predictions, flows)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary.to_csv(args.output_dir / "calendar_56_scope_audit.csv", index=False)
    if args.snapshots is not None:
        snapshots = pd.read_parquet(
            args.snapshots,
            columns=["forecast_date", "bakery_id", "product_id", "generated_at"],
        )
        timing = snapshot_timing_audit(predictions, snapshots)
        timing.to_csv(args.output_dir / "snapshot_timing_audit.csv", index=False)
        metadata["snapshot_timing"] = timing.to_dict(orient="records")
        print(timing.to_string(index=False))
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

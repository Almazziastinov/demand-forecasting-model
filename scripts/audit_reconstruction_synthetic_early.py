"""Pseudo-stockout audit of the frozen restoration rule on early data only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.calibrate_post_last_sale_demand import build_cases  # noqa: E402
from scripts.run_chronos2_fixed_origin import _sha256  # noqa: E402


KEYS = ["date", "bakery_id", "product_id"]
START = pd.Timestamp("2026-06-01")
END = pd.Timestamp("2026-07-05")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hourly", type=Path, required=True)
    parser.add_argument("--target-panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    hourly = pd.read_parquet(args.hourly)
    hourly = hourly.loc[hourly["date"].between(START, END)]
    cases = build_cases(hourly, [12, 15, 18])
    columns = [
        *KEYS, "observed_sales_qty", "reference_sales", "reference_days",
        "reference_last_sale_hour", "bakery_last_sale_hour",
        "broad_stockout_signal", "release_qty", "incoming_move_qty",
        "outgoing_move_qty", "written_off_qty",
    ]
    panel = pd.read_parquet(args.target_panel, columns=columns)
    cases = cases.merge(panel, on=KEYS, how="inner", validate="many_to_one")
    raw_residual = (
        cases["release_qty"] + cases["incoming_move_qty"]
        - cases["outgoing_move_qty"] - cases["observed_sales_qty"]
        - cases["written_off_qty"]
    )
    compatible_sales = (
        cases["observed_sales_qty"] - cases["observed"] - cases["true_hidden"]
    ).abs().le(1.0)
    eligible = (
        compatible_sales & ~cases["broad_stockout_signal"]
        & raw_residual.ge(1.0)
        & cases["true_hidden"].gt(0.0)
    )
    cases = cases.loc[eligible].copy()
    if cases.empty:
        raise ValueError("No source-compatible pseudo-stockout cases")
    signal = (
        cases["observed"].ge(3.0)
        & cases["reference_days"].ge(3)
        & cases["reference_sales"].gt(1.15 * cases["observed"])
        & cases["reference_last_sale_hour"].ge(cases["cutoff"] + 1.5)
        & cases["bakery_last_sale_hour"].ge(cases["cutoff"] + 2.0)
    )
    cases["identified"] = signal
    cases["predicted_hidden"] = np.minimum.reduce(
        [
            (
                (cases["reference_sales"] - cases["observed"])
                .clip(lower=0).fillna(0).to_numpy()
            ),
            (0.5 * cases["observed"]).to_numpy(),
            np.full(len(cases), 15.0),
        ]
    )
    cases.loc[~signal, "predicted_hidden"] = 0.0
    cases["absolute_error"] = (
        cases["predicted_hidden"] - cases["true_hidden"]
    ).abs()
    records = []
    for cutoff, group in cases.groupby("cutoff"):
        hidden = float(group["true_hidden"].sum())
        predicted = float(group["predicted_hidden"].sum())
        records.append(
            {
                "cutoff_hour": int(cutoff),
                "cases": len(group),
                "identified_cases": int(group["identified"].sum()),
                "identification_recall_pct": 100 * float(group["identified"].mean()),
                "true_hidden_units": hidden,
                "predicted_hidden_units": predicted,
                "aggregate_recovery_pct": 100 * predicted / hidden,
                "reconstruction_wape_pct": (
                    100 * float(group["absolute_error"].sum()) / hidden
                ),
            }
        )
    summary = pd.DataFrame(records)
    args.output_dir.mkdir(parents=True)
    summary.to_csv(args.output_dir / "by_cutoff.csv", index=False)
    metadata = {
        "production_write": False,
        "start": str(START.date()),
        "end": str(END.date()),
        "synthetic_censoring_not_natural_lost_demand": True,
        "eligibility": (
            "Observed sales agree across sources within one unit, full-day "
            "sales continue to at least 21:00, no broad natural stockout "
            "flag, positive flow residual, and positive hidden tail."
        ),
        "hourly_sha256": _sha256(args.hourly),
        "target_panel_sha256": _sha256(args.target_panel),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

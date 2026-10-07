"""Build an auditable research demand target without production writes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_tournament.demand_regularization import (  # noqa: E402
    SOURCE_COLUMNS,
    adjust_demand_panel,
    dense_candidate_panel,
    select_origin_pairs,
)
from src.model_tournament.series_diagnostics import stage_diagnostics  # noqa: E402


TRAIN_ORIGINS = pd.to_datetime(
    ["2026-02-02", "2026-03-02", "2026-03-30", "2026-04-27", "2026-05-25", "2026-06-21"]
).tolist()
EVAL_ORIGINS = pd.to_datetime(
    ["2026-07-06", "2026-07-20", "2026-08-03", "2026-08-17"]
).tolist()
START = pd.Timestamp("2026-01-01")
END = pd.Timestamp("2026-08-31")
TRAIN_CUTOFF = pd.Timestamp("2026-07-05")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.max_pairs is not None and args.max_pairs < 1:
        raise ValueError("max-pairs must be positive")
    flows = pd.read_parquet(args.input, columns=SOURCE_COLUMNS)
    flows = flows.loc[flows["date"].between(START, END)].copy()
    if flows.empty or flows["date"].min() > START or flows["date"].max() < END:
        raise ValueError("Source does not cover the declared historical period")
    pairs = select_origin_pairs(flows, TRAIN_ORIGINS + EVAL_ORIGINS)
    full_pair_count = len(pairs)
    if args.max_pairs is not None:
        pairs = pairs.head(args.max_pairs)
    dense = dense_candidate_panel(flows, pairs, start=START, end=END)
    del flows
    adjusted = adjust_demand_panel(dense)
    del dense
    diagnostics = stage_diagnostics(adjusted, cutoff=TRAIN_CUTOFF)
    diagnostic_summary = diagnostics.groupby("stage", as_index=False).agg(
        series=("bakery_id", "size"),
        median_cv=("cv", "median"),
        median_acf7=("acf7", "median"),
        median_pacf7=("pacf7", "median"),
        median_weekly_seasonality_strength=("weekly_seasonality_strength", "median"),
        median_prior_weekday_r2=("prior_weekday_r2", "median"),
        median_mean_stability=("mean_second_to_first", "median"),
        median_variance_stability=("variance_second_to_first", "median"),
    )
    sales_total = float(adjusted["observed_sales_qty"].sum())
    summary = {
        "research_only": True,
        "production_write": False,
        "target_is_proxy_not_observed_ground_truth": True,
        "source_path": str(args.input.resolve()),
        "source_sha256": _hash(args.input),
        "start": str(START.date()),
        "end": str(END.date()),
        "train_cutoff": str(TRAIN_CUTOFF.date()),
        "train_origins": [str(day.date()) for day in TRAIN_ORIGINS],
        "evaluation_origins": [str(day.date()) for day in EVAL_ORIGINS],
        "scope_pairs_before_limit": full_pair_count,
        "scope_pairs": len(pairs),
        "max_pairs": args.max_pairs,
        "rows": len(adjusted),
        "observed_sales_total": sales_total,
        "restoration_uplift_total": float(adjusted["restoration_uplift_qty"].sum()),
        "outlier_reduction_total": float(adjusted["outlier_reduction_qty"].sum()),
        "reconstructed_demand_total": float(adjusted["reconstructed_demand_qty"].sum()),
        "net_uplift_percent": (
            100 * (adjusted["reconstructed_demand_qty"].sum() / sales_total - 1)
        ),
        "restored_rows": int(adjusted["is_restored"].sum()),
        "reduced_rows": int(adjusted["is_reduced_spike"].sum()),
        "diagnostic_series": int(len(diagnostics) // 3),
        "notes": (
            "Same-weekday references use strictly prior observations; same-day "
            "cross-bakery peer ratio is used only for historical spike labels. "
            "Diagnostics use training dates only. R2 is a descriptive prior-weekday "
            "one-step fit and is not a holdout forecast score."
        ),
    }
    args.output_dir.mkdir(parents=True)
    adjusted.to_parquet(args.output_dir / "panel.parquet", index=False)
    diagnostics.to_parquet(args.output_dir / "series_diagnostics.parquet", index=False)
    diagnostic_summary.to_csv(args.output_dir / "diagnostic_summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(diagnostic_summary.to_string(index=False))


if __name__ == "__main__":
    main()

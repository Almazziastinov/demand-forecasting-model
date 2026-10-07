"""Test target regularization with a frozen weighted-weekday baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_chronos2_fixed_origin import _origins, _sha256, _summary  # noqa: E402
from src.model_tournament.fixed_origin import (  # noqa: E402
    KEYS,
    _same_weekday_formula,
    build_fixed_origin_panel,
    validate_flows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--evaluation-origins", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    origins = _origins(args.evaluation_origins)
    panel = validate_flows(pd.read_parquet(args.input))
    detail_parts = []
    for origin in origins:
        rows = build_fixed_origin_panel(panel, origin)
        history = panel.loc[
            panel["date"].between(origin - pd.Timedelta(days=55), origin)
        ]
        demand_history = history[
            ["date", *KEYS, "reconstructed_demand_qty"]
        ].rename(columns={"reconstructed_demand_qty": "observed_sales_qty"})
        predictions = {
            "weighted_weekday_raw_sales": rows["weighted_weekday_sales"].to_numpy(),
            "weighted_weekday_reconstructed": _same_weekday_formula(
                rows, demand_history
            ),
        }
        for name, values in predictions.items():
            columns = ["forecast_origin", "date", "lead_days", *KEYS, "actual"]
            part = rows[columns].copy()
            part["model"] = name
            part["prediction"] = values
            detail_parts.append(part)
    detail = pd.concat(detail_parts, ignore_index=True)
    truth = panel[["date", *KEYS, "broad_stockout_signal"]]
    detail = detail.merge(truth, on=["date", *KEYS], how="left", validate="many_to_one")
    if detail["broad_stockout_signal"].isna().any():
        raise ValueError("Evaluation days lack stockout flags")
    args.output_dir.mkdir(parents=True)
    detail.to_parquet(args.output_dir / "detail.parquet", index=False)
    leaderboard = _summary(detail, ["model"])
    leaderboard.to_csv(args.output_dir / "leaderboard.csv", index=False)
    _summary(detail, ["forecast_origin", "model"]).to_csv(
        args.output_dir / "by_origin.csv", index=False
    )
    _summary(detail.loc[~detail["broad_stockout_signal"]], ["model"]).to_csv(
        args.output_dir / "clean_subset.csv", index=False
    )
    _summary(detail, ["date", "model"]).to_csv(
        args.output_dir / "by_day.csv", index=False
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "production_write": False,
                "primary_target": "original_observed_sales_qty",
                "comparison": (
                    "identical frozen SKU-days, raw versus reconstructed history"
                ),
                "evaluation_origins": [str(origin.date()) for origin in origins],
                "input_sha256": _sha256(args.input),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()

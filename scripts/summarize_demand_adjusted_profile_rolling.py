"""Aggregate guarded demand-adjusted profile metrics across time cutoffs."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


BASELINE = "observed_sales_profile"
VARIANT = "demand_adjusted_guarded_routing"
DEFAULT_SCOPES = [
    "all_holdout",
    "clean_bakery_days",
    "clean_sku_days",
    "adjusted_pairs_clean_sku_days",
    "new_tier1_member_clean_sku_days",
]


def collect_cutoff_metrics(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in paths:
        metrics = pd.read_csv(path / "metrics.csv")
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        lookup = metrics.set_index(["variant", "scope"])
        for scope in DEFAULT_SCOPES:
            if (BASELINE, scope) not in lookup.index or (VARIANT, scope) not in lookup.index:
                continue
            base = lookup.loc[(BASELINE, scope)]
            variant = lookup.loc[(VARIANT, scope)]
            rows.append(
                {
                    "cutoff": summary["train_end"],
                    "scope": scope,
                    "baseline_wape": base["wape"],
                    "variant_wape": variant["wape"],
                    "wape_delta": variant["wape"] - base["wape"],
                    "baseline_sku_day_wape": base["sku_day_wape"],
                    "variant_sku_day_wape": variant["sku_day_wape"],
                    "sku_day_wape_delta": variant["sku_day_wape"] - base["sku_day_wape"],
                    "underforecast_qty_delta": variant["underforecast_qty"] - base["underforecast_qty"],
                    "overforecast_qty_delta": variant["overforecast_qty"] - base["overforecast_qty"],
                }
            )
    return pd.DataFrame(rows).sort_values(["cutoff", "scope"]).reset_index(drop=True)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("scope", as_index=False)
        .agg(
            cutoffs=("cutoff", "nunique"),
            wape_wins=("wape_delta", lambda value: int(value.lt(0).sum())),
            mean_wape_delta=("wape_delta", "mean"),
            sku_day_wape_wins=("sku_day_wape_delta", lambda value: int(value.lt(0).sum())),
            mean_sku_day_wape_delta=("sku_day_wape_delta", "mean"),
            mean_underforecast_qty_delta=("underforecast_qty_delta", "mean"),
            mean_overforecast_qty_delta=("overforecast_qty_delta", "mean"),
        )
        .sort_values("scope")
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    detail = collect_cutoff_metrics(args.inputs)
    aggregate = summarize(detail)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / "cutoff_metrics.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(args.output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "summary.json").write_text(
        json.dumps(aggregate.to_dict("records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()

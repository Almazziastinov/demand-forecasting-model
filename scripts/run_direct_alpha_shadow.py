"""Build the frozen Direct alpha=.25 candidate as local shadow artifacts.

The runner intentionally has no database client, loader, activation command or
production table arguments. Its input must already contain Direct/floor causal
features for the requested forecast horizon.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments_v2.direct_alpha_allocation import (  # noqa: E402
    DirectAlphaAllocationConfig,
    build_selected_direct_plan,
)


DAY = ["date", "bakery_id"]


def concentration(rows: pd.DataFrame, column: str) -> dict[str, float | int]:
    totals = rows.groupby(DAY)[column].transform("sum")
    shares = rows[column] / totals.replace(0.0, pd.NA)
    top = shares.groupby([rows[key] for key in DAY]).max()
    return {
        "top_share_max": float(top.max()),
        "bakery_days_ge20": int(top.ge(0.20).sum()),
        "bakery_days_ge30": int(top.ge(0.30).sum()),
        "bakery_days_ge40": int(top.ge(0.40).sum()),
    }


def run_shadow(input_path: Path, output_dir: Path) -> dict:
    rows = pd.read_parquet(input_path)
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    result = build_selected_direct_plan(rows, DirectAlphaAllocationConfig())
    history_mass = result.groupby(DAY)["broad_56_mean"].transform("sum")
    result["cold_start_fallback"] = history_mass.le(0.0)
    if "incumbent_sku_forecast" in result.columns:
        result.loc[result["cold_start_fallback"], "selected_sku_forecast"] = result.loc[
            result["cold_start_fallback"], "incumbent_sku_forecast"
        ].clip(lower=0.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_dir / "shadow_rows.parquet", index=False)
    result[["date", "bakery_id", "product_id", "selected_sku_forecast"]].rename(
        columns={"selected_sku_forecast": "forecast_qty"}
    ).to_csv(output_dir / "shadow_sku_day.csv", index=False, encoding="utf-8-sig")

    base_total = float(result["direct_p50"].sum())
    selected_total = float(result["selected_sku_forecast"].sum())
    summary = {
        "contract": "direct_alpha_025_floor_tail_v1",
        "scope": {
            "dates": int(result["date"].nunique()),
            "bakeries": int(result["bakery_id"].nunique()),
            "products": int(result["product_id"].nunique()),
            "rows": int(len(result)),
        },
        "volume": {
            "direct_p50": base_total,
            "selected": selected_total,
            "delta": selected_total - base_total,
        },
        "tail_cap_rows": int(result["tail_cap_applied"].sum()),
        "cold_start_fallback_bakery_days": int(
            result.loc[result["cold_start_fallback"], DAY].drop_duplicates().shape[0]
        ),
        "concentration": concentration(result, "selected_sku_forecast"),
        "database_write": False,
        "activation": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_shadow(args.input, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

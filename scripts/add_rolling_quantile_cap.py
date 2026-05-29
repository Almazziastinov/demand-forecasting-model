"""Add rolling-quantile cap column to existing bakery_daily_sales.csv.

This avoids re-running the full raw rebuild. It loads the processed dataset,
applies `add_rolling_quantile_capped_base_target` with the standard parameters,
and overwrites the file in place.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.experiments_v2.sales_cleaning import add_rolling_quantile_capped_base_target


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "data" / "processed" / "bakery_daily_sales.csv"
DEFAULT_SUMMARY = ROOT / "data" / "processed" / "bakery_daily_sales_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=str(DEFAULT_PATH))
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--window", type=int, default=26)
    parser.add_argument("--min-periods", type=int, default=8)
    parser.add_argument("--lower-quantile", type=float, default=0.05)
    parser.add_argument("--upper-quantile", type=float, default=0.95)
    args = parser.parse_args()

    path = Path(args.path)
    print(f"loading {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    print(f"rows: {len(df)}")

    df = add_rolling_quantile_capped_base_target(
        df,
        value_col="bakery_sales",
        entity_cols=["bakery_id"],
        seasonal_cols=["dow"],
        window=args.window,
        min_periods=args.min_periods,
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile,
        capped_col="bakery_sales_base_rolling_quantile_capped",
    )
    print(
        "rolling_quantile_capped_rows:",
        int(df["rolling_quantile_base_target_capped_flag"].sum()),
    )
    print(
        "mean_bakery_sales_base_rolling_quantile_capped:",
        float(df["bakery_sales_base_rolling_quantile_capped"].mean()),
    )

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"wrote {path}")

    summary_path = Path(args.summary_path)
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["mean_bakery_sales_base_rolling_quantile_capped"] = round(
            float(
                pd.to_numeric(
                    df["bakery_sales_base_rolling_quantile_capped"], errors="coerce"
                ).mean()
            ),
            6,
        )
        summary["rolling_quantile_capped_rows"] = int(
            df["rolling_quantile_base_target_capped_flag"].sum()
        )
        summary["rolling_quantile_window"] = args.window
        summary["rolling_quantile_min_periods"] = args.min_periods
        summary["rolling_quantile_lower"] = args.lower_quantile
        summary["rolling_quantile_upper"] = args.upper_quantile
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"updated {summary_path}")


if __name__ == "__main__":
    main()

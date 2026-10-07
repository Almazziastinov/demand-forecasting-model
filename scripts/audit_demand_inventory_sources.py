"""Audit local pilot stock balance against the frozen broad demand panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_chronos2_fixed_origin import _sha256  # noqa: E402


KEYS = ["date", "bakery_id", "product_id"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-panel", type=Path, required=True)
    parser.add_argument("--pilot-balance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    panel = pd.read_parquet(
        args.target_panel,
        columns=[*KEYS, "observed_sales_qty", "restoration_uplift_qty"],
    )
    pilot = pd.read_csv(
        args.pilot_balance,
        usecols=[
            *KEYS, "qty_sold", "qty_produced", "qty_received", "qty_sent",
            "stock_balance", "hourly_daily_sales_agree",
        ],
    )
    pilot["date"] = pd.to_datetime(pilot["date"]).dt.normalize()
    if panel.duplicated(KEYS).any() or pilot.duplicated(KEYS).any():
        raise ValueError("Source contains duplicate SKU-days")
    arithmetic_balance = (
        pilot["qty_produced"] + pilot["qty_received"]
        - pilot["qty_sent"] - pilot["qty_sold"]
    )
    balance_delta = (pilot["stock_balance"] - arithmetic_balance).abs()
    joined = panel.merge(pilot, on=KEYS, how="inner", validate="one_to_one")
    sales_delta = (joined["observed_sales_qty"] - joined["qty_sold"]).abs()
    compatible = joined.loc[sales_delta.le(1.0)]
    restored = compatible["restoration_uplift_qty"].gt(0)
    result = {
        "production_write": False,
        "pilot_rows": len(pilot),
        "pilot_date_min": str(pilot["date"].min().date()),
        "pilot_date_max": str(pilot["date"].max().date()),
        "pilot_bakeries": int(pilot["bakery_id"].nunique()),
        "pilot_balance_equals_daily_arithmetic_rows": int(
            np.isclose(balance_delta, 0.0, atol=1e-8).sum()
        ),
        "pilot_balance_negative_rows": int(pilot["stock_balance"].lt(0).sum()),
        "matched_sku_days": len(joined),
        "matched_sales_disagree_gt1": int(sales_delta.gt(1.0).sum()),
        "sales_compatible_sku_days": len(compatible),
        "restored_compatible_sku_days": int(restored.sum()),
        "restored_compatible_with_positive_balance_ge3": int(
            (restored & compatible["stock_balance"].ge(3)).sum()
        ),
        "restored_compatible_with_negative_balance_lt_minus1": int(
            (restored & compatible["stock_balance"].lt(-1)).sum()
        ),
        "interpretation": (
            "stock_balance is a same-day arithmetic residual, not an "
            "independently measured physical inventory label"
        ),
        "target_panel_sha256": _sha256(args.target_panel),
        "pilot_balance_sha256": _sha256(args.pilot_balance),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

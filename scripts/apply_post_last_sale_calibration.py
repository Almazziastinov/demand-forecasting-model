"""Apply frozen cutoff-hour coefficients to an existing stockout demand label."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("reports/relaxed_stockout_network_20260826/sku_day_demand.csv"),
    )
    parser.add_argument(
        "--coefficients",
        type=Path,
        default=Path("reports/post_last_sale_calibration_20260826/calibration_coefficients.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/calibrated_stockout_network_20260826/sku_day_demand.csv"),
    )
    args = parser.parse_args()

    demand = pd.read_csv(args.source, encoding="utf-8-sig", low_memory=False)
    coefficients = pd.read_csv(args.coefficients).sort_values("cutoff")
    selected = demand["is_clear_stockout"].fillna(False).astype(bool)
    last_sale = pd.to_numeric(demand["last_sale_hour"], errors="coerce")
    multiplier = np.interp(
        last_sale.fillna(coefficients["cutoff"].min()),
        coefficients["cutoff"],
        coefficients["rate_multiplier"],
    )
    raw_lost = pd.to_numeric(demand["raw_imputed_demand"], errors="coerce").fillna(0.0)
    demand["calibration_multiplier"] = np.where(selected, multiplier, 0.0)
    demand["imputed_demand"] = np.where(selected, raw_lost * multiplier, 0.0)
    observed = pd.to_numeric(demand["demand_lower_bound"], errors="coerce").fillna(0.0)
    demand["demand_point_estimate"] = observed + demand["imputed_demand"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    demand.to_csv(args.output, index=False, encoding="utf-8-sig")
    stockouts = demand[selected]
    print(
        f"rows={len(demand)} stockouts={len(stockouts)} "
        f"old_capped_lost={pd.read_csv(args.source, usecols=['imputed_demand'])['imputed_demand'].sum():.3f} "
        f"calibrated_lost={stockouts['imputed_demand'].sum():.3f}"
    )


if __name__ == "__main__":
    main()

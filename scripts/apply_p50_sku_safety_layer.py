"""Apply causal SKU caps and safe redistribution to comparable P50 forecasts."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.backtest_retrained_p50_comparable import (  # noqa: E402
    add_end_day_lost,
    load_detail,
)


SOURCE = Path(
    os.environ.get(
        "P50_SAFETY_SOURCE",
        ROOT / "reports/retrained_p50_comparable_daily_100_20260910/rows.parquet",
    )
)
OUTPUT = Path(
    os.environ.get(
        "P50_SAFETY_OUTPUT", ROOT / "reports/retrained_p50_sku_safety_20260910"
    )
)
KEYS = ["date", "bakery_id", "product_id"]
MULTIPLIERS = [1.1, 1.2, 1.3, 1.5]
REDISTRIBUTION_WEIGHT = os.environ.get("P50_REDISTRIBUTION_WEIGHT", "lost")


def add_causal_history(rows: pd.DataFrame) -> pd.DataFrame:
    history = add_end_day_lost(load_detail()).sort_values(KEYS)
    grouped = history.groupby(["bakery_id", "product_id"], sort=False)
    history["recent_sales_p75"] = grouped["sold_qty"].transform(
        lambda values: values.shift(1).rolling(14, min_periods=3).quantile(0.75)
    )
    history["recent_lost_sum"] = grouped["lost"].transform(
        lambda values: values.shift(1).rolling(14, min_periods=1).sum()
    )
    return rows.merge(
        history[KEYS + ["recent_sales_p75", "recent_lost_sum"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )


def redistribute_group(group: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    result = group.copy()
    proposed = result["alpha25_tail_capped"].to_numpy(dtype=float)
    direct = result["direct_forecast"].to_numpy(dtype=float)
    recent_p75 = result["recent_sales_p75"].to_numpy(dtype=float)
    lost = result["recent_lost_sum"].fillna(0.0).to_numpy(dtype=float)
    history_cap = np.where(np.isfinite(recent_p75), recent_p75 * multiplier, direct)
    cap = np.maximum(direct, history_cap)
    allocated = np.minimum(proposed, cap)
    remainder = float(proposed.sum() - allocated.sum())
    if REDISTRIBUTION_WEIGHT == "lost":
        eligible = lost > 0
        redistribution_weight = lost
    elif REDISTRIBUTION_WEIGHT == "direct":
        eligible = direct > 0
        redistribution_weight = direct
    else:
        raise ValueError(
            "P50_REDISTRIBUTION_WEIGHT must be either 'lost' or 'direct'"
        )
    for _ in range(10):
        headroom = np.maximum(cap - allocated, 0.0)
        weights = np.where(
            eligible & (headroom > 1e-9), redistribution_weight, 0.0
        )
        if remainder <= 1e-9 or weights.sum() <= 0:
            break
        addition = np.minimum(remainder * weights / weights.sum(), headroom)
        allocated += addition
        remainder -= float(addition.sum())
    result["alpha25_tail_capped"] = allocated
    result["sku_cap"] = cap
    result["unallocated_bakery_total"] = remainder
    return result


def main() -> None:
    rows = pd.read_parquet(SOURCE)
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    rows = add_causal_history(rows)
    rows["_row_id"] = np.arange(len(rows))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    for multiplier in MULTIPLIERS:
        safe = pd.concat(
            [
                redistribute_group(group, multiplier)
                for _, group in rows.groupby(["date", "bakery_id"], sort=False)
            ],
            ignore_index=True,
        )
        safe = safe.sort_values("_row_id").reset_index(drop=True)
        label = str(multiplier).replace(".", "")
        safe.drop(columns="_row_id").to_parquet(
            OUTPUT / f"rows_cap_{label}.parquet", index=False
        )
        diagnostics.append(
            {
                "multiplier": multiplier,
                "production_before": rows["alpha25_tail_capped"].sum(),
                "production_after": safe["alpha25_tail_capped"].sum(),
                "unallocated": safe.groupby(["date", "bakery_id"])[
                    "unallocated_bakery_total"
                ]
                .first()
                .sum(),
                "capped_rows": int(
                    safe["alpha25_tail_capped"]
                    .lt(rows["alpha25_tail_capped"].to_numpy() - 1e-9)
                    .sum()
                ),
            }
        )
    pd.DataFrame(diagnostics).to_csv(OUTPUT / "diagnostics.csv", index=False)
    print(pd.DataFrame(diagnostics).to_string(index=False))


if __name__ == "__main__":
    main()

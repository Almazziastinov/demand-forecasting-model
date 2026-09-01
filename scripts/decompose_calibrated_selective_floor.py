"""Decompose the selected calibrated SKU floor into useful and surplus uplift."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.search_calibrated_selective_sku_floor import add_causal_reference

ROWS = ROOT / "reports/calibrated_quantile_operational_balance_20260826/rows.parquet"
LABELS = ROOT / "reports/calibrated_stockout_network_20260826/sku_day_demand.csv"
OUTPUT = ROOT / "reports/calibrated_selective_floor_decomposition_20260826"
KEYS = ["date", "bakery_id", "product_id"]


def summarize(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    result = frame.groupby(keys, as_index=False, dropna=False).agg(
        rows=("product_id", "size"),
        added=("floor_added", "sum"),
        useful=("useful_added", "sum"),
        surplus_added=("surplus_added", "sum"),
        base_under=("base_under", "sum"),
        final_under=("final_under", "sum"),
    )
    result["efficiency"] = result["useful"] / result["added"].replace(0.0, np.nan)
    return result.sort_values("added", ascending=False)


def main() -> None:
    rows = pd.read_parquet(ROWS)
    labels = pd.read_csv(
        LABELS,
        usecols=[*KEYS, "demand_point_estimate"],
        encoding="utf-8-sig",
    ).rename(columns={"demand_point_estimate": "history_demand"})
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    history = labels.rename(columns={"history_demand": "demand"})
    rows = add_causal_reference(rows, history)

    rows["base_plan"] = rows["predictive_forecast"] * rows["p50_new"]
    eligible = rows["history_n"] >= 6
    floor = 0.95 * rows["history_p67"]
    rows["center_plan"] = np.where(
        eligible,
        np.maximum(rows["base_plan"], np.minimum(floor, rows["base_plan"] + 8.0)),
        rows["base_plan"],
    )
    rows["floor_added"] = rows["center_plan"] - rows["base_plan"]
    rows["base_under"] = (rows["demand"] - rows["base_plan"]).clip(lower=0.0)
    rows["final_under"] = (rows["demand"] - rows["center_plan"]).clip(lower=0.0)
    rows["useful_added"] = rows["base_under"] - rows["final_under"]
    rows["surplus_added"] = rows["floor_added"] - rows["useful_added"]
    rows["history_band"] = pd.cut(
        rows["history_n"],
        bins=[-1, 0, 2, 5, 7, np.inf],
        labels=["0", "1-2", "3-5", "6-7", "8+"],
    ).astype(str)
    dates = sorted(rows["date"].unique())
    rows["split"] = np.where(rows["date"].isin(dates[:4]), "calibration", "test")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(OUTPUT / "rows.parquet", index=False)
    for name, keys in {
        "by_split": ["split"],
        "by_date": ["date"],
        "by_history_band": ["history_band"],
        "by_category": ["category"],
        "by_bakery": ["bakery_id"],
        "by_product": ["product_id"],
        "by_split_category": ["split", "category"],
    }.items():
        summarize(rows, keys).to_csv(OUTPUT / f"{name}.csv", index=False, encoding="utf-8-sig")

    selected = rows[rows["floor_added"] > 0]
    payload = {
        "rows": int(len(rows)),
        "uplifted_rows": int(len(selected)),
        "added": float(rows["floor_added"].sum()),
        "useful": float(rows["useful_added"].sum()),
        "surplus_added": float(rows["surplus_added"].sum()),
        "efficiency": float(rows["useful_added"].sum() / rows["floor_added"].sum()),
        "base_under": float(rows["base_under"].sum()),
        "final_under": float(rows["final_under"].sum()),
    }
    print(pd.Series(payload).to_string())
    print("\nBy split")
    print(summarize(rows, ["split"]).to_string(index=False))
    print("\nBy history")
    print(summarize(rows, ["history_band"]).to_string(index=False))
    print("\nProducts with largest surplus addition")
    print(summarize(rows, ["product_id"]).sort_values("surplus_added", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()

"""Search a causal same-weekday SKU floor on calibrated lost-demand labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "reports/calibrated_quantile_operational_balance_20260826/rows.parquet"
LABELS = ROOT / "reports/calibrated_stockout_network_20260826/sku_day_demand.csv"
OUTPUT = ROOT / "reports/calibrated_selective_sku_floor_20260826"
KEYS = ["date", "bakery_id", "product_id"]


def add_causal_reference(rows: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    parts = []
    history = history.copy()
    history["dow"] = history["date"].dt.dayofweek
    for date in sorted(rows["date"].unique()):
        date = pd.Timestamp(date)
        sample = history[
            history["date"].between(date - pd.Timedelta(days=56), date - pd.Timedelta(days=1))
            & history["dow"].eq(date.dayofweek)
            & history["demand"].gt(0)
        ]
        reference = sample.groupby(["bakery_id", "product_id"], as_index=False).agg(
            history_n=("demand", "size"),
            history_p67=("demand", lambda values: values.quantile(0.67)),
        )
        part = rows[rows["date"].eq(date)].merge(
            reference, on=["bakery_id", "product_id"], how="left", validate="many_to_one"
        )
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    result["history_n"] = result["history_n"].fillna(0).astype(int)
    result["history_p67"] = result["history_p67"].fillna(0.0)
    return result


def balance(plan: np.ndarray, demand: np.ndarray) -> tuple[float, float, float, float]:
    error = plan - demand
    surplus = float(np.maximum(error, 0).sum())
    under = float(np.maximum(-error, 0).sum())
    return float(plan.sum()), surplus, under, surplus + under


def main() -> None:
    rows = pd.read_parquet(ROWS)
    labels = pd.read_csv(
        LABELS,
        usecols=[*KEYS, "demand_point_estimate"],
        encoding="utf-8-sig",
    ).rename(columns={"demand_point_estimate": "demand"})
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    rows = add_causal_reference(rows, labels)
    dates = sorted(rows["date"].unique())
    calibration_dates = set(dates[:4])
    rows["split"] = np.where(rows["date"].isin(calibration_dates), "calibration", "test")

    actual_by_split = {}
    for split, part in rows.groupby("split"):
        actual_by_split[split] = balance(
            part["available_to_sell"].to_numpy(), part["demand"].to_numpy()
        )[2]

    grid = []
    base_variants = [50, 67, 75, 85, 95]
    for quantile in base_variants:
        new_factor_col = f"p{quantile:02d}_new"
        factor_col = new_factor_col if new_factor_col in rows.columns else f"p{quantile:02d}"
        for uplift in [0.0, 0.02]:
            base = rows["predictive_forecast"].to_numpy() * rows[factor_col].to_numpy() * (1 + uplift)
            for min_n in [3, 4, 5, 6, 7, 8]:
                eligible = rows["history_n"].to_numpy() >= min_n
                for scale in np.arange(0.60, 1.31, 0.05):
                    floor = rows["history_p67"].to_numpy() * scale
                    for cap in [2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100]:
                        plan = np.where(eligible, np.maximum(base, np.minimum(floor, base + cap)), base)
                        record = {
                            "base": f"P{quantile}" + ("_+2%" if uplift else ""),
                            "min_n": min_n,
                            "scale": round(float(scale), 2),
                            "cap": cap,
                        }
                        for split in ["calibration", "test"]:
                            mask = rows["split"].eq(split).to_numpy()
                            volume, surplus, under, imbalance = balance(
                                plan[mask], rows.loc[mask, "demand"].to_numpy()
                            )
                            record.update(
                                {
                                    f"{split}_volume": volume,
                                    f"{split}_surplus": surplus,
                                    f"{split}_under": under,
                                    f"{split}_imbalance": imbalance,
                                    f"{split}_actual_under": actual_by_split[split],
                                }
                            )
                        record["volume"] = record["calibration_volume"] + record["test_volume"]
                        record["surplus"] = record["calibration_surplus"] + record["test_surplus"]
                        record["underbake"] = record["calibration_under"] + record["test_under"]
                        record["imbalance"] = record["surplus"] + record["underbake"]
                        record["passes_both_actual_under"] = (
                            record["calibration_under"] <= record["calibration_actual_under"]
                            and record["test_under"] <= record["test_actual_under"]
                        )
                        grid.append(record)

    result = pd.DataFrame(grid)
    feasible = result[result["passes_both_actual_under"]].sort_values(
        ["surplus", "underbake", "imbalance"]
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT / "grid.csv", index=False)
    feasible.head(100).to_csv(OUTPUT / "feasible_top100.csv", index=False)
    print(f"candidates={len(result)} feasible={len(feasible)}")
    print(feasible.head(20).to_string(index=False))


if __name__ == "__main__":
    main()

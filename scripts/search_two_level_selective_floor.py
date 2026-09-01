"""Select product-specific floor caps on calibration and evaluate frozen test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "reports/calibrated_selective_floor_decomposition_20260826/rows.parquet"
OUTPUT = ROOT / "reports/two_level_selective_floor_20260826"


def metrics(plan: np.ndarray, demand: np.ndarray) -> tuple[float, float, float, float]:
    error = plan - demand
    surplus = float(np.maximum(error, 0).sum())
    under = float(np.maximum(-error, 0).sum())
    return float(plan.sum()), surplus, under, surplus + under


def main() -> None:
    rows = pd.read_parquet(ROWS)
    base = rows["base_plan"].to_numpy()
    demand = rows["demand"].to_numpy()
    n8 = rows["history_n"].to_numpy() >= 8
    standard_floor = 0.95 * rows["history_p67"].to_numpy()
    standard = np.where(n8, np.maximum(base, np.minimum(standard_floor, base + 8.0)), base)

    calibration = rows["split"].eq("calibration").to_numpy()
    standard_added = standard - base
    base_under = np.maximum(demand - base, 0.0)
    standard_under = np.maximum(demand - standard, 0.0)
    calibration_rows = rows.loc[calibration, ["product_id"]].copy()
    calibration_rows["added"] = standard_added[calibration]
    calibration_rows["useful"] = base_under[calibration] - standard_under[calibration]
    calibration_rows["remaining_under"] = standard_under[calibration]
    product_stats = calibration_rows.groupby("product_id", as_index=False).agg(
        calibration_added=("added", "sum"),
        calibration_useful=("useful", "sum"),
        calibration_remaining_under=("remaining_under", "sum"),
    )
    product_stats["calibration_efficiency"] = (
        product_stats["calibration_useful"]
        / product_stats["calibration_added"].replace(0.0, np.nan)
    )

    grid = []
    for efficiency_threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
        for remaining_under_threshold in [50, 100, 200, 400, 800]:
            selected_products = set(
                product_stats.loc[
                    product_stats["calibration_added"].ge(20)
                    & product_stats["calibration_efficiency"].ge(efficiency_threshold)
                    & product_stats["calibration_remaining_under"].ge(remaining_under_threshold),
                    "product_id",
                ]
            )
            selected = rows["product_id"].isin(selected_products).to_numpy() & n8
            for scale in [0.95, 1.00, 1.05, 1.10]:
                expanded_floor = scale * rows["history_p67"].to_numpy()
                for expanded_cap in [10, 12, 15, 20, 30]:
                    plan = standard.copy()
                    plan[selected] = np.maximum(
                        base[selected],
                        np.minimum(expanded_floor[selected], base[selected] + expanded_cap),
                    )
                    record = {
                        "efficiency_threshold": efficiency_threshold,
                        "remaining_under_threshold": remaining_under_threshold,
                        "scale": scale,
                        "expanded_cap": expanded_cap,
                        "selected_products": len(selected_products),
                    }
                    for split in ["calibration", "test"]:
                        mask = rows["split"].eq(split).to_numpy()
                        volume, surplus, under, imbalance = metrics(plan[mask], demand[mask])
                        record.update(
                            {
                                f"{split}_volume": volume,
                                f"{split}_surplus": surplus,
                                f"{split}_under": under,
                                f"{split}_imbalance": imbalance,
                            }
                        )
                    record["volume"] = record["calibration_volume"] + record["test_volume"]
                    record["surplus"] = record["calibration_surplus"] + record["test_surplus"]
                    record["underbake"] = record["calibration_under"] + record["test_under"]
                    record["imbalance"] = record["calibration_imbalance"] + record["test_imbalance"]
                    grid.append(record)

    result = pd.DataFrame(grid)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT / "grid.csv", index=False)
    product_stats.to_csv(OUTPUT / "calibration_product_stats.csv", index=False)

    balanced = result.sort_values(["calibration_imbalance", "calibration_under"]).iloc[0]
    under_center_pool = result[
        result["calibration_under"]
        <= 0.90 * metrics(standard[calibration], demand[calibration])[2]
    ]
    under_center = under_center_pool.sort_values(
        ["calibration_surplus", "calibration_under"]
    ).iloc[0]
    under_first = result.sort_values(["calibration_under", "calibration_surplus"]).iloc[0]
    chosen = pd.DataFrame(
        [
            {"selection": "balanced", **balanced.to_dict()},
            {"selection": "under_center", **under_center.to_dict()},
            {"selection": "under_first", **under_first.to_dict()},
        ]
    )
    chosen.to_csv(OUTPUT / "calibration_selected.csv", index=False)
    print("Standard n>=8")
    for split in ["calibration", "test"]:
        mask = rows["split"].eq(split).to_numpy()
        print(split, metrics(standard[mask], demand[mask]))
    print("\nCalibration-selected candidates")
    print(chosen.to_string(index=False))


if __name__ == "__main__":
    main()

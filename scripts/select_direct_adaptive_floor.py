"""Select adaptive floor parameters on the blocked fold and evaluate current fold."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "reports/direct_uplift_floor_20260827/rows.parquet"
OPERATIONAL = ROOT / "reports/rolling_actual_state_comparison_20260826/rows.parquet"
OUTPUT = ROOT / "reports/direct_uplift_floor_20260827"
KEYS = ["date", "bakery_id", "product_id"]


def balance(plan: np.ndarray, demand: np.ndarray) -> dict[str, float]:
    error = plan - demand
    return {
        "volume": float(plan.sum()),
        "surplus": float(np.maximum(error, 0).sum()),
        "underbake": float(np.maximum(-error, 0).sum()),
        "imbalance": float(np.abs(error).sum()),
    }


def make_plan(rows: pd.DataFrame, params: dict[str, float]) -> np.ndarray:
    base = rows["direct_uplift_p50"].to_numpy()
    eligible = (
        rows["floor_history_n"].ge(params["min_n"])
        & rows["historical_stockout_rate"].ge(params["min_stockout_rate"])
        & rows["historical_lost_mean"].ge(params["min_lost_mean"])
    ).to_numpy()
    floor = rows["floor_demand_p67"].to_numpy() * params["scale"]
    cap = np.minimum(base + params["unit_cap"], base * (1 + params["relative_cap"]))
    return np.where(eligible, np.maximum(base, np.minimum(floor, cap)), base)


def main() -> None:
    rows = pd.read_parquet(ROWS)
    actual = pd.read_parquet(OPERATIONAL)[KEYS + ["available_to_sell"]]
    rows = rows.merge(actual, on=KEYS, how="left", validate="one_to_one")
    blocked = rows[rows["fold"].eq("blocked")].copy()
    current = rows[rows["fold"].eq("current")].copy()
    blocked_actual = balance(
        blocked["available_to_sell"].to_numpy(), blocked["demand"].to_numpy()
    )

    records = []
    for min_n in [4, 6, 8]:
        for min_stockout_rate in [0.10, 0.25, 0.50, 0.75]:
            for min_lost_mean in [0.5, 1.0, 2.0, 4.0]:
                for scale in [0.80, 0.90, 1.00]:
                    for unit_cap in [5.0, 10.0, 15.0, 25.0]:
                        for relative_cap in [0.10, 0.25, 0.50]:
                            params = {
                                "min_n": min_n,
                                "min_stockout_rate": min_stockout_rate,
                                "min_lost_mean": min_lost_mean,
                                "scale": scale,
                                "unit_cap": unit_cap,
                                "relative_cap": relative_cap,
                            }
                            result = balance(
                                make_plan(blocked, params), blocked["demand"].to_numpy()
                            )
                            records.append(
                                {
                                    **params,
                                    **{
                                        f"blocked_{key}": value
                                        for key, value in result.items()
                                    },
                                    "passes_actual_under": result["underbake"]
                                    <= blocked_actual["underbake"],
                                }
                            )
    grid = pd.DataFrame(records)
    feasible = grid[grid["passes_actual_under"]].sort_values(
        ["blocked_surplus", "blocked_underbake", "blocked_imbalance"]
    )
    if feasible.empty:
        selected = grid.sort_values(
            ["blocked_underbake", "blocked_surplus", "blocked_imbalance"]
        ).iloc[0]
        selection_rule = "minimum blocked underbake; no candidate beat actual"
    else:
        selected = feasible.iloc[0]
        selection_rule = (
            "minimum blocked surplus among candidates beating actual underbake"
        )
    params = {
        key: float(selected[key])
        for key in [
            "min_n",
            "min_stockout_rate",
            "min_lost_mean",
            "scale",
            "unit_cap",
            "relative_cap",
        ]
    }
    rows["direct_uplift_selected_floor"] = make_plan(rows, params)
    summary = {
        "selection_rule": selection_rule,
        "selected_params": params,
        "blocked_actual": blocked_actual,
        "blocked": {
            "direct_uplift_p50": balance(
                blocked["direct_uplift_p50"].to_numpy(),
                blocked["demand"].to_numpy(),
            ),
            "selected_floor": balance(
                rows.loc[
                    rows["fold"].eq("blocked"), "direct_uplift_selected_floor"
                ].to_numpy(),
                blocked["demand"].to_numpy(),
            ),
        },
        "current": {
            "actual": balance(
                current["available_to_sell"].to_numpy(), current["demand"].to_numpy()
            ),
            "direct_uplift_p50": balance(
                current["direct_uplift_p50"].to_numpy(), current["demand"].to_numpy()
            ),
            "selected_floor": balance(
                rows.loc[
                    rows["fold"].eq("current"), "direct_uplift_selected_floor"
                ].to_numpy(),
                current["demand"].to_numpy(),
            ),
        },
        "production_write": False,
    }
    grid.to_csv(OUTPUT / "adaptive_floor_grid.csv", index=False)
    rows.to_parquet(OUTPUT / "selected_rows.parquet", index=False)
    (OUTPUT / "selected_floor_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

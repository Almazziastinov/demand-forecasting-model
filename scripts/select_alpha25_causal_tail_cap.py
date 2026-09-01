"""Select an expanding-history causal tail cap for the approved alpha=.25 candidate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "reports/weighted_direct_soft_normalization_20260827/rows.parquet"
OUTPUT = ROOT / "reports/alpha25_tail_cap_20260827"
FOLDS = ["2026-07-20", "2026-07-27", "2026-08-10", "2026-08-17"]
GROUP = ["scenario", "date", "bakery_id"]
BASE = "original_alpha_025_floor"


def apply_cap(
    rows: pd.DataFrame, min_n: int, share_threshold: float, p67_scale: float
) -> tuple[pd.Series, pd.Series]:
    total = rows.groupby(GROUP)[BASE].transform("sum")
    share = rows[BASE] / total.replace(0, np.nan)
    bound = rows["floor_demand_p67"] * p67_scale
    eligible = (
        rows["floor_history_n"].ge(min_n)
        & share.gt(share_threshold)
        & rows[BASE].gt(bound)
        & bound.gt(0)
    )
    capped = rows[BASE].where(~eligible, np.minimum(rows[BASE], bound))
    return capped, eligible


def metrics(plan: pd.Series, demand: pd.Series) -> dict[str, float]:
    error = plan - demand
    return {
        "volume": float(plan.sum()),
        "surplus": float(error.clip(lower=0).sum()),
        "underbake": float((-error).clip(lower=0).sum()),
        "imbalance": float(error.abs().sum()),
    }


def select(prior: pd.DataFrame) -> dict[str, float]:
    base = metrics(prior[BASE], prior["scenario_demand"])
    records = []
    for min_n in [4, 6, 8]:
        for share_threshold in [0.15, 0.18, 0.20]:
            for p67_scale in [1.0, 1.10, 1.20, 1.30, 1.50]:
                plan, eligible = apply_cap(prior, min_n, share_threshold, p67_scale)
                result = metrics(plan, prior["scenario_demand"])
                records.append(
                    {
                        "min_n": min_n,
                        "share_threshold": share_threshold,
                        "p67_scale": p67_scale,
                        "capped_rows": int(eligible.sum()),
                        **result,
                        "surplus_saved": base["surplus"] - result["surplus"],
                        "underbake_added": result["underbake"] - base["underbake"],
                    }
                )
    grid = pd.DataFrame(records)
    feasible = grid[
        grid["capped_rows"].gt(0) & grid["surplus_saved"].gt(grid["underbake_added"])
    ].sort_values(["imbalance", "underbake", "capped_rows"])
    selected = feasible.iloc[0] if not feasible.empty else grid.iloc[0]
    return {
        "min_n": int(selected["min_n"]),
        "share_threshold": float(selected["share_threshold"]),
        "p67_scale": float(selected["p67_scale"]),
    }


def main() -> None:
    rows = pd.read_parquet(ROWS)
    rows["alpha25_tail_capped"] = rows[BASE]
    selections = []
    calibrated = rows[rows["scenario"].eq("calibrated")]
    for index, fold in enumerate(FOLDS):
        if index == 0:
            selections.append({"fold": fold, "status": "calibration_no_cap"})
            continue
        prior = calibrated[calibrated["rolling_fold"].isin(FOLDS[:index])]
        params = select(prior)
        mask = rows["rolling_fold"].eq(fold)
        capped, eligible = apply_cap(rows[mask], **params)
        rows.loc[mask, "alpha25_tail_capped"] = capped
        selections.append(
            {
                "fold": fold,
                "status": "selected_on_prior_folds",
                **params,
                "capped_rows_all_scenarios": int(eligible.sum()),
            }
        )

    evaluation = rows[~rows["rolling_fold"].eq(FOLDS[0])]
    summary = {}
    for scenario, part in evaluation.groupby("scenario"):
        summary[scenario] = {
            BASE: metrics(part[BASE], part["scenario_demand"]),
            "alpha25_tail_capped": metrics(
                part["alpha25_tail_capped"], part["scenario_demand"]
            ),
            "capped_rows": int(part["alpha25_tail_capped"].lt(part[BASE] - 1e-9).sum()),
        }
    case = rows[
        rows["scenario"].eq("calibrated")
        & rows["date"].eq(pd.Timestamp("2026-07-27"))
        & rows["bakery_id"].eq(244)
        & rows["product_id"].eq(11018)
    ]
    summary["bakery_244_sku_11018_2026_07_27"] = (
        case[["actual_sold", "scenario_demand", BASE, "alpha25_tail_capped"]]
        .iloc[0]
        .to_dict()
    )
    summary["production_write"] = False
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(OUTPUT / "rows.parquet", index=False)
    pd.DataFrame(selections).to_csv(
        OUTPUT / "selections.csv", index=False, encoding="utf-8-sig"
    )
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(pd.DataFrame(selections).to_string(index=False))


if __name__ == "__main__":
    main()

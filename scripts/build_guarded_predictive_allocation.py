"""Build a causal coverage-filled and guarded SKU allocation research candidate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports/rolling_actual_state_comparison_20260826/rows.parquet"
OUTPUT = ROOT / "reports/guarded_predictive_allocation_20260827"
CATEGORY_GROUP = ["date", "bakery_id", "category"]
DAY_GROUP = ["date", "bakery_id"]
MAX_FLOOR_UPLIFT_RATIO = 0.25
MAX_SKU_DAY_SHARE = 0.20


def fill_predictive_zeros(group: pd.DataFrame) -> pd.Series:
    total = float(group["predictive_forecast"].sum())
    if total <= 0:
        return group["causal_trend_forecast"].clip(lower=0.0)
    current_share = group["predictive_forecast"].clip(lower=0.0) / total
    prior = group["causal_trend_forecast"].clip(lower=0.0)
    prior_sum = float(prior.sum())
    prior_share = prior / prior_sum if prior_sum > 0 else current_share
    missing = group["predictive_raw"].le(1e-12)
    missing_mass = min(float(prior_share[missing].sum()), 1.0)
    result = pd.Series(0.0, index=group.index)
    result.loc[missing] = prior_share.loc[missing]
    retained = ~missing
    retained_mass = float(current_share.loc[retained].sum())
    if retained.any() and retained_mass > 0:
        result.loc[retained] = current_share.loc[retained] * (1.0 - missing_mass) / retained_mass
    elif retained.any():
        result.loc[retained] = (1.0 - missing_mass) / retained.sum()
    return result * total


def cap_day_shares(group: pd.DataFrame, column: str) -> pd.Series:
    values = group[column].clip(lower=0.0).astype(float).copy()
    total = float(values.sum())
    if total <= 0:
        return values
    feasible_cap = max(MAX_SKU_DAY_SHARE, 1.0 / len(values)) * total
    for _ in range(20):
        over = values > feasible_cap + 1e-10
        if not over.any():
            break
        excess = float((values.loc[over] - feasible_cap).sum())
        values.loc[over] = feasible_cap
        under = ~over
        capacity = feasible_cap - values.loc[under]
        eligible = under & capacity.reindex(values.index, fill_value=0.0).gt(1e-10)
        if not eligible.any() or excess <= 1e-12:
            break
        weights = values.loc[eligible]
        if float(weights.sum()) <= 0:
            weights = pd.Series(1.0, index=weights.index)
        addition = excess * weights / float(weights.sum())
        addition = np.minimum(addition, feasible_cap - values.loc[eligible])
        values.loc[eligible] += addition
        remainder = total - float(values.sum())
        if remainder > 1e-9:
            spare = (feasible_cap - values).clip(lower=0.0)
            if float(spare.sum()) > 0:
                values += remainder * spare / float(spare.sum())
    if not np.isclose(values.sum(), total, atol=1e-6):
        values *= total / float(values.sum())
    return values


def main() -> None:
    rows = pd.read_parquet(INPUT)
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    filled_parts = [fill_predictive_zeros(group) for _, group in rows.groupby(DAY_GROUP, sort=False)]
    rows["predictive_filled"] = pd.concat(filled_parts).sort_index()
    rows["p50_predictive_filled"] = rows["predictive_filled"] * rows["p50_factor"]

    eligible = rows["history_n"].fillna(0).ge(8)
    historical_floor = 0.95 * rows["history_p67"].fillna(0.0)
    raw_floor = np.where(
        eligible,
        np.maximum(
            rows["p50_predictive_filled"],
            np.minimum(historical_floor, rows["p50_predictive_filled"] + 8.0),
        ),
        rows["p50_predictive_filled"],
    )
    rows["filled_raw_floor"] = raw_floor
    rows["floor_increment"] = rows["filled_raw_floor"] - rows["p50_predictive_filled"]
    day_base = rows.groupby(DAY_GROUP)["p50_predictive_filled"].transform("sum")
    day_increment = rows.groupby(DAY_GROUP)["floor_increment"].transform("sum")
    allowed_increment = MAX_FLOOR_UPLIFT_RATIO * day_base
    increment_scale = np.minimum(1.0, allowed_increment / day_increment.replace(0.0, np.nan)).fillna(1.0)
    rows["filled_volume_guard"] = rows["p50_predictive_filled"] + rows["floor_increment"] * increment_scale

    capped_parts = [
        cap_day_shares(group, "filled_volume_guard")
        for _, group in rows.groupby(DAY_GROUP, sort=False)
    ]
    rows["guarded_predictive_floor"] = pd.concat(capped_parts).sort_index()

    filled_day_delta = (
        rows.groupby(DAY_GROUP)[["predictive_filled", "predictive_forecast"]].sum().eval(
            "predictive_filled - predictive_forecast"
        ).abs().max()
    )
    day_guard_delta = (
        rows.groupby(DAY_GROUP)[["guarded_predictive_floor", "filled_volume_guard"]].sum().eval(
            "guarded_predictive_floor - filled_volume_guard"
        ).abs().max()
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(OUTPUT / "rows.parquet", index=False)
    print(f"rows={len(rows)} filled_day_max_delta={filled_day_delta:.12g} day_guard_max_delta={day_guard_delta:.12g}")
    print(
        rows[["predictive_forecast", "predictive_filled", "p50_predictive_filled", "filled_raw_floor", "filled_volume_guard", "guarded_predictive_floor"]]
        .sum()
        .to_string()
    )


if __name__ == "__main__":
    main()

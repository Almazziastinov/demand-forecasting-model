"""Compare forecast candidates after identical stock, batch and capacity rules."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/forecast_embedded"))

from app.db import get_client  # noqa: E402


ROWS = ROOT / "reports/alpha25_tail_cap_20260827/rows.parquet"
ACTUAL = ROOT / "reports/rolling_actual_state_comparison_20260826/rows.parquet"
OUTPUT = ROOT / "reports/alpha25_operational_candidate_comparison_20260827"
KEYS = ["date", "bakery_id", "product_id"]
DAY = ["scenario", "date", "bakery_id"]
EVALUATION_FOLDS = {"2026-07-27", "2026-08-10", "2026-08-17"}
VARIANTS = {
    "current": "incumbent_sku_forecast",
    "direct_p50": "direct_p50",
    "previous_final": "direct_uplift_adaptive_floor",
    "alpha25_tail_cap": "alpha25_tail_capped",
}
CORE_CATEGORIES = {
    "Выпечка сытная",
    "Выпечка сладкая",
    "Пироги сытные",
    "Пироги сладкие",
}
PIE_CATEGORIES = {"Пироги сытные", "Пироги сладкие"}


def load_operational_metadata() -> tuple[pd.DataFrame, int]:
    client = get_client()
    meta = client.query_df(
        """
        select product_id, kratnost
        from baking_sku_meta final
        where is_active = 1 and scope = 'base'
        """
    )
    capacity = client.query_df(
        """
        select bakers_count
        from baking_capacity_config final
        where is_active = 1 and bakery_id is null
        order by valid_from desc
        limit 1
        """
    )
    meta["product_id"] = pd.to_numeric(meta["product_id"]).astype("int64")
    meta["kratnost"] = (
        pd.to_numeric(meta["kratnost"]).fillna(1).clip(lower=1).astype(int)
    )
    daily_cap = int(capacity.iloc[0, 0]) * 600
    return meta.drop_duplicates("product_id", keep="last"), daily_cap


def round_up(values: pd.Series, multiples: pd.Series) -> pd.Series:
    return pd.Series(
        [0.0 if value <= 0 else float(math.ceil(value / multiple - 1e-12) * multiple)
         for value, multiple in zip(values, multiples, strict=True)],
        index=values.index,
    )


def cap_day(group: pd.DataFrame, cap: int) -> pd.DataFrame:
    result = group.copy()
    core = result["has_meta"] & result["category"].isin(CORE_CATEGORIES)
    excess = float(result.loc[core, "production_plan"].sum()) - cap
    result["capacity_reduction"] = 0.0
    if excess <= 1e-9:
        result["capacity_binding"] = False
        return result
    candidates = result.loc[core].copy()
    candidates["priority_group"] = candidates["is_core_sku"].astype(int)
    candidates["priority_ratio"] = candidates["broad_56_mean"] / candidates[
        "production_plan"
    ].replace(0, np.nan)
    candidates = candidates.sort_values(
        ["priority_group", "priority_ratio", "broad_56_mean"],
        ascending=[True, True, True],
    )
    for index, row in candidates.iterrows():
        if excess <= 1e-9:
            break
        multiple = int(row["effective_multiple"])
        steps = min(
            int(result.at[index, "production_plan"] // multiple),
            math.ceil(excess / multiple),
        )
        reduction = float(steps * multiple)
        result.at[index, "production_plan"] -= reduction
        result.at[index, "capacity_reduction"] = reduction
        excess -= reduction
    result["capacity_binding"] = True
    return result


def operate(rows: pd.DataFrame, source: str, cap: int) -> pd.DataFrame:
    result = rows.copy()
    result["raw_forecast"] = result[source].clip(lower=0.0)
    result["net_need"] = (
        result["raw_forecast"] - result["opening_stock"]
    ).clip(lower=0.0)
    result["production_plan"] = result["raw_forecast"]
    covered = result["has_meta"]
    result.loc[covered, "production_plan"] = round_up(
        result.loc[covered, "net_need"], result.loc[covered, "effective_multiple"]
    )
    result = (
        result.groupby(DAY, group_keys=True, sort=False)
        .apply(cap_day, cap=cap, include_groups=False)
        .reset_index()
    )
    result["total_to_sell"] = np.where(
        result["has_meta"],
        result["production_plan"] + result["opening_stock"],
        result["production_plan"],
    )
    return result


def score(rows: pd.DataFrame, variant: str) -> dict[str, float | int | str]:
    error = rows["total_to_sell"] - rows["scenario_demand"]
    return {
        "scenario": str(rows["scenario"].iloc[0]),
        "variant": variant,
        "volume": float(rows["total_to_sell"].sum()),
        "surplus": float(error.clip(lower=0).sum()),
        "underbake": float((-error).clip(lower=0).sum()),
        "imbalance": float(error.abs().sum()),
        "production": float(rows["production_plan"].sum()),
        "capacity_reduction": float(rows["capacity_reduction"].sum()),
        "capacity_binding_days": int(
            rows.loc[rows["capacity_binding"], ["date", "bakery_id"]]
            .drop_duplicates()
            .shape[0]
        ),
    }


def main() -> None:
    rows = pd.read_parquet(ROWS)
    actual = pd.read_parquet(ACTUAL)
    for frame in (rows, actual):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    opening = actual[KEYS + ["opening_stock"]].drop_duplicates(KEYS)
    rows = rows.merge(opening, on=KEYS, how="left", validate="many_to_one")
    rows["opening_stock"] = rows["opening_stock"].fillna(0.0).clip(lower=0.0)
    meta, cap = load_operational_metadata()
    rows = rows.merge(meta, on="product_id", how="left", validate="many_to_one")
    rows["has_meta"] = rows["kratnost"].notna()
    rows["effective_multiple"] = rows["kratnost"].fillna(1).astype(int)
    rows.loc[rows["category"].isin(PIE_CATEGORIES), "effective_multiple"] = 4
    rows = rows[rows["rolling_fold"].isin(EVALUATION_FOLDS)].copy()

    summaries = []
    outputs = []
    for scenario, scenario_rows in rows.groupby("scenario", sort=False):
        for variant, source in VARIANTS.items():
            operated = operate(scenario_rows, source, cap)
            operated["variant"] = variant
            summaries.append(score(operated, variant))
            outputs.append(operated)
    summary = pd.DataFrame(summaries)
    output_rows = pd.concat(outputs, ignore_index=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT / "summary.csv", index=False, encoding="utf-8-sig")
    output_rows.to_parquet(OUTPUT / "rows.parquet", index=False)
    (OUTPUT / "metadata.json").write_text(
        json.dumps({"daily_core_cap": cap, "production_write": False}, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

"""Apply SKU multiples and a daily core-bakery capacity screen to alpha=.25."""

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
ACTUAL_STATE = ROOT / "reports/rolling_actual_state_comparison_20260826/rows.parquet"
OUTPUT = ROOT / "reports/alpha25_operational_constraints_20260827"
GROUP = ["scenario", "date", "bakery_id"]
KEYS = ["date", "bakery_id", "product_id"]
SOURCE = "alpha25_tail_capped"
CORE_CATEGORIES = {
    "Выпечка сытная",
    "Выпечка сладкая",
    "Пироги сытные",
    "Пироги сладкие",
}
PIE_CATEGORIES = {"Пироги сытные", "Пироги сладкие"}


def load_metadata() -> tuple[pd.DataFrame, int]:
    client = get_client()
    meta = client.query_df(
        """
        select product_id, kratnost, is_on_demand
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
        pd.to_numeric(meta["kratnost"]).fillna(1).astype(int).clip(lower=1)
    )
    daily_core_cap = int(capacity["bakers_count"].iloc[0]) * 600
    return meta.drop_duplicates("product_id", keep="last"), daily_core_cap


def round_up(value: float, multiple: int) -> float:
    if value <= 0:
        return 0.0
    return float(math.ceil(value / multiple - 1e-12) * multiple)


def apply_capacity(group: pd.DataFrame, daily_core_cap: int) -> pd.DataFrame:
    result = group.copy()
    core_mask = result["is_bakeable_meta"] & result["category"].isin(CORE_CATEGORIES)
    total = float(result.loc[core_mask, "rounded_plan"].sum())
    result["capacity_reduction"] = 0.0
    if total <= daily_core_cap:
        result["capacity_binding"] = False
        return result

    candidates = result.loc[core_mask].copy()
    candidates["priority_group"] = candidates["is_core_sku"].astype(int)
    candidates["priority_ratio"] = candidates["broad_56_mean"] / candidates[
        "rounded_plan"
    ].replace(0, np.nan)
    candidates = candidates.sort_values(
        ["priority_group", "priority_ratio", "broad_56_mean"],
        ascending=[True, True, True],
    )
    excess = total - daily_core_cap
    for index, row in candidates.iterrows():
        if excess <= 1e-9:
            break
        multiple = int(row["effective_multiple"])
        available_steps = int(result.at[index, "rounded_plan"] // multiple)
        steps = min(available_steps, math.ceil(excess / multiple))
        reduction = float(steps * multiple)
        result.at[index, "rounded_plan"] -= reduction
        result.at[index, "capacity_reduction"] += reduction
        excess -= reduction
    result["capacity_binding"] = True
    return result


def score(rows: pd.DataFrame, column: str) -> dict[str, float]:
    error = rows[column] - rows["scenario_demand"]
    return {
        "volume": float(rows[column].sum()),
        "surplus": float(error.clip(lower=0).sum()),
        "underbake": float((-error).clip(lower=0).sum()),
        "imbalance": float(error.abs().sum()),
    }


def main() -> None:
    rows = pd.read_parquet(ROWS)
    actual_state = pd.read_parquet(ACTUAL_STATE)
    actual_state["date"] = pd.to_datetime(actual_state["date"]).dt.normalize()
    opening_stock = actual_state[KEYS + ["opening_stock"]].drop_duplicates(KEYS)
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    rows = rows.merge(opening_stock, on=KEYS, how="left", validate="many_to_one")
    rows["opening_stock"] = rows["opening_stock"].fillna(0.0).clip(lower=0.0)
    meta, daily_core_cap = load_metadata()
    rows = rows.merge(meta, on="product_id", how="left", validate="many_to_one")
    rows["is_bakeable_meta"] = rows["kratnost"].notna()
    rows["effective_multiple"] = rows["kratnost"].fillna(1).astype(int)
    rows.loc[rows["category"].isin(PIE_CATEGORIES), "effective_multiple"] = 4
    rows["net_need"] = (rows[SOURCE] - rows["opening_stock"]).clip(lower=0.0)
    rows["rounded_plan"] = [
        round_up(value, int(multiple)) if bakeable else float(source)
        for value, source, multiple, bakeable in zip(
            rows["net_need"],
            rows[SOURCE],
            rows["effective_multiple"],
            rows["is_bakeable_meta"],
            strict=True,
        )
    ]
    rows["total_to_sell_before_capacity"] = np.where(
        rows["is_bakeable_meta"],
        rows["rounded_plan"] + rows["opening_stock"],
        rows["rounded_plan"],
    )
    rows["rounding_increment"] = rows["total_to_sell_before_capacity"] - rows[SOURCE]
    rows = (
        rows.groupby(GROUP, group_keys=True, sort=False)
        .apply(apply_capacity, daily_core_cap=daily_core_cap, include_groups=False)
        .reset_index()
    )
    rows["alpha25_operational"] = np.where(
        rows["is_bakeable_meta"],
        rows["rounded_plan"] + rows["opening_stock"],
        rows["rounded_plan"],
    )

    evaluation = rows[~rows["rolling_fold"].eq("2026-07-20")]
    summary = {
        "daily_core_cap": daily_core_cap,
        "metadata": {
            "products": int(meta["product_id"].nunique()),
            "covered_rows": int(evaluation["is_bakeable_meta"].sum()),
            "covered_products": int(
                evaluation.loc[evaluation["is_bakeable_meta"], "product_id"].nunique()
            ),
        },
        "rounding": {
            "increment": float(evaluation["rounding_increment"].sum()),
            "rows_increased": int(evaluation["rounding_increment"].gt(1e-9).sum()),
            "opening_stock": float(
                evaluation.loc[evaluation["is_bakeable_meta"], "opening_stock"].sum()
            ),
        },
        "capacity": {
            "binding_bakery_days": int(
                evaluation.loc[evaluation["capacity_binding"], GROUP]
                .drop_duplicates()
                .shape[0]
            ),
            "reduction": float(evaluation["capacity_reduction"].sum()),
        },
        "scenarios": {},
        "production_write": False,
    }
    for scenario, part in evaluation.groupby("scenario"):
        summary["scenarios"][scenario] = {
            SOURCE: score(part, SOURCE),
            "alpha25_operational": score(part, "alpha25_operational"),
        }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(OUTPUT / "rows.parquet", index=False)
    meta.to_csv(
        OUTPUT / "baking_sku_meta_snapshot.csv", index=False, encoding="utf-8-sig"
    )
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

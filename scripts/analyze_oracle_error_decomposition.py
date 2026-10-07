"""Decompose Direct errors into bakery volume, SKU placement, and stock policy.

This is a research-only diagnostic over the frozen 2026-08-24..31 holdout.
Oracle variants use the corrected reconstructed demand from that holdout and
must not be interpreted as deployable forecasts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_freshness_calibrated_economics import simulate_group  # noqa: E402


SOURCE = ROOT / "reports/three_prod_models_august_holdout_20260911"
OUTPUT = ROOT / "reports/oracle_error_decomposition_20260913"
KEYS = ["date", "bakery_id", "product_id"]
GROUP_KEYS = ["date", "bakery_id"]
DIRECT_PLAN = "plan_1"


def _allocate_oracle_variants(rows: pd.DataFrame) -> pd.DataFrame:
    result = rows.copy()
    direct_total = result.groupby(GROUP_KEYS)[DIRECT_PLAN].transform("sum")
    demand_total = result.groupby(GROUP_KEYS)["demand"].transform("sum")

    direct_share = result[DIRECT_PLAN] / direct_total.replace(0.0, np.nan)
    demand_share = result["demand"] / demand_total.replace(0.0, np.nan)

    # Perfect bakery-day demand total, but the Direct SKU proportions remain.
    result["oracle_bakery_total_direct_shares"] = (
        demand_total * direct_share.fillna(demand_share).fillna(0.0)
    )
    # Direct bakery-day total, but perfect hindsight SKU proportions.
    result["direct_total_oracle_sku_shares"] = (
        direct_total * demand_share.fillna(0.0)
    )
    # Perfect hindsight demand at both bakery-day and SKU levels. Remaining
    # loss comes from the downstream inventory/old-stock policy.
    result["oracle_total_oracle_sku"] = result["demand"]
    return result


def _simulate(rows: pd.DataFrame, plan_column: str) -> pd.DataFrame:
    return pd.concat(
        [
            simulate_group(group, plan_column)
            for _, group in rows.groupby(["bakery_id", "product_id"], sort=False)
        ],
        ignore_index=True,
    )


def _summarize_variant(
    label: str,
    plan: pd.Series,
    demand: pd.Series,
    simulation: pd.DataFrame,
    demand_total: float,
    direct_gp: float,
    actual_gp: float,
    actual_cases: int,
) -> dict[str, float | int | str]:
    gross_profit = float(simulation["gross_profit"].sum())
    gap = actual_gp - direct_gp
    return {
        "variant": label,
        "plan_qty": float(plan.sum()),
        "plan_wape_vs_demand_pct": 100
        * float(np.abs(plan.to_numpy() - demand.to_numpy()).sum())
        / demand_total,
        "production_qty": float(simulation["production"].sum()),
        "served_qty": float(simulation["served"].sum()),
        "lost_qty": float(simulation["lost"].sum()),
        "lost_cases_ge_1": int(simulation["lost"].ge(1.0).sum()),
        "case_delta_vs_actual": int(simulation["lost"].ge(1.0).sum() - actual_cases),
        "writeoff_qty": float(simulation["writeoff"].sum()),
        "revenue": float(simulation["revenue"].sum()),
        "production_cost": float(simulation["production_cost"].sum()),
        "gross_profit": gross_profit,
        "gp_delta_vs_direct": gross_profit - direct_gp,
        "gp_delta_vs_actual": gross_profit - actual_gp,
        "direct_gap_closed_pct": 100 * (gross_profit - direct_gp) / gap,
        "service_pct": 100 * float(simulation["served"].sum()) / demand_total,
    }


def _level_diagnostics(rows: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    paired = rows[KEYS + ["demand", DIRECT_PLAN]].merge(
        actual[KEYS + ["production"]], on=KEYS, validate="one_to_one"
    )
    levels = {
        "network_day": ["date"],
        "bakery_day": ["date", "bakery_id"],
        "product_network_day": ["date", "product_id"],
        "bakery_product_day": KEYS,
    }
    records = []
    for label, keys in levels.items():
        grouped = paired.groupby(keys, dropna=False)[
            ["demand", DIRECT_PLAN, "production"]
        ].sum()
        denominator = float(grouped["demand"].sum())
        records.append(
            {
                "level": label,
                "groups": len(grouped),
                "actual_production_wape_pct": 100
                * float((grouped["production"] - grouped["demand"]).abs().sum())
                / denominator,
                "direct_plan_wape_pct": 100
                * float((grouped[DIRECT_PLAN] - grouped["demand"]).abs().sum())
                / denominator,
                "actual_demand_correlation": grouped["production"].corr(grouped["demand"]),
                "direct_demand_correlation": grouped[DIRECT_PLAN].corr(grouped["demand"]),
            }
        )
    return pd.DataFrame(records)


def _concentration(actual: pd.DataFrame, direct: pd.DataFrame) -> pd.DataFrame:
    paired = actual[KEYS + ["gross_profit"]].merge(
        direct[KEYS + ["gross_profit"]],
        on=KEYS,
        suffixes=("_actual", "_direct"),
        validate="one_to_one",
    )
    paired["gp_delta"] = paired["gross_profit_direct"] - paired["gross_profit_actual"]
    net_loss = -float(paired["gp_delta"].sum())
    records = []
    for level, key, top_n in [
        ("bakery", "bakery_id", 10),
        ("product", "product_id", 20),
    ]:
        delta = paired.groupby(key)["gp_delta"].sum()
        losses = (-delta[delta.lt(0.0)]).sort_values(ascending=False)
        records.append(
            {
                "level": level,
                "negative_entities": int(delta.lt(0.0).sum()),
                "positive_entities": int(delta.gt(0.0).sum()),
                "gross_negative_gp": float(losses.sum()),
                "top_n": top_n,
                "top_n_share_of_gross_negative_pct": 100
                * float(losses.head(top_n).sum())
                / float(losses.sum()),
                "top_n_share_of_net_loss_pct": 100
                * float(losses.head(top_n).sum())
                / net_loss,
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = pd.read_parquet(SOURCE / "evaluation_input.parquet")
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    rows = _allocate_oracle_variants(rows)

    actual = pd.read_parquet(SOURCE / "economics_actual_state.parquet")
    direct = pd.read_parquet(SOURCE / "economics_direct_alpha_025_v1.parquet")
    actual_gp = float(actual["gross_profit"].sum())
    direct_gp = float(direct["gross_profit"].sum())
    actual_cases = int(actual["lost"].ge(1.0).sum())
    demand_total = float(rows["demand"].sum())

    variants = {
        "direct_alpha_025_v1": DIRECT_PLAN,
        "oracle_bakery_total_direct_shares": "oracle_bakery_total_direct_shares",
        "direct_total_oracle_sku_shares": "direct_total_oracle_sku_shares",
        "oracle_total_oracle_sku": "oracle_total_oracle_sku",
    }
    records = []
    for label, column in variants.items():
        simulation = direct if label == "direct_alpha_025_v1" else _simulate(rows, column)
        records.append(
            _summarize_variant(
                label,
                rows[column],
                rows["demand"],
                simulation,
                demand_total,
                direct_gp,
                actual_gp,
                actual_cases,
            )
        )
        simulation.to_parquet(OUTPUT / f"economics_{label}.parquet", index=False)

    summary = pd.DataFrame(records)
    summary.to_csv(OUTPUT / "oracle_summary.csv", index=False, encoding="utf-8-sig")
    _level_diagnostics(rows, actual).to_csv(
        OUTPUT / "level_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    _concentration(actual, direct).to_csv(
        OUTPUT / "loss_concentration.csv", index=False, encoding="utf-8-sig"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

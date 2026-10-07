"""Extended stateful Direct guardrail backtest for 2026-08-24..2026-09-09."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.backtest_direct_demand_guardrail import KEYS, add_guardrails  # noqa: E402
from scripts.evaluate_three_model_august_holdout import (  # noqa: E402
    add_causal_freshness_priors,
    add_plan_columns,
    add_reconciliation,
    load_base_rows,
)
from scripts.experiment_direct_old_stock_credit import simulate_group  # noqa: E402


OUTPUT = ROOT / "reports/direct_demand_guardrail_aug24_sep09_20260914"
CANONICAL_DEMAND = (
    ROOT
    / "reports/historical_daily_retraining_gate_loss075_full_20260910"
    / "predictions.parquet"
)
SEPTEMBER_ROWS = ROOT / "reports/live_pilot_model_versions_20260911/forecast_rows.parquet"


def load_august() -> pd.DataFrame:
    rows, _ = load_base_rows()
    rows = add_plan_columns(rows)
    direct_column = next(
        column
        for column, label in rows.attrs["plan_labels"].items()
        if label == "direct_alpha_025_v1"
    )
    rows = add_causal_freshness_priors(rows)
    rows["forecast_qty"] = rows[direct_column]
    rows["source_period"] = "aug24_31_canonical"
    return rows


def load_september() -> pd.DataFrame:
    rows = pd.read_parquet(SEPTEMBER_ROWS)
    rows = rows[rows["variant"].eq("direct_alpha_025_v1")].copy()
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    rows["source_period"] = "sep01_09_live"
    return rows


def common_columns() -> list[str]:
    return KEYS + [
        "source_period",
        "forecast_qty",
        "demand",
        "observed_sales_qty",
        "observed_sales_amount",
        "release_qty",
        "incoming_move_qty",
        "outgoing_move_qty",
        "written_off_qty",
        "unit_cost",
        "unit_price",
        "old_share_prior",
        "fresh_price_prior",
        "old_price_prior",
        "actual_sold_old",
        "actual_sold_fresh",
    ]


def load_rows() -> pd.DataFrame:
    columns = common_columns()
    august = load_august()
    september = load_september()
    missing = sorted((set(columns) - set(august.columns)) | (set(columns) - set(september.columns)))
    if missing:
        raise RuntimeError(f"Missing extended-economics columns: {missing}")
    rows = pd.concat([august[columns], september[columns]], ignore_index=True)
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    numeric = [column for column in columns if column not in KEYS + ["source_period"]]
    rows[numeric] = rows[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    rows[numeric] = rows[numeric].clip(lower=0.0)
    rows = rows.sort_values(KEYS).drop_duplicates(KEYS, keep="last").reset_index(drop=True)
    return add_reconciliation(rows)


def load_demand_history(rows: pd.DataFrame) -> pd.DataFrame:
    history = pd.read_parquet(CANONICAL_DEMAND, columns=KEYS + ["demand"])
    history["date"] = pd.to_datetime(history["date"]).dt.normalize()
    september = rows[rows["date"].ge("2026-09-01")][KEYS + ["demand"]]
    history = pd.concat([history, september], ignore_index=True).drop_duplicates(KEYS, keep="last")
    history["dow"] = history["date"].dt.dayofweek
    history["is_weekend"] = history["dow"].ge(5)
    return history


def simulate_stateful(rows: pd.DataFrame, plan_column: str) -> pd.DataFrame:
    return pd.concat(
        [
            simulate_group(group, plan_column, "full", 1.0)
            for _, group in rows.groupby(["bakery_id", "product_id"], sort=False)
        ],
        ignore_index=True,
    )


def actual_rows(rows: pd.DataFrame) -> pd.DataFrame:
    actual = rows[KEYS + [
        "source_period",
        "demand",
        "release_qty",
        "observed_sales_qty",
        "observed_sales_amount",
        "written_off_qty",
        "unit_cost",
    ]].copy()
    actual["production"] = actual["release_qty"]
    actual["served"] = actual["observed_sales_qty"]
    actual["lost"] = (actual["demand"] - actual["served"]).clip(lower=0.0)
    actual["writeoff"] = actual["written_off_qty"]
    actual["gross_profit"] = actual["observed_sales_amount"] - actual["production"] * actual["unit_cost"]
    return actual


def aggregate(rows: pd.DataFrame, variant: str, period: str) -> dict[str, object]:
    return {
        "variant": variant,
        "period": period,
        "rows": len(rows),
        "production": rows["production"].sum(),
        "served": rows["served"].sum(),
        "lost": rows["lost"].sum(),
        "writeoff": rows["writeoff"].sum(),
        "gross_profit": rows["gross_profit"].sum(),
    }


def main() -> None:
    rows = load_rows()
    history = load_demand_history(rows)
    guards = add_guardrails(
        rows[KEYS + ["forecast_qty"]],
        history[KEYS + ["demand", "dow", "is_weekend"]],
    )
    rows = rows.merge(
        guards[KEYS + ["plain_anchor_28", "plain_fixed_lower_28", "plain_fixed_upper_28"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    rows["symmetric_guard_28"] = rows["forecast_qty"].clip(
        lower=rows["plain_fixed_lower_28"],
        upper=rows["plain_fixed_upper_28"],
    ).fillna(rows["forecast_qty"])

    simulations = {
        "actual_state": actual_rows(rows),
        "direct": simulate_stateful(rows, "forecast_qty"),
        "symmetric_guard_28": simulate_stateful(rows, "symmetric_guard_28"),
    }
    source_period = rows[KEYS + ["source_period"]]
    for variant in ["direct", "symmetric_guard_28"]:
        simulations[variant] = simulations[variant].merge(
            source_period, on=KEYS, how="left", validate="one_to_one"
        )

    periods = {
        "aug24_31": (pd.Timestamp("2026-08-24"), pd.Timestamp("2026-08-31")),
        "sep01_09": (pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-09")),
        "combined": (pd.Timestamp("2026-08-24"), pd.Timestamp("2026-09-09")),
    }
    summary_rows = []
    for period, (start, end) in periods.items():
        for variant, simulation in simulations.items():
            selected = simulation[simulation["date"].between(start, end)]
            summary_rows.append(aggregate(selected, variant, period))
    summary = pd.DataFrame(summary_rows)
    for period, indices in summary.groupby("period").groups.items():
        block = summary.loc[indices]
        actual = block[block["variant"].eq("actual_state")].iloc[0]
        direct = block[block["variant"].eq("direct")].iloc[0]
        summary.loc[indices, "gross_profit_delta_vs_actual"] = block["gross_profit"] - actual["gross_profit"]
        summary.loc[indices, "gross_profit_delta_vs_direct"] = block["gross_profit"] - direct["gross_profit"]
        summary.loc[indices, "production_delta_vs_actual"] = block["production"] - actual["production"]
        summary.loc[indices, "lost_delta_vs_direct"] = block["lost"] - direct["lost"]
        summary.loc[indices, "writeoff_delta_vs_direct"] = block["writeoff"] - direct["writeoff"]

    daily_parts = []
    for variant, simulation in simulations.items():
        daily = simulation.groupby("date", as_index=False).agg(
            production=("production", "sum"),
            served=("served", "sum"),
            lost=("lost", "sum"),
            writeoff=("writeoff", "sum"),
            gross_profit=("gross_profit", "sum"),
        )
        daily["variant"] = variant
        daily_parts.append(daily)
    daily_summary = pd.concat(daily_parts, ignore_index=True)

    direct_delta = rows["symmetric_guard_28"] - rows["forecast_qty"]
    coverage = pd.DataFrame(
        [
            {
                "rows": len(rows),
                "dates": rows["date"].nunique(),
                "bakeries": rows["bakery_id"].nunique(),
                "products": rows["product_id"].nunique(),
                "anchor_rows": int(rows["plain_anchor_28"].notna().sum()),
                "anchor_pct": 100 * rows["plain_anchor_28"].notna().mean(),
                "raised_rows": int(direct_delta.gt(1e-9).sum()),
                "raised_qty": direct_delta.clip(lower=0.0).sum(),
                "capped_rows": int(direct_delta.lt(-1e-9).sum()),
                "capped_qty": (-direct_delta.clip(upper=0.0)).sum(),
            }
        ]
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(OUTPUT / "decision_rows.parquet", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False, encoding="utf-8-sig")
    daily_summary.to_csv(OUTPUT / "daily.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(OUTPUT / "coverage.csv", index=False, encoding="utf-8-sig")
    for variant, simulation in simulations.items():
        simulation.to_parquet(OUTPUT / f"economics_{variant}.parquet", index=False)
    print(summary.to_string(index=False))
    print("\nCoverage")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()

"""Full causal 2026 walk-forward for fixed and prior-year seasonal demand guards."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.backtest_regime_aware_guardrail_aug24_sep09 as regime_helpers  # noqa: E402
from scripts.backtest_direct_demand_guardrail import KEYS, add_guardrails  # noqa: E402
from scripts.backtest_direct_demand_guardrail_aug24_sep09 import (  # noqa: E402
    actual_rows,
    simulate_stateful,
)
from scripts.evaluate_three_model_august_holdout import (  # noqa: E402
    add_causal_freshness_priors,
    add_reconciliation,
)


OUTPUT = ROOT / "reports/regime_aware_guardrail_full_walkforward_20260915"
PREDICTIONS = ROOT / "reports/historical_daily_retraining_gate_loss075_full_20260910/predictions.parquet"
PANEL = ROOT / ".codex_tmp/historical_clickhouse_panel/causal_panel.parquet"
MAPPING = ROOT / "reports/markup_price_mapping_20260826/mapped_products.csv"
START = pd.Timestamp("2026-03-01")
END = pd.Timestamp("2026-08-31")
TARGET_MONTHS = (3, 4, 5, 6, 7, 8)
VARIANT_COLUMNS = {
    "direct": "forecast_qty",
    "floor_only": "floor_only",
    "symmetric_guard_28": "symmetric_guard_28",
    "seasonal_guard": "seasonal_guard",
    "regime_aware_guard": "regime_aware_guard",
}


def load_rows() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    prediction_columns = KEYS + [
        "direct_plan",
        "demand",
        "observed_sales_qty",
        "observed_sales_amount",
    ]
    rows = pd.read_parquet(PREDICTIONS, columns=prediction_columns)
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    rows = rows[rows["date"].between(START, END)].copy()
    rows = rows.rename(columns={"direct_plan": "forecast_qty"})

    panel_columns = KEYS + [
        "release_qty",
        "incoming_move_qty",
        "outgoing_move_qty",
        "written_off_qty",
    ]
    panel = pd.read_parquet(
        PANEL,
        columns=panel_columns + ["observed_sales_qty"],
    )
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    flows = panel.loc[panel["date"].between(START, END), panel_columns]
    rows = rows.merge(flows, on=KEYS, how="left", validate="one_to_one")

    numeric = [
        "forecast_qty",
        "demand",
        "observed_sales_qty",
        "observed_sales_amount",
        "release_qty",
        "incoming_move_qty",
        "outgoing_move_qty",
        "written_off_qty",
    ]
    rows[numeric] = rows[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    rows[numeric] = rows[numeric].clip(lower=0.0)

    mapping = pd.read_csv(MAPPING, encoding="utf-8-sig")
    mapping = mapping[mapping["valid_economics"].astype(bool)].copy()
    mapping["product_id"] = pd.to_numeric(mapping["product_id"], errors="coerce")
    mapping = mapping.dropna(subset=["product_id"])
    mapping["product_id"] = mapping["product_id"].astype(int)
    mapping = mapping.sort_values("unit_price").drop_duplicates("product_id", keep="last")
    source_rows = len(rows)
    source_demand = float(rows["demand"].sum())
    rows = rows.merge(
        mapping[["product_id", "unit_cost", "unit_price"]],
        on="product_id",
        how="inner",
        validate="many_to_one",
    )
    coverage = {
        "source_rows": float(source_rows),
        "priced_rows": float(len(rows)),
        "priced_rows_pct": 100.0 * len(rows) / source_rows,
        "priced_demand_pct": 100.0 * float(rows["demand"].sum()) / source_demand,
    }
    rows["source_period"] = rows["date"].dt.to_period("M").astype(str)
    rows = rows.sort_values(KEYS).drop_duplicates(KEYS, keep="last").reset_index(drop=True)
    return rows, panel, coverage


def load_current_demand_history() -> pd.DataFrame:
    history = pd.read_parquet(PREDICTIONS, columns=KEYS + ["demand"])
    history["date"] = pd.to_datetime(history["date"]).dt.normalize()
    history = history[history["date"].le(END)].copy()
    history["dow"] = history["date"].dt.dayofweek
    history["is_weekend"] = history["dow"].ge(5)
    return history


def prior_year_history(panel: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    pairs = rows[["bakery_id", "product_id"]].drop_duplicates()
    history = panel[
        panel["date"].between("2025-02-01", "2025-08-31")
    ][KEYS + ["observed_sales_qty"]].copy()
    history = history.merge(pairs, on=["bakery_id", "product_id"], how="inner")
    history = history.rename(columns={"observed_sales_qty": "sales_qty"})
    history["sales_qty"] = pd.to_numeric(history["sales_qty"], errors="coerce").fillna(0.0).clip(lower=0.0)
    history["month"] = history["date"].dt.month
    history["is_weekend"] = history["date"].dt.dayofweek.ge(5)
    return history


def add_variants(
    rows: pd.DataFrame,
    demand_history: pd.DataFrame,
    prior_history: pd.DataFrame,
) -> pd.DataFrame:
    print("Building causal 28-day anchors...", flush=True)
    guards = add_guardrails(
        rows[KEYS + ["forecast_qty"]],
        demand_history[KEYS + ["demand", "dow", "is_weekend"]],
    )
    work = rows.merge(
        guards[KEYS + ["plain_anchor_28", "plain_fixed_lower_28", "plain_fixed_upper_28"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    regime_helpers.TARGET_MONTHS = TARGET_MONTHS
    early = work[work["date"].dt.day.le(regime_helpers.SEASONAL_DECAY_DAYS)].copy()
    print(f"Building prior-year regimes for {len(early):,} early-month rows...", flush=True)
    seasonal = regime_helpers.build_seasonal_priors(early, prior_history)
    current = regime_helpers.build_current_growth(early, demand_history)
    early_signals = seasonal.merge(current, on=KEYS, validate="one_to_one")
    work = work.merge(early_signals, on=KEYS, how="left", validate="one_to_one")
    defaults = {
        "prior_year_seasonal_ratio": 1.0,
        "seasonal_weight": 0.0,
        "effective_seasonal_ratio": 1.0,
        "prior_year_pair_days": 0.0,
        "current_growth_ratio": 1.0,
        "current_growth_comparable": False,
    }
    for column, value in defaults.items():
        work[column] = work[column].fillna(value)

    direct = work["forecast_qty"]
    lower = work["plain_fixed_lower_28"].fillna(direct)
    fixed_upper = work["plain_fixed_upper_28"].fillna(direct)
    work["floor_only"] = np.maximum(direct, lower)
    work["symmetric_guard_28"] = direct.clip(lower=lower, upper=fixed_upper)

    anchor = work["plain_anchor_28"].fillna(direct)
    seasonal_anchor = anchor * work["effective_seasonal_ratio"]
    work["seasonal_upper"] = seasonal_anchor + np.maximum(0.15 * seasonal_anchor, 2.0)
    work["seasonal_guard"] = work["floor_only"].clip(upper=work["seasonal_upper"])
    work["regime_confirmed"] = (
        work["prior_year_seasonal_ratio"].ge(regime_helpers.SEASONAL_MIN_RATIO)
        & work["current_growth_comparable"].astype(bool)
        & work["current_growth_ratio"].ge(regime_helpers.CURRENT_MIN_RATIO)
    )
    work["regime_upper"] = fixed_upper
    confirmed = work["regime_confirmed"]
    work.loc[confirmed, "regime_upper"] = work.loc[confirmed, "seasonal_upper"]
    work["regime_aware_guard"] = work["floor_only"].clip(upper=work["regime_upper"])
    return work


def monthly_summary(
    simulations: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    records = []
    for variant, simulation in simulations.items():
        work = simulation.copy()
        work["month"] = work["date"].dt.to_period("M").astype(str)
        grouped = work.groupby("month", as_index=False).agg(
            rows=("date", "size"),
            production=("production", "sum"),
            served=("served", "sum"),
            lost=("lost", "sum"),
            writeoff=("writeoff", "sum"),
            gross_profit=("gross_profit", "sum"),
        )
        grouped["variant"] = variant
        records.append(grouped)
    summary = pd.concat(records, ignore_index=True)
    for month, indices in summary.groupby("month").groups.items():
        block = summary.loc[indices]
        actual = block[block["variant"].eq("actual_state")].iloc[0]
        direct = block[block["variant"].eq("direct")].iloc[0]
        summary.loc[indices, "gross_profit_delta_vs_actual"] = block["gross_profit"] - actual["gross_profit"]
        summary.loc[indices, "gross_profit_delta_vs_direct"] = block["gross_profit"] - direct["gross_profit"]
        summary.loc[indices, "production_delta_vs_actual"] = block["production"] - actual["production"]
        summary.loc[indices, "lost_delta_vs_direct"] = block["lost"] - direct["lost"]
        summary.loc[indices, "writeoff_delta_vs_direct"] = block["writeoff"] - direct["writeoff"]
    return summary.sort_values(["month", "variant"])


def case_summary(
    simulations: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    actual = simulations["actual_state"][KEYS + ["lost"]].rename(columns={"lost": "actual_lost"})
    direct = simulations["direct"][KEYS + ["lost"]].rename(columns={"lost": "direct_lost"})
    records = []
    for variant, simulation in simulations.items():
        if variant == "actual_state":
            continue
        paired = simulation[KEYS + ["lost"]].merge(actual, on=KEYS, validate="one_to_one")
        paired = paired.merge(direct, on=KEYS, validate="one_to_one")
        paired["month"] = paired["date"].dt.to_period("M").astype(str)
        for month, group in paired.groupby("month"):
            actual_case = group["actual_lost"].ge(1.0)
            model_case = group["lost"].ge(1.0)
            records.append(
                {
                    "month": month,
                    "variant": variant,
                    "lost_cases": int(model_case.sum()),
                    "new_cases_vs_actual": int((~actual_case & model_case).sum()),
                    "fixed_cases_vs_actual": int((actual_case & ~model_case).sum()),
                    "improved_rows_vs_direct": int(((group["direct_lost"] - group["lost"]) >= 1.0).sum()),
                    "worsened_rows_vs_direct": int(((group["lost"] - group["direct_lost"]) >= 1.0).sum()),
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    print("Loading canonical historical inputs...", flush=True)
    rows, panel, coverage = load_rows()
    demand_history = load_current_demand_history()
    prior_history = prior_year_history(panel, rows)
    print(f"Rows: {len(rows):,}; coverage: {coverage}", flush=True)
    print("Adding freshness priors and reconciliation...", flush=True)
    rows = add_causal_freshness_priors(rows)
    rows = add_reconciliation(rows)
    rows = add_variants(rows, demand_history, prior_history)

    simulations: dict[str, pd.DataFrame] = {"actual_state": actual_rows(rows)}
    for variant, column in VARIANT_COLUMNS.items():
        print(f"Simulating {variant}...", flush=True)
        simulations[variant] = simulate_stateful(rows, column)

    monthly = monthly_summary(simulations)
    cases = case_summary(simulations)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUTPUT / "monthly_summary.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(OUTPUT / "case_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([coverage]).to_csv(OUTPUT / "coverage.csv", index=False, encoding="utf-8-sig")
    decision_columns = KEYS + [
        "forecast_qty",
        "demand",
        "plain_anchor_28",
        "plain_fixed_lower_28",
        "plain_fixed_upper_28",
        "prior_year_seasonal_ratio",
        "seasonal_weight",
        "effective_seasonal_ratio",
        "current_growth_ratio",
        "current_growth_comparable",
        "regime_confirmed",
        "floor_only",
        "symmetric_guard_28",
        "seasonal_upper",
        "seasonal_guard",
        "regime_aware_guard",
    ]
    rows[decision_columns].to_parquet(OUTPUT / "decision_rows.parquet", index=False)
    economics_columns = KEYS + ["production", "served", "lost", "writeoff", "gross_profit"]
    for variant, simulation in simulations.items():
        simulation[economics_columns].to_parquet(OUTPUT / f"economics_{variant}.parquet", index=False)
    print("\nMONTHLY", flush=True)
    print(monthly.to_string(index=False), flush=True)
    print("\nCASES", flush=True)
    print(cases.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

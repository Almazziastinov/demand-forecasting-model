"""Apply the 28-day demand floor and gated cap to the canonical August test."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.backtest_direct_demand_guardrail import (  # noqa: E402
    KEYS,
    add_guardrails,
)
from scripts.evaluate_three_model_august_holdout import (  # noqa: E402
    add_causal_freshness_priors,
    add_plan_columns,
    add_reconciliation,
    load_base_rows,
)
from scripts.experiment_direct_old_stock_credit import simulate_group  # noqa: E402


OUTPUT = ROOT / "reports/canonical_direct_demand_guardrail_20260914"
SOURCE_OUTPUT = ROOT / "reports/three_prod_models_august_holdout_20260911"
DEMAND_HISTORY = (
    ROOT
    / "reports/historical_daily_retraining_gate_loss075_full_20260910"
    / "predictions.parquet"
)
CAUSAL_PANEL = ROOT / ".codex_tmp/historical_clickhouse_panel/causal_panel.parquet"


def load_canonical_history() -> pd.DataFrame:
    history = pd.read_parquet(
        DEMAND_HISTORY,
        columns=KEYS
        + [
            "direct_plan",
            "observed_sales_qty",
            "lost_reconstructed",
            "demand",
        ],
    )
    history["date"] = pd.to_datetime(history["date"]).dt.normalize()
    flows = pd.read_parquet(
        CAUSAL_PANEL,
        columns=KEYS
        + [
            "release_qty",
            "incoming_move_qty",
            "outgoing_move_qty",
            "written_off_qty",
        ],
    )
    flows["date"] = pd.to_datetime(flows["date"]).dt.normalize()
    history = history.merge(
        flows,
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    numeric = [
        "direct_plan",
        "observed_sales_qty",
        "lost_reconstructed",
        "demand",
        "release_qty",
        "incoming_move_qty",
        "outgoing_move_qty",
        "written_off_qty",
    ]
    history[numeric] = history[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    available = (
        history["release_qty"]
        + history["incoming_move_qty"]
        - history["outgoing_move_qty"]
    ).clip(lower=0.0)
    history["observable_closing"] = (
        available - history["observed_sales_qty"] - history["written_off_qty"]
    ).clip(lower=0.0)
    history["dow"] = history["date"].dt.dayofweek
    history["is_weekend"] = history["dow"].ge(5)
    return history


def add_upper_gate(rows: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    lookup = history.set_index(KEYS)[
        ["direct_plan", "demand", "lost_reconstructed", "observable_closing"]
    ].to_dict("index")
    pair_history = {
        key: group.sort_values("date")
        for key, group in history.groupby(["bakery_id", "product_id"], sort=False)
    }
    gate_rows = []
    for row in rows[KEYS].itertuples(index=False):
        confirmations = 0
        comparable = 0
        for lag in [7, 14, 21]:
            key = (row.date - pd.Timedelta(days=lag), row.bakery_id, row.product_id)
            observation = lookup.get(key)
            if observation is None or pd.isna(observation["direct_plan"]):
                continue
            comparable += 1
            demand = float(observation["demand"])
            threshold = demand + max(0.15 * demand, 2.0)
            confirmed_overforecast = (
                float(observation["direct_plan"]) > threshold
                and float(observation["lost_reconstructed"]) <= 1e-9
                and float(observation["observable_closing"]) > 0.0
            )
            confirmations += int(confirmed_overforecast)

        pair = pair_history.get((row.bakery_id, row.product_id))
        no_growth = False
        if pair is not None:
            age = (row.date - pair["date"]).dt.days
            same_type = pair["is_weekend"].eq(row.date.dayofweek >= 5)
            recent = pair[same_type & age.between(1, 14)]["demand"]
            prior = pair[same_type & age.between(15, 28)]["demand"]
            if len(recent) >= 2 and len(prior) >= 2:
                no_growth = float(recent.mean()) <= float(prior.mean()) * 1.05 + 1.0
        gate_rows.append(
            {
                "date": row.date,
                "bakery_id": row.bakery_id,
                "product_id": row.product_id,
                "upper_confirmations": confirmations,
                "upper_comparable_lags": comparable,
                "upper_no_growth": no_growth,
                "upper_gate": confirmations >= 2 and no_growth,
            }
        )
    return rows.merge(pd.DataFrame(gate_rows), on=KEYS, how="left", validate="one_to_one")


def summarize(
    label: str,
    simulation: pd.DataFrame,
    actual_lost: pd.DataFrame,
    actual_gp: float,
) -> dict[str, object]:
    paired = actual_lost.merge(
        simulation[KEYS + ["lost"]], on=KEYS, how="inner", validate="one_to_one"
    )
    actual_case = paired["actual_lost"].ge(1.0)
    model_case = paired["lost"].ge(1.0)
    delta = paired["actual_lost"] - paired["lost"]
    return {
        "variant": label,
        "production": simulation["production"].sum(),
        "served": simulation["served"].sum(),
        "lost": simulation["lost"].sum(),
        "writeoff": simulation["writeoff"].sum(),
        "gross_profit": simulation["gross_profit"].sum(),
        "gross_profit_delta_vs_actual": simulation["gross_profit"].sum() - actual_gp,
        "improved_actual_lost_cases": int((actual_case & delta.ge(1.0)).sum()),
        "new_lost_cases": int((~actual_case & model_case).sum()),
        "worsened_rows": int(delta.le(-1.0).sum()),
        "net_recovered_units_vs_actual": delta.sum(),
    }


def main() -> None:
    rows, _ = load_base_rows()
    rows = add_plan_columns(rows)
    direct_plan_column = next(
        column
        for column, label in rows.attrs["plan_labels"].items()
        if label == "direct_alpha_025_v1"
    )
    rows = add_causal_freshness_priors(rows)
    rows = add_reconciliation(rows)

    history = load_canonical_history()
    guard_input = rows[KEYS].copy()
    guard_input["forecast_qty"] = rows[direct_plan_column].to_numpy()
    guard = add_guardrails(
        guard_input,
        history[KEYS + ["demand", "dow", "is_weekend"]],
    )
    rows = rows.merge(
        guard[
            KEYS
            + [
                "plain_fixed_lower_28",
                "plain_fixed_upper_28",
                "plain_anchor_28",
            ]
        ],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    rows = add_upper_gate(rows, history)

    direct = rows[direct_plan_column].clip(lower=0.0)
    lower = rows["plain_fixed_lower_28"]
    upper = rows["plain_fixed_upper_28"]
    rows["demand_floor_28"] = np.maximum(direct, lower.fillna(direct))
    for band_pct in [10, 20, 25, 30]:
        width = np.maximum(rows["plain_anchor_28"] * (band_pct / 100.0), 2.0)
        band_lower = (rows["plain_anchor_28"] - width).clip(lower=0.0)
        rows[f"demand_floor_band_{band_pct}"] = np.maximum(
            direct, band_lower.fillna(direct)
        )
    rows["symmetric_guard_28"] = direct.clip(lower=lower, upper=upper).fillna(direct)
    rows["floor_gated_upper_28"] = rows["demand_floor_28"]
    gated = rows["upper_gate"].fillna(False) & upper.notna()
    rows.loc[gated, "floor_gated_upper_28"] = np.minimum(
        rows.loc[gated, "demand_floor_28"], upper[gated]
    )

    actual = pd.read_parquet(
        SOURCE_OUTPUT / "economics_actual_state.parquet",
        columns=KEYS + ["lost", "gross_profit"],
    )
    actual_lost = actual[KEYS + ["lost"]].rename(columns={"lost": "actual_lost"})
    actual_gp = float(actual["gross_profit"].sum())

    variants = {
        "current_full_credit": direct_plan_column,
        "demand_floor_band_10": "demand_floor_band_10",
        "demand_floor_28": "demand_floor_28",
        "demand_floor_band_20": "demand_floor_band_20",
        "demand_floor_band_25": "demand_floor_band_25",
        "demand_floor_band_30": "demand_floor_band_30",
        "symmetric_guard_28": "symmetric_guard_28",
        "floor_gated_upper_28": "floor_gated_upper_28",
    }
    simulations = {}
    summaries = []
    for label, plan_column in variants.items():
        simulation = pd.concat(
            [
                simulate_group(group, plan_column, "full", 1.0)
                for _, group in rows.groupby(["bakery_id", "product_id"], sort=False)
            ],
            ignore_index=True,
        )
        simulations[label] = simulation
        summaries.append(summarize(label, simulation, actual_lost, actual_gp))

    summary = pd.DataFrame(summaries)
    baseline = summary[summary["variant"].eq("current_full_credit")].iloc[0]
    for column in ["production", "lost", "writeoff", "gross_profit", "new_lost_cases"]:
        summary[f"{column}_delta_vs_direct"] = summary[column] - baseline[column]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT / "summary.csv", index=False, encoding="utf-8-sig")
    rows[
        KEYS
        + [
            direct_plan_column,
            "plain_anchor_28",
            "plain_fixed_lower_28",
            "plain_fixed_upper_28",
            "upper_confirmations",
            "upper_no_growth",
            "upper_gate",
            "demand_floor_28",
            "demand_floor_band_10",
            "demand_floor_band_20",
            "demand_floor_band_25",
            "demand_floor_band_30",
            "symmetric_guard_28",
            "floor_gated_upper_28",
        ]
    ].to_parquet(OUTPUT / "decision_rows.parquet", index=False)
    for label, simulation in simulations.items():
        simulation.to_parquet(OUTPUT / f"economics_{label}.parquet", index=False)
    print(summary.to_string(index=False))
    print("\nCoverage")
    print(
        {
            "rows": len(rows),
            "anchor_rows": int(rows["plain_anchor_28"].notna().sum()),
            "floor_changed_rows": int((rows["demand_floor_28"] > direct + 1e-9).sum()),
            "upper_gate_rows": int(rows["upper_gate"].fillna(False).sum()),
            "upper_gate_changed_rows": int(
                (rows["floor_gated_upper_28"] < rows["demand_floor_28"] - 1e-9).sum()
            ),
        }
    )


if __name__ == "__main__":
    main()

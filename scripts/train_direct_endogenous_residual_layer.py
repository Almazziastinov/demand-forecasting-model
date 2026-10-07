"""Train Direct residual risk on endogenous inventory-simulation outcomes."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_three_model_august_holdout import (  # noqa: E402
    add_causal_freshness_priors,
)
from scripts.experiment_direct_risk_reallocation import (  # noqa: E402
    build_candidate,
    evaluate,
)
from scripts.run_freshness_calibrated_economics import simulate_group  # noqa: E402
from scripts.train_direct_residual_layer import (  # noqa: E402
    TRAIN_END,
    VALIDATION_END,
    VALIDATION_START,
    add_derived_features,
    build_test_rows,
    classification_metrics,
    fit_models,
    quantile_candidates,
    score_model,
)
from src.experiments_v2.direct_alpha_allocation import (  # noqa: E402
    DirectAlphaAllocationConfig,
    build_selected_direct_plan,
)


ROLLING = ROOT / "reports/rolling_direct_uplift_floor_20260827/rows.parquet"
DEMAND = (
    ROOT
    / "reports/historical_daily_retraining_gate_loss075_full_20260910"
    / "predictions.parquet"
)
PANEL = ROOT / ".codex_tmp/historical_clickhouse_panel/causal_panel.parquet"
MAPPING = ROOT / "reports/markup_price_mapping_20260826/mapped_products.csv"
ECONOMICS_DIR = ROOT / "reports/three_prod_models_august_holdout_20260911"
TEST_INPUT = ECONOMICS_DIR / "evaluation_input.parquet"
OUTPUT = ROOT / "reports/direct_endogenous_residual_layer_20260911"
KEYS = ["date", "bakery_id", "product_id"]


def add_sequence_id(rows: pd.DataFrame) -> pd.DataFrame:
    result = rows.sort_values(["bakery_id", "product_id", "date"]).copy()
    groups = result.groupby(["bakery_id", "product_id"], sort=False)
    date_gap = groups["date"].diff().dt.days.ne(1)
    fold_change = groups["rolling_fold"].shift().ne(result["rolling_fold"])
    result["new_sequence"] = date_gap | fold_change
    result["sequence_id"] = result.groupby(
        ["bakery_id", "product_id"], sort=False
    )["new_sequence"].cumsum()
    return result.sort_index()


def add_segmented_reconciliation(rows: pd.DataFrame) -> pd.DataFrame:
    reconciled: list[tuple[int, float, float]] = []
    group_keys = ["bakery_id", "product_id", "sequence_id"]
    for _, group in rows.groupby(group_keys, sort=False):
        carry = 0.0
        for index, row in group.sort_values("date").iterrows():
            old = carry
            old_reconciliation = max(float(row["actual_sold_old"] - old), 0.0)
            old += old_reconciliation
            old -= min(old, row["actual_sold_old"])
            other_out = row["outgoing_move_qty"] + row["written_off_qty"]
            from_old = min(old, other_out)
            fresh_required = row["actual_sold_fresh"] + other_out - from_old
            fresh_known = row["release_qty"] + row["incoming_move_qty"]
            fresh_reconciliation = max(float(fresh_required - fresh_known), 0.0)
            carry = max(float(fresh_known + fresh_reconciliation - fresh_required), 0.0)
            reconciled.append((index, old_reconciliation, fresh_reconciliation))
    values = pd.DataFrame(
        reconciled,
        columns=["row_index", "old_reconciliation_in", "fresh_reconciliation_in"],
    ).set_index("row_index")
    result = rows.join(values, how="left")
    columns = ["old_reconciliation_in", "fresh_reconciliation_in"]
    result[columns] = result[columns].fillna(0.0)
    result["reconciliation_in"] = result[columns].sum(axis=1)
    return result


def build_endogenous_history() -> pd.DataFrame:
    rows = pd.read_parquet(ROLLING)
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    rows = rows[rows["scenario"].eq("calibrated")].copy()
    pilot_ids = pd.read_parquet(TEST_INPUT, columns=["bakery_id"])[
        "bakery_id"
    ].unique()
    rows = rows[rows["bakery_id"].isin(pilot_ids)].copy()
    rows["loss_scale"] = 1.0
    rows = build_selected_direct_plan(rows, DirectAlphaAllocationConfig())

    demand = pd.read_parquet(
        DEMAND,
        columns=KEYS + ["demand", "observed_sales_qty", "observed_sales_amount"],
    )
    demand["date"] = pd.to_datetime(demand["date"]).dt.normalize()
    rows = rows.merge(demand, on=KEYS, how="inner", validate="one_to_one")

    flows = pd.read_parquet(
        PANEL,
        columns=KEYS
        + ["release_qty", "incoming_move_qty", "outgoing_move_qty", "written_off_qty"],
    )
    flows["date"] = pd.to_datetime(flows["date"]).dt.normalize()
    rows = rows.merge(flows, on=KEYS, how="left", validate="one_to_one")
    flow_columns = [
        "release_qty",
        "incoming_move_qty",
        "outgoing_move_qty",
        "written_off_qty",
    ]
    rows[flow_columns] = rows[flow_columns].fillna(0.0).clip(lower=0.0)

    mapping = pd.read_csv(MAPPING, encoding="utf-8-sig")
    mapping = (
        mapping[mapping["valid_economics"].astype(bool)]
        .sort_values("unit_price")
        .drop_duplicates("product_id", keep="last")
    )
    rows = rows.merge(
        mapping[["product_id", "unit_cost", "unit_price"]],
        on="product_id",
        how="inner",
        validate="many_to_one",
    )
    rows = add_causal_freshness_priors(rows)
    rows = add_sequence_id(rows)
    rows = add_segmented_reconciliation(rows)

    simulations = []
    group_keys = ["bakery_id", "product_id", "sequence_id"]
    for _, group in rows.groupby(group_keys, sort=False):
        simulations.append(simulate_group(group, "selected_sku_forecast"))
    simulation = pd.concat(simulations, ignore_index=True).rename(
        columns={"lost": "endogenous_lost", "served": "endogenous_served"}
    )
    rows = rows.merge(
        simulation[KEYS + ["endogenous_lost", "endogenous_served"]],
        on=KEYS,
        validate="one_to_one",
    )
    rows = add_derived_features(rows)
    rows["residual_qty"] = rows["endogenous_lost"]
    rows["residual_case"] = rows["endogenous_lost"].ge(1.0).astype("int8")
    rows["certain_residual"] = (
        rows["observed_sales_qty"] - rows["endogenous_served"]
    ).ge(1.0)
    rows["label_weight"] = np.where(
        rows["residual_case"].eq(1),
        np.where(rows["certain_residual"], 1.0, 0.35),
        0.70,
    )
    return rows


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    history = build_endogenous_history()
    history.to_parquet(OUTPUT / "endogenous_training_panel.parquet", index=False)
    train = history[history["date"].le(TRAIN_END)].copy()
    validation = history[
        history["date"].between(VALIDATION_START, VALIDATION_END)
    ].copy()
    classifier, severity = fit_models(train, validation)
    validation = score_model(validation, classifier, severity)

    economics_input = pd.read_parquet(TEST_INPUT)
    economics_input["date"] = pd.to_datetime(economics_input["date"]).dt.normalize()
    test = build_test_rows(economics_input[KEYS])
    test = score_model(test, classifier, severity)
    direct = pd.read_parquet(
        ECONOMICS_DIR / "economics_direct_alpha_025_v1.parquet"
    )
    direct_labels = direct[KEYS + ["lost", "served"]].rename(
        columns={"lost": "residual_qty", "served": "endogenous_served"}
    )
    test_scored = test.merge(direct_labels, on=KEYS, validate="one_to_one")
    test_scored["residual_case"] = test_scored["residual_qty"].ge(1.0).astype("int8")

    metric_rows = [
        classification_metrics(validation, "validation_2026-08-17_23"),
        classification_metrics(test_scored, "test_2026-08-24_31"),
    ]
    pd.DataFrame(metric_rows).to_csv(
        OUTPUT / "classifier_metrics.csv", index=False, encoding="utf-8-sig"
    )

    actual = pd.read_parquet(ECONOMICS_DIR / "economics_actual_state.parquet")
    actual_cases = actual["lost"].ge(1.0)
    actual_gp = float(actual["gross_profit"].sum())
    records = [
        {
            "variant": "direct_alpha_025_v1",
            "transferred_qty": 0.0,
            "lost_cases": int(direct["lost"].ge(1.0).sum()),
            "case_delta_vs_actual": int(
                direct["lost"].ge(1.0).sum() - actual_cases.sum()
            ),
            "lost_qty": float(direct["lost"].sum()),
            "writeoff_qty": float(direct["writeoff"].sum()),
            "service_pct": 100 * direct["served"].sum() / direct["demand"].sum(),
            "gross_profit": float(direct["gross_profit"].sum()),
            "gp_delta_vs_actual": float(direct["gross_profit"].sum() - actual_gp),
        }
    ]

    allocator_features = test.copy()
    allocator_features["predicted_stockout_probability"] = allocator_features[
        "residual_probability"
    ]
    allocator_features["predicted_lost_if_stockout"] = allocator_features[
        "residual_severity"
    ]
    allocator_features["predictive_uplift"] = allocator_features[
        "residual_expected"
    ]
    candidates = quantile_candidates(validation)
    for index, candidate in enumerate(candidates):
        plan_column = f"endogenous_residual_plan_{index}"
        plan = build_candidate(allocator_features, candidate)
        economics_input[plan_column] = plan
        transferred = 0.5 * float(
            (plan - allocator_features["selected_sku_forecast"]).abs().sum()
        )
        record, simulation = evaluate(
            economics_input,
            plan_column,
            f"endogenous_{candidate.name}",
            actual_cases,
            actual_gp,
            transferred,
        )
        records.append(record)
        simulation.to_parquet(
            OUTPUT / f"economics_endogenous_{candidate.name}.parquet"
        )

    summary = pd.DataFrame(records).sort_values(
        ["lost_cases", "gross_profit"], ascending=[True, False]
    )
    summary.to_csv(OUTPUT / "economics_summary.csv", index=False, encoding="utf-8-sig")
    test_scored.to_parquet(OUTPUT / "test_scores.parquet", index=False)
    joblib.dump(classifier, OUTPUT / "residual_classifier.joblib")
    joblib.dump(severity, OUTPUT / "residual_severity.joblib")
    metadata = {
        "train_end": str(TRAIN_END.date()),
        "validation": [str(VALIDATION_START.date()), str(VALIDATION_END.date())],
        "test": ["2026-08-24", "2026-08-31"],
        "label": "endogenous Direct inventory-simulation lost >= 1",
        "sequence_reset": "rolling fold change or calendar gap greater than one day",
        "candidates": [asdict(candidate) for candidate in candidates],
        "production_write": False,
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

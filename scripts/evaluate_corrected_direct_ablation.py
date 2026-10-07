"""Ablate corrected Direct base and loss components on the August holdout."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.backtest_direct_bakery_sku_allocation import FEATURES  # noqa: E402
from scripts.build_direct_uplift_floor_candidates import add_floor_reference  # noqa: E402
from scripts.evaluate_weighted_loss_direct import KEYS, SOURCE, summarize  # noqa: E402
from scripts.run_freshness_calibrated_economics import simulate_group  # noqa: E402
from src.experiments_v2.direct_alpha_allocation import (  # noqa: E402
    DAY_KEYS,
    DirectAlphaAllocationConfig,
    build_selected_direct_plan,
)


CURRENT = ROOT / "models/direct_alpha_025_v1"
CORRECTED = ROOT / "models/direct_corrected_demand_gate30_v1"
INPUTS = ROOT / "reports/weighted_loss_direct_august_holdout_20260914/direct_days"
OUTPUT = ROOT / "reports/corrected_direct_august_holdout_20260914"
FLOOR_COLUMNS = [
    "floor_history_n",
    "floor_demand_p67",
    "historical_stockout_rate",
    "historical_lost_mean",
    "historical_volume",
]


def load_floor(path: Path) -> pd.DataFrame:
    parquet = path / "floor_history.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    return pd.read_csv(path / "floor_history.csv.gz")


def build_plan(
    base_path: Path,
    loss_path: Path,
    *,
    use_uplift: bool = True,
    use_floor: bool = True,
) -> pd.DataFrame:
    direct = joblib.load(base_path / "direct_model.joblib")
    classifier = joblib.load(loss_path / "stockout_classifier.joblib")
    severity = joblib.load(loss_path / "lost_severity_model.joblib")
    metadata = json.loads((base_path / "metadata.json").read_text(encoding="utf-8"))
    factors = {int(key): value for key, value in metadata["p50_factors"].items()}
    floor = load_floor(loss_path)
    floor["date"] = pd.to_datetime(floor["date"]).dt.normalize()
    floor["product_id"] = floor["product_id"].astype("int64")
    parts = []
    for input_path in sorted(INPUTS.glob("*/shadow_input.parquet")):
        rows = pd.read_parquet(input_path).drop(columns=FLOOR_COLUMNS, errors="ignore")
        rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
        rows["direct_raw_demand"] = np.maximum(direct.predict(rows[FEATURES]), 1e-9)
        groups = [rows[key] for key in DAY_KEYS]
        raw_total = rows["direct_raw_demand"].groupby(groups).transform("sum")
        bakery_total = rows["incumbent_sku_forecast"].groupby(groups).transform("sum")
        rows["direct_forecast"] = rows["direct_raw_demand"] / raw_total * bakery_total
        rows["direct_p50"] = rows["direct_forecast"] * rows["bakery_id"].map(factors).fillna(
            metadata["p50_fallback"]
        )
        rows["predicted_stockout_probability"] = classifier.predict_proba(rows[FEATURES])[:, 1]
        rows["predicted_lost_if_stockout"] = np.expm1(severity.predict(rows[FEATURES])).clip(
            min=0.0
        )
        rows["predictive_uplift"] = (
            rows["predicted_stockout_probability"] * rows["predicted_lost_if_stockout"]
            if use_uplift
            else 0.0
        )
        rows["loss_scale"] = 1.0
        if use_floor:
            rows = add_floor_reference(rows, floor)
        else:
            rows["floor_history_n"] = 0
            rows["floor_demand_p67"] = 0.0
            rows["historical_stockout_rate"] = 0.0
            rows["historical_lost_mean"] = 0.0
            rows["historical_volume"] = 0.0
        result = build_selected_direct_plan(rows, DirectAlphaAllocationConfig())
        parts.append(result[KEYS + ["selected_sku_forecast"]])
    return pd.concat(parts, ignore_index=True)


def simulate_volume_neutral(plan: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = pd.read_parquet(SOURCE / "evaluation_input.parquet")
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    plan = plan.rename(columns={"selected_sku_forecast": "candidate_plan"})
    rows = rows.merge(plan, on=KEYS, how="left", validate="one_to_one")
    rows["candidate_plan"] = rows["candidate_plan"].fillna(0.0).clip(lower=0.0)
    groups = [rows["date"], rows["bakery_id"]]
    current_total = rows["plan_1"].groupby(groups).transform("sum")
    candidate_total = rows["candidate_plan"].groupby(groups).transform("sum")
    rows["neutral_plan"] = rows["candidate_plan"] * (
        current_total / candidate_total.replace(0.0, pd.NA)
    ).fillna(1.0)
    simulation = pd.concat(
        [
            simulate_group(group, "neutral_plan")
            for _, group in rows.groupby(["bakery_id", "product_id"], sort=False)
        ],
        ignore_index=True,
    )
    simulation.to_parquet(OUTPUT / f"economics_{label}.parquet", index=False)
    return simulation


def main() -> None:
    variants = {
        "new_base_only": (CORRECTED, CURRENT, False, False),
        "new_base_old_floor_only": (CORRECTED, CURRENT, False, True),
        "new_base_old_uplift_only": (CORRECTED, CURRENT, True, False),
        "new_base_old_loss": (CORRECTED, CURRENT, True, True),
        "old_base_new_loss": (CURRENT, CORRECTED, True, True),
    }
    actual = pd.read_parquet(SOURCE / "economics_actual_state.parquet")
    current = pd.read_parquet(SOURCE / "economics_direct_alpha_025_v1.parquet")
    actual_gp = float(actual["gross_profit"].sum())
    current_gp = float(current["gross_profit"].sum())
    summaries = [summarize("direct_alpha_025_v1", current, actual_gp)]
    for label, (base_path, loss_path, use_uplift, use_floor) in variants.items():
        result_path = OUTPUT / f"economics_{label}.parquet"
        if result_path.exists():
            simulation = pd.read_parquet(result_path)
        else:
            plan = build_plan(
                base_path,
                loss_path,
                use_uplift=use_uplift,
                use_floor=use_floor,
            )
            simulation = simulate_volume_neutral(plan, label)
        summaries.append(summarize(label, simulation, actual_gp))
    full = pd.read_parquet(OUTPUT / "economics_corrected_direct_gate30_volume_neutral.parquet")
    summaries.append(summarize("new_base_new_loss", full, actual_gp))
    summary = pd.DataFrame(summaries)
    summary["gp_delta_vs_current_direct"] = summary["gross_profit"] - current_gp
    summary.to_csv(OUTPUT / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

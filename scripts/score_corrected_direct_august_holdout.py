"""Score corrected-demand Direct artifacts on frozen August shadow inputs."""

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
from scripts.evaluate_weighted_loss_direct import KEYS  # noqa: E402
from src.experiments_v2.direct_alpha_allocation import (  # noqa: E402
    DAY_KEYS,
    DirectAlphaAllocationConfig,
    build_selected_direct_plan,
)


ARTIFACTS = ROOT / "models/direct_corrected_demand_gate30_v1"
INPUTS = ROOT / "reports/weighted_loss_direct_august_holdout_20260914/direct_days"
OUTPUT = ROOT / "reports/corrected_direct_august_holdout_20260914"
FLOOR_COLUMNS = [
    "floor_history_n",
    "floor_demand_p67",
    "historical_stockout_rate",
    "historical_lost_mean",
    "historical_volume",
]


def main() -> None:
    direct = joblib.load(ARTIFACTS / "direct_model.joblib")
    classifier = joblib.load(ARTIFACTS / "stockout_classifier.joblib")
    severity = joblib.load(ARTIFACTS / "lost_severity_model.joblib")
    metadata = json.loads((ARTIFACTS / "metadata.json").read_text(encoding="utf-8"))
    factors = {int(key): value for key, value in metadata["p50_factors"].items()}
    labels = pd.read_parquet(ARTIFACTS / "floor_history.parquet")
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    labels["product_id"] = labels["product_id"].astype("int64")

    plan_parts = []
    detail_parts = []
    for input_path in sorted(INPUTS.glob("*/shadow_input.parquet")):
        rows = pd.read_parquet(input_path).drop(columns=FLOOR_COLUMNS, errors="ignore")
        rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
        rows["direct_raw_demand"] = np.maximum(direct.predict(rows[FEATURES]), 1e-9)
        raw_total = rows.groupby(DAY_KEYS)["direct_raw_demand"].transform("sum")
        bakery_total = rows.groupby(DAY_KEYS)["incumbent_sku_forecast"].transform("sum")
        rows["direct_forecast"] = rows["direct_raw_demand"] / raw_total * bakery_total
        rows["p50_factor"] = rows["bakery_id"].map(factors).fillna(metadata["p50_fallback"])
        rows["direct_p50"] = rows["direct_forecast"] * rows["p50_factor"]
        rows["predicted_stockout_probability"] = classifier.predict_proba(rows[FEATURES])[:, 1]
        rows["predicted_lost_if_stockout"] = np.expm1(severity.predict(rows[FEATURES])).clip(min=0.0)
        rows["predictive_uplift"] = (
            rows["predicted_stockout_probability"] * rows["predicted_lost_if_stockout"]
        )
        rows["loss_scale"] = 1.0
        rows = add_floor_reference(rows, labels)
        result = build_selected_direct_plan(rows, DirectAlphaAllocationConfig())
        plan_parts.append(result[KEYS + ["selected_sku_forecast"]])
        detail_parts.append(result)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = pd.concat(plan_parts, ignore_index=True)
    detail = pd.concat(detail_parts, ignore_index=True)
    plan.to_parquet(OUTPUT / "plan.parquet", index=False)
    detail.to_parquet(OUTPUT / "detail.parquet", index=False)
    summary = {
        "variant": metadata["version"],
        "dates": int(plan["date"].nunique()),
        "bakeries": int(plan["bakery_id"].nunique()),
        "rows": int(len(plan)),
        "plan_sum": float(plan["selected_sku_forecast"].sum()),
        "production_write": False,
    }
    (OUTPUT / "score_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

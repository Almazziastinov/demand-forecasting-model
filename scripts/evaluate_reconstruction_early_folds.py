"""Screen frozen demand-target variants only on pre-July time folds.

This is a cheap baseline screen, not a substitute for LoRA validation. Later
July–September windows are never read or scored by this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_chronos2_fixed_origin import _sha256, _summary  # noqa: E402
from src.model_tournament.demand_target_contract import (  # noqa: E402
    validate_demand_target_contract,
)
from src.model_tournament.fixed_origin import (  # noqa: E402
    KEYS,
    _same_weekday_formula,
    build_fixed_origin_panel,
    validate_flows,
)
from src.model_tournament.inventory_evidence import inventory_uncertainty  # noqa: E402
from src.model_tournament.series_diagnostics import stage_diagnostics  # noqa: E402


ORIGINS = pd.to_datetime(
    ["2026-03-30", "2026-04-27", "2026-05-25", "2026-06-21"]
).tolist()
TRAIN_CUTOFF = pd.Timestamp("2026-07-05")
KEY_COLUMNS = ["date", *KEYS]
def _candidate_panels(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    evidence = inventory_uncertainty(panel)
    work = panel.merge(evidence, on=KEY_COLUMNS, how="left", validate="one_to_one")
    if work["high_confidence_inventory_context"].isna().any():
        raise ValueError("Inventory evidence misses target rows")
    work["guarded_uplift_qty"] = work["restoration_uplift_qty"].where(
        work["high_confidence_inventory_context"], 0.0
    )
    sales = work["observed_sales_qty"]
    reduction = work["outlier_reduction_qty"]
    work["inventory_guard_restored_qty"] = sales + work["guarded_uplift_qty"]
    work["inventory_guard_full_qty"] = (
        work["inventory_guard_restored_qty"] - reduction
    )
    work["inventory_guard_half_qty"] = (
        sales + 0.5 * work["guarded_uplift_qty"] - reduction
    )
    for demand, uplift in (
        ("inventory_guard_full_qty", work["guarded_uplift_qty"]),
        ("inventory_guard_half_qty", 0.5 * work["guarded_uplift_qty"]),
    ):
        contract = pd.DataFrame(
            {
                "sales": sales,
                "demand": work[demand],
                "uplift": uplift,
                "reduction": reduction,
            }
        )
        validate_demand_target_contract(
            contract,
            sales_column="sales",
            demand_column="demand",
            uplift_column="uplift",
            reduction_column="reduction",
            require_global_uplift=True,
        )
    summary = {
        "restored_rows_before_guard": int(work["restoration_uplift_qty"].gt(0).sum()),
        "restored_rows_after_guard": int(work["guarded_uplift_qty"].gt(0).sum()),
        "gross_uplift_before_guard": float(work["restoration_uplift_qty"].sum()),
        "gross_uplift_after_guard": float(work["guarded_uplift_qty"].sum()),
        "suppressed_possible_carryover_rows": int(
            (work["restoration_uplift_qty"].gt(0) & work["possible_carryover"]).sum()
        ),
        "suppressed_flow_inconsistent_rows": int(
            (
                work["restoration_uplift_qty"].gt(0)
                & work["same_day_supply_inconsistent"]
            ).sum()
        ),
        "outlier_reduction_total": float(reduction.sum()),
        "guard_full_net_uplift_pct": (
            100 * (work["inventory_guard_full_qty"].sum() / sales.sum() - 1)
        ),
        "guard_half_net_uplift_pct": (
            100 * (work["inventory_guard_half_qty"].sum() / sales.sum() - 1)
        ),
    }
    return work, summary


def _forecast_component(
    rows: pd.DataFrame, history: pd.DataFrame, column: str
) -> pd.Series:
    source = history[[*KEY_COLUMNS, column]].rename(
        columns={column: "observed_sales_qty"}
    )
    return pd.Series(_same_weekday_formula(rows, source), index=rows.index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    panel = validate_flows(pd.read_parquet(args.input))
    if panel["date"].max() < TRAIN_CUTOFF:
        raise ValueError("Target panel lacks the full early-fold outcome period")
    work, evidence_summary = _candidate_panels(panel)
    parts = []
    for origin in ORIGINS:
        rows = build_fixed_origin_panel(work, origin)
        history = work.loc[
            work["date"].between(origin - pd.Timedelta(days=55), origin)
        ]
        raw = rows["weighted_weekday_sales"]
        uplift = _forecast_component(rows, history, "restoration_uplift_qty")
        guarded = _forecast_component(rows, history, "guarded_uplift_qty")
        reduction = _forecast_component(rows, history, "outlier_reduction_qty")
        predictions = {
            "raw_sales": raw,
            "current_full": raw + uplift - reduction,
            "inventory_guard_full": raw + guarded - reduction,
            "inventory_guard_half": raw + 0.5 * guarded - reduction,
        }
        for name, values in predictions.items():
            columns = ["forecast_origin", "date", "lead_days", *KEYS, "actual"]
            detail = rows[columns].copy()
            detail["model"] = name
            detail["prediction"] = values.to_numpy(dtype=float)
            parts.append(detail)
    detail = pd.concat(parts, ignore_index=True)
    flags = work[[*KEY_COLUMNS, "broad_stockout_signal"]]
    detail = detail.merge(flags, on=KEY_COLUMNS, how="left", validate="many_to_one")
    if detail["broad_stockout_signal"].isna().any():
        raise ValueError("Early-fold SKU-days lack target-panel flags")
    leaderboard = _summary(detail, ["model"])
    clean = _summary(detail.loc[~detail["broad_stockout_signal"]], ["model"])
    raw_all = leaderboard.set_index("model").loc["raw_sales"]
    raw_clean = clean.set_index("model").loc["raw_sales"]
    candidates = leaderboard.merge(
        clean[["model", "wmape_pct"]].rename(columns={"wmape_pct": "clean_wmape_pct"}),
        on="model", validate="one_to_one",
    )
    candidates["passes_screen"] = (
        candidates["model"].ne("raw_sales")
        & candidates["wmape_pct"].le(raw_all["wmape_pct"])
        & candidates["clean_wmape_pct"].le(raw_clean["wmape_pct"] + 0.1)
        & candidates["under_qty"].le(raw_all["under_qty"])
    )
    accepted = candidates.loc[candidates["passes_screen"]].sort_values(
        ["wmape_pct", "clean_wmape_pct"]
    )
    selected = str(accepted.iloc[0]["model"]) if not accepted.empty else None
    # Full per-series diagnostics use only the early-fold training period.
    diagnostic_input = work.copy()
    diagnostic_input["demand_after_restoration"] = work["inventory_guard_restored_qty"]
    diagnostic_input["reconstructed_demand_qty"] = work["inventory_guard_full_qty"]
    diagnostics = stage_diagnostics(diagnostic_input, cutoff=TRAIN_CUTOFF)
    args.output_dir.mkdir(parents=True)
    detail.to_parquet(args.output_dir / "detail.parquet", index=False)
    candidates.to_csv(args.output_dir / "leaderboard.csv", index=False)
    clean.to_csv(args.output_dir / "clean_subset.csv", index=False)
    _summary(detail, ["forecast_origin", "model"]).to_csv(
        args.output_dir / "by_origin.csv", index=False
    )
    diagnostics.to_parquet(args.output_dir / "series_diagnostics.parquet", index=False)
    target_columns = [
        *KEY_COLUMNS, "observed_sales_qty", "restoration_uplift_qty",
        "guarded_uplift_qty", "outlier_reduction_qty",
        "inventory_guard_full_qty", "inventory_guard_half_qty",
        "possible_carryover", "same_day_supply_inconsistent",
        "prior_day_positive_residual_qty", "same_day_flow_residual_qty",
    ]
    work[target_columns].to_parquet(
        args.output_dir / "target_variants.parquet", index=False
    )
    metadata = {
        "production_write": False,
        "evaluation_target": "original_observed_sales_qty",
        "screen_only_not_lora_validation": True,
        "origins": [str(origin.date()) for origin in ORIGINS],
        "last_evaluation_day": str(TRAIN_CUTOFF.date()),
        "selection_rule": (
            "overall WMAPE <= raw; clean WMAPE <= raw+0.1pp; "
            "underforecast quantity <= raw"
        ),
        "selected_variant": selected,
        "input_sha256": _sha256(args.input),
        "inventory_evidence_is_proxy_not_physical_stock": True,
        "evidence_summary": evidence_summary,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    display_columns = [
        "model", "wmape_pct", "clean_wmape_pct", "under_qty", "passes_screen"
    ]
    print(candidates[display_columns].to_string(index=False))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

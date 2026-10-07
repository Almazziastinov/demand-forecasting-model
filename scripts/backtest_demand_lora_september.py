"""Independent September comparison of sales- and demand-trained LoRA.

Uses an August 31 origin and frozen September outcome facts. A bakery missing
any outcome day is excluded in full, never silently counted as zero sales.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.finetune_chronos2_fixed_origin import require_eval_mode  # noqa: E402
from scripts.finetune_chronos2_reconstructed_demand import _target_flows  # noqa: E402
from scripts.run_chronos2_fixed_origin import (  # noqa: E402
    MODEL_VARIANTS,
    _sha256,
    _summary,
    align_chronos_predictions,
)
from src.model_tournament.fixed_origin import (  # noqa: E402
    KEYS,
    _same_weekday_formula,
    build_fixed_origin_panel,
    validate_flows,
)
from src.model_tournament.nhits_candidate import INPUT_DAYS, dense_history  # noqa: E402


ORIGIN = pd.Timestamp("2026-08-31")
HORIZON_DAYS = 14


def _checked_adapters(
    demand_report: Path, sales_report: Path, target_panel: Path
) -> tuple[Path, Path, str, str]:
    demand_meta = json.loads(
        (demand_report / "metadata.json").read_text(encoding="utf-8")
    )
    sales_meta = json.loads(
        (sales_report / "metadata.json").read_text(encoding="utf-8")
    )
    _, repo, revision = MODEL_VARIANTS["small"]
    if _sha256(target_panel) != demand_meta["input_sha256"]:
        raise ValueError("Demand model target input hash differs from frozen panel")
    if pd.Timestamp(demand_meta["train_cutoff"]) >= ORIGIN:
        raise ValueError("Demand adapter trained through the September origin")
    if pd.Timestamp(sales_meta["train_cutoff"]) >= ORIGIN:
        raise ValueError("Sales adapter trained through the September origin")
    for meta in (demand_meta, sales_meta):
        if meta["model_repo"] != repo or meta["model_revision"] != revision:
            raise ValueError("Adapters do not share the pinned Chronos-2 Small base")
        if meta["num_steps"] != 1000 or not meta["score_valid"]:
            raise ValueError("Expected verified 1,000-step LoRA adapters")
    demand_adapter = (
        demand_report / "checkpoint" / "finetuned-ckpt" / "adapter_model.safetensors"
    )
    sales_adapter = Path(sales_meta["adapter_file"])
    verification = json.loads(
        (demand_report / "checkpoint_verification.json").read_text(encoding="utf-8")
    )
    if _sha256(demand_adapter) != verification["adapter_sha256"]:
        raise ValueError("Demand adapter hash differs from reload verification")
    if _sha256(sales_adapter) != sales_meta["adapter_sha256"]:
        raise ValueError("Sales adapter hash differs from verified report")
    return demand_adapter, sales_adapter, repo, revision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-panel", type=Path, required=True)
    parser.add_argument("--frozen-input", type=Path, required=True)
    parser.add_argument("--demand-report", type=Path, required=True)
    parser.add_argument("--sales-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    demand_adapter, sales_adapter, repo, revision = _checked_adapters(
        args.demand_report, args.sales_report, args.target_panel
    )
    frozen_meta = json.loads(
        (args.frozen_input / "metadata.json").read_text(encoding="utf-8")
    )
    frozen_path = args.frozen_input / "frozen_flows.parquet"
    if _sha256(frozen_path) != frozen_meta["frozen_flows_sha256"]:
        raise ValueError("September outcome source hash differs from frozen metadata")
    target_panel = validate_flows(pd.read_parquet(args.target_panel))
    target_flows = _target_flows(target_panel)
    future = validate_flows(pd.read_parquet(frozen_path))
    future = future.loc[
        future["date"].between(
            ORIGIN + pd.Timedelta(days=1), ORIGIN + pd.Timedelta(days=HORIZON_DAYS)
        )
    ].copy()
    days = pd.date_range(ORIGIN + pd.Timedelta(days=1), periods=HORIZON_DAYS)
    daily_bakeries = {
        day: set(future.loc[future["date"].eq(day), "bakery_id"])
        for day in days
    }
    if any(len(bakeries) < 50 for bakeries in daily_bakeries.values()):
        raise ValueError("September outcome source has insufficient daily coverage")
    stable_bakeries = set.intersection(*daily_bakeries.values())
    if len(stable_bakeries) < 50:
        raise ValueError("Fewer than 50 bakeries have complete daily coverage")
    all_bakeries = set.union(*daily_bakeries.values())
    flows = validate_flows(
        pd.concat(
            [
                target_panel[["date", *KEYS, "observed_sales_qty", "release_qty"]],
                future[["date", *KEYS, "observed_sales_qty", "release_qty"]],
            ],
            ignore_index=True,
        )
    )
    rows = build_fixed_origin_panel(flows, ORIGIN)
    rows = rows.loc[rows["bakery_id"].isin(stable_bakeries)].copy()
    raw_context = dense_history(
        target_panel, rows[KEYS], ORIGIN - pd.Timedelta(days=INPUT_DAYS - 1), ORIGIN
    )
    demand_context = dense_history(
        target_flows, rows[KEYS], ORIGIN - pd.Timedelta(days=INPUT_DAYS - 1), ORIGIN
    )

    import torch
    from chronos import Chronos2Pipeline
    from peft import PeftModel

    torch.set_num_threads(4)
    predict_args = {
        "id_column": "unique_id",
        "timestamp_column": "ds",
        "target": "y",
        "prediction_length": HORIZON_DAYS,
        "quantile_levels": [0.5],
        "batch_size": args.batch_size,
    }
    details = []
    base = Chronos2Pipeline.from_pretrained(repo, revision=revision, device_map="cpu")
    require_eval_mode(base)
    for context, model_id in (
        (raw_context, "chronos2_zero_shot_raw_context"),
        (demand_context, "chronos2_zero_shot_demand_context"),
    ):
        predicted = base.predict_df(context, **predict_args)
        details.append(align_chronos_predictions(rows, predicted, model_id=model_id))
    del base
    for adapter, context, model_id in (
        (sales_adapter, raw_context, "chronos2_lora_raw_sales"),
        (demand_adapter, demand_context, "chronos2_lora_reconstructed_demand"),
    ):
        base = Chronos2Pipeline.from_pretrained(
            repo, revision=revision, device_map="cpu"
        )
        adapted = Chronos2Pipeline(
            model=PeftModel.from_pretrained(base.model, adapter.parent)
        )
        require_eval_mode(adapted)
        predicted = adapted.predict_df(context, **predict_args)
        details.append(align_chronos_predictions(rows, predicted, model_id=model_id))
        del adapted, base
    for model_id, values in (
        ("weighted_weekday_raw_sales", rows["weighted_weekday_sales"].to_numpy()),
        (
            "weighted_weekday_reconstructed",
            _same_weekday_formula(
                rows,
                target_flows.loc[
                    target_flows["date"].between(
                        ORIGIN - pd.Timedelta(days=INPUT_DAYS - 1), ORIGIN
                    )
                ],
            ),
        ),
    ):
        columns = ["forecast_origin", "date", "lead_days", *KEYS, "actual"]
        baseline = rows[columns].copy()
        baseline["model"] = model_id
        baseline["prediction"] = values
        details.append(baseline)
    detail = pd.concat(details, ignore_index=True)
    leaderboard = _summary(detail, ["model"]).sort_values("wmape_pct")
    args.output_dir.mkdir(parents=True)
    detail.to_parquet(args.output_dir / "detail.parquet", index=False)
    leaderboard.to_csv(args.output_dir / "leaderboard.csv", index=False)
    _summary(detail, ["date", "model"]).to_csv(
        args.output_dir / "by_day.csv", index=False
    )
    metadata = {
        "production_write": False,
        "evaluation_type": "retrospective_fixed_origin",
        "origin": str(ORIGIN.date()),
        "outcome_dates": [str(days.min().date()), str(days.max().date())],
        "target": "observed_sales_qty",
        "missing_bakery_days_are_not_zero_filled": True,
        "stable_bakery_count": len(stable_bakeries),
        "excluded_incomplete_bakeries": sorted(all_bakeries - stable_bakeries),
        "pairs": len(rows) // HORIZON_DAYS,
        "rows_per_model": len(rows),
        "target_panel_sha256": _sha256(args.target_panel),
        "frozen_outcome_sha256": _sha256(frozen_path),
        "as_of_origin_fact_arrival_verified": False,
        "production_target_and_scope_parity_verified": False,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()

"""Research-only recent 14-day backtest of a verified Chronos-2 LoRA adapter.

The SKU scope and 56-day model context stop at the requested origin. Future
observed sales are used only for scoring. Historical fact-arrival times are
not reconstructed, so this is not a certified 08:00 as-of replay.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.finetune_chronos2_fixed_origin import require_eval_mode  # noqa: E402
from scripts.run_chronos2_fixed_origin import (  # noqa: E402
    MODEL_VARIANTS,
    _sha256,
    _summary,
    align_chronos_predictions,
)
from src.model_tournament.fixed_origin import validate_flows  # noqa: E402
from src.model_tournament.nhits_candidate import (  # noqa: E402
    evaluation_history_and_rows,
)


def checked_adapter(report: Path, origin: pd.Timestamp) -> tuple[Path, str, str]:
    """Require the previously verified, pre-origin 1,000-step adapter."""
    meta = json.loads((report / "metadata.json").read_text(encoding="utf-8"))
    _, repo, revision = MODEL_VARIANTS["small"]
    if (
        not meta.get("score_valid")
        or meta["model_repo"] != repo
        or meta["model_revision"] != revision
        or meta["num_steps"] != 1000
        or pd.Timestamp(meta["train_cutoff"]) >= origin
    ):
        raise ValueError("Adapter is not a verified pre-origin Small checkpoint")
    adapter_file = Path(meta["adapter_file"])
    if _sha256(adapter_file) != meta["adapter_sha256"]:
        raise ValueError("Adapter hash differs from the verified report")
    return adapter_file, repo, revision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-input", type=Path, required=True)
    parser.add_argument("--adapter-report", type=Path, required=True)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    origin = pd.Timestamp(args.origin).normalize()
    input_meta = json.loads(
        (args.frozen_input / "metadata.json").read_text(encoding="utf-8")
    )
    input_file = args.frozen_input / "frozen_flows.parquet"
    if _sha256(input_file) != input_meta["frozen_flows_sha256"]:
        raise ValueError("Frozen input hash differs from metadata")
    flows = validate_flows(pd.read_parquet(input_file))
    last = origin + pd.Timedelta(days=14)
    if flows["date"].max() < last:
        raise ValueError("Frozen input lacks the full 14-day outcome window")
    daily_bakeries = (
        flows.loc[flows["date"].between(origin + pd.Timedelta(days=1), last)]
        .groupby("date")["bakery_id"]
        .nunique()
        .reindex(pd.date_range(origin + pd.Timedelta(days=1), last))
    )
    if not daily_bakeries.ge(50).all():
        raise ValueError("Outcome facts lack 50-bakery daily coverage")
    context, rows = evaluation_history_and_rows(flows, origin)
    if context["ds"].max() != origin:
        raise ValueError("Model context crosses forecast origin")
    adapter_file, repo, revision = checked_adapter(args.adapter_report, origin)

    try:
        import torch
        from chronos import Chronos2Pipeline
        from peft import PeftModel
    except ImportError as exc:
        raise SystemExit("Research environment needs Chronos-2 and PEFT") from exc
    torch.set_num_threads(4)
    base = Chronos2Pipeline.from_pretrained(repo, revision=revision, device_map="cpu")
    require_eval_mode(base)
    common_predict_args = {
        "id_column": "unique_id",
        "timestamp_column": "ds",
        "target": "y",
        "prediction_length": 14,
        "quantile_levels": [0.5],
        "batch_size": args.batch_size,
    }
    zero_predicted = base.predict_df(context, **common_predict_args)
    zero = align_chronos_predictions(
        rows, zero_predicted, model_id="chronos2_small_zero_shot"
    )
    adapted = Chronos2Pipeline(
        model=PeftModel.from_pretrained(base.model, adapter_file.parent)
    )
    require_eval_mode(adapted)
    lora_predicted = adapted.predict_df(context, **common_predict_args)
    lora = align_chronos_predictions(
        rows, lora_predicted, model_id="chronos2_small_lora_1000"
    )
    baseline = rows[
        ["forecast_origin", "date", "lead_days", "bakery_id", "product_id", "actual"]
    ].copy()
    baseline["model"] = "weighted_weekday_sales_formula"
    baseline["prediction"] = rows["weighted_weekday_sales"].to_numpy(dtype=float)
    detail = pd.concat([zero, lora, baseline], ignore_index=True)
    expected_rows = len(rows)
    if any(len(part) != expected_rows for part in (zero, lora, baseline)):
        raise ValueError("Comparison models do not share the same SKU-day population")
    leaderboard = _summary(detail, ["model"]).sort_values("wmape_pct")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    detail.to_parquet(args.output_dir / "detail.parquet", index=False)
    leaderboard.to_csv(args.output_dir / "leaderboard.csv", index=False)
    _summary(detail, ["date", "model"]).to_csv(
        args.output_dir / "by_date.csv", index=False
    )
    _summary(detail, ["lead_days", "model"]).to_csv(
        args.output_dir / "by_lead.csv", index=False
    )
    meta = {
        "production_write": False,
        "score_valid": True,
        "evaluation_type": "retrospective_fixed_origin",
        "origin": str(origin.date()),
        "target_start": str((origin + pd.Timedelta(days=1)).date()),
        "target_end": str(last.date()),
        "target": "observed_sales_qty",
        "scope": "positive release in origin-55..origin; every pair on all 14 days",
        "scope_bakeries": int(rows["bakery_id"].nunique()),
        "scope_pairs": len(rows) // 14,
        "rows_per_model": expected_rows,
        "min_outcome_daily_bakery_coverage": int(daily_bakeries.min()),
        "input_path": str(input_file.resolve()),
        "input_sha256": input_meta["frozen_flows_sha256"],
        "model_repo": repo,
        "model_revision": revision,
        "adapter_sha256": _sha256(adapter_file),
        "created_utc": datetime.now(ZoneInfo("UTC")).isoformat(),
        "as_of_origin_fact_arrival_verified": False,
        "production_target_and_scope_parity_verified": False,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()

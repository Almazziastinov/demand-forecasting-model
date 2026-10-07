"""Research-only Chronos-2 LoRA training on a reconstructed demand proxy.

Evaluation uses frozen future SKU-days and original observed sales, never the
smoothed training target as the primary score. No production data is changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.finetune_chronos2_fixed_origin import (  # noqa: E402
    HORIZON_DAYS,
    require_eval_mode,
    training_arrays,
)
from scripts.run_chronos2_fixed_origin import (  # noqa: E402
    MODEL_VARIANTS,
    _origins,
    _sha256,
    _summary,
    align_chronos_predictions,
)
from src.model_tournament.demand_target_contract import (  # noqa: E402
    validate_demand_target_contract,
)
from src.model_tournament.fixed_origin import KEYS, validate_flows  # noqa: E402
from src.model_tournament.nhits_candidate import (  # noqa: E402
    INPUT_DAYS,
    dense_history,
    evaluation_history_and_rows,
    training_history,
)


MODEL_ID = "chronos2_small_lora_reconstructed_demand_h14"
ZERO_SHOT_ID = "chronos2_small_zero_shot_reconstructed_context_h14"


def _target_flows(panel: pd.DataFrame) -> pd.DataFrame:
    validate_demand_target_contract(
        panel,
        sales_column="observed_sales_qty",
        demand_column="reconstructed_demand_qty",
        uplift_column="restoration_uplift_qty",
        reduction_column="outlier_reduction_qty",
        require_global_uplift=True,
    )
    target = panel[["date", *KEYS, "release_qty", "reconstructed_demand_qty"]].copy()
    return target.rename(columns={"reconstructed_demand_qty": "observed_sales_qty"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-origins", required=True)
    parser.add_argument("--evaluation-origins", required=True)
    parser.add_argument("--train-start", default="2026-01-01")
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prediction-batch-size", type=int, default=128)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    train_origins = _origins(args.train_origins)
    eval_origins = _origins(args.evaluation_origins)
    train_cutoff = max(train_origins) + pd.Timedelta(days=HORIZON_DAYS)
    if train_cutoff >= min(eval_origins):
        raise ValueError("Training labels must end before the first evaluation origin")
    if args.num_steps < 1 or args.batch_size < 1 or args.prediction_batch_size < 1:
        raise ValueError("Steps and batch sizes must be positive")
    if args.max_pairs is not None and args.max_pairs < 1:
        raise ValueError("max-pairs must be positive")

    try:
        import peft
        import torch
        from chronos import Chronos2Pipeline
    except ImportError as exc:
        raise SystemExit("Research Chronos-2 and PEFT environment is required") from exc

    source_summary_path = args.input.with_name("summary.json")
    if not source_summary_path.exists():
        raise ValueError("Demand panel must have an auditable summary.json")
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    if source_summary.get("train_cutoff") != str(train_cutoff.date()):
        raise ValueError("Demand panel cutoff differs from training cutoff")
    panel = validate_flows(pd.read_parquet(args.input))
    target_flows = _target_flows(panel)
    train, train_counts = training_history(
        target_flows,
        train_origins,
        start=pd.Timestamp(args.train_start),
        input_days=INPUT_DAYS,
        horizon_days=HORIZON_DAYS,
        max_pairs=args.max_pairs,
    )
    if train["ds"].max() != train_cutoff:
        raise ValueError("Training history does not end at the declared cutoff")
    arrays = training_arrays(train)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    _, model_repo, model_revision = MODEL_VARIANTS["small"]
    pretrained = Chronos2Pipeline.from_pretrained(
        model_repo, revision=model_revision, device_map="cpu"
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    finetuned = pretrained.fit(
        inputs=arrays,
        prediction_length=HORIZON_DAYS,
        context_length=INPUT_DAYS,
        finetune_mode="lora",
        learning_rate=1e-5,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        output_dir=args.output_dir / "checkpoint",
        optim="adamw_torch",
        seed=42,
        disable_tqdm=True,
        remove_printer_callback=True,
    )
    require_eval_mode(finetuned)
    # ``fit`` may mutate the source pipeline; reload the pinned base so the
    # comparator is genuinely zero-shot on the identical reconstructed input.
    zero_shot = Chronos2Pipeline.from_pretrained(
        model_repo, revision=model_revision, device_map="cpu"
    )
    require_eval_mode(zero_shot)

    details = []
    counts = []
    for origin in eval_origins:
        _, rows = evaluation_history_and_rows(panel, origin, max_pairs=args.max_pairs)
        context = dense_history(
            target_flows,
            rows[KEYS],
            origin - pd.Timedelta(days=INPUT_DAYS - 1),
            origin,
        )
        for model, model_id in ((zero_shot, ZERO_SHOT_ID), (finetuned, MODEL_ID)):
            prediction = model.predict_df(
                context,
                id_column="unique_id",
                timestamp_column="ds",
                target="y",
                prediction_length=HORIZON_DAYS,
                quantile_levels=[0.5],
                batch_size=args.prediction_batch_size,
            )
            details.append(
                align_chronos_predictions(rows, prediction, model_id=model_id)
            )
        counts.append(
            {"origin": str(origin.date()), "pairs": len(rows) // HORIZON_DAYS}
        )

    detail = pd.concat(details, ignore_index=True)
    # The target proxy is added for secondary audit only; primary actual is raw sales.
    demand_truth = panel[
        ["date", *KEYS, "reconstructed_demand_qty", "broad_stockout_signal"]
    ]
    detail = detail.merge(
        demand_truth, on=["date", *KEYS], how="left", validate="many_to_one"
    )
    if detail["reconstructed_demand_qty"].isna().any():
        raise ValueError("Evaluation truth is missing from candidate panel")
    metadata = {
        "production_write": False,
        "score_valid": True,
        "primary_evaluation_target": "original_observed_sales_qty",
        "training_target": "reconstructed_demand_qty",
        "training_target_is_proxy_not_ground_truth": True,
        "train_origins": [str(day.date()) for day in train_origins],
        "train_cutoff": str(train_cutoff.date()),
        "evaluation_origins": [str(day.date()) for day in eval_origins],
        "scope": "positive release in origin-55..origin; full future SKU-day grid",
        "train_counts": train_counts,
        "train_series": len(arrays),
        "evaluation_counts": counts,
        "max_pairs": args.max_pairs,
        "comparable_full_scope": (
            args.max_pairs is None and source_summary.get("max_pairs") is None
        ),
        "target_panel_summary_sha256": _sha256(source_summary_path),
        "num_steps": args.num_steps,
        "batch_size": args.batch_size,
        "seed": 42,
        "learning_rate": 1e-5,
        "model_repo": model_repo,
        "model_revision": model_revision,
        "inference_mode": "eval",
        "input_sha256": _sha256(args.input),
        "chronos_forecasting_version": version("chronos-forecasting"),
        "peft_version": peft.__version__,
        "torch_version": torch.__version__,
    }
    detail.to_parquet(args.output_dir / "detail.parquet", index=False)
    leaderboard = _summary(detail, ["model"])
    leaderboard.to_csv(args.output_dir / "leaderboard.csv", index=False)
    _summary(detail, ["forecast_origin", "model"]).to_csv(
        args.output_dir / "by_origin.csv", index=False
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()

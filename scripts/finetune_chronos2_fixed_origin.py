"""Research-only LoRA fine-tuning of Chronos-2 Small on causal sales history.

Requires ``chronos-forecasting==2.3.2`` and ``peft==0.19.1``. Uses local
Parquet input and never connects to production services or publishes forecasts.
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

from scripts.run_chronos2_fixed_origin import (  # noqa: E402
    MODEL_VARIANTS,
    _origins,
    _sha256,
    _summary,
    align_chronos_predictions,
)
from src.model_tournament.fixed_origin import validate_flows  # noqa: E402
from src.model_tournament.nhits_candidate import (  # noqa: E402
    INPUT_DAYS,
    evaluation_history_and_rows,
    training_history,
)


MODEL_ID = "new_chronos2_small_lora_h14"
HORIZON_DAYS = 14


def training_arrays(train: pd.DataFrame) -> list[np.ndarray]:
    """Convert one daily training series per pair without dropping zero days."""
    if train.duplicated(["unique_id", "ds"]).any():
        raise ValueError("Training history has duplicate series-day keys")
    arrays = [
        group["y"].to_numpy(dtype=np.float32)
        for _, group in train.sort_values(["unique_id", "ds"]).groupby(
            "unique_id", sort=True
        )
    ]
    if not arrays or any(len(series) < INPUT_DAYS + HORIZON_DAYS for series in arrays):
        raise ValueError("Training history contains a series shorter than 70 days")
    if any(not np.isfinite(series).all() or (series < 0).any() for series in arrays):
        raise ValueError("Training history contains invalid values")
    return arrays


def require_eval_mode(pipeline: object) -> None:
    """Keep inference deterministic after Hugging Face Trainer returns."""
    model = pipeline.model
    model.eval()
    if model.training:
        raise RuntimeError("Fine-tuned model did not enter evaluation mode")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-origins", required=True)
    parser.add_argument("--evaluation-origins", required=True)
    parser.add_argument("--train-start", default="2026-01-01")
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prediction-batch-size", type=int, default=128)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    train_origins = _origins(args.train_origins)
    eval_origins = _origins(args.evaluation_origins)
    train_cutoff = max(train_origins) + pd.Timedelta(days=HORIZON_DAYS)
    if train_cutoff >= min(eval_origins):
        raise ValueError("Training labels must end before the first evaluation origin")
    if args.num_steps < 1 or args.batch_size < 1 or args.prediction_batch_size < 1:
        raise ValueError("Steps and batch sizes must be positive")
    if args.max_pairs is not None and args.max_pairs < 1:
        raise ValueError("max-pairs must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")

    try:
        import peft
        import torch
        from chronos import Chronos2Pipeline
    except ImportError as exc:
        raise SystemExit(
            "Install chronos-forecasting==2.3.2 and peft==0.19.1 "
            "in a research environment"
        ) from exc

    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    flows = validate_flows(pd.read_parquet(args.input))
    train, train_counts = training_history(
        flows,
        train_origins,
        start=pd.Timestamp(args.train_start),
        input_days=INPUT_DAYS,
        horizon_days=HORIZON_DAYS,
        max_pairs=args.max_pairs,
    )
    if train["ds"].max() != train_cutoff:
        raise ValueError("Training history does not end at the declared cutoff")
    arrays = training_arrays(train)
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
    # Chronos2Pipeline.predict_df does not switch the Trainer's model to eval.
    # Without this, dropout makes scores stochastic and saved adapters disagree.
    require_eval_mode(finetuned)

    detail_parts = []
    eval_counts = []
    for origin in eval_origins:
        context, rows = evaluation_history_and_rows(
            flows, origin, max_pairs=args.max_pairs
        )
        predicted = finetuned.predict_df(
            context,
            id_column="unique_id",
            timestamp_column="ds",
            target="y",
            prediction_length=HORIZON_DAYS,
            quantile_levels=[0.5],
            batch_size=args.prediction_batch_size,
        )
        detail_parts.append(
            align_chronos_predictions(rows, predicted, model_id=MODEL_ID)
        )
        eval_counts.append(
            {"origin": str(origin.date()), "pairs": len(rows) // 14, "rows": len(rows)}
        )

    detail = pd.concat(detail_parts, ignore_index=True)
    leaderboard = _summary(detail, ["model"])
    metadata = {
        "production_write": False,
        "score_valid": True,
        "target": "observed_sales_qty",
        "scope": (
            "positive release in origin-55..origin; "
            "all eligible pairs on every future date"
        ),
        "model": MODEL_ID,
        "model_repo": model_repo,
        "model_revision": model_revision,
        "finetune_mode": "lora",
        "lora_config": "chronos-forecasting 2.3.2 default",
        "learning_rate": 1e-5,
        "num_steps": args.num_steps,
        "batch_size": args.batch_size,
        "prediction_batch_size": args.prediction_batch_size,
        "seed": 42,
        "train_origins": [str(origin.date()) for origin in train_origins],
        "train_start": args.train_start,
        "train_cutoff": str(train_cutoff.date()),
        "train_counts": train_counts,
        "train_series": len(arrays),
        "evaluation_origins": [str(origin.date()) for origin in eval_origins],
        "evaluation_counts": eval_counts,
        "max_pairs": args.max_pairs,
        "comparable_full_scope": args.max_pairs is None,
        "inference_mode": "eval",
        "missing_sparse_fact_rows_as_zero": True,
        "as_of_fact_arrival_verified": False,
        "production_target_and_scope_parity_verified": False,
        "input_path": str(args.input.resolve()),
        "input_sha256": _sha256(args.input),
        "chronos_forecasting_version": version("chronos-forecasting"),
        "peft_version": peft.__version__,
        "torch_version": torch.__version__,
    }
    detail.to_parquet(args.output_dir / "detail.parquet", index=False)
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

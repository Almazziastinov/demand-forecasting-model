"""Rescore a saved Chronos-2 LoRA adapter in deterministic eval mode.

This repairs research reports generated while the Trainer's returned model was
still in training mode. It reads only local files and writes a new report.
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


MODEL_ID = "new_chronos2_small_lora_h14_eval"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    metadata_path = args.training_report / "metadata.json"
    old_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _, model_repo, model_revision = MODEL_VARIANTS["small"]
    if (
        old_metadata["model_repo"] != model_repo
        or old_metadata["model_revision"] != model_revision
    ):
        raise ValueError("Training report is not from the pinned Chronos-2 Small")
    if old_metadata["max_pairs"] is not None:
        raise ValueError("Cannot rescore a smoke-test checkpoint as full scope")
    input_path = Path(old_metadata["input_path"])
    if _sha256(input_path) != old_metadata["input_sha256"]:
        raise ValueError("Frozen input hash no longer matches training report")
    origins = [pd.Timestamp(value) for value in old_metadata["evaluation_origins"]]
    if pd.Timestamp(old_metadata["train_cutoff"]) >= min(origins):
        raise ValueError("Training labels reach evaluation period")
    checkpoint = args.training_report / "checkpoint" / "finetuned-ckpt"
    adapter_file = checkpoint / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise FileNotFoundError(adapter_file)

    try:
        import torch
        from chronos import Chronos2Pipeline
        from peft import PeftModel
    except ImportError as exc:
        raise SystemExit(
            "Install chronos-forecasting==2.3.2 and peft==0.19.1 "
            "in a research environment"
        ) from exc

    torch.set_num_threads(4)
    base = Chronos2Pipeline.from_pretrained(
        model_repo, revision=model_revision, device_map="cpu"
    )
    pipeline = Chronos2Pipeline(
        model=PeftModel.from_pretrained(base.model, checkpoint)
    )
    require_eval_mode(pipeline)

    flows = validate_flows(pd.read_parquet(input_path))
    parts = []
    eval_counts = []
    for origin in origins:
        context, rows = evaluation_history_and_rows(flows, origin)
        predicted = pipeline.predict_df(
            context,
            id_column="unique_id",
            timestamp_column="ds",
            target="y",
            prediction_length=14,
            quantile_levels=[0.5],
            batch_size=old_metadata["prediction_batch_size"],
        )
        parts.append(align_chronos_predictions(rows, predicted, model_id=MODEL_ID))
        eval_counts.append(
            {"origin": str(origin.date()), "pairs": len(rows) // 14, "rows": len(rows)}
        )
    detail = pd.concat(parts, ignore_index=True)
    leaderboard = _summary(detail, ["model"])
    source_metadata = {
        key: value
        for key, value in old_metadata.items()
        if key not in {"score_valid", "invalid_reason", "superseded_by"}
    }
    metadata = {
        **source_metadata,
        "score_valid": True,
        "model": MODEL_ID,
        "inference_mode": "eval",
        "supersedes_train_mode_report": str(args.training_report.resolve()),
        "adapter_file": str(adapter_file.resolve()),
        "adapter_sha256": _sha256(adapter_file),
        "evaluation_counts": eval_counts,
        "torch_version": torch.__version__,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    detail.to_parquet(args.output_dir / "detail.parquet", index=False)
    leaderboard.to_csv(args.output_dir / "leaderboard.csv", index=False)
    _summary(detail, ["forecast_origin", "model"]).to_csv(
        args.output_dir / "by_origin.csv", index=False
    )
    _summary(detail, ["lead_days", "model"]).to_csv(
        args.output_dir / "by_lead.csv", index=False
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()

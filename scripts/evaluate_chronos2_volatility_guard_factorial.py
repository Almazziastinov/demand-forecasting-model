"""Cross LoRA training targets and inference contexts on the same frozen days."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.finetune_chronos2_fixed_origin import require_eval_mode  # noqa: E402
from scripts.finetune_chronos2_volatility_guard import (  # noqa: E402
    EVAL_ORIGINS,
    GUARDED_ID,
    _checked_existing,
    guarded_flows,
)
from scripts.run_chronos2_fixed_origin import (  # noqa: E402
    _sha256,
    _summary,
    align_chronos_predictions,
)
from src.model_tournament.fixed_origin import KEYS, validate_flows  # noqa: E402
from src.model_tournament.nhits_candidate import (  # noqa: E402
    INPUT_DAYS,
    dense_history,
    evaluation_history_and_rows,
)

RAW_ID = "new_chronos2_small_lora_h14_eval"
RAW_TRAIN_GUARD_CONTEXT = "raw_lora_guarded_context_h14"
GUARD_TRAIN_RAW_CONTEXT = "guarded_lora_raw_context_h14"


def _load_adapter(metadata: dict, checkpoint: Path) -> object:
    from chronos import Chronos2Pipeline
    from peft import PeftModel

    base = Chronos2Pipeline.from_pretrained(
        metadata["model_repo"], revision=metadata["model_revision"], device_map="cpu"
    )
    pipeline = Chronos2Pipeline(model=PeftModel.from_pretrained(base.model, checkpoint))
    require_eval_mode(pipeline)
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guard-report", type=Path, required=True)
    parser.add_argument("--raw-report", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    guarded_metadata = json.loads(
        (args.guard_report / "metadata.json").read_text(encoding="utf-8")
    )
    raw_metadata = json.loads(
        (args.raw_report / "metadata.json").read_text(encoding="utf-8")
    )
    if not guarded_metadata["score_valid"] or not raw_metadata["score_valid"]:
        raise ValueError("One source report is invalid")
    if guarded_metadata["model_revision"] != raw_metadata["model_revision"]:
        raise ValueError("LoRAs use different base revisions")
    if guarded_metadata["panel_sha256"] != _sha256(args.input):
        raise ValueError("Input panel differs from guarded training")
    if guarded_metadata["registry_sha256"] != _sha256(args.registry):
        raise ValueError("Input registry differs from guarded training")
    guard_checkpoint = args.guard_report / "checkpoint" / "finetuned-ckpt"
    raw_checkpoint = Path(raw_metadata["adapter_file"]).parent
    if (
        _sha256(guard_checkpoint / "adapter_model.safetensors")
        != guarded_metadata["adapter_sha256"]
    ):
        raise ValueError("Guarded adapter checksum mismatch")
    if (
        _sha256(raw_checkpoint / "adapter_model.safetensors")
        != raw_metadata["adapter_sha256"]
    ):
        raise ValueError("Raw adapter checksum mismatch")

    import torch

    torch.set_num_threads(4)
    raw_model = _load_adapter(raw_metadata, raw_checkpoint)
    guard_model = _load_adapter(guarded_metadata, guard_checkpoint)
    panel = validate_flows(pd.read_parquet(args.input))
    registry = pd.read_parquet(args.registry)
    rows_parts = []
    cross_parts = []
    for origin in EVAL_ORIGINS:
        raw_context, rows = evaluation_history_and_rows(panel, origin)
        guarded_context = dense_history(
            guarded_flows(panel, registry, origin),
            rows[KEYS],
            origin - pd.Timedelta(days=INPUT_DAYS - 1),
            origin,
        )
        for model, context, model_id in (
            (raw_model, guarded_context, RAW_TRAIN_GUARD_CONTEXT),
            (guard_model, raw_context, GUARD_TRAIN_RAW_CONTEXT),
        ):
            prediction = model.predict_df(
                context,
                id_column="unique_id",
                timestamp_column="ds",
                target="y",
                prediction_length=14,
                quantile_levels=[0.5],
                batch_size=128,
            )
            cross_parts.append(
                align_chronos_predictions(rows, prediction, model_id=model_id)
            )
        rows_parts.append(rows)
        print(f"Cross-scored {origin.date()}", flush=True)
    truth = pd.concat(rows_parts, ignore_index=True)
    guard_existing = _checked_existing(
        args.guard_report / "detail.parquet", truth, GUARDED_ID
    )
    raw_existing = _checked_existing(args.raw_report / "detail.parquet", truth, RAW_ID)
    detail = pd.concat([*cross_parts, guard_existing, raw_existing], ignore_index=True)
    keys = ["forecast_origin", "date", "lead_days", *KEYS, "model"]
    if detail.duplicated(keys).any():
        raise ValueError("Duplicate factorial prediction key")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    detail.to_parquet(args.output_dir / "detail.parquet", index=False)
    leaderboard = _summary(detail, ["model"])
    leaderboard.to_csv(args.output_dir / "leaderboard.csv", index=False)
    _summary(detail, ["forecast_origin", "model"]).to_csv(
        args.output_dir / "by_origin.csv", index=False
    )
    metadata = {
        "production_write": False,
        "score_valid": True,
        "design": (
            "2x2 training target by inference context, "
            "identical observed-sales truth"
        ),
        "evaluation_origins": [str(x.date()) for x in EVAL_ORIGINS],
        "guard_report_metadata_sha256": _sha256(args.guard_report / "metadata.json"),
        "raw_report_metadata_sha256": _sha256(args.raw_report / "metadata.json"),
        "panel_sha256": _sha256(args.input),
        "registry_sha256": _sha256(args.registry),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(leaderboard.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

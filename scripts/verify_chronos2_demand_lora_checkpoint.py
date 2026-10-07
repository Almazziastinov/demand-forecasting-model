"""Independently reload and verify a saved demand-LoRA score in eval mode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.finetune_chronos2_fixed_origin import require_eval_mode  # noqa: E402
from scripts.finetune_chronos2_reconstructed_demand import (  # noqa: E402
    MODEL_ID,
    _target_flows,
)
from scripts.run_chronos2_fixed_origin import (  # noqa: E402
    _sha256,
    align_chronos_predictions,
)
from src.model_tournament.fixed_origin import KEYS, validate_flows  # noqa: E402
from src.model_tournament.nhits_candidate import (  # noqa: E402
    INPUT_DAYS,
    dense_history,
    evaluation_history_and_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.max_pairs < 1:
        raise ValueError("max-pairs must be positive")
    metadata = json.loads(
        (args.training_report / "metadata.json").read_text(encoding="utf-8")
    )
    if _sha256(args.input) != metadata["input_sha256"]:
        raise ValueError("Frozen target input hash differs from training")
    checkpoint = args.training_report / "checkpoint" / "finetuned-ckpt"
    adapter = checkpoint / "adapter_model.safetensors"
    if not adapter.is_file():
        raise FileNotFoundError(adapter)
    import torch
    from chronos import Chronos2Pipeline
    from peft import PeftModel

    torch.set_num_threads(4)
    base = Chronos2Pipeline.from_pretrained(
        metadata["model_repo"],
        revision=metadata["model_revision"],
        device_map="cpu",
    )
    pipeline = Chronos2Pipeline(model=PeftModel.from_pretrained(base.model, checkpoint))
    require_eval_mode(pipeline)
    panel = validate_flows(pd.read_parquet(args.input))
    target = _target_flows(panel)
    origin = pd.Timestamp(metadata["evaluation_origins"][0])
    _, rows = evaluation_history_and_rows(panel, origin, max_pairs=args.max_pairs)
    context = dense_history(
        target,
        rows[KEYS],
        origin - pd.Timedelta(days=INPUT_DAYS - 1),
        origin,
    )
    predicted = pipeline.predict_df(
        context,
        id_column="unique_id",
        timestamp_column="ds",
        target="y",
        prediction_length=14,
        quantile_levels=[0.5],
        batch_size=metadata["batch_size"] * 8,
    )
    checked = align_chronos_predictions(rows, predicted, model_id=MODEL_ID)
    saved = pd.read_parquet(args.training_report / "detail.parquet")
    saved = saved.loc[saved["model"].eq(MODEL_ID)]
    keys = ["forecast_origin", "date", "lead_days", *KEYS]
    joined = checked.merge(
        saved[[*keys, "actual", "prediction"]],
        on=keys,
        how="left",
        validate="one_to_one",
        suffixes=("_reloaded", "_saved"),
    )
    if joined["prediction_saved"].isna().any():
        raise ValueError("Saved report lacks verification SKU-days")
    difference = np.abs(joined["prediction_reloaded"] - joined["prediction_saved"])
    max_difference = float(difference.max())
    if max_difference > 1e-4:
        raise ValueError(
            f"Reloaded adapter differs from saved scores by {max_difference}"
        )
    result = {
        "verified": True,
        "origin": str(origin.date()),
        "pairs": len(rows) // 14,
        "rows": len(joined),
        "max_absolute_prediction_difference": max_difference,
        "adapter_sha256": _sha256(adapter),
        "score_target": "original_observed_sales_qty",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

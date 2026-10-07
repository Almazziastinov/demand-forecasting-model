"""Reload the saved guarded LoRA and verify its frozen predictions."""

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
from scripts.finetune_chronos2_volatility_guard import (  # noqa: E402
    GUARDED_ID,
    guarded_flows,
)
from scripts.run_chronos2_fixed_origin import _sha256, align_chronos_predictions  # noqa: E402
from src.model_tournament.fixed_origin import KEYS, validate_flows  # noqa: E402
from src.model_tournament.nhits_candidate import (  # noqa: E402
    INPUT_DAYS,
    dense_history,
    evaluation_history_and_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=100)
    args = parser.parse_args()
    if args.max_pairs < 1:
        raise ValueError("max-pairs must be positive")
    metadata = json.loads((args.report / "metadata.json").read_text(encoding="utf-8"))
    if not metadata["score_valid"] or metadata["inference_mode"] != "eval":
        raise ValueError("Report is not a valid eval-mode score")
    if _sha256(args.input) != metadata["panel_sha256"]:
        raise ValueError("Panel checksum differs")
    if _sha256(args.registry) != metadata["registry_sha256"]:
        raise ValueError("Guard registry checksum differs")
    checkpoint = args.report / "checkpoint" / "finetuned-ckpt"
    adapter = checkpoint / "adapter_model.safetensors"
    if _sha256(adapter) != metadata["adapter_sha256"]:
        raise ValueError("Adapter checksum differs")

    import torch
    from chronos import Chronos2Pipeline
    from peft import PeftModel

    torch.set_num_threads(4)
    base = Chronos2Pipeline.from_pretrained(
        metadata["model_repo"], revision=metadata["model_revision"], device_map="cpu"
    )
    pipeline = Chronos2Pipeline(model=PeftModel.from_pretrained(base.model, checkpoint))
    require_eval_mode(pipeline)
    panel = validate_flows(pd.read_parquet(args.input))
    registry = pd.read_parquet(args.registry)
    origin = pd.Timestamp(metadata["evaluation_origins"][0])
    _, rows = evaluation_history_and_rows(panel, origin, max_pairs=args.max_pairs)
    context_flows = guarded_flows(panel, registry, origin)
    context = dense_history(
        context_flows, rows[KEYS], origin - pd.Timedelta(days=INPUT_DAYS - 1), origin
    )
    predicted = pipeline.predict_df(
        context,
        id_column="unique_id",
        timestamp_column="ds",
        target="y",
        prediction_length=14,
        quantile_levels=[0.5],
        batch_size=128,
    )
    checked = align_chronos_predictions(rows, predicted, model_id=GUARDED_ID)
    saved = pd.read_parquet(args.report / "detail.parquet")
    saved = saved.loc[saved["model"].eq(GUARDED_ID)]
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
    if not np.array_equal(
        joined["actual_reloaded"].to_numpy(), joined["actual_saved"].to_numpy()
    ):
        raise ValueError("Saved report truth differs")
    maximum = float(
        np.abs(joined["prediction_reloaded"] - joined["prediction_saved"]).max()
    )
    if maximum > 1e-4:
        raise ValueError(f"Reloaded checkpoint differs by {maximum}")
    result = {
        "verified": True,
        "origin": str(origin.date()),
        "rows": len(joined),
        "max_absolute_prediction_difference": maximum,
        "adapter_sha256": metadata["adapter_sha256"],
    }
    (args.report / "checkpoint_verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

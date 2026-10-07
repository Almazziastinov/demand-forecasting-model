"""Offline A/B: Chronos-2 Small LoRA on guarded sales versus frozen raw LoRA.

The 50% uplift is a *training proxy*, never an evaluation label or production write.
"""

from __future__ import annotations

import argparse
import hashlib
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
    _sha256,
    _summary,
    align_chronos_predictions,
)
from src.model_tournament.fixed_origin import KEYS, validate_flows  # noqa: E402
from src.model_tournament.nhits_candidate import (  # noqa: E402
    INPUT_DAYS,
    dense_history,
    evaluation_history_and_rows,
    training_history,
)

TRAIN_ORIGINS = pd.to_datetime(
    ["2026-02-02", "2026-03-02", "2026-03-30", "2026-04-27", "2026-05-25", "2026-06-21"]
).tolist()
EVAL_ORIGINS = pd.to_datetime(
    ["2026-07-06", "2026-07-20", "2026-08-03", "2026-08-17"]
).tolist()
RAW_TRAIN_SHA256 = "6cc3d1d752edd5eff1aa2d666eea489adf4326539569ae19f127250533c9b2d8"
RAW_ADAPTER_SHA256 = "0c9321bdad07ec247bb7d3a45813ca146dbade4f4478d35c1fc751fcfeb843cc"
GUARDED_ID = "chronos2_small_lora_guarded_half_h14"
ZERO_SHOT_ID = "chronos2_small_zero_shot_guarded_context_h14"
KEY = ["forecast_origin", "date", "lead_days", *KEYS]


def guarded_flows(
    panel: pd.DataFrame, registry: pd.DataFrame, as_of: pd.Timestamp
) -> pd.DataFrame:
    """Apply only finalized adjustments through as_of-7 to observed history."""
    columns = ["date", *KEYS, "guarded_total_uplift", "observed_sales_qty"]
    if registry.duplicated(["date", *KEYS]).any():
        raise ValueError("Guard registry contains duplicate SKU-days")
    adjustments = registry.loc[
        registry["date"].le(as_of - pd.Timedelta(days=7)), columns
    ]
    if not np.isfinite(adjustments["guarded_total_uplift"]).all():
        raise ValueError("Invalid guard uplift")
    if (adjustments["guarded_total_uplift"] < 0).any():
        raise ValueError("Negative guard uplift")
    target = panel[["date", *KEYS, "release_qty", "observed_sales_qty"]].merge(
        adjustments,
        on=["date", *KEYS],
        how="left",
        validate="one_to_one",
        suffixes=("", "_registry"),
    )
    matched = target["observed_sales_qty_registry"].notna()
    if not np.allclose(
        target.loc[matched, "observed_sales_qty"],
        target.loc[matched, "observed_sales_qty_registry"],
        atol=1e-6,
    ):
        raise ValueError("Guard registry sales do not match panel")
    uplift = target["guarded_total_uplift"].fillna(0.0)
    target["observed_sales_qty"] = target["observed_sales_qty"] + 0.5 * uplift
    return target[["date", *KEYS, "release_qty", "observed_sales_qty"]]


def _array_digest(arrays: list[np.ndarray]) -> str:
    return hashlib.sha256(np.concatenate(arrays).tobytes()).hexdigest()


def _checked_existing(
    path: Path, rows: pd.DataFrame, expected_model: str
) -> pd.DataFrame:
    old = pd.read_parquet(path)
    old = old.loc[old["model"].eq(expected_model)].copy()
    if old.empty or old.duplicated(KEY).any() or rows.duplicated(KEY).any():
        raise ValueError(f"Invalid reference prediction keys: {path}")
    joined = rows[[*KEY, "actual"]].merge(
        old,
        on=KEY,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_new", "_old"),
    )
    if len(joined) != len(rows) or not joined["_merge"].eq("both").all():
        raise ValueError(f"Reference predictions do not cover frozen scope: {path}")
    if not np.array_equal(
        joined["actual_new"].to_numpy(), joined["actual_old"].to_numpy()
    ):
        raise ValueError(f"Reference sales truth differs: {path}")
    if not np.isfinite(joined["prediction"].to_numpy()).all():
        raise ValueError(f"Reference predictions invalid: {path}")
    return old[[*KEY, "actual", "model", "prediction"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--raw-lora-report", type=Path, required=True)
    parser.add_argument("--raw-zero-shot-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    source_metadata = json.loads(
        (args.registry.parent / "metadata.json").read_text(encoding="utf-8")
    )
    if source_metadata["input_sha256"] != _sha256(args.input):
        raise ValueError("Guard registry was made from a different panel")
    raw_metadata = json.loads(
        (args.raw_lora_report / "metadata.json").read_text(encoding="utf-8")
    )
    if (
        not raw_metadata.get("score_valid")
        or raw_metadata.get("inference_mode") != "eval"
    ):
        raise ValueError("Raw LoRA report is not verified eval-mode inference")
    if raw_metadata["adapter_sha256"] != RAW_ADAPTER_SHA256:
        raise ValueError("Unexpected raw LoRA adapter")
    if _sha256(Path(raw_metadata["adapter_file"])) != RAW_ADAPTER_SHA256:
        raise ValueError("Raw LoRA adapter checksum mismatch")
    if raw_metadata["train_origins"] != [str(x.date()) for x in TRAIN_ORIGINS]:
        raise ValueError("Raw LoRA train origins differ")
    if raw_metadata["evaluation_origins"] != [str(x.date()) for x in EVAL_ORIGINS]:
        raise ValueError("Raw LoRA evaluation origins differ")
    if raw_metadata["num_steps"] != 1000 or raw_metadata["batch_size"] != 16:
        raise ValueError("Raw LoRA hyperparameters differ")
    if raw_metadata["model_revision"] != MODEL_VARIANTS["small"][2]:
        raise ValueError("Raw LoRA model revision differs")

    panel = validate_flows(pd.read_parquet(args.input))
    registry = pd.read_parquet(args.registry)
    train_cutoff = max(TRAIN_ORIGINS) + pd.Timedelta(days=HORIZON_DAYS)
    if train_cutoff >= min(EVAL_ORIGINS):
        raise ValueError("Train labels overlap evaluation")
    raw_train, raw_counts = training_history(
        panel, TRAIN_ORIGINS, start=pd.Timestamp("2026-01-01")
    )
    raw_arrays = training_arrays(raw_train)
    if _array_digest(raw_arrays) != RAW_TRAIN_SHA256:
        raise ValueError("Raw training arrays differ from frozen control")
    train_flows = guarded_flows(panel, registry, train_cutoff)
    train, train_counts = training_history(
        train_flows, TRAIN_ORIGINS, start=pd.Timestamp("2026-01-01")
    )
    if not train[["unique_id", "ds"]].equals(raw_train[["unique_id", "ds"]]):
        raise ValueError("Guard training array shape/order differs from raw")
    if train_counts != raw_counts or train["ds"].max() != train_cutoff:
        raise ValueError("Guard train scope or cutoff differs")
    arrays = training_arrays(train)
    affected = int((train["y"].to_numpy() != raw_train["y"].to_numpy()).sum())
    if affected == 0:
        raise ValueError("Guard did not change the training target")

    try:
        import peft
        import torch
        from chronos import Chronos2Pipeline
    except ImportError as exc:
        raise SystemExit("Chronos-2/PEFT research environment is required") from exc
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    _, model_repo, model_revision = MODEL_VARIANTS["small"]
    base = Chronos2Pipeline.from_pretrained(
        model_repo, revision=model_revision, device_map="cpu"
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    print(
        f"Training guarded LoRA: {len(arrays)} series, {affected} changed SKU-days",
        flush=True,
    )
    finetuned = base.fit(
        inputs=arrays,
        prediction_length=HORIZON_DAYS,
        context_length=INPUT_DAYS,
        finetune_mode="lora",
        learning_rate=1e-5,
        num_steps=1000,
        batch_size=16,
        output_dir=args.output_dir / "checkpoint",
        optim="adamw_torch",
        seed=42,
        disable_tqdm=True,
        remove_printer_callback=True,
    )
    require_eval_mode(finetuned)
    zero_shot = Chronos2Pipeline.from_pretrained(
        model_repo, revision=model_revision, device_map="cpu"
    )
    require_eval_mode(zero_shot)

    parts = []
    truth_rows = []
    for origin in EVAL_ORIGINS:
        _, rows = evaluation_history_and_rows(panel, origin)
        context_flows = guarded_flows(panel, registry, origin)
        context = dense_history(
            context_flows,
            rows[KEYS],
            origin - pd.Timedelta(days=INPUT_DAYS - 1),
            origin,
        )
        for pipeline, model_id in ((zero_shot, ZERO_SHOT_ID), (finetuned, GUARDED_ID)):
            predicted = pipeline.predict_df(
                context,
                id_column="unique_id",
                timestamp_column="ds",
                target="y",
                prediction_length=HORIZON_DAYS,
                quantile_levels=[0.5],
                batch_size=128,
            )
            parts.append(align_chronos_predictions(rows, predicted, model_id=model_id))
        truth_rows.append(rows)
        print(f"Scored origin {origin.date()} ({len(rows)} SKU-days)", flush=True)
    truth = pd.concat(truth_rows, ignore_index=True)
    raw_lora = _checked_existing(
        args.raw_lora_report / "detail.parquet", truth, raw_metadata["model"]
    )
    raw_zero_meta = json.loads(
        (args.raw_zero_shot_report / "metadata.json").read_text(encoding="utf-8")
    )
    if raw_zero_meta["model_revision"] != model_revision:
        raise ValueError("Raw zero-shot revision differs")
    raw_zero = _checked_existing(
        args.raw_zero_shot_report / "detail.parquet",
        truth,
        "new_chronos2_small_zero_shot_h14",
    )
    detail = pd.concat([*parts, raw_lora, raw_zero], ignore_index=True)
    baseline = truth[[*KEY, "actual"]].copy()
    baseline["model"] = "weighted_weekday_sales"
    baseline["prediction"] = truth["weighted_weekday_sales"].to_numpy()
    detail = pd.concat([detail, baseline], ignore_index=True)
    if detail.duplicated([*KEY, "model"]).any():
        raise ValueError("Duplicate evaluation predictions")
    detail.to_parquet(args.output_dir / "detail.parquet", index=False)
    leaderboard = _summary(detail, ["model"])
    leaderboard.to_csv(args.output_dir / "leaderboard.csv", index=False)
    _summary(detail, ["forecast_origin", "model"]).to_csv(
        args.output_dir / "by_origin.csv", index=False
    )
    metadata = {
        "production_write": False,
        "score_valid": True,
        "primary_evaluation_target": "unchanged_observed_sales_qty",
        "training_target": "observed_sales_plus_half_guarded_uplift",
        "training_target_is_proxy_not_ground_truth": True,
        "guard_availability_lag_days": 7,
        "train_origins": [str(x.date()) for x in TRAIN_ORIGINS],
        "train_cutoff": str(train_cutoff.date()),
        "evaluation_origins": [str(x.date()) for x in EVAL_ORIGINS],
        "train_counts": train_counts,
        "changed_training_sku_days": affected,
        "raw_training_sha256": RAW_TRAIN_SHA256,
        "guarded_training_sha256": _array_digest(arrays),
        "num_steps": 1000,
        "batch_size": 16,
        "seed": 42,
        "learning_rate": 1e-5,
        "model_repo": model_repo,
        "model_revision": model_revision,
        "inference_mode": "eval",
        "panel_sha256": _sha256(args.input),
        "registry_sha256": _sha256(args.registry),
        "raw_lora_detail_sha256": _sha256(args.raw_lora_report / "detail.parquet"),
        "raw_zero_detail_sha256": _sha256(args.raw_zero_shot_report / "detail.parquet"),
        "adapter_sha256": _sha256(
            args.output_dir
            / "checkpoint"
            / "finetuned-ckpt"
            / "adapter_model.safetensors"
        ),
        "chronos_forecasting_version": version("chronos-forecasting"),
        "peft_version": peft.__version__,
        "torch_version": torch.__version__,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(leaderboard.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

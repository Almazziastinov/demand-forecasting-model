"""Make local-only forward Chronos-2 forecasts from a frozen fact snapshot.

No future labels are read or scored. The first already-started Moscow business
date is excluded so the report contains only prospective target dates.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.finetune_chronos2_fixed_origin import require_eval_mode  # noqa: E402
from scripts.run_chronos2_fixed_origin import MODEL_VARIANTS, _sha256  # noqa: E402
from src.model_tournament.fixed_origin import validate_flows  # noqa: E402
from src.model_tournament.nhits_candidate import (  # noqa: E402
    evaluation_history_and_rows,
)


KEYS = ["date", "bakery_id", "product_id"]


def align_shadow_predictions(
    rows: pd.DataFrame, predicted: pd.DataFrame, model_id: str
) -> pd.DataFrame:
    """Match every frozen SKU-day, without carrying synthetic future truth."""
    required = {"unique_id", "ds", "0.5"}
    if not required.issubset(predicted.columns):
        raise ValueError("Chronos output lacks required prediction columns")
    if predicted.duplicated(["unique_id", "ds"]).any():
        raise ValueError("Chronos output has duplicate SKU-day keys")
    expected = rows[KEYS].copy()
    expected["unique_id"] = (
        expected["bakery_id"].astype(str) + ":" + expected["product_id"].astype(str)
    )
    result = expected.merge(
        predicted[["unique_id", "ds", "0.5"]],
        left_on=["unique_id", "date"],
        right_on=["unique_id", "ds"],
        how="left",
        validate="one_to_one",
    )
    values = pd.to_numeric(result["0.5"], errors="coerce").to_numpy(dtype=float)
    if len(predicted) != len(expected) or not np.isfinite(values).all():
        raise ValueError("Chronos output does not cover the frozen SKU-day scope")
    output = result[KEYS].copy()
    output["model"] = model_id
    output["prediction"] = np.maximum(values, 0.0)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-input", type=Path, required=True)
    parser.add_argument("--adapter-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    frozen_metadata = json.loads(
        (args.frozen_input / "metadata.json").read_text(encoding="utf-8")
    )
    input_file = args.frozen_input / "frozen_flows.parquet"
    if _sha256(input_file) != frozen_metadata["frozen_flows_sha256"]:
        raise ValueError("Frozen fact file hash differs from metadata")
    origin = pd.Timestamp(frozen_metadata["origin"])
    prospective_start = pd.Timestamp(frozen_metadata["first_prospective_date"])
    if (
        prospective_start <= origin
        or prospective_start > origin + pd.Timedelta(days=14)
    ):
        raise ValueError("No prospective target dates remain")
    if datetime.now(ZoneInfo("Europe/Moscow")).date() >= prospective_start.date():
        raise ValueError("Prospective start has passed; freeze a new shadow input")
    flows = validate_flows(pd.read_parquet(input_file))
    if flows["date"].max() != origin:
        raise ValueError("Frozen facts must end exactly at the declared origin")
    context, rows = evaluation_history_and_rows(flows, origin)
    if rows["date"].min() != origin + pd.Timedelta(days=1):
        raise ValueError("Unexpected forecast window")

    adapter_metadata = json.loads(
        (args.adapter_report / "metadata.json").read_text(encoding="utf-8")
    )
    _, model_repo, model_revision = MODEL_VARIANTS["small"]
    if (
        not adapter_metadata.get("score_valid")
        or adapter_metadata["model_repo"] != model_repo
        or adapter_metadata["model_revision"] != model_revision
        or adapter_metadata["num_steps"] != 1000
    ):
        raise ValueError("Adapter is not the verified 1,000-step Small checkpoint")
    adapter_file = Path(adapter_metadata["adapter_file"])
    if _sha256(adapter_file) != adapter_metadata["adapter_sha256"]:
        raise ValueError("Saved adapter hash differs from verified report")

    try:
        import torch
        from chronos import Chronos2Pipeline
        from peft import PeftModel
    except ImportError as exc:
        raise SystemExit("Research environment needs Chronos-2 and PEFT") from exc
    torch.set_num_threads(4)
    base = Chronos2Pipeline.from_pretrained(
        model_repo, revision=model_revision, device_map="cpu"
    )
    require_eval_mode(base)
    zero_predicted = base.predict_df(
        context, id_column="unique_id", timestamp_column="ds", target="y",
        prediction_length=14, quantile_levels=[0.5], batch_size=args.batch_size,
    )
    zero = align_shadow_predictions(rows, zero_predicted, "chronos2_small_zero_shot")
    adapted = Chronos2Pipeline(
        model=PeftModel.from_pretrained(base.model, adapter_file.parent)
    )
    require_eval_mode(adapted)
    lora_predicted = adapted.predict_df(
        context, id_column="unique_id", timestamp_column="ds", target="y",
        prediction_length=14, quantile_levels=[0.5], batch_size=args.batch_size,
    )
    lora = align_shadow_predictions(rows, lora_predicted, "chronos2_small_lora_1000")
    prospect = pd.concat([zero, lora], ignore_index=True)
    prospect = prospect.loc[prospect["date"] >= prospective_start].copy()
    if prospect.empty:
        raise ValueError("No prospective predictions remain")
    incumbent = pd.read_parquet(args.frozen_input / "incumbent.parquet")
    incumbent["date"] = pd.to_datetime(incumbent["date"]).dt.normalize()
    incumbent = incumbent.loc[
        incumbent["date"] >= prospective_start, KEYS + ["prediction"]
    ]
    common = prospect[KEYS].drop_duplicates().merge(incumbent, on=KEYS, how="inner")
    incumbent_rows = common[KEYS + ["prediction"]].copy()
    incumbent_rows["model"] = "incumbent_active_at_freeze"
    prospect = pd.concat([prospect, incumbent_rows], ignore_index=True)
    forecast_finished = datetime.now(ZoneInfo("UTC"))
    if (
        forecast_finished.astimezone(ZoneInfo("Europe/Moscow")).date()
        >= prospective_start.date()
    ):
        raise ValueError("Forecast completed after the prospective window began")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    prospect.to_parquet(args.output_dir / "predictions.parquet", index=False)
    summary = (
        prospect.groupby("model", as_index=False)
        .agg(rows=("prediction", "size"), prediction_sum=("prediction", "sum"))
    )
    summary.to_csv(args.output_dir / "forecast_summary.csv", index=False)
    metadata = {
        "production_write": False,
        "score_valid": None,
        "reason_unscored": "future observed-sales labels not yet available",
        "origin": str(origin.date()),
        "first_prospective_date": str(prospective_start.date()),
        "last_prospective_date": str(rows["date"].max().date()),
        "forecast_finished_utc": forecast_finished.isoformat(),
        "model_repo": model_repo,
        "model_revision": model_revision,
        "adapter_sha256": adapter_metadata["adapter_sha256"],
        "frozen_flows_sha256": frozen_metadata["frozen_flows_sha256"],
        "active_run_id_at_freeze": frozen_metadata["active_run_id_at_extraction"],
        "chronos_scope_pairs": len(rows) // 14,
        "incumbent_overlap_rows": len(common),
        "forecast_rows": len(prospect),
        "training_target": "observed_sales_qty",
        "production_target_and_scope_parity_verified": False,
        "as_of_origin_fact_arrival_verified": False,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
